"""Price Predictor — the GPU half of the model.

Consumes the Market Analyser's output *and* the raw tokenised 12-hour window
(the brief asks for both), and emits the next 25 minutes of log-returns as
coefficients of the smooth basis in :mod:`btcpred.models.basis`.

Architecture is a pre-norm transformer encoder with SwiGLU feed-forwards and
RoPE positional encoding. Two details are specific to this problem:

**Dual input path.** The analyser memory is the processed view; the raw tokens
are projected in alongside it and summed. If the predictor could only see the
analyser's output, the analyser would be an information bottleneck that early
training has no way to widen — and the gradient reaching the CPU stack would
be the only thing determining whether the predictor ever sees clean data. The
skip connection makes the analyser *additive*: worst case it contributes
nothing and the predictor still works; best case it does the heavy lifting.

**Non-uniform positions.** Tokens are not equally spaced in time (300 of them
cover one second each, 120 cover five minutes each). Feeding an ordinal index
to RoPE would tell the model that the gap between token 0 and 1 equals the gap
between token 659 and 658, which is wrong by a factor of 300. We feed RoPE the
*log of seconds-before-now* instead, so geometric time distance maps to linear
phase — distances the model reasons about match distances in the market.

Three heads are available; which one a run uses is what makes the ensemble
members genuinely different rather than four seeds of the same thing:

``point``
    ``K`` coefficients, trained with Huber on the path. Fast, sharp, the
    baseline.
``quantile``
    ``3K`` coefficients -> p10 / p50 / p90 paths, pinball loss. Gives the
    trading layer an honest uncertainty band instead of a false point.
``distribution``
    ``K`` median coefficients plus a categorical over terminal log-return
    bins, trained with a soft (label-smoothed-to-Gaussian) cross-entropy.
    Handles the fat tails that a squared or Huber loss quietly ignores.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .basis import PathSynthesiser

HeadType = Literal["point", "quantile", "distribution"]
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    positions: torch.Tensor, head_dim: int, base: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for arbitrary real-valued positions.

    ``positions``: ``[T]`` float. Unlike the usual integer-index RoPE this
    accepts the log-time coordinates described in the module docstring.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, dtype=torch.float32) / half))
    ang = positions.float()[:, None] * inv_freq[None, :]  # [T, half]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x``: ``[B, H, T, D]``; ``cos``/``sin``: ``[T, D/2]``."""
    x1, x2 = x.chunk(2, dim=-1)
    c = cos.to(x.dtype)[None, None]
    s = sin.to(x.dtype)[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Bidirectional over the *context* window: every token in it is in the
        # past, so there is nothing to mask. Causality is enforced by which
        # seconds the replay lets us read, not by an attention mask.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: int):
        super().__init__()
        hidden = int(2 * mult * d_model / 3 / 64 + 0.5) * 64  # keep it matmul-friendly
        self.w12 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float,
                 n_layers: int):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.n2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_mult)
        self.drop = nn.Dropout(dropout)
        # Scale residual branches by 1/sqrt(2L) so activation variance stays
        # O(1) through a deep stack without a warmup-dependent hack.
        scale = (2.0 * n_layers) ** -0.5
        with torch.no_grad():
            self.attn.proj.weight.mul_(scale)
            self.ffn.w3.weight.mul_(scale)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.n1(x), cos, sin))
        return x + self.drop(self.ffn(self.n2(x)))


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class PricePredictor(nn.Module):
    """Predicts the next ``horizon`` seconds of log-return from the anchor.

    Parameters
    ----------
    token_offsets:
        ``[n_tokens]`` seconds-before-cursor at which each token's bin ends,
        from :meth:`PyramidSpec.token_end_offsets`. Drives the log-time RoPE.
    n_bins:
        Size of the terminal-return categorical, ``distribution`` head only.
    """

    def __init__(
        self,
        token_offsets: torch.Tensor,
        n_features: int = 14,
        d_analyser: int = 512,
        d_state: int = 512,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        ffn_mult: int = 4,
        n_coeffs: int = 48,
        horizon: int = 1500,
        head: HeadType = "point",
        dropout: float = 0.0,
        n_bins: int = 101,
        bin_range_bps: float = 400.0,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.head_type = head
        self.horizon = horizon
        self.n_coeffs = n_coeffs
        self.grad_checkpointing = grad_checkpointing
        self.n_bins = n_bins

        self.in_raw = nn.Linear(n_features, d_model)
        self.in_mem = nn.Linear(d_analyser, d_model)
        self.state_token = nn.Linear(d_state, d_model)

        # Log-time positions. +1 keeps log finite at offset 0 (the newest
        # token); the 0.5 scale spreads 0..12h over ~5 radians of base phase,
        # which empirically keeps attention logits in a sane range.
        pos = torch.log1p(token_offsets.float()) * 0.5
        # The state token is prepended; give it its own position below all
        # real tokens so it is never confused with a point in time.
        pos = torch.cat([torch.full((1,), -1.0), pos])
        cos, sin = build_rope_cache(pos, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, ffn_mult, dropout, n_layers) for _ in range(n_layers)
        )
        self.final_norm = RMSNorm(d_model)
        self.synth = PathSynthesiser(horizon, n_coeffs)

        n_out = {"point": n_coeffs, "quantile": len(QUANTILES) * n_coeffs,
                 "distribution": n_coeffs}[head]
        # Pool = [state token, newest token, mean over tokens].
        self.head = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, n_out)
        )
        # Start at "no move predicted". A head that starts with random
        # coefficients emits multi-percent moves on step 0 and the first few
        # hundred updates are spent undoing that.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        if head == "distribution":
            self.bin_head = nn.Sequential(
                nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, n_bins)
            )
            edges = torch.linspace(-bin_range_bps, bin_range_bps, n_bins) / 10_000.0
            self.register_buffer("bin_centres", edges, persistent=False)

    # ------------------------------------------------------------------ #
    def forward(
        self, tokens: torch.Tensor, memory: torch.Tensor, state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        ``tokens``  ``[B, T, n_features]``  raw pyramid tokens
        ``memory``  ``[B, T, d_analyser]``  analyser per-token output
        ``state``   ``[B, d_state]``        analyser pooled summary
        """
        x = self.in_raw(tokens) + self.in_mem(memory)
        x = torch.cat([self.state_token(state).unsqueeze(1), x], dim=1)

        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.final_norm(x)

        pooled = torch.cat([x[:, 0], x[:, -1], x[:, 1:].mean(dim=1)], dim=-1)
        raw = self.head(pooled)

        out: dict[str, torch.Tensor] = {}
        if self.head_type == "point":
            out["coeffs"] = raw
            out["path"] = self.synth(raw)
        elif self.head_type == "quantile":
            coeffs = raw.view(-1, len(QUANTILES), self.n_coeffs)
            paths = self.synth(coeffs)  # [B, Q, H]
            # Enforce monotone quantiles by construction: predict p50 plus
            # non-negative spreads. A model free to cross its own quantiles
            # will, and a crossed band is worse than no band for sizing.
            p50 = paths[:, 1]
            lo = p50 - F.softplus(p50 - paths[:, 0])
            hi = p50 + F.softplus(paths[:, 2] - p50)
            out["coeffs"] = coeffs
            out["quantile_paths"] = torch.stack([lo, p50, hi], dim=1)
            out["path"] = p50
        else:  # distribution
            out["coeffs"] = raw
            out["path"] = self.synth(raw)
            out["bin_logits"] = self.bin_head(pooled)
        return out

    # ------------------------------------------------------------------ #
    def expected_terminal(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean terminal log-return implied by the ``distribution`` head."""
        if "bin_logits" not in out:
            return out["path"][:, -1]
        p = F.softmax(out["bin_logits"].float(), dim=-1)
        return (p * self.bin_centres[None, :]).sum(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Decay matrices, never norms or biases — the usual, but explicit.

        Applying weight decay to RMSNorm gains pulls them toward zero and
        slowly strangles the residual stream; it is a classic silent
        regression in from-scratch transformer code.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 else decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


def soft_bin_targets(
    terminal: torch.Tensor, centres: torch.Tensor, sigma_bps: float = 8.0
) -> torch.Tensor:
    """Gaussian-smoothed one-hot target over return bins (HL-Gauss).

    A hard one-hot over 101 bins throws away the fact that neighbouring bins
    are nearly as correct, which makes the categorical loss needlessly
    high-variance. Smoothing with a Gaussian of the measurement scale recovers
    most of the efficiency of regression while keeping the tails.
    """
    sigma = sigma_bps / 10_000.0
    d = (terminal[:, None] - centres[None, :]) / sigma
    logits = -0.5 * d.pow(2)
    return torch.softmax(logits, dim=-1)
