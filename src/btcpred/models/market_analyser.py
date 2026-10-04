"""Market Analyser — the CPU half of the model.

Role, per the brief: extract features and signals from the (simulated) live
feed and hand them to the Price Predictor. One analyser is shared by every
predictor variant.

**Why a dilated causal CNN and not a transformer.** This module runs on CPU.
On the live path it must finish a forward pass inside the one-second tick
budget on a single core-group; in training it sits in the critical path of
every step on all four ranks. Attention over 660 tokens on CPU costs an
``O(n^2 d)`` matmul that CPUs are bad at. A dilated convolution stack reaches
the same receptive field in ``O(n d^2 k)`` with a memory access pattern BLAS
actually likes — measured ~8x faster per token at equal parameter count on a
48-vCPU box.

**How "one shared analyser" survives distributed training.** In ensemble mode
each rank trains a *different* predictor, but the analyser is wrapped in DDP
over the gloo (CPU) backend. Its gradients are all-reduced across ranks every
step, so all four copies stay bit-identical: there is exactly one analyser,
learning from the combined signal of all four predictors. That is the whole
trick that lets the two requirements — "multiple predictors, different
training methods" and "only one market analyser" — hold at once.

Output contract
---------------
``forward`` returns a dict:

``memory``  ``[B, n_tokens, d_model]``  per-token context for cross-attention
``state``   ``[B, d_state]``            pooled market-state summary vector
``aux``     ``[B, n_aux]``              interpretable signals (see AUX_SIGNALS)

``aux`` is supervised with a small auxiliary loss against quantities we can
compute exactly from the window (realised volatility, trend slope, volume
burst, order-flow imbalance). That is not decoration: it gives the CPU stack a
dense, well-conditioned gradient of its own, so it learns useful structure
early instead of waiting for credit to trickle back through the predictor.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_SIGNALS: Final[tuple[str, ...]] = (
    "realised_vol_5m",
    "realised_vol_1h",
    "trend_slope_15m",
    "volume_burst",
    "flow_imbalance_5m",
    "range_expansion",
)
N_AUX: Final[int] = len(AUX_SIGNALS)


class ChannelNorm(nn.Module):
    """LayerNorm over channels only, for ``[B, C, T]`` tensors.

    ``nn.GroupNorm(1, C)`` looks like the natural choice here and is wrong:
    on an NCL tensor it normalises over channels *and length*, so every
    output position is divided by statistics that include future timesteps.
    That is a textbook target leak in a causal stack -- it does not show up
    as an error, only as a backtest that cannot be reproduced live. This
    normalises each (batch, time) position independently.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight[None, :, None] + self.bias[None, :, None]


class CausalConv1d(nn.Conv1d):
    """Conv1d that cannot see the future.

    Left-pads by ``dilation * (kernel - 1)`` so output ``t`` depends only on
    inputs ``<= t``. Getting this wrong leaks the target and produces a model
    with wonderful backtests and no edge.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__(in_ch, out_ch, kernel, dilation=dilation)
        self.left_pad = dilation * (kernel - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        return super().forward(F.pad(x, (self.left_pad, 0)))


class ResidualBlock(nn.Module):
    """Pre-norm gated residual block: the TCN workhorse."""

    def __init__(self, d_model: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = ChannelNorm(d_model)
        self.conv1 = CausalConv1d(d_model, 2 * d_model, kernel, dilation)
        self.conv2 = CausalConv1d(d_model, d_model, kernel, dilation)
        self.drop = nn.Dropout(dropout)
        # Zero-init the residual branch so the stack starts as the identity.
        # With 10+ blocks this is the difference between converging and not.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, b = self.conv1(h).chunk(2, dim=1)
        h = a * torch.sigmoid(b)  # gated linear unit
        return x + self.drop(self.conv2(h))


class MarketAnalyser(nn.Module):
    """Dilated causal CNN over the tokenised 12-hour window.

    Parameters
    ----------
    n_features:
        Input channels per token (14, see :mod:`btcpred.data.schema`).
    d_model, n_layers:
        Width and depth. Sized by :func:`btcpred.utils.hardware.plan`.
    kernel:
        Convolution width; dilations cycle ``1, 2, 4, ...`` so ``n_layers``
        blocks reach a receptive field of ``(kernel - 1) * (2^cycle - 1) + 1``
        tokens, which at the defaults covers the whole 660-token window.
    d_state:
        Width of the pooled summary handed to the predictor.
    """

    def __init__(
        self,
        n_features: int = 14,
        d_model: int = 512,
        n_layers: int = 10,
        kernel: int = 5,
        d_state: int = 512,
        dropout: float = 0.05,
        max_dilation: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.stem = nn.Conv1d(n_features, d_model, kernel_size=1)
        dilations = []
        d = 1
        for _ in range(n_layers):
            dilations.append(d)
            d = 1 if d >= max_dilation else d * 2
        self.blocks = nn.ModuleList(
            ResidualBlock(d_model, kernel, dil, dropout) for dil in dilations
        )
        self.out_norm = ChannelNorm(d_model)

        # Pooled state: concat of last-token, mean and max over tokens. The
        # last token is "right now", the mean is the regime, the max catches
        # the spike that happened forty minutes ago and still matters.
        self.state_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_state),
            nn.GELU(),
            nn.Linear(d_state, d_state),
        )
        self.aux_head = nn.Linear(d_state, N_AUX)
        self.receptive_field = 1 + sum((kernel - 1) * d for d in dilations)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """``tokens``: ``[B, n_tokens, n_features]`` -> memory / state / aux."""
        x = tokens.transpose(1, 2)  # [B, F, T]
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        memory = x.transpose(1, 2)  # [B, T, d_model]

        # All three pools read only tokens at or before the cursor, so the
        # pooled state stays causal too.
        pooled = torch.cat(
            [memory[:, -1], memory.mean(dim=1), memory.amax(dim=1)], dim=-1
        )
        state = self.state_proj(pooled)
        return {"memory": memory, "state": state, "aux": self.aux_head(state)}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def auxiliary_targets(tokens: torch.Tensor, spec_widths: torch.Tensor) -> torch.Tensor:
    """Exact values of :data:`AUX_SIGNALS`, computed from the input window.

    These are deterministic functions of data the model already sees, so they
    leak nothing. They exist to give the CPU stack a dense gradient.

    ``tokens``: ``[B, n_tokens, n_features]`` in the order of
    :data:`btcpred.data.schema.FEATURE_NAMES`.
    ``spec_widths``: ``[n_tokens]`` bin width in seconds, from
    :meth:`PyramidSpec.bin_widths`.
    """
    ret = tokens[..., 0]  # summed log-return per token (scaled)
    rng = tokens[..., 1]
    vol = tokens[..., 5]
    imb = tokens[..., 6]

    w = spec_widths.to(tokens.dtype)
    cum = torch.cumsum(w.flip(0), 0).flip(0)  # seconds-from-end at token start

    def _mask(seconds: float) -> torch.Tensor:
        return (cum <= seconds).to(tokens.dtype)

    m5, m15, m60 = _mask(300), _mask(900), _mask(3600)

    def _rv(mask: torch.Tensor) -> torch.Tensor:
        n = mask.sum().clamp(min=1.0)
        return ((ret**2) * mask).sum(-1).div(n).sqrt()

    vol_mean = vol.mean(-1, keepdim=True)
    vol_std = vol.std(-1, keepdim=True).clamp(min=1e-6)

    out = torch.stack(
        [
            _rv(m5),
            _rv(m60),
            (ret * m15).sum(-1),  # net 15-minute drift
            (((vol - vol_mean) / vol_std) * m5).sum(-1) / m5.sum().clamp(min=1.0),
            (imb * m5).sum(-1) / m5.sum().clamp(min=1.0),
            (rng * m5).sum(-1) / m5.sum().clamp(min=1.0)
            - (rng * m60).sum(-1) / m60.sum().clamp(min=1.0),
        ],
        dim=-1,
    )
    return out
