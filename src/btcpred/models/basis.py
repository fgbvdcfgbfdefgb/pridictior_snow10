"""Smooth function basis for the 1500-step forecast path.

The predictor does not emit 1500 independent numbers. It emits ``K`` (default
48) coefficients of a fixed smooth basis, and the path is synthesised as
``path = coeffs @ B``.

Three reasons, all of which the brief asks for directly:

1. **Stability.** The brief says predictions drive real trading and must not
   jump. A head with 1500 free outputs can produce a sawtooth; a 48-term
   band-limited expansion physically cannot. Smoothness is a property of the
   parameterisation, not a penalty we hope the optimiser respects.
2. **Capacity.** 1500 outputs from a 1536-wide model is a 2.3M-parameter head
   that mostly learns to copy its neighbours. 48 outputs is 74k.
3. **Conditioning.** Adjacent horizons are ~0.999 correlated. Regressing them
   independently spends the whole gradient budget on a direction the data
   barely distinguishes.

The basis is a half-cycle sine family, ``phi_k(h) = sin(pi (k + 1/2) (h+1) / H)``.
It vanishes at ``h = -1`` (the forecast origin), so every synthesised path
starts at the anchor price by construction and cannot open with a jump
discontinuity — a free, hard guarantee that a plain MLP head does not give you.

Coefficients are scaled by ``1 / (k + 1)`` before synthesis so that producing a
high-frequency wiggle requires a correspondingly large logit. The model can
still do it when the data demands, but the prior leans smooth.
"""

from __future__ import annotations

import functools

import numpy as np
import torch


@functools.lru_cache(maxsize=8)
def sine_basis(horizon: int, n_coeffs: int) -> np.ndarray:
    """``[n_coeffs, horizon]`` float32 synthesis matrix, already decay-scaled."""
    h = np.arange(1, horizon + 1, dtype=np.float64)[None, :]
    k = np.arange(n_coeffs, dtype=np.float64)[:, None]
    phi = np.sin(np.pi * (k + 0.5) * h / horizon)
    decay = 1.0 / (k + 1.0)
    return (phi * decay).astype(np.float32)


class PathSynthesiser(torch.nn.Module):
    """Turns coefficient vectors into cumulative log-return paths.

    Input ``[..., n_coeffs]`` -> output ``[..., horizon]``, where element ``h``
    is the predicted ``log(P[t+1+h]) - log(P[t])``.
    """

    def __init__(self, horizon: int = 1500, n_coeffs: int = 48) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_coeffs = n_coeffs
        self.register_buffer(
            "basis", torch.from_numpy(sine_basis(horizon, n_coeffs)), persistent=False
        )

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.shape[-1] != self.n_coeffs:
            raise ValueError(
                f"expected {self.n_coeffs} coefficients, got {coeffs.shape[-1]}"
            )
        return coeffs @ self.basis

    def project(self, path: torch.Tensor) -> torch.Tensor:
        """Least-squares inverse: path ``[..., horizon]`` -> coeffs.

        Used by the self-distillation variant, which needs to express a
        teacher's realised path in the student's coefficient space, and by the
        diagnostics that report how much of the true path the basis can
        represent at all.
        """
        b = self.basis  # [K, H]
        gram = b @ b.transpose(0, 1)  # [K, K]
        rhs = path @ b.transpose(0, 1)  # [..., K]
        return torch.linalg.solve(gram.unsqueeze(0), rhs.unsqueeze(-1)).squeeze(-1)


def representable_fraction(paths: np.ndarray, horizon: int, n_coeffs: int) -> float:
    """Share of path variance the basis can capture. A sanity check, not a loss.

    If this is below ~0.97 on real data the basis is too narrow and the model
    is being handicapped before training starts; raise ``n_coeffs``.
    """
    b = sine_basis(horizon, n_coeffs).astype(np.float64)
    gram = b @ b.T
    coeffs = np.linalg.solve(gram, (paths @ b.T).T).T
    recon = coeffs @ b
    resid = ((paths - recon) ** 2).sum()
    total = (paths**2).sum()
    return float(1.0 - resid / max(total, 1e-12))
