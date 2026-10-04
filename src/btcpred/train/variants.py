"""The ensemble: several predictors, genuinely different training methods.

The brief asks for multiple Price Predictor instances trained by *different
methods*, sharing one Market Analyser. Four seeds of the same recipe would
satisfy the letter of that and none of the value — correlated models average
to the same model. These four differ along the three axes that actually
decorrelate a forecaster:

1. **What it predicts** — a point path, a quantile band, or a full terminal
   distribution. Different output geometry means different failure modes.
2. **How it is optimised** — AdamW at different betas and learning rates,
   plus one member on a slower, EMA-distilled trajectory.
3. **What it is penalised for** — the stability-weighted member pays five
   times as much for flickering, and ends up much calmer and slightly blunter.

With four A10s, rank ``k`` trains variant ``k``. With fewer GPUs the variants
are assigned round-robin; with more, extras duplicate with different seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..models.price_predictor import HeadType
from .objective import ObjectiveConfig


@dataclass
class VariantConfig:
    name: str
    head: HeadType = "point"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    #: Exponential-moving-average teacher. ``0`` disables it.
    ema_decay: float = 0.0
    #: Weight on the student-teacher agreement term when ``ema_decay > 0``.
    distill_weight: float = 0.0
    dropout: float = 0.0
    #: Per-variant overrides merged onto the base objective config.
    objective_overrides: dict = field(default_factory=dict)
    seed: int = 0

    def objective(self, base: ObjectiveConfig) -> ObjectiveConfig:
        return replace(base, **self.objective_overrides)


DEFAULT_VARIANTS: list[VariantConfig] = [
    VariantConfig(
        name="point-adamw",
        head="point",
        lr=3e-4,
        betas=(0.9, 0.95),
        seed=1,
    ),
    VariantConfig(
        name="quantile-pinball",
        head="quantile",
        lr=2e-4,
        betas=(0.9, 0.98),
        weight_decay=0.02,
        seed=2,
        # The pinball term already supplies most of the path gradient, so the
        # Huber path weight is dialled back to stop the two fighting.
        objective_overrides={"w_path": 0.5},
    ),
    VariantConfig(
        name="distribution-hlgauss",
        head="distribution",
        lr=2.5e-4,
        betas=(0.9, 0.95),
        seed=3,
        # The categorical head owns the terminal question; let it.
        objective_overrides={"w_terminal": 1.0, "w_bins": 1.5},
    ),
    VariantConfig(
        name="point-ema-stable",
        head="point",
        lr=1.5e-4,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        ema_decay=0.9995,
        distill_weight=0.5,
        dropout=0.05,
        seed=4,
        # The calm member: heavy consistency, heavy jump penalty. Expect
        # slightly worse MAE and markedly better stability, which is the
        # trade the live trading layer usually wants.
        objective_overrides={"w_consistency": 2.5, "w_jump": 0.25},
    ),
]


def variant_for_rank(rank: int, variants: list[VariantConfig] | None = None) -> VariantConfig:
    variants = variants or DEFAULT_VARIANTS
    base = variants[rank % len(variants)]
    if rank >= len(variants):
        # Extra ranks duplicate a recipe but must not duplicate its seed, or
        # the "ensemble" silently collapses to fewer effective members.
        return replace(base, name=f"{base.name}-r{rank}", seed=base.seed + 100 * rank)
    return base
