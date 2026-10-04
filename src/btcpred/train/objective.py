"""The real-time reward / loss function.

There are no epochs. At every simulated market second the model issues a
1500-step forecast, and that forecast is scored against what the market
actually did in the following 1500 seconds — the real, non-simulated archive
that was downloaded in Phase 1. The scalar that comes out is both the reward
signal and the training loss.

Five terms, each earning its place:

``path``
    Huber on the predicted vs realised cumulative log-return path. Huber, not
    MSE: one-second crypto returns have tails that make a squared loss chase
    outliers and flatten everything else. ``delta`` is set in units of
    log-return so it is interpretable as "errors beyond this many bps stop
    being penalised quadratically".

``terminal``
    Extra weight on ``h = 1500`` itself. The brief's actual question is "where
    is the price in 25 minutes"; the path is a means to that end, and without
    this term the loss is dominated by the 1499 easy near-horizons.

``direction``
    Soft sign agreement on the terminal move. MAE in bps can be excellent
    while directional accuracy sits at 50%, and a trading model that is
    beautifully calibrated about magnitude and wrong about sign loses money.

``consistency``
    The stability requirement, as a loss. At second ``t`` the model's forecast
    for wall-clock time ``t + k`` should agree with what it said about the same
    wall-clock instant one second ago. We shift the previous (detached)
    prediction by one step and penalise disagreement on the 1499-step overlap.
    This is what stops the forecast flickering between ticks.

``jump``
    Second difference of the path. The basis already bounds curvature, so this
    is a light belt-and-braces term that mostly matters early in training.

Weighting between them is in :class:`ObjectiveConfig`; the defaults were
chosen so that at initialisation every term contributes within an order of
magnitude of the others, which is the only principled starting point when you
cannot tune on a validation set you do not yet have.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..models.price_predictor import QUANTILES, soft_bin_targets


@dataclass
class ObjectiveConfig:
    huber_delta_bps: float = 15.0
    w_path: float = 1.0
    w_terminal: float = 2.0
    w_direction: float = 0.3
    w_consistency: float = 0.5
    w_jump: float = 0.05
    w_aux: float = 0.1
    w_bins: float = 1.0
    direction_temp_bps: float = 5.0
    #: Down-weight targets built from forward-filled (no-trade) seconds.
    #: 0 would discard them entirely; 0.25 keeps a little signal because a
    #: genuinely silent market is itself information.
    filled_weight: float = 0.25
    #: Horizon weighting: emphasise later horizons slightly, since they carry
    #: the decision-relevant information and are intrinsically harder.
    horizon_tilt: float = 0.5
    quantiles: tuple[float, ...] = field(default=QUANTILES)


def horizon_weights(horizon: int, tilt: float, device, dtype) -> torch.Tensor:
    """``[H]`` weights rising from ~1 at h=0 to ~1+tilt at h=H, mean-normalised."""
    h = torch.linspace(0.0, 1.0, horizon, device=device, dtype=dtype)
    w = 1.0 + tilt * h
    return w / w.mean()


def _huber(err: torch.Tensor, delta: float) -> torch.Tensor:
    a = err.abs()
    return torch.where(a <= delta, 0.5 * err.pow(2) / delta, a - 0.5 * delta)


class RealTimeObjective:
    """Stateful scorer: holds the previous prediction for the consistency term.

    One instance per training process. ``prev`` is kept detached and shaped
    ``[B, H]``; stream ``k`` of the batch is the same market stream from one
    step to the next (:class:`ParallelReplay` guarantees this), which is what
    makes the one-step shift meaningful.
    """

    def __init__(self, cfg: ObjectiveConfig, horizon: int = 1500) -> None:
        self.cfg = cfg
        self.horizon = horizon
        self.prev: torch.Tensor | None = None
        self.prev_index: torch.Tensor | None = None
        self.prev_realised: torch.Tensor | None = None

    def reset(self) -> None:
        self.prev = None
        self.prev_index = None
        self.prev_realised = None

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        out: dict[str, torch.Tensor],
        target_path: torch.Tensor,
        fill_frac: torch.Tensor,
        index: torch.Tensor,
        aux_pred: torch.Tensor | None = None,
        aux_true: torch.Tensor | None = None,
        bin_centres: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Score one batch of forecasts.

        ``out``          predictor output dict
        ``target_path``  ``[B, H]`` realised log(P[t+1+h]) - log(P[t])
        ``fill_frac``    ``[B]`` fraction of the target window that was filled
        ``index``        ``[B]`` cache index of each stream's cursor
        """
        cfg = self.cfg
        pred = out["path"]
        b, h = pred.shape
        dev, dt = pred.device, pred.dtype
        delta = cfg.huber_delta_bps / 10_000.0

        hw = horizon_weights(h, cfg.horizon_tilt, dev, dt)
        # Sample weight: trust targets made of real prints more than filled ones.
        sw = (1.0 - fill_frac * (1.0 - cfg.filled_weight)).to(dt).clamp(min=0.0)
        sw = sw / sw.mean().clamp(min=1e-6)

        logs: dict[str, float] = {}

        # --- path -----------------------------------------------------
        err = pred - target_path
        path_loss = (_huber(err, delta) * hw[None, :]).mean(dim=1)
        path_loss = (path_loss * sw).mean()
        total = cfg.w_path * path_loss
        logs["loss/path"] = float(path_loss.detach())

        # --- terminal -------------------------------------------------
        term_err = pred[:, -1] - target_path[:, -1]
        term_loss = (_huber(term_err, delta) * sw).mean()
        total = total + cfg.w_terminal * term_loss
        logs["loss/terminal"] = float(term_loss.detach())

        # --- direction ------------------------------------------------
        # Hinge on sign agreement, scaled by how decisive the true move was.
        # The obvious form, `1 - tanh(pred) * tanh(true)`, is a trap: when the
        # market barely moved both tanhs are near zero and the term returns
        # ~1 even for a perfect forecast, creating a standing incentive to
        # over-predict magnitude. A hinge is exactly zero whenever the signs
        # agree, and its weight fades out on the near-zero moves that are
        # noise anyway.
        temp = cfg.direction_temp_bps / 10_000.0
        p_dir = torch.tanh(pred[:, -1] / temp)
        t_dir = torch.tanh(target_path[:, -1].detach() / temp)
        dir_loss = (torch.clamp(-p_dir * t_dir, min=0.0) * sw).mean()
        total = total + cfg.w_direction * dir_loss
        logs["loss/direction"] = float(dir_loss.detach())

        # --- consistency ----------------------------------------------
        cons = torch.zeros((), device=dev, dtype=dt)
        if self.prev is not None and self.prev.shape == pred.shape:
            # Only compare streams that actually advanced by exactly one
            # second; a stream that wrapped to the start of its span has no
            # meaningful predecessor.
            ok = (index - self.prev_index == 1).to(dt) if self.prev_index is not None else torch.ones(b, device=dev, dtype=dt)
            if ok.any() and self.prev_realised is not None:
                # Re-anchoring. The previous forecast, issued at t-1, says
                #     prev[k] = log P(t + k) - log P(t - 1)
                # The current one, issued at t, says
                #     pred[j] = log P(t + 1 + j) - log P(t)
                # The same wall-clock instant is k = j + 1, and the two differ
                # by the one-step return r = log P(t) - log P(t-1), which the
                # market has already printed and which we recorded last step
                # as that step's target_path[:, 0]. Subtracting it is the
                # whole point: without it this term would punish the model for
                # the price having moved, which is not instability.
                r = self.prev_realised.unsqueeze(1)
                aligned_prev = self.prev[:, 1:] - r
                cons_err = pred[:, :-1] - aligned_prev
                cons = ((_huber(cons_err, delta).mean(dim=1)) * ok * sw).sum() / ok.sum().clamp(min=1.0)
                total = total + cfg.w_consistency * cons
        logs["loss/consistency"] = float(cons.detach())
        self.prev = pred.detach()
        self.prev_index = index.detach().clone()
        # target_path[:, 0] is log P(t+1) - log P(t): next step's re-anchor.
        self.prev_realised = target_path[:, 0].detach().clone()

        # --- jump -----------------------------------------------------
        d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        # Second differences of a real price path are ~1e-9, so rescale into
        # a range where the weight is interpretable. Note this term is biased
        # toward over-smoothing (a true random walk scores ~0.02 here), which
        # is why w_jump is small: it is a guard rail, not an objective.
        jump = d2.pow(2).mean() * 1e8
        total = total + cfg.w_jump * jump
        logs["loss/jump"] = float(jump.detach())

        # --- head-specific --------------------------------------------
        if "quantile_paths" in out:
            qp = out["quantile_paths"]  # [B, Q, H]
            qs = torch.tensor(cfg.quantiles, device=dev, dtype=dt)[None, :, None]
            diff = target_path.unsqueeze(1) - qp
            pinball = torch.maximum(qs * diff, (qs - 1.0) * diff)
            q_loss = ((pinball.mean(dim=(1, 2))) * sw).mean()
            total = total + q_loss
            logs["loss/pinball"] = float(q_loss.detach())

        if "bin_logits" in out and bin_centres is not None:
            tgt = soft_bin_targets(target_path[:, -1].detach(), bin_centres)
            ce = -(tgt * F.log_softmax(out["bin_logits"].float(), dim=-1)).sum(-1)
            bin_loss = (ce * sw).mean()
            total = total + cfg.w_bins * bin_loss
            logs["loss/bins"] = float(bin_loss.detach())

        # --- analyser auxiliary ---------------------------------------
        if aux_pred is not None and aux_true is not None:
            aux_loss = F.smooth_l1_loss(aux_pred, aux_true.to(aux_pred.dtype))
            total = total + cfg.w_aux * aux_loss
            logs["loss/aux"] = float(aux_loss.detach())

        logs["loss/total"] = float(total.detach())
        return total, logs


# --------------------------------------------------------------------------- #
# Metrics (reported, never optimised)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batch_metrics(
    pred: torch.Tensor, target: torch.Tensor, prev_pred: torch.Tensor | None
) -> dict[str, float]:
    """Human-readable diagnostics in basis points."""
    bps = 10_000.0
    out = {
        "mae_bps/terminal": float((pred[:, -1] - target[:, -1]).abs().mean() * bps),
        "mae_bps/5m": float((pred[:, 299] - target[:, 299]).abs().mean() * bps),
        "mae_bps/path": float((pred - target).abs().mean() * bps),
        "dir_acc": float(
            ((pred[:, -1] > 0) == (target[:, -1] > 0)).float().mean()
        ),
        "move_bps/pred": float(pred[:, -1].abs().mean() * bps),
        "move_bps/true": float(target[:, -1].abs().mean() * bps),
    }
    # A naive "price does not move" forecast is the benchmark that matters:
    # beating it is the entire job, and plenty of published models do not.
    out["mae_bps/naive_terminal"] = float(target[:, -1].abs().mean() * bps)
    out["skill_vs_naive"] = 1.0 - out["mae_bps/terminal"] / max(
        out["mae_bps/naive_terminal"], 1e-9
    )
    if prev_pred is not None and prev_pred.shape == pred.shape:
        out["stability_bps"] = float(
            (pred[:, :-1] - prev_pred[:, 1:]).abs().mean() * bps
        )
    return out
