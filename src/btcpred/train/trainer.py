"""Epoch-free, second-by-second online trainer.

One step = one simulated market second, replayed across ``B`` parallel streams.
There is no shuffling, no epoch counter and no held-out sampling during
training: the loop walks market time forward and is scored, every second,
against what the market actually did over the following 25 minutes.

Device topology
---------------
::

    tokens (numpy, CPU)
        |
        v
    MarketAnalyser  -- CPU, wrapped in DDP/gloo so all ranks share one model
        |  memory [B,T,Da], state [B,Ds]
        v   (.to(cuda), autograd follows the tensors back)
    PricePredictor  -- GPU, one *different* variant per rank (ensemble mode)
        |
        v
    RealTimeObjective -- GPU

Autograd crosses the device boundary on its own; the backward pass runs the
predictor's half on the GPU and the analyser's half on the CPU. That is what
makes "analyser on CPU, predictor on GPU, trained end to end" literally true
rather than two models bolted together.

A word on throughput
--------------------
The CPU analyser is the bottleneck by a wide margin — a 500M-FLOP/token conv
stack on 48 vCPUs is roughly two orders of magnitude slower than the same
stack on an A10. The default honours the brief and keeps it on CPU; pass
``--analyser-device cuda`` to move it (it costs ~1 GB of VRAM) when you care
more about steps/second than about the letter of the layout. The trainer logs
the measured split every ``--log-every`` steps so the cost is visible rather
than theoretical.

No-peek
-------
``ParallelReplay`` only ever hands the model cache rows at or before each
cursor, and the analyser's convolutions are causal (see
:class:`~btcpred.models.market_analyser.ChannelNorm` for the subtle part).
The realised future reaches the objective and nothing else.
"""

from __future__ import annotations

import json
import logging
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data.cache import FeatureCache
from ..data.pyramid import PyramidSpec
from ..data.replay import HORIZON_SECONDS, ParallelReplay
from ..data.schema import N_FEATURES
from ..models.market_analyser import MarketAnalyser, auxiliary_targets
from ..models.price_predictor import PricePredictor
from ..utils import dist as D
from ..utils.hardware import Hardware, SizingPlan, detect, plan
from .objective import ObjectiveConfig, RealTimeObjective, batch_metrics
from .variants import VariantConfig, variant_for_rank

LOG = logging.getLogger("btcpred.train")


class _Stopper:
    """Turn SIGINT/SIGTERM into a clean checkpoint instead of a lost run."""

    def __init__(self) -> None:
        self.stop = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle)
            except ValueError:  # not in main thread
                pass

    def _handle(self, *_args) -> None:
        if self.stop:
            raise KeyboardInterrupt("second signal: hard exit")
        LOG.warning("stop requested -- finishing step and checkpointing")
        self.stop = True


class OnlineTrainer:
    def __init__(
        self,
        cache_dir: Path,
        out_dir: Path,
        *,
        spec: PyramidSpec = PyramidSpec(),
        horizon: int = HORIZON_SECONDS,
        n_coeffs: int = 48,
        batch: int | None = None,
        max_steps: int = 100_000,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        objective: ObjectiveConfig | None = None,
        variant: VariantConfig | None = None,
        analyser_device: str = "cpu",
        mode: str = "ensemble",
        memory_headroom: float = 0.80,
        max_d_model: int = 2048,
        log_every: int = 25,
        ckpt_every: int = 2_000,
        eval_every: int = 5_000,
        lr_warmup: int = 500,
        seed: int | None = None,
        override_plan: dict | None = None,
    ) -> None:
        self.info = D.init()
        self.out_dir = Path(out_dir)
        self.spec = spec
        self.horizon = horizon
        self.max_steps = max_steps
        self.log_every = log_every
        self.ckpt_every = ckpt_every
        self.eval_every = eval_every
        self.lr_warmup = lr_warmup
        self.mode = mode

        self.variant = variant or variant_for_rank(self.info.rank)
        torch.manual_seed(seed if seed is not None else self.variant.seed)
        np.random.seed((seed if seed is not None else self.variant.seed) + 17)

        # ---- hardware & sizing --------------------------------------
        self.hw: Hardware = detect()
        self.plan: SizingPlan = plan(
            self.hw, spec.n_tokens, N_FEATURES, n_coeffs,
            memory_headroom=memory_headroom, max_d_model=max_d_model,
        )
        if override_plan:
            for k, v in override_plan.items():
                if v is not None and hasattr(self.plan, k):
                    setattr(self.plan, k, v)
        if self.info.is_main:
            LOG.info("hardware:\n%s", self.hw.summary())
            for note in self.plan.notes:
                LOG.info("sizing: %s", note)

        torch.set_num_threads(max(1, self.plan.analyser_threads))
        self.batch = batch or self.plan.micro_batch

        # ---- data ----------------------------------------------------
        self.cache = FeatureCache(cache_dir)
        if self.info.is_main:
            LOG.info("cache: %s", self.cache.describe())
        n = len(self.cache)
        # Chronological split. Random splits leak catastrophically here:
        # neighbouring seconds are ~identical, so a shuffled validation set
        # measures memorisation, not forecasting.
        self.test_start = int(n * (1.0 - test_fraction))
        self.val_start = int(n * (1.0 - test_fraction - val_fraction))
        self.train_range = (0, self.val_start)
        self.val_range = (self.val_start, self.test_start)
        self.test_range = (self.test_start, n)

        self.replay = ParallelReplay(
            self.cache, self.batch, spec, horizon,
            index_range=self.train_range,
            # Different ranks walk different phases of the archive so the
            # shared analyser sees four decorrelated views per step.
            seed=1000 + self.info.rank,
        )

        # ---- models ---------------------------------------------------
        self.analyser_device = torch.device(analyser_device)
        self.gpu = self.info.device

        analyser = MarketAnalyser(
            n_features=N_FEATURES,
            d_model=self.plan.analyser_d_model,
            n_layers=self.plan.analyser_layers,
            d_state=self.plan.analyser_d_model,
        ).to(self.analyser_device)

        predictor = PricePredictor(
            token_offsets=torch.from_numpy(spec.token_end_offsets().copy()),
            n_features=N_FEATURES,
            d_analyser=self.plan.analyser_d_model,
            d_state=self.plan.analyser_d_model,
            d_model=self.plan.predictor_d_model,
            n_layers=self.plan.predictor_layers,
            n_heads=self.plan.predictor_heads,
            ffn_mult=self.plan.predictor_ffn_mult,
            n_coeffs=n_coeffs,
            horizon=horizon,
            head=self.variant.head,
            dropout=self.variant.dropout,
            grad_checkpointing=self.plan.activation_checkpointing,
        ).to(self.gpu)

        self.analyser_raw = analyser
        self.predictor_raw = predictor

        if self.info.distributed:
            # The analyser is all-reduced on every rank -> one shared model.
            self.analyser = DDP(analyser, device_ids=None, broadcast_buffers=False)
            if mode == "ddp":
                self.predictor = DDP(
                    predictor,
                    device_ids=[self.info.local_rank] if self.gpu.type == "cuda" else None,
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                )
            else:
                # Ensemble: each rank's predictor is its own model and must
                # NOT be synchronised. Wrapping it in DDP here would silently
                # average four different architectures' gradients.
                self.predictor = predictor
        else:
            self.analyser = analyser
            self.predictor = predictor

        self.teacher: PricePredictor | None = None
        if self.variant.ema_decay > 0:
            import copy

            self.teacher = copy.deepcopy(predictor).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

        # ---- optimisation ---------------------------------------------
        self.opt_pred = torch.optim.AdamW(
            self.predictor_raw.param_groups(self.variant.weight_decay),
            lr=self.variant.lr, betas=self.variant.betas, eps=1e-8,
        )
        self.opt_anal = torch.optim.AdamW(
            self.analyser_raw.parameters(),
            # The analyser gets gradient from every rank, so its effective
            # batch is world_size times the predictor's. Scale its LR down
            # accordingly or it runs away from the predictors it feeds.
            lr=self.variant.lr * 0.5 / math.sqrt(max(self.info.world_size, 1)),
            betas=(0.9, 0.95), weight_decay=0.01,
        )

        self.amp = self.plan.precision in ("bf16", "fp16") and self.gpu.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.plan.precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp and self.amp_dtype is torch.float16
        )

        self.objective = RealTimeObjective(
            self.variant.objective(objective or ObjectiveConfig()), horizon
        )
        self.bin_widths = torch.from_numpy(spec.bin_widths().copy())

        self.step = 0
        self.best_score = float("inf")
        self.stopper = _Stopper()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / f"log_rank{self.info.rank}.jsonl"
        self._timing = {"analyser": 0.0, "predictor": 0.0, "data": 0.0, "backward": 0.0}

        if self.info.is_main:
            self._write_run_card()

    # ------------------------------------------------------------------ #
    def _write_run_card(self) -> None:
        card = {
            "hardware": self.hw.summary(),
            "plan": self.plan.as_dict(),
            "variant": asdict(self.variant),
            "objective": asdict(self.objective.cfg),
            "spec_levels": list(self.spec.levels),
            "n_tokens": self.spec.n_tokens,
            "horizon": self.horizon,
            "batch_per_rank": self.batch,
            "world_size": self.info.world_size,
            "mode": self.mode,
            "cache": self.cache.describe(),
            "splits": {
                "train": self.train_range,
                "val": self.val_range,
                "test": self.test_range,
            },
            "predictor_params": self.predictor_raw.num_parameters(),
            "analyser_params": self.analyser_raw.num_parameters(),
        }
        (self.out_dir / "run_card.json").write_text(json.dumps(card, indent=2))
        LOG.info(
            "predictor %.1fM params | analyser %.1fM params | batch %d x %d ranks",
            card["predictor_params"] / 1e6, card["analyser_params"] / 1e6,
            self.batch, self.info.world_size,
        )

    def _lr_at(self, step: int) -> float:
        """Linear warmup, then flat.

        No decay schedule: the run has no fixed horizon (it is a stream), and
        a cosine schedule to an arbitrary ``max_steps`` would make the model's
        final quality depend on a number someone typed on the command line.
        """
        if step >= self.lr_warmup:
            return 1.0
        return (step + 1) / max(self.lr_warmup, 1)

    # ------------------------------------------------------------------ #
    def _forward(self, tokens_np: np.ndarray, grad: bool = True):
        tokens_cpu = torch.from_numpy(tokens_np)

        t0 = time.perf_counter()
        a_in = tokens_cpu.to(self.analyser_device, non_blocking=True)
        with torch.set_grad_enabled(grad):
            a_out = self.analyser(a_in)
        t1 = time.perf_counter()

        tokens_gpu = tokens_cpu.to(self.gpu, non_blocking=True)
        memory = a_out["memory"].to(self.gpu, non_blocking=True)
        state = a_out["state"].to(self.gpu, non_blocking=True)

        with torch.set_grad_enabled(grad), torch.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.amp
        ):
            out = self.predictor(tokens_gpu, memory, state)
        t2 = time.perf_counter()

        self._timing["analyser"] += t1 - t0
        self._timing["predictor"] += t2 - t1
        return out, a_out, tokens_cpu

    def train_step(self) -> dict[str, float]:
        t0 = time.perf_counter()
        batch = self.replay.step()
        target = torch.from_numpy(
            batch["future"] - batch["anchor"][:, None]
        ).to(self.gpu)
        fill = torch.from_numpy(batch["fill_frac"]).to(self.gpu)
        index = torch.from_numpy(batch["index"]).to(self.gpu)
        self._timing["data"] += time.perf_counter() - t0

        out, a_out, tokens_cpu = self._forward(batch["tokens"], grad=True)

        aux_true = auxiliary_targets(tokens_cpu, self.bin_widths)
        loss, logs = self.objective(
            out,
            target.float(),
            fill,
            index,
            aux_pred=a_out["aux"],
            aux_true=aux_true,
            bin_centres=getattr(self.predictor_raw, "bin_centres", None),
        )

        if self.teacher is not None:
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=self.amp_dtype, enabled=self.amp
            ):
                t_out = self.teacher(
                    tokens_cpu.to(self.gpu),
                    a_out["memory"].detach().to(self.gpu),
                    a_out["state"].detach().to(self.gpu),
                )
            distill = torch.nn.functional.smooth_l1_loss(
                out["path"], t_out["path"].detach(), beta=1e-4
            )
            loss = loss + self.variant.distill_weight * distill
            logs["loss/distill"] = float(distill.detach())

        t3 = time.perf_counter()
        self.opt_pred.zero_grad(set_to_none=True)
        self.opt_anal.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt_pred)
        gn_p = nn.utils.clip_grad_norm_(
            self.predictor_raw.parameters(), self.variant.grad_clip
        )
        gn_a = nn.utils.clip_grad_norm_(
            self.analyser_raw.parameters(), self.variant.grad_clip
        )

        scale = self._lr_at(self.step)
        for g in self.opt_pred.param_groups:
            g["lr"] = self.variant.lr * scale
        for g in self.opt_anal.param_groups:
            g["lr"] = (
                self.variant.lr * 0.5 / math.sqrt(max(self.info.world_size, 1)) * scale
            )

        self.scaler.step(self.opt_pred)
        self.scaler.update()
        self.opt_anal.step()
        self._timing["backward"] += time.perf_counter() - t3

        if self.teacher is not None:
            with torch.no_grad():
                d = self.variant.ema_decay
                for tp, sp in zip(self.teacher.parameters(), self.predictor_raw.parameters()):
                    tp.lerp_(sp.detach(), 1.0 - d)

        logs["grad_norm/predictor"] = float(gn_p)
        logs["grad_norm/analyser"] = float(gn_a)
        logs["lr"] = self.variant.lr * scale
        logs.update(
            batch_metrics(
                out["path"].detach().float(), target.float(), self.objective.prev
            )
        )
        return logs

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, index_range: tuple[int, int], n_steps: int = 300) -> dict[str, float]:
        """Walk-forward evaluation on an unseen, strictly later time span."""
        self.predictor_raw.eval()
        self.analyser_raw.eval()
        replay = ParallelReplay(
            self.cache, self.batch, self.spec, self.horizon,
            index_range=index_range, seed=7, jitter=False,
        )
        acc: dict[str, list[float]] = {}
        prev = None
        for _ in range(n_steps):
            b = replay.step()
            target = torch.from_numpy(b["future"] - b["anchor"][:, None]).to(self.gpu)
            out, _, _ = self._forward(b["tokens"], grad=False)
            m = batch_metrics(out["path"].float(), target.float(), prev)
            prev = out["path"].float()
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        self.predictor_raw.train()
        self.analyser_raw.train()
        return {k: float(np.mean(v)) for k, v in acc.items()}

    # ------------------------------------------------------------------ #
    def save(self, tag: str, extra: dict | None = None) -> Path:
        path = self.out_dir / f"{tag}.pt"
        torch.save(
            {
                "step": self.step,
                "variant": asdict(self.variant),
                "plan": self.plan.as_dict(),
                "spec_levels": list(self.spec.levels),
                "horizon": self.horizon,
                "n_coeffs": self.predictor_raw.n_coeffs,
                "analyser": self.analyser_raw.state_dict(),
                "predictor": self.predictor_raw.state_dict(),
                "opt_pred": self.opt_pred.state_dict(),
                "opt_anal": self.opt_anal.state_dict(),
                "metrics": extra or {},
            },
            path,
        )
        return path

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.analyser_raw.load_state_dict(ck["analyser"])
        self.predictor_raw.load_state_dict(ck["predictor"])
        self.opt_pred.load_state_dict(ck["opt_pred"])
        self.opt_anal.load_state_dict(ck["opt_anal"])
        self.step = ck["step"]
        LOG.info("resumed %s at step %d", path, self.step)

    # ------------------------------------------------------------------ #
    def run(self) -> dict[str, float]:
        LOG.info(
            "rank %d training variant '%s' for up to %d market-seconds",
            self.info.rank, self.variant.name, self.max_steps,
        )
        fh = self.log_path.open("a")
        t_start = time.perf_counter()
        window: dict[str, list[float]] = {}
        last_val: dict[str, float] = {}

        try:
            while self.step < self.max_steps and not self.stopper.stop:
                logs = self.train_step()
                for k, v in logs.items():
                    window.setdefault(k, []).append(v)
                self.step += 1

                if self.step % self.log_every == 0:
                    agg = {k: float(np.mean(v)) for k, v in window.items()}
                    window.clear()
                    elapsed = time.perf_counter() - t_start
                    tot = sum(self._timing.values()) or 1.0
                    rec = {
                        "step": self.step,
                        "rank": self.info.rank,
                        "variant": self.variant.name,
                        "steps_per_s": self.step / elapsed,
                        "archive_seen": self.replay.epochs_equivalent,
                        "time_frac": {k: v / tot for k, v in self._timing.items()},
                        **agg,
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    LOG.info(
                        "step %6d | loss %.5f | term MAE %.2f bps | dir %.3f | "
                        "skill %+.3f | stab %.2f bps | %.2f step/s",
                        self.step, agg.get("loss/total", float("nan")),
                        agg.get("mae_bps/terminal", float("nan")),
                        agg.get("dir_acc", float("nan")),
                        agg.get("skill_vs_naive", float("nan")),
                        agg.get("stability_bps", float("nan")),
                        rec["steps_per_s"],
                    )

                if self.eval_every and self.step % self.eval_every == 0:
                    last_val = self.evaluate(self.val_range)
                    LOG.info("VAL step %d: %s", self.step, _fmt(last_val))
                    fh.write(json.dumps({"step": self.step, "split": "val", **last_val}) + "\n")
                    fh.flush()
                    score = _selection_score(last_val)
                    if score < self.best_score:
                        self.best_score = score
                        self.save(f"best_{self.variant.name}", last_val)
                        LOG.info("new best for %s (score %.4f)", self.variant.name, score)

                if self.ckpt_every and self.step % self.ckpt_every == 0:
                    self.save(f"last_{self.variant.name}")
        finally:
            self.save(f"last_{self.variant.name}", last_val)
            fh.close()

        if not last_val:
            last_val = self.evaluate(self.val_range)
        D.barrier(self.info)
        return last_val


def _fmt(d: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in sorted(d.items()))


def _selection_score(m: dict[str, float]) -> float:
    """Single number used to pick 'best'. Lower is better.

    Terminal accuracy is the headline, but a model that is 2% more accurate
    and twice as jittery is not an improvement for something that places
    orders — so stability enters the score directly, and directional accuracy
    gets a modest pull because sign is what a position actually keys on.
    """
    mae = m.get("mae_bps/terminal", 1e9)
    stab = m.get("stability_bps", 0.0)
    dir_acc = m.get("dir_acc", 0.5)
    return mae + 0.5 * stab + 20.0 * max(0.0, 0.5 - dir_acc)
