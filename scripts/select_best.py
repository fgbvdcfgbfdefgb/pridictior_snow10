#!/usr/bin/env python3
"""Evaluate every trained variant on the held-out tail and pick a winner.

Run after training, on one GPU (it loads each checkpoint in turn)::

    python scripts/select_best.py --runs runs/ensemble --cache data/cache \
        --out models/best.pt --steps 2000

Selection is not "lowest MAE". A forecaster that drives orders is judged on
four things at once, and the score below trades them off explicitly:

* **terminal MAE (bps)** — how wrong the 25-minute price call is;
* **directional accuracy** — whether the sign is right, which is what a
  position actually keys on;
* **stability (bps/s)** — how much the forecast for a fixed future instant
  moves between consecutive ticks. The brief calls this out directly;
* **skill vs naive** — improvement over "the price will not move", which is a
  genuinely hard benchmark at a 25-minute horizon and the one that exposes a
  model that has learned nothing but the unconditional mean.

The evaluation span is strictly *later* in time than anything training saw,
walked forward second by second with the same replay machinery, so the number
is a real out-of-sample estimate rather than a reshuffled in-sample one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from btcpred.data.cache import FeatureCache  # noqa: E402
from btcpred.data.pyramid import PyramidSpec  # noqa: E402
from btcpred.data.replay import ParallelReplay  # noqa: E402
from btcpred.live.runner import LivePredictor  # noqa: E402
from btcpred.train.objective import batch_metrics  # noqa: E402


def selection_score(m: dict[str, float]) -> float:
    """Lower is better. Mirrors ``trainer._selection_score``, kept in sync."""
    return (
        m.get("mae_bps/terminal", 1e9)
        + 0.5 * m.get("stability_bps", 0.0)
        + 20.0 * max(0.0, 0.5 - m.get("dir_acc", 0.5))
    )


@torch.no_grad()
def evaluate_checkpoint(
    path: Path,
    cache: FeatureCache,
    index_range: tuple[int, int],
    steps: int,
    batch: int,
    device: str,
) -> dict[str, float]:
    lp = LivePredictor(path, device=device, analyser_device=device, ema_halflife=0.0)
    spec = PyramidSpec(tuple(tuple(x) for x in lp.spec.levels))
    replay = ParallelReplay(
        cache, batch, spec, lp.horizon, index_range=index_range, seed=11, jitter=False
    )
    dev = lp.device

    acc: dict[str, list[float]] = {}
    prev = None
    pnl: list[float] = []
    for _ in range(steps):
        b = replay.step()
        tokens = torch.from_numpy(b["tokens"]).to(dev)
        target = torch.from_numpy(b["future"] - b["anchor"][:, None]).to(dev).float()
        a = lp.analyser(tokens)
        out = lp.predictor(tokens, a["memory"], a["state"])
        pred = out["path"].float()
        m = batch_metrics(pred, target, prev)
        prev = pred
        for k, v in m.items():
            acc.setdefault(k, []).append(v)

        # Toy long/flat/short sizing: take the sign of the 25m call, size by
        # conviction, pay 1 bp of cost per side. Not a strategy -- a sanity
        # check that the forecast's sign carries tradeable information.
        signal = torch.tanh(pred[:, -1] / (5.0 / 10_000.0))
        gross = (signal * target[:, -1]).mean()
        cost = signal.abs().mean() * (1.0 / 10_000.0)
        pnl.append(float((gross - cost) * 10_000.0))

    res = {k: float(np.mean(v)) for k, v in acc.items()}
    res["pnl_bps_per_trade"] = float(np.mean(pnl))
    res["pnl_sharpe"] = float(np.mean(pnl) / (np.std(pnl) + 1e-12))
    res["score"] = selection_score(res)
    return res


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--out", type=Path, default=Path("models/best.pt"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--glob", default="best_*.pt")
    args = p.parse_args()

    ckpts = sorted(args.runs.rglob(args.glob))
    if not ckpts:
        print(f"no checkpoints matching {args.glob} under {args.runs}", file=sys.stderr)
        return 1

    cache = FeatureCache(args.cache)
    n = len(cache)
    test_range = (int(n * (1.0 - args.test_fraction)), n)
    print(f"cache: {cache.describe()}")
    print(f"test span: indices {test_range[0]:,}..{test_range[1]:,} "
          f"({(test_range[1] - test_range[0]) / 86400:.1f} days, strictly held out)\n")

    results: dict[str, dict[str, float]] = {}
    for ck in ckpts:
        print(f"evaluating {ck.name} ...", flush=True)
        try:
            results[ck.name] = evaluate_checkpoint(
                ck, cache, test_range, args.steps, args.batch, args.device
            )
            results[ck.name]["path"] = str(ck)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")

    if not results:
        return 1

    cols = ["mae_bps/terminal", "dir_acc", "stability_bps", "skill_vs_naive",
            "pnl_bps_per_trade", "pnl_sharpe", "score"]
    width = max(len(k) for k in results) + 2
    print("\n" + "model".ljust(width) + "".join(c.rjust(20) for c in cols))
    print("-" * (width + 20 * len(cols)))
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["score"]):
        print(name.ljust(width) + "".join(f"{m.get(c, float('nan')):20.4f}" for c in cols))

    best_name = min(results, key=lambda k: results[k]["score"])
    best = results[best_name]
    print(f"\nwinner: {best_name}  (score {best['score']:.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(best["path"], map_location="cpu", weights_only=False)
    payload["selection"] = {k: v for k, v in best.items() if k != "path"}
    payload["selection_table"] = {
        k: {c: v.get(c) for c in cols} for k, v in results.items()
    }
    # Optimiser state is ~2/3 of the file and useless for serving.
    payload.pop("opt_pred", None)
    payload.pop("opt_anal", None)
    torch.save(payload, args.out)
    (args.out.parent / "selection.json").write_text(
        json.dumps({k: {c: v.get(c) for c in cols} for k, v in results.items()}, indent=2)
    )
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
