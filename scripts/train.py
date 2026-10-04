#!/usr/bin/env python3
"""Train the Market Analyser + Price Predictor ensemble.

Single GPU / CPU smoke test::

    python scripts/train.py --cache data/cache --out runs/smoke \
        --max-steps 200 --batch 2 --d-model 128 --layers 4

Four A10s on Snowflake, one predictor variant per GPU, one shared analyser::

    torchrun --standalone --nproc_per_node=4 scripts/train.py \
        --cache data/cache --out runs/ensemble --mode ensemble \
        --max-steps 400000

Four A10s all training the *same* predictor with data-parallel gradients::

    torchrun --standalone --nproc_per_node=4 scripts/train.py \
        --cache data/cache --out runs/ddp --mode ddp --variant point-adamw
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from btcpred.data.pyramid import PyramidSpec  # noqa: E402
from btcpred.train.objective import ObjectiveConfig  # noqa: E402
from btcpred.train.trainer import OnlineTrainer  # noqa: E402
from btcpred.train.variants import DEFAULT_VARIANTS, variant_for_rank  # noqa: E402
from btcpred.utils import dist as D  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--out", type=Path, default=Path("runs/default"))
    p.add_argument("--max-steps", type=int, default=400_000,
                   help="market-seconds to replay (there are no epochs)")
    p.add_argument("--batch", type=int, default=None,
                   help="parallel streams per rank; default from hardware plan")
    p.add_argument("--mode", choices=("ensemble", "ddp"), default="ensemble")
    p.add_argument("--variant", default=None,
                   help=f"force one variant: {[v.name for v in DEFAULT_VARIANTS]}")
    p.add_argument("--analyser-device", default="cpu", choices=("cpu", "cuda"),
                   help="the brief says CPU; 'cuda' is ~100x faster per step")
    p.add_argument("--horizon", type=int, default=1500, help="forecast seconds")
    p.add_argument("--coeffs", type=int, default=48, help="smooth-basis terms")
    p.add_argument("--memory-headroom", type=float, default=0.80,
                   help="fraction of each GPU the sizing planner may occupy")
    p.add_argument("--max-d-model", type=int, default=2048,
                   help="cap on predictor width chosen by the planner")
    p.add_argument("--d-model", type=int, default=None, help="override predictor width")
    p.add_argument("--layers", type=int, default=None, help="override predictor depth")
    p.add_argument("--analyser-d-model", type=int, default=None)
    p.add_argument("--analyser-layers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=2_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", 0))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [r{rank}] %(levelname)s %(message)s",
    )

    variant = None
    if args.variant:
        matches = [v for v in DEFAULT_VARIANTS if v.name == args.variant]
        if not matches:
            p.error(f"unknown variant {args.variant!r}; "
                    f"choose from {[v.name for v in DEFAULT_VARIANTS]}")
        variant = matches[0]
    if args.mode == "ddp" and variant is None:
        # In ddp mode every rank must build the *same* architecture, or the
        # all-reduce will fail on mismatched parameter shapes.
        variant = DEFAULT_VARIANTS[0]
    if variant is None:
        variant = variant_for_rank(rank)
    if args.lr is not None:
        variant.lr = args.lr

    trainer = OnlineTrainer(
        cache_dir=args.cache,
        out_dir=args.out,
        spec=PyramidSpec(),
        horizon=args.horizon,
        n_coeffs=args.coeffs,
        batch=args.batch,
        max_steps=args.max_steps,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        objective=ObjectiveConfig(),
        variant=variant,
        analyser_device=args.analyser_device,
        mode=args.mode,
        memory_headroom=args.memory_headroom,
        max_d_model=args.max_d_model,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        eval_every=args.eval_every,
        seed=args.seed,
        override_plan={
            "predictor_d_model": args.d_model,
            "predictor_layers": args.layers,
            "analyser_d_model": args.analyser_d_model,
            "analyser_layers": args.analyser_layers,
            "predictor_heads": max(4, args.d_model // 64) if args.d_model else None,
        },
    )
    if args.resume:
        trainer.load(args.resume)

    metrics = trainer.run()
    if trainer.info.is_main:
        print("\nfinal validation metrics:")
        for k, v in sorted(metrics.items()):
            print(f"  {k:28s} {v: .5f}")
    D.shutdown(trainer.info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
