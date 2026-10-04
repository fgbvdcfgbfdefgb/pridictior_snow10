#!/usr/bin/env python3
"""Expand Parquet shards into the memory-mapped training cache.

Run once after downloading (or after adding new months). Offline — no network.

    python scripts/build_cache.py --shards data/btcusdt_1s --cache data/cache
    python scripts/build_cache.py --shards data/btcusdt_1s --cache data/cache \
        --start 2025-01 --end 2026-10
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from btcpred.data.cache import FeatureCache, build_cache  # noqa: E402
from btcpred.data.schema import N_FEATURES  # noqa: E402


def _month(s: str) -> tuple[int, int]:
    y, m = s.split("-")
    return int(y), int(m)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", type=Path, default=Path("data/btcusdt_1s"))
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--start", type=_month, default=None, help="first month YYYY-MM")
    p.add_argument("--end", type=_month, default=None, help="last month YYYY-MM")
    p.add_argument("--force", action="store_true", help="rebuild even if present")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if (args.cache / "manifest.json").exists() and not args.force:
        cache = FeatureCache(args.cache)
        print(f"cache already built: {cache.describe()}")
        print("pass --force to rebuild")
        return 0

    shard_dir = args.shards
    tmp: Path | None = None
    if args.start or args.end:
        # Symlink the selected months into a scratch dir so the contiguity
        # check in discover_shards still has a clean, gap-free view.
        tmp = Path(tempfile.mkdtemp(prefix="btcpred-shards-"))
        lo = args.start or (0, 0)
        hi = args.end or (9999, 12)
        n = 0
        for f in sorted(args.shards.glob("*.parquet")):
            stem = f.stem.replace("-partial", "")
            y, m = int(stem.split("-")[-2]), int(stem.split("-")[-1])
            if lo <= (y, m) <= hi:
                (tmp / f.name).symlink_to(f.resolve())
                n += 1
        if not n:
            print(f"no shards in range {args.start}..{args.end}", file=sys.stderr)
            return 1
        shard_dir = tmp
        print(f"selected {n} shard(s) from {args.shards}")

    try:
        manifest = build_cache(shard_dir, args.cache)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    gb = manifest.n_seconds * (N_FEATURES * 4 + 4 + 1) / 1e9
    cache = FeatureCache(args.cache)
    print()
    print(cache.describe())
    print(f"cache size: {gb:.2f} GB in {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
