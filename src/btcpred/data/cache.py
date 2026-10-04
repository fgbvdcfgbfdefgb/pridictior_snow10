"""Build and open the memory-mapped feature cache.

The Parquet shards are the *archival* format: small, compressed, git-friendly.
They are useless for training, because decoding 24 MB of zstd-delta to answer
a random 12-hour window request would dominate the step time.

So once, offline, we expand every shard into three flat memory-mapped arrays
laid out on a single contiguous second-by-second timeline:

``features.f32``   ``[T, 14]``  the model input, see :mod:`.features`
``logprice.f32``   ``[T]``      log(close), the regression target space
``flags.u8``       ``[T]``      1 where the second was synthesised

Memory mapping is the whole point. On Snowflake with 100 GB of RAM the page
cache holds the hot span of the file, every dataloader worker shares those
same physical pages, and a random window read is a pointer offset instead of
a decode. Twelve months costs ~1.8 GB; the full 2020-to-now history costs
~12 GB, which still fits comfortably.

The cache is derived data and is **not** committed to git. Rebuild it with::

    python scripts/build_cache.py --shards data/btcusdt_1s --cache data/cache
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .features import build_features, log_price
from .schema import N_FEATURES, SYMBOL, month_start_epoch

LOG = logging.getLogger("btcpred.cache")

MANIFEST = "manifest.json"
FEATURES_FILE = "features.f32"
LOGPRICE_FILE = "logprice.f32"
FLAGS_FILE = "flags.u8"


def _parse_shard_name(path: Path) -> tuple[int, int, bool]:
    """``BTCUSDT-1s-2026-09[-partial].parquet`` -> ``(2026, 9, partial)``."""
    stem = path.stem
    partial = stem.endswith("-partial")
    if partial:
        stem = stem[: -len("-partial")]
    parts = stem.split("-")
    return int(parts[-2]), int(parts[-1]), partial


def discover_shards(shard_dir: Path) -> list[tuple[int, int, Path, int]]:
    """Ordered ``(year, month, path, n_rows)``, contiguity-checked.

    A missing month in the middle of the range is fatal rather than silently
    stitched: the cache is indexed by *absolute second offset*, so a hole
    would shift every subsequent timestamp and quietly invalidate the
    calendar features and every evaluation window.
    """
    found: dict[tuple[int, int], Path] = {}
    for path in sorted(shard_dir.glob("*.parquet")):
        year, month, _ = _parse_shard_name(path)
        key = (year, month)
        # A completed shard always beats a partial one for the same month.
        if key in found and "partial" in path.name:
            continue
        found[key] = path

    if not found:
        raise FileNotFoundError(
            f"no parquet shards in {shard_dir}. Run "
            f"`python -m btcpred.data.download_binance --out {shard_dir}` first."
        )

    keys = sorted(found)
    out: list[tuple[int, int, Path, int]] = []
    for i, key in enumerate(keys):
        if i:
            py, pm = keys[i - 1]
            expect = (py + 1, 1) if pm == 12 else (py, pm + 1)
            if key != expect:
                raise ValueError(
                    f"gap in shard sequence: {expect} is missing between "
                    f"{keys[i - 1]} and {key}. Download it or narrow --start."
                )
        path = found[key]
        out.append((key[0], key[1], path, pq.read_metadata(path).num_rows))
    return out


@dataclass(frozen=True)
class CacheManifest:
    symbol: str
    start_epoch: int
    n_seconds: int
    n_features: int
    months: list[dict]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @staticmethod
    def load(cache_dir: Path) -> "CacheManifest":
        data = json.loads((cache_dir / MANIFEST).read_text())
        return CacheManifest(**data)

    def epoch_at(self, index: int) -> int:
        return self.start_epoch + index

    def index_at(self, epoch: int) -> int:
        return epoch - self.start_epoch


def build_cache(shard_dir: Path, cache_dir: Path, chunk: int = 1_000_000) -> CacheManifest:
    shards = discover_shards(shard_dir)
    total = sum(s[3] for s in shards)
    start_epoch = month_start_epoch(shards[0][0], shards[0][1])
    cache_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(
        "building cache: %d shards, %s seconds, %.2f GB features",
        len(shards), f"{total:,}", total * N_FEATURES * 4 / 1e9,
    )

    feats = np.lib.format.open_memmap(
        cache_dir / FEATURES_FILE, mode="w+", dtype=np.float32,
        shape=(total, N_FEATURES),
    )
    lprice = np.lib.format.open_memmap(
        cache_dir / LOGPRICE_FILE, mode="w+", dtype=np.float32, shape=(total,)
    )
    fl = np.lib.format.open_memmap(
        cache_dir / FLAGS_FILE, mode="w+", dtype=np.uint8, shape=(total,)
    )

    offset = 0
    months: list[dict] = []
    for year, month, path, n_rows in shards:
        table = pq.read_table(path)
        cols = {name: table.column(name).to_numpy() for name in table.column_names}
        del table

        build_features(
            c=cols["c"], o_c=cols["o_c"], h_c=cols["h_c"], l_c=cols["l_c"],
            v=cols["v"], tb=cols["tb"], n=cols["n"], flags=cols["flags"],
            start_epoch=month_start_epoch(year, month),
            out=feats[offset : offset + n_rows],
        )
        lprice[offset : offset + n_rows] = log_price(cols["c"])
        fl[offset : offset + n_rows] = cols["flags"].astype(np.uint8)

        months.append(
            {
                "year": year, "month": month, "rows": n_rows, "offset": offset,
                "file": path.name,
                "filled_frac": float(cols["flags"].mean()),
            }
        )
        LOG.info("  %04d-%02d  rows=%s  offset=%s", year, month, f"{n_rows:,}", f"{offset:,}")
        offset += n_rows
        del cols

    feats.flush(); lprice.flush(); fl.flush()
    del feats, lprice, fl

    manifest = CacheManifest(
        symbol=SYMBOL, start_epoch=start_epoch, n_seconds=total,
        n_features=N_FEATURES, months=months,
    )
    (cache_dir / MANIFEST).write_text(manifest.to_json())
    LOG.info("cache written to %s", cache_dir)
    return manifest


class FeatureCache:
    """Read-only mmap view over a built cache. Safe to share across workers."""

    def __init__(self, cache_dir: Path) -> None:
        self.dir = Path(cache_dir)
        if not (self.dir / MANIFEST).exists():
            raise FileNotFoundError(
                f"no cache manifest in {self.dir}. Run scripts/build_cache.py."
            )
        self.manifest = CacheManifest.load(self.dir)
        self.features = np.load(self.dir / FEATURES_FILE, mmap_mode="r")
        self.logprice = np.load(self.dir / LOGPRICE_FILE, mmap_mode="r")
        self.flags = np.load(self.dir / FLAGS_FILE, mmap_mode="r")
        if self.features.shape[0] != self.manifest.n_seconds:
            raise ValueError("cache manifest and feature file disagree on length")

    def __len__(self) -> int:
        return self.manifest.n_seconds

    @property
    def start_epoch(self) -> int:
        return self.manifest.start_epoch

    def describe(self) -> str:
        import datetime as dt

        a = dt.datetime.fromtimestamp(self.start_epoch, dt.timezone.utc)
        b = dt.datetime.fromtimestamp(self.start_epoch + len(self) - 1, dt.timezone.utc)
        return (
            f"{self.manifest.symbol} 1s  {a:%Y-%m-%d %H:%M}Z .. {b:%Y-%m-%d %H:%M}Z  "
            f"({len(self):,} seconds, {len(self) / 86400:.1f} days, "
            f"{float(np.asarray(self.flags[::997]).mean()) * 100:.2f}% filled)"
        )
