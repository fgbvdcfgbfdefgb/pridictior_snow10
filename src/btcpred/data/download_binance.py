"""Download the complete BTCUSDT 1-second kline history from Binance.

Source: ``https://data.binance.vision`` — Binance's public S3 archive. No API
key, no account, no rate limit worth worrying about. Spot 1s klines for
BTCUSDT begin at 2020-01; this project defaults to 2022-01 onward (see
``schema.FIRST_MONTH``), but ``--start 2020-01`` fetches the full archive.

The script is *resumable*: a month whose Parquet shard already exists and
passes the row-count check is skipped. Interrupt it and re-run it; it picks up
where it left off.

Two quirks of the upstream data are handled here, and getting either wrong
silently corrupts every downstream timestamp:

1. **Timestamp units change.** Binance switched spot kline timestamps from
   milliseconds to *microseconds* on 2025-01-01. We sniff the magnitude of the
   first open time rather than hard-coding a date.
2. **Seconds with no trades are missing rows.** A "1s kline" is only emitted
   when something traded. We reindex onto the dense 1-second grid and
   forward-fill price (open=high=low=close=last close, volume=0), flagging
   every synthesised row so the trainer can tell them apart.

Monthly archives appear a few days after month end; for the trailing period we
fall back to stitching the daily archives together.

Usage
-----
::

    python -m btcpred.data.download_binance --out data/btcusdt_1s
    python -m btcpred.data.download_binance --out data/btcusdt_1s \\
        --start 2020-01 --end 2026-10 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gc
import io
import shutil
import logging
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    COLUMNS,
    FIRST_MONTH,
    FLAG_FILLED,
    FLAG_REAL,
    PRICE_SCALE,
    SYMBOL,
    VOLUME_SCALE,
    iter_months,
    month_length_seconds,
    month_start_epoch,
    shard_name,
)

LOG = logging.getLogger("btcpred.download")

BASE = "https://data.binance.vision/data/spot"
RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]
USE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
               "trades", "taker_base"]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _download_to_file(url: str, dest: Path, retries: int = 5) -> bool:
    """Stream a URL to disk. Returns False on a clean 404."""
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "btcpred/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            LOG.warning("HTTP %s on %s (attempt %d)", exc.code, url, attempt + 1)
        except Exception as exc:  # noqa: BLE001 - network is network
            LOG.warning("%s on %s (attempt %d)", exc, url, attempt + 1)
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"giving up on {url}")


def _month_urls(year: int, month: int) -> list[str]:
    """Monthly archive if published, else the month's daily archives.

    The monthly roll-up lands a few days after month end, so recent months
    must be stitched from dailies. We probe the monthly URL with a cheap HEAD
    rather than downloading it speculatively.
    """
    monthly = (
        f"{BASE}/monthly/klines/{SYMBOL}/1s/"
        f"{SYMBOL}-1s-{year:04d}-{month:02d}.zip"
    )
    try:
        req = urllib.request.Request(
            monthly, method="HEAD", headers={"User-Agent": "btcpred/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60):
            return [monthly]
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    except Exception:  # noqa: BLE001 - fall through to dailies
        pass

    LOG.info("%04d-%02d: no monthly archive, using daily files", year, month)
    urls: list[str] = []
    day = dt.date(year, month, 1)
    today = dt.datetime.now(dt.timezone.utc).date()
    while day.month == month and day < today:
        urls.append(
            f"{BASE}/daily/klines/{SYMBOL}/1s/{SYMBOL}-1s-{day.isoformat()}.zip"
        )
        day += dt.timedelta(days=1)
    return urls


class _DenseMonth:
    """Dense 1-second accumulator for one calendar month.

    Allocated once (~100 MB for a 31-day month) and scattered into chunk by
    chunk, so peak memory is independent of how many rows the archive holds.
    """

    def __init__(self, year: int, month: int) -> None:
        self.start = month_start_epoch(year, month)
        self.n_rows = month_length_seconds(year, month)
        n = self.n_rows
        self.close = np.zeros(n, dtype=np.int64)  # cents
        self.o_c = np.zeros(n, dtype=np.int32)
        self.h_c = np.zeros(n, dtype=np.int32)
        self.l_c = np.zeros(n, dtype=np.int32)
        self.v = np.zeros(n, dtype=np.int64)  # milli-BTC
        self.tb = np.zeros(n, dtype=np.int64)
        self.n = np.zeros(n, dtype=np.int32)
        self.flags = np.full(n, FLAG_FILLED, dtype=np.int8)
        self.seen = 0

    def scatter(self, chunk: pd.DataFrame) -> None:
        ts = _normalise_open_time(chunk["open_time"].to_numpy())
        idx = (ts - self.start).astype(np.int64)
        keep = (idx >= 0) & (idx < self.n_rows)
        if not keep.all():
            idx = idx[keep]
            chunk = chunk.loc[keep]
            if idx.size == 0:
                return

        c = np.rint(chunk["close"].to_numpy() * PRICE_SCALE).astype(np.int64)
        self.close[idx] = c
        self.o_c[idx] = (
            np.rint(chunk["open"].to_numpy() * PRICE_SCALE).astype(np.int64) - c
        ).astype(np.int32)
        self.h_c[idx] = (
            np.rint(chunk["high"].to_numpy() * PRICE_SCALE).astype(np.int64) - c
        ).astype(np.int32)
        self.l_c[idx] = (
            np.rint(chunk["low"].to_numpy() * PRICE_SCALE).astype(np.int64) - c
        ).astype(np.int32)
        self.v[idx] = np.rint(chunk["volume"].to_numpy() * VOLUME_SCALE)
        self.tb[idx] = np.rint(chunk["taker_base"].to_numpy() * VOLUME_SCALE)
        self.n[idx] = chunk["trades"].to_numpy().astype(np.int32)
        self.flags[idx] = FLAG_REAL
        self.seen += int(idx.size)

    def last_real_index(self) -> int:
        """Index of the final second that carried a real print, or -1."""
        real = np.flatnonzero(self.flags == FLAG_REAL)
        return int(real[-1]) if real.size else -1

    def finalise(self, limit: int | None = None) -> pa.Table:
        """Forward-fill silent seconds and emit the on-disk table.

        ``limit`` truncates the shard to the first ``limit`` rows, used for the
        in-progress current month so we never ship hours of fabricated future.
        """
        missing = self.flags == FLAG_FILLED
        n_missing = int(missing.sum())
        if n_missing:
            if missing[0]:
                # Leading gap (month starts mid-silence): back-fill from the
                # first real print so no row carries a price of zero.
                first_real = int(np.argmax(~missing))
                self.close[:first_real] = self.close[first_real]
                missing[:first_real] = True
            src = np.where(missing, 0, np.arange(self.n_rows, dtype=np.int64))
            np.maximum.accumulate(src, out=src)
            self.close = self.close[src]
            # A silent second has no trade: OHLC collapse to the last close
            # (deltas are zero, already) and volume/count stay at zero.
        end = self.n_rows if limit is None else min(limit, self.n_rows)
        if n_missing:
            # Report only over the retained span: for the in-progress current
            # month the tail beyond `limit` is future, not a data gap, and
            # counting it would make coverage look alarmingly bad.
            kept_missing = int(missing[:end].sum())
            LOG.info(
                "  %d/%d retained seconds (%.2f%%) had no trades -> filled",
                kept_missing, end, 100.0 * kept_missing / max(end, 1),
            )
        if end < self.n_rows:
            for name in ("close", "o_c", "h_c", "l_c", "v", "tb", "n", "flags"):
                setattr(self, name, getattr(self, name)[:end])
        return pa.table(
            {
                "c": pa.array(self.close.astype(np.int32)),
                "o_c": pa.array(self.o_c),
                "h_c": pa.array(self.h_c),
                "l_c": pa.array(self.l_c),
                "v": pa.array(self.v),
                "tb": pa.array(self.tb),
                "n": pa.array(self.n),
                "flags": pa.array(self.flags),
            }
        )


def _feed_zip(dense: _DenseMonth, zip_path: Path, chunk_rows: int) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            has_header = fh.read(10).lower().startswith(b"open_time")
        with zf.open(name) as fh:
            reader = pd.read_csv(
                fh,
                header=0 if has_header else None,
                names=RAW_COLUMNS,
                usecols=[RAW_COLUMNS.index(c) for c in USE_COLUMNS],
                dtype=np.float64,
                chunksize=chunk_rows,
            )
            for chunk in reader:
                chunk.columns = USE_COLUMNS
                dense.scatter(chunk)


def _normalise_open_time(open_time: np.ndarray) -> np.ndarray:
    """Return open times as integer unix *seconds*, whatever the input unit.

    Binance used milliseconds through 2024 and microseconds from 2025-01-01.
    A millisecond timestamp for any plausible date is ~1.7e12; a microsecond
    one is ~1.7e15. The 1e14 threshold sits between the two and stays valid
    for centuries in both directions.
    """
    first = float(open_time[0])
    divisor = 1_000_000 if first > 1e14 else 1_000
    return (open_time.astype(np.int64) // divisor).astype(np.int64)


def write_shard(table: pa.Table, path: Path, level: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(
        table,
        tmp,
        compression="zstd",
        compression_level=level,
        use_dictionary=False,
        column_encoding={k: "DELTA_BINARY_PACKED" for k in COLUMNS},
        data_page_size=1 << 20,
    )
    tmp.replace(path)


def shard_is_complete(path: Path, year: int, month: int) -> bool:
    """True only for a full-month shard already on disk with the right length."""
    if not path.exists():
        return False
    try:
        meta = pq.read_metadata(path)
    except Exception:  # noqa: BLE001 - truncated file from an interrupted run
        return False
    return meta.num_rows == month_length_seconds(year, month)


def download_month(
    year: int,
    month: int,
    out_dir: Path,
    level: int = 12,
    chunk_rows: int = 250_000,
    tmp_dir: Path | None = None,
) -> str:
    path = out_dir / shard_name(year, month)
    if shard_is_complete(path, year, month):
        return f"{year:04d}-{month:02d} skip (complete)"

    urls = _month_urls(year, month)
    if not urls:
        return f"{year:04d}-{month:02d} UNAVAILABLE"

    dense = _DenseMonth(year, month)
    tmp_dir = tmp_dir or out_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = tmp_dir / f".{SYMBOL}-1s-{year:04d}-{month:02d}.part.zip"
    try:
        for url in urls:
            if not _download_to_file(url, tmp_zip):
                LOG.warning("missing archive %s", url)
                continue
            _feed_zip(dense, tmp_zip, chunk_rows)
    finally:
        tmp_zip.unlink(missing_ok=True)

    if dense.seen == 0:
        return f"{year:04d}-{month:02d} UNAVAILABLE (no rows)"

    # The current month is still being written by the market; stop the shard
    # at the last second that actually printed rather than forward-filling to
    # the end of the month.
    last_real = dense.last_real_index()
    complete = last_real == dense.n_rows - 1
    coverage = 100.0 * dense.seen / (last_real + 1)
    table = dense.finalise(limit=None if complete else last_real + 1)
    if not complete:
        path = out_dir / shard_name(year, month).replace(
            ".parquet", "-partial.parquet"
        )
    del dense
    gc.collect()
    write_shard(table, path, level=level)
    mb = path.stat().st_size / 1e6
    tag = "ok" if complete else "ok (partial, month in progress)"
    return (
        f"{year:04d}-{month:02d} {tag} ({table.num_rows:,} rows, "
        f"{coverage:.1f}% real, {mb:.1f} MB)"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_month(s: str) -> tuple[int, int]:
    y, m = s.split("-")
    return int(y), int(m)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/btcusdt_1s"))
    p.add_argument("--start", type=_parse_month, default=FIRST_MONTH,
                   help="first month, YYYY-MM (default 2022-01; 2020-01 is the\nearliest Binance publishes)")
    p.add_argument("--end", type=_parse_month, default=None,
                   help="last month, YYYY-MM (default: current month)")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel month downloads; each needs ~1.5 GB RAM")
    p.add_argument("--zstd-level", type=int, default=12)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    now = dt.datetime.now(dt.timezone.utc)
    end = args.end or (now.year, now.month)
    months = iter_months(args.start, end)
    print(f"{len(months)} month(s): {months[0]} .. {months[-1]} -> {args.out}")

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(download_month, y, m, args.out, args.zstd_level): (y, m)
            for y, m in months
        }
        for i, fut in enumerate(cf.as_completed(futs), 1):
            y, m = futs[fut]
            try:
                msg = fut.result()
            except Exception as exc:  # noqa: BLE001
                msg = f"{y:04d}-{m:02d} FAILED: {exc}"
            print(f"[{i:3d}/{len(months)}] {msg}", flush=True)

    total = sum(f.stat().st_size for f in args.out.glob("*.parquet"))
    print(f"done in {time.time() - t0:.0f}s, {total / 1e9:.2f} GB on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
