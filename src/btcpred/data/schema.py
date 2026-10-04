"""Canonical on-disk schema for the 1-second BTCUSDT dataset.

The dataset is stored as one Parquet shard per calendar month on a *dense,
gap-free* 1-second grid. Because the grid is implicit, no timestamp column is
stored: row ``i`` of shard ``YYYY-MM`` is the kline whose open time is
``month_start_utc + i`` seconds. This removes ~25% of the file size and makes
random access an O(1) integer offset, which the replay simulator relies on.

Columns (all integer, delta-encoded + zstd on disk):

==========  =======  ==========================================================
name        dtype    meaning
==========  =======  ==========================================================
c           int32    close price in cents (USDT * 100)
o_c         int32    open  - close, in cents
h_c         int32    high  - close, in cents
l_c         int32    low   - close, in cents
v           int64    base-asset volume in milli-BTC (BTC * 1000)
tb          int64    taker-buy base volume in milli-BTC
n           int32    number of trades
flags       int8     0 = real kline, 1 = synthetically forward-filled gap
==========  =======  ==========================================================

Storing deltas against the close rather than absolute OHLC matters: the deltas
are tiny integers (usually < 2000 cents) and compress ~10x better than four
correlated absolute price series.

``flags`` is the honest part of the contract. Binance 1s klines are emitted
only for seconds in which at least one trade printed; quiet seconds are simply
absent from the CSV. We forward-fill them to keep the grid dense and mark them,
so the trainer can down-weight (or the evaluator can exclude) synthetic rows
instead of silently learning from fabricated ticks.
"""

from __future__ import annotations

import datetime as _dt
from typing import Final

import numpy as np

SYMBOL: Final[str] = "BTCUSDT"

PRICE_SCALE: Final[int] = 100  # cents per USDT
VOLUME_SCALE: Final[int] = 1000  # milli-BTC per BTC

#: Default first month for the dataset.
#:
#: Binance actually publishes 1s spot klines for BTCUSDT back to 2020-01, and
#: the downloader will happily fetch them with ``--start 2020-01``. The
#: default starts at 2022-01 instead: that window spans the 2022 bear market,
#: the 2023 recovery, the 2024-25 bull run and the current regime, which is
#: enough regime diversity without the 2020-21 era whose microstructure
#: (thinner books, different fee tiers, pre-ETF flow) is least like today's.
FIRST_MONTH: Final[tuple[int, int]] = (2022, 1)

#: Earliest month that exists upstream, for reference and validation.
EARLIEST_AVAILABLE_MONTH: Final[tuple[int, int]] = (2020, 1)

COLUMNS: Final[dict[str, np.dtype]] = {
    "c": np.dtype(np.int32),
    "o_c": np.dtype(np.int32),
    "h_c": np.dtype(np.int32),
    "l_c": np.dtype(np.int32),
    "v": np.dtype(np.int64),
    "tb": np.dtype(np.int64),
    "n": np.dtype(np.int32),
    "flags": np.dtype(np.int8),
}

FLAG_REAL: Final[int] = 0
FLAG_FILLED: Final[int] = 1

#: Feature channels handed to the models, in order. See
#: :func:`btcpred.data.features.build_features`.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log_return",  # log(c_t / c_{t-1})
    "hl_range",  # (h - l) / c
    "body",  # (c - o) / c
    "upper_wick",  # (h - max(o, c)) / c
    "lower_wick",  # (min(o, c) - l) / c
    "log_volume",  # log1p(volume in BTC)
    "taker_imbalance",  # 2 * tb / v - 1, in [-1, 1]
    "log_trades",  # log1p(number of trades)
    "avg_trade_size",  # log1p(v / n)
    "is_filled",  # 1.0 if the second was synthesised
    "tod_sin",  # time of day, sine
    "tod_cos",  # time of day, cosine
    "dow_sin",  # day of week, sine
    "dow_cos",  # day of week, cosine
)

N_FEATURES: Final[int] = len(FEATURE_NAMES)


def month_start_epoch(year: int, month: int) -> int:
    """Unix seconds at 00:00:00 UTC on the first day of ``year``-``month``."""
    return int(
        _dt.datetime(year, month, 1, tzinfo=_dt.timezone.utc).timestamp()
    )


def month_length_seconds(year: int, month: int) -> int:
    """Number of 1-second rows a complete shard for this month must contain."""
    if month == 12:
        nxt = _dt.datetime(year + 1, 1, 1, tzinfo=_dt.timezone.utc)
    else:
        nxt = _dt.datetime(year, month + 1, 1, tzinfo=_dt.timezone.utc)
    cur = _dt.datetime(year, month, 1, tzinfo=_dt.timezone.utc)
    return int((nxt - cur).total_seconds())


def iter_months(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """Inclusive list of ``(year, month)`` pairs from ``start`` to ``end``."""
    y, m = start
    out: list[tuple[int, int]] = []
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def shard_name(year: int, month: int) -> str:
    return f"{SYMBOL}-1s-{year:04d}-{month:02d}.parquet"
