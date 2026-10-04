"""Turn raw 1-second klines into the model's float32 feature matrix.

Everything here is *causal and scale-free*. The model must not see absolute
price levels: BTC traded at 7k in 2020 and well north of that today, and a
network that memorises levels will fall over the moment it meets a new regime.
So every channel is either a ratio, a log-return, a bounded imbalance, or a
calendar encoding. The absolute close is kept alongside, outside the feature
matrix, purely to reconstruct predicted prices at the end.

The output is written once to a memory-mapped ``.npy`` cache by
``scripts/build_cache.py`` and thereafter mmapped read-only by every
dataloader worker, which is what makes 200M+ seconds tractable.
"""

from __future__ import annotations

import numpy as np

from .schema import FEATURE_NAMES, N_FEATURES, PRICE_SCALE, VOLUME_SCALE

SECONDS_PER_DAY = 86_400
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY

# Channel-wise scale factors chosen so each feature lands roughly in [-3, 3].
# They are constants, not fitted statistics: a fitted scaler is one more thing
# to keep in sync between training, Snowflake and the live notebook, and the
# natural units of these quantities are stable enough not to need it.
_RETURN_SCALE = 2_000.0  # 1 bp of log-return -> 0.2
_RANGE_SCALE = 2_000.0
_WICK_SCALE = 2_000.0


def build_features(
    c: np.ndarray,
    o_c: np.ndarray,
    h_c: np.ndarray,
    l_c: np.ndarray,
    v: np.ndarray,
    tb: np.ndarray,
    n: np.ndarray,
    flags: np.ndarray,
    start_epoch: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Build the ``[T, N_FEATURES]`` float32 feature matrix for one shard.

    Parameters mirror :mod:`btcpred.data.schema` exactly. ``start_epoch`` is
    the unix second of row 0 and is only used for the calendar channels.

    ``out`` may be a writable mmap slice, which lets the cache builder stream
    shard by shard without ever holding the whole dataset in RAM.
    """
    t = c.shape[0]
    if out is None:
        out = np.empty((t, N_FEATURES), dtype=np.float32)
    assert out.shape == (t, N_FEATURES), out.shape

    close = c.astype(np.float64) / PRICE_SCALE
    # Guard against a zero close leaking in from a malformed shard; a division
    # by zero here would poison the cache with inf and be a nightmare to trace.
    np.maximum(close, 1e-8, out=close)

    high = close + h_c.astype(np.float64) / PRICE_SCALE
    low = close + l_c.astype(np.float64) / PRICE_SCALE
    open_ = close + o_c.astype(np.float64) / PRICE_SCALE

    log_close = np.log(close)
    ret = np.empty(t, dtype=np.float64)
    ret[0] = 0.0
    np.subtract(log_close[1:], log_close[:-1], out=ret[1:])

    vol = v.astype(np.float64) / VOLUME_SCALE
    taker = tb.astype(np.float64) / VOLUME_SCALE
    trades = n.astype(np.float64)

    out[:, 0] = ret * _RETURN_SCALE
    out[:, 1] = (high - low) / close * _RANGE_SCALE
    out[:, 2] = (close - open_) / close * _RANGE_SCALE
    out[:, 3] = (high - np.maximum(open_, close)) / close * _WICK_SCALE
    out[:, 4] = (np.minimum(open_, close) - low) / close * _WICK_SCALE
    out[:, 5] = np.log1p(vol)
    # Taker imbalance is undefined on a silent second; 0 (balanced) is the
    # right neutral prior there, and the is_filled channel tells the model
    # that the 0 is "unknown" rather than "measured balanced".
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(vol > 0, 2.0 * taker / np.maximum(vol, 1e-12) - 1.0, 0.0)
    out[:, 6] = np.clip(imb, -1.0, 1.0)
    out[:, 7] = np.log1p(trades)
    out[:, 8] = np.log1p(np.where(trades > 0, vol / np.maximum(trades, 1.0), 0.0))
    out[:, 9] = flags.astype(np.float32)

    tsec = start_epoch + np.arange(t, dtype=np.int64)
    tod = (tsec % SECONDS_PER_DAY).astype(np.float64) * (2 * np.pi / SECONDS_PER_DAY)
    dow = (tsec % SECONDS_PER_WEEK).astype(np.float64) * (2 * np.pi / SECONDS_PER_WEEK)
    out[:, 10] = np.sin(tod)
    out[:, 11] = np.cos(tod)
    out[:, 12] = np.sin(dow)
    out[:, 13] = np.cos(dow)

    if not np.isfinite(out).all():
        bad = np.argwhere(~np.isfinite(out))
        raise ValueError(
            f"non-finite feature produced at {bad[:5].tolist()} "
            f"(channels {[FEATURE_NAMES[i] for i in sorted(set(bad[:, 1]))]})"
        )
    return out


def log_price(c: np.ndarray) -> np.ndarray:
    """Natural log of the close price, float32, for target construction."""
    return np.log(c.astype(np.float64) / PRICE_SCALE).astype(np.float32)
