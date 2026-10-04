"""Multi-resolution tokenisation of the 12-hour context window.

A 12-hour window at 1-second resolution is 43,200 steps. Feeding that to a
transformer is not an option, and feeding a uniformly downsampled version
throws away exactly the information a 25-minute-ahead forecast needs most:
what happened in the last few minutes.

So the window is tokenised at *decaying* resolution — every second near the
cursor, coarse bars far from it:

=========================  ========  ========
span (seconds before now)  bin size  tokens
=========================  ========  ========
0 - 300                    1 s       300
300 - 1,800                10 s      150
1,800 - 7,200              60 s      90
7,200 - 43,200             300 s     120
=========================  ========  ========

660 tokens total, covering the full 12 hours, with the last five minutes at
native tick resolution. Cost is linear in tokens; a uniform 1-second encoding
would have been 65x larger for no extra information about the deep past.

Pooling is **not** uniform across channels, and this is the subtle part. A
log-return is additive over time: the 10-second return is the *sum* of ten
1-second returns, not their mean. A high-low range is an extent: it takes the
*max*. Volume is additive. Getting this wrong produces tokens whose scale
silently depends on the bin width, which the model then has to spend capacity
undoing.

The same function is used by the trainer, the offline evaluator and the live
notebook. That is deliberate: tokenisation skew between training and serving
is the single most common way a forecaster that looks good offline dies in
production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from .schema import FEATURE_NAMES, N_FEATURES

Reducer = Literal["sum", "mean", "max", "last"]

#: How each feature channel collapses when several seconds share a bin.
CHANNEL_REDUCERS: Final[dict[str, Reducer]] = {
    "log_return": "sum",  # additive over time
    "hl_range": "max",  # an extent, not an average
    "body": "sum",  # open-to-close change is additive like a return
    "upper_wick": "max",
    "lower_wick": "max",
    "log_volume": "sum",  # log1p of a sum is close enough and keeps scale sane
    "taker_imbalance": "mean",
    "log_trades": "sum",
    "avg_trade_size": "mean",
    "is_filled": "mean",  # fraction of the bin that was synthesised
    "tod_sin": "last",  # calendar channels describe the bin's end instant
    "tod_cos": "last",
    "dow_sin": "last",
    "dow_cos": "last",
}

DEFAULT_LEVELS: Final[tuple[tuple[int, int], ...]] = (
    (300, 1),
    (1_800, 10),
    (7_200, 60),
    (43_200, 300),
)


@dataclass(frozen=True)
class PyramidSpec:
    """Immutable description of a tokenisation layout.

    ``levels`` is a tuple of ``(span_back_seconds, bin_seconds)`` with strictly
    increasing spans. Each level covers from the previous level's span back to
    its own, so the levels tile the window without overlap.
    """

    levels: tuple[tuple[int, int], ...] = DEFAULT_LEVELS

    def __post_init__(self) -> None:
        spans = [s for s, _ in self.levels]
        if spans != sorted(spans) or len(set(spans)) != len(spans):
            raise ValueError("pyramid levels must have strictly increasing spans")
        prev = 0
        for span, bin_s in self.levels:
            if (span - prev) % bin_s:
                raise ValueError(
                    f"level {span}s/{bin_s}s does not tile evenly "
                    f"({span - prev} is not a multiple of {bin_s})"
                )
            prev = span

    @property
    def context_seconds(self) -> int:
        """Total lookback, i.e. the span of the coarsest level."""
        return self.levels[-1][0]

    @property
    def n_tokens(self) -> int:
        prev, total = 0, 0
        for span, bin_s in self.levels:
            total += (span - prev) // bin_s
            prev = span
        return total

    def bin_widths(self) -> np.ndarray:
        """``[n_tokens]`` width in seconds of each token, oldest token first."""
        chunks = []
        prev = 0
        for span, bin_s in self.levels:
            chunks.append(np.full((span - prev) // bin_s, bin_s, dtype=np.int32))
            prev = span
        # Levels are listed newest-first; the token sequence runs oldest-first.
        return np.concatenate(chunks[::-1])

    def token_end_offsets(self) -> np.ndarray:
        """Seconds-before-cursor at which each token's bin *ends*, oldest first.

        Offset 0 means the bin ends at the cursor itself (the newest token).
        """
        widths = self.bin_widths()
        # Cumulative from the newest end backwards.
        ends = np.cumsum(widths[::-1]) - widths[::-1]
        return ends[::-1].astype(np.int32)


def _reducer_codes(spec_names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    code = {"sum": 0, "mean": 1, "max": 2, "last": 3}
    return np.array([code[CHANNEL_REDUCERS[n]] for n in spec_names], dtype=np.int8)


_CODES = _reducer_codes()


def tokenize(window: np.ndarray, spec: PyramidSpec = PyramidSpec()) -> np.ndarray:
    """Pool a ``[context_seconds, N_FEATURES]`` window into ``[n_tokens, N_FEATURES]``.

    ``window`` must be ordered oldest-to-newest, with its final row being the
    cursor second. Returns float32, oldest token first.
    """
    ctx = spec.context_seconds
    if window.shape != (ctx, N_FEATURES):
        raise ValueError(
            f"expected window of shape ({ctx}, {N_FEATURES}), got {window.shape}"
        )

    out = np.empty((spec.n_tokens, N_FEATURES), dtype=np.float32)
    write = spec.n_tokens  # we fill from the newest token backwards
    read = ctx  # exclusive end index into `window`

    for span, bin_s in spec.levels:
        prev_span = 0 if span == spec.levels[0][0] else _prev_span(spec, span)
        n_bins = (span - prev_span) // bin_s
        seg = window[read - n_bins * bin_s : read]
        read -= n_bins * bin_s
        block = seg.reshape(n_bins, bin_s, N_FEATURES)

        pooled = np.empty((n_bins, N_FEATURES), dtype=np.float32)
        pooled[:, _CODES == 0] = block[:, :, _CODES == 0].sum(axis=1)
        pooled[:, _CODES == 1] = block[:, :, _CODES == 1].mean(axis=1)
        pooled[:, _CODES == 2] = block[:, :, _CODES == 2].max(axis=1)
        pooled[:, _CODES == 3] = block[:, -1, _CODES == 3]

        out[write - n_bins : write] = pooled
        write -= n_bins

    assert write == 0 and read == 0, (write, read)
    return out


def tokenize_batch(windows: np.ndarray, spec: PyramidSpec = PyramidSpec()) -> np.ndarray:
    """Vectorised :func:`tokenize` over a leading batch dimension.

    ``windows`` is ``[B, context_seconds, N_FEATURES]``; returns
    ``[B, n_tokens, N_FEATURES]``. This is the hot path in the dataloader, so
    it avoids the per-sample Python loop entirely.
    """
    b, ctx, f = windows.shape
    if (ctx, f) != (spec.context_seconds, N_FEATURES):
        raise ValueError(
            f"expected [B, {spec.context_seconds}, {N_FEATURES}], got {windows.shape}"
        )

    out = np.empty((b, spec.n_tokens, N_FEATURES), dtype=np.float32)
    write, read = spec.n_tokens, ctx
    for span, bin_s in spec.levels:
        prev_span = 0 if span == spec.levels[0][0] else _prev_span(spec, span)
        n_bins = (span - prev_span) // bin_s
        seg = windows[:, read - n_bins * bin_s : read, :]
        read -= n_bins * bin_s
        block = seg.reshape(b, n_bins, bin_s, f)

        pooled = np.empty((b, n_bins, f), dtype=np.float32)
        pooled[:, :, _CODES == 0] = block[:, :, :, _CODES == 0].sum(axis=2)
        pooled[:, :, _CODES == 1] = block[:, :, :, _CODES == 1].mean(axis=2)
        pooled[:, :, _CODES == 2] = block[:, :, :, _CODES == 2].max(axis=2)
        pooled[:, :, _CODES == 3] = block[:, :, -1, _CODES == 3]

        out[:, write - n_bins : write, :] = pooled
        write -= n_bins

    assert write == 0 and read == 0, (write, read)
    return out


def _prev_span(spec: PyramidSpec, span: int) -> int:
    spans = [s for s, _ in spec.levels]
    return spans[spans.index(span) - 1]
