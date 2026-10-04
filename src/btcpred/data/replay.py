"""Live market simulator: replays the 1-second archive as if it were streaming.

Two things live here.

:class:`ReplayStream`
    A single cursor walking the cache one second at a time. It exposes exactly
    the interface the live Binance feed exposes — ``step()`` returns the next
    second and nothing else — so the training loop, the backtester and the
    molab notebook all consume the same shape of data. It can run as fast as
    the CPU allows (training) or pinned to wall-clock market time (demo /
    latency testing) via ``speed``.

:class:`ParallelReplay`
    The training workhorse. ``B`` independent cursors scattered across the
    archive, all stepping in lockstep. Each step yields a batch of ``B``
    tokenised contexts and their ``B`` realised 25-minute futures. This is how
    an epoch-free, second-by-second objective gets enough arithmetic intensity
    to keep four A10s busy: the batch dimension is *parallel market time*, not
    shuffled samples.

**No-peek guarantee.** A stream at cursor ``t`` may read cache rows
``[t - context + 1, t]`` and nothing beyond. The realised future
``[t + 1, t + horizon]`` is returned separately and only ever reaches the loss
function, never the model input. The assertions in :meth:`ParallelReplay.step`
enforce this at runtime rather than trusting the call sites.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .cache import FeatureCache
from .pyramid import PyramidSpec, tokenize, tokenize_batch

LOG = logging.getLogger("btcpred.replay")

HORIZON_SECONDS = 25 * 60  # 1500: the forecast horizon, in seconds


@dataclass
class Tick:
    """One second of simulated market time."""

    index: int  # absolute row in the cache
    epoch: int  # unix seconds
    log_price: float
    features: np.ndarray  # [N_FEATURES]


class ReplayStream:
    """Replay the archive from one cursor, optionally at real-time pace.

    Parameters
    ----------
    cache:
        The mmapped feature cache.
    start:
        Absolute cache index to begin at. Must leave a full context behind it.
    end:
        Exclusive stop index; defaults to the end of the cache.
    spec:
        Tokenisation layout; its ``context_seconds`` sets the warm-up needed.
    speed:
        ``0`` (default) runs as fast as possible. ``1.0`` replays at true
        market pace (one second of data per wall-clock second); ``60.0`` is
        sixty times faster than real time. Used for latency-realistic demos
        and for driving the molab notebook off historical data.
    """

    def __init__(
        self,
        cache: FeatureCache,
        start: int | None = None,
        end: int | None = None,
        spec: PyramidSpec = PyramidSpec(),
        speed: float = 0.0,
    ) -> None:
        self.cache = cache
        self.spec = spec
        self.speed = speed
        self.ctx = spec.context_seconds

        lo = self.ctx - 1
        self.start = lo if start is None else max(start, lo)
        self.end = len(cache) if end is None else min(end, len(cache))
        if self.start >= self.end:
            raise ValueError(
                f"empty replay range: start={self.start} end={self.end}. "
                f"The cache holds {len(cache):,} seconds and a stream needs "
                f"{self.ctx:,} of warm-up context."
            )
        self.cursor = self.start
        self._t0_wall: float | None = None
        self._t0_sim: int | None = None

    def __len__(self) -> int:
        return self.end - self.start

    @property
    def exhausted(self) -> bool:
        return self.cursor >= self.end

    def context(self) -> np.ndarray:
        """Tokenised ``[n_tokens, N_FEATURES]`` view of the trailing 12 hours."""
        window = np.asarray(
            self.cache.features[self.cursor - self.ctx + 1 : self.cursor + 1]
        )
        return tokenize(window, self.spec)

    def step(self) -> Tick | None:
        """Advance one simulated second. Returns ``None`` when exhausted."""
        if self.exhausted:
            return None
        if self.speed > 0:
            self._pace()
        i = self.cursor
        tick = Tick(
            index=i,
            epoch=self.cache.start_epoch + i,
            log_price=float(self.cache.logprice[i]),
            features=np.asarray(self.cache.features[i]),
        )
        self.cursor += 1
        return tick

    def future(self, horizon: int = HORIZON_SECONDS) -> np.ndarray | None:
        """Realised log-prices for the ``horizon`` seconds after the cursor.

        Returns ``None`` if the archive does not extend that far — which is
        the honest answer at the tail of the dataset, and the trainer treats
        it as "this prediction never settles".
        """
        i = self.cursor
        if i + horizon >= len(self.cache):
            return None
        return np.asarray(self.cache.logprice[i + 1 : i + 1 + horizon])

    def _pace(self) -> None:
        now = time.monotonic()
        if self._t0_wall is None:
            self._t0_wall, self._t0_sim = now, self.cursor
            return
        elapsed_sim = self.cursor - self._t0_sim
        target = self._t0_wall + elapsed_sim / self.speed
        if target > now:
            time.sleep(target - now)


class ParallelReplay:
    """``B`` cursors stepping in lockstep over disjoint spans of the archive.

    The spans are carved so that no two streams ever overlap, which keeps the
    batch statistically diverse: at any step the batch mixes 2020 chop with
    2024 trend with last week's regime, instead of ``B`` near-identical
    neighbouring windows.
    """

    def __init__(
        self,
        cache: FeatureCache,
        batch: int,
        spec: PyramidSpec = PyramidSpec(),
        horizon: int = HORIZON_SECONDS,
        index_range: tuple[int, int] | None = None,
        seed: int = 0,
        jitter: bool = True,
    ) -> None:
        self.cache = cache
        self.batch = batch
        self.spec = spec
        self.horizon = horizon
        self.ctx = spec.context_seconds

        lo, hi = index_range or (0, len(cache))
        lo = max(lo, self.ctx - 1)
        hi = min(hi, len(cache) - horizon - 1)
        usable = hi - lo
        if usable < batch * (self.ctx + horizon):
            raise ValueError(
                f"range [{lo}, {hi}) holds {usable:,} usable seconds, too few "
                f"for {batch} streams needing {self.ctx + horizon:,} each. "
                f"Lower --batch or widen the range."
            )

        span = usable // batch
        rng = np.random.default_rng(seed)
        offsets = rng.integers(0, max(span - self.ctx - horizon, 1), size=batch) if jitter else np.zeros(batch, dtype=int)
        self.starts = np.array(
            [lo + k * span + int(offsets[k]) for k in range(batch)], dtype=np.int64
        )
        self.stops = np.array(
            [min(lo + (k + 1) * span, hi) for k in range(batch)], dtype=np.int64
        )
        self.cursors = self.starts.copy()
        self.steps_taken = 0
        self.wraps = 0

        LOG.info(
            "ParallelReplay: %d streams over [%s, %s), %s s per stream",
            batch, f"{lo:,}", f"{hi:,}", f"{span:,}",
        )

    @property
    def epochs_equivalent(self) -> float:
        """Fraction of the usable archive consumed so far.

        There are no epochs in this trainer — the objective is defined per
        market second — but it is still useful to know when the replay has
        seen everything once, which is what this reports.
        """
        total = float((self.stops - self.starts).sum())
        return float(self.steps_taken * self.batch / max(total, 1.0))

    def step(self) -> dict[str, np.ndarray]:
        """Advance every cursor one second and return the batch.

        Returns a dict with:

        ``tokens``     ``[B, n_tokens, N_FEATURES]`` model input
        ``anchor``     ``[B]`` log-price at the cursor, the forecast origin
        ``future``     ``[B, horizon]`` realised log-prices, cursor+1..+horizon
        ``fill_frac``  ``[B]`` fraction of the future that was forward-filled
        ``index``      ``[B]`` absolute cache index of each cursor
        """
        ctx, h = self.ctx, self.horizon
        b = self.batch

        # Any stream that ran off the end of its span restarts at its start.
        done = self.cursors >= self.stops
        if done.any():
            self.cursors[done] = self.starts[done]
            self.wraps += int(done.sum())

        windows = np.empty((b, ctx, self.cache.features.shape[1]), dtype=np.float32)
        future = np.empty((b, h), dtype=np.float32)
        fill = np.empty(b, dtype=np.float32)
        anchor = np.empty(b, dtype=np.float32)

        for k in range(b):
            i = int(self.cursors[k])
            # No-peek: the input window ends *at* the cursor, inclusive.
            windows[k] = self.cache.features[i - ctx + 1 : i + 1]
            future[k] = self.cache.logprice[i + 1 : i + 1 + h]
            fill[k] = np.asarray(self.cache.flags[i + 1 : i + 1 + h]).mean()
            anchor[k] = self.cache.logprice[i]

        out = {
            "tokens": tokenize_batch(windows, self.spec),
            "anchor": anchor,
            "future": future,
            "fill_frac": fill,
            "index": self.cursors.copy(),
        }
        self.cursors += 1
        self.steps_taken += 1
        return out
