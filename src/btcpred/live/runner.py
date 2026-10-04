"""Inference engine: one object that turns a stream of ticks into a forecast.

The same class serves the molab live notebook and the offline backtester, which
is the point — if serving used different code from evaluation, the numbers in
the README would be fiction.

Responsibilities:

* keep a rolling 12-hour ring buffer of 1-second features;
* fill gaps in the live feed the same way the archive builder does, so a
  quiet second looks identical in training and in production;
* tokenise with the exact :class:`PyramidSpec` the checkpoint was trained on
  (read from the checkpoint, never assumed);
* run analyser -> predictor and return prices, not log-returns;
* smooth across ticks with an EMA so the displayed curve does not flicker.

The EMA deserves a note. The model is already trained for inter-tick
consistency and emits a band-limited path, so raw output is fairly calm. The
EMA is a second, cheap layer of defence with a tunable time constant: at
``ema_halflife=5`` seconds it removes residual tick noise while still letting
the forecast move decisively on a real regime change within ~15 seconds.
Setting it to 0 disables smoothing entirely if you want to see the raw model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data.features import build_features
from ..data.pyramid import PyramidSpec, tokenize
from ..data.schema import N_FEATURES, PRICE_SCALE, VOLUME_SCALE
from ..models.market_analyser import MarketAnalyser
from ..models.price_predictor import PricePredictor

LOG = logging.getLogger("btcpred.live")


@dataclass
class Bar:
    """One second of market data, as the live feed delivers it."""

    epoch: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    taker_buy: float = 0.0
    trades: int = 0
    filled: bool = False


@dataclass
class Forecast:
    """What the model says at one instant."""

    epoch: int  # anchor time
    anchor_price: float
    horizon_s: int
    prices: np.ndarray  # [H] predicted price path
    log_returns: np.ndarray  # [H] predicted cumulative log-return
    quantile_prices: np.ndarray | None = None  # [3, H] if the head has them
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def terminal_price(self) -> float:
        return float(self.prices[-1])

    @property
    def terminal_bps(self) -> float:
        return float(self.log_returns[-1] * 10_000.0)


class RingBuffer:
    """Fixed-capacity rolling window of raw bars, kept on a dense 1s grid.

    Gaps are forward-filled on insert, exactly as
    :mod:`btcpred.data.download_binance` does for the archive. A feed that
    drops three seconds must not produce a different window shape from an
    archive that had no trades for three seconds.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.c = np.zeros(capacity, dtype=np.float64)
        self.o = np.zeros(capacity, dtype=np.float64)
        self.h = np.zeros(capacity, dtype=np.float64)
        self.l = np.zeros(capacity, dtype=np.float64)
        self.v = np.zeros(capacity, dtype=np.float64)
        self.tb = np.zeros(capacity, dtype=np.float64)
        self.n = np.zeros(capacity, dtype=np.float64)
        self.flags = np.zeros(capacity, dtype=np.int8)
        self.last_epoch: int | None = None
        self.filled = 0

    @property
    def ready(self) -> bool:
        return self.filled >= self.capacity

    @property
    def warmup_remaining(self) -> int:
        return max(0, self.capacity - self.filled)

    def _shift(self, k: int) -> None:
        if k <= 0:
            return
        if k >= self.capacity:
            k = self.capacity
        for arr in (self.c, self.o, self.h, self.l, self.v, self.tb, self.n, self.flags):
            arr[:-k] = arr[k:]
        self.filled = min(self.capacity, self.filled + k)

    def push(self, bar: Bar) -> None:
        if self.last_epoch is not None:
            gap = bar.epoch - self.last_epoch
            if gap <= 0:
                # Duplicate or out-of-order tick: overwrite the last slot
                # rather than corrupting the grid.
                self._write(-1, bar)
                return
            if gap > 1:
                synth = Bar(
                    epoch=0, open=self.c[-1], high=self.c[-1], low=self.c[-1],
                    close=self.c[-1], filled=True,
                )
                for _ in range(min(gap - 1, self.capacity)):
                    self._shift(1)
                    self._write(-1, synth)
        self._shift(1)
        self._write(-1, bar)
        self.last_epoch = bar.epoch

    def _write(self, i: int, bar: Bar) -> None:
        self.c[i] = bar.close
        self.o[i] = bar.open
        self.h[i] = bar.high
        self.l[i] = bar.low
        self.v[i] = bar.volume
        self.tb[i] = bar.taker_buy
        self.n[i] = bar.trades
        self.flags[i] = 1 if bar.filled else 0

    def features(self, end_epoch: int) -> np.ndarray:
        """``[capacity, N_FEATURES]`` window ending at ``end_epoch``."""
        cc = np.rint(self.c * PRICE_SCALE).astype(np.int64)
        return build_features(
            c=cc.astype(np.int32),
            o_c=(np.rint(self.o * PRICE_SCALE) - cc).astype(np.int32),
            h_c=(np.rint(self.h * PRICE_SCALE) - cc).astype(np.int32),
            l_c=(np.rint(self.l * PRICE_SCALE) - cc).astype(np.int32),
            v=np.rint(self.v * VOLUME_SCALE).astype(np.int64),
            tb=np.rint(self.tb * VOLUME_SCALE).astype(np.int64),
            n=self.n.astype(np.int32),
            flags=self.flags,
            start_epoch=end_epoch - self.capacity + 1,
        )


class LivePredictor:
    """Load a checkpoint and forecast from a tick stream."""

    def __init__(
        self,
        checkpoint: Path | str,
        device: str | torch.device = "cuda",
        analyser_device: str | torch.device | None = None,
        ema_halflife: float = 5.0,
        compile_model: bool = False,
    ) -> None:
        ck = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.analyser_device = torch.device(analyser_device or self.device)

        self.spec = PyramidSpec(tuple(tuple(x) for x in ck["spec_levels"]))
        self.horizon = int(ck["horizon"])
        plan = ck["plan"]
        variant = ck["variant"]
        self.variant_name = variant["name"]

        self.analyser = MarketAnalyser(
            n_features=N_FEATURES,
            d_model=plan["analyser_d_model"],
            n_layers=plan["analyser_layers"],
            d_state=plan["analyser_d_model"],
        )
        self.analyser.load_state_dict(ck["analyser"])
        self.analyser.eval().to(self.analyser_device)

        self.predictor = PricePredictor(
            token_offsets=torch.from_numpy(self.spec.token_end_offsets().copy()),
            n_features=N_FEATURES,
            d_analyser=plan["analyser_d_model"],
            d_state=plan["analyser_d_model"],
            d_model=plan["predictor_d_model"],
            n_layers=plan["predictor_layers"],
            n_heads=plan["predictor_heads"],
            ffn_mult=plan["predictor_ffn_mult"],
            n_coeffs=int(ck["n_coeffs"]),
            horizon=self.horizon,
            head=variant["head"],
        )
        self.predictor.load_state_dict(ck["predictor"])
        self.predictor.eval().to(self.device)

        if compile_model:
            # Worth it on the RTX PRO 6000 for a long-lived session; the first
            # call pays ~30s of compilation.
            self.predictor = torch.compile(self.predictor)

        self.buffer = RingBuffer(self.spec.context_seconds)
        self.ema_halflife = ema_halflife
        self._ema: np.ndarray | None = None
        self._last_anchor_log: float | None = None
        self.metrics = ck.get("metrics", {})

        LOG.info(
            "loaded %s (step %s) | predictor %.1fM params | horizon %ds | device %s",
            self.variant_name, ck.get("step"),
            sum(p.numel() for p in self.predictor.parameters()) / 1e6,
            self.horizon, self.device,
        )

    # ------------------------------------------------------------------ #
    def push(self, bar: Bar) -> None:
        self.buffer.push(bar)

    @property
    def ready(self) -> bool:
        return self.buffer.ready

    @torch.no_grad()
    def predict(self) -> Forecast | None:
        """Forecast from the current buffer, or ``None`` during warm-up."""
        import time

        if not self.buffer.ready or self.buffer.last_epoch is None:
            return None
        t0 = time.perf_counter()

        window = self.buffer.features(self.buffer.last_epoch)
        tokens = torch.from_numpy(tokenize(window, self.spec)).unsqueeze(0)

        a = self.analyser(tokens.to(self.analyser_device))
        out = self.predictor(
            tokens.to(self.device),
            a["memory"].to(self.device),
            a["state"].to(self.device),
        )
        path = out["path"][0].float().cpu().numpy()

        # EMA across ticks. The previous forecast referred to an anchor one
        # second earlier, so re-anchor it before blending -- otherwise the
        # smoother fights the market's actual drift and lags systematically.
        anchor_price = float(self.buffer.c[-1])
        anchor_log = float(np.log(max(anchor_price, 1e-9)))
        if self._ema is not None and self.ema_halflife > 0:
            shift = (
                0.0 if self._last_anchor_log is None
                else anchor_log - self._last_anchor_log
            )
            prev = np.concatenate([self._ema[1:] - shift, [self._ema[-1] - shift]])
            alpha = 1.0 - 0.5 ** (1.0 / self.ema_halflife)
            self._ema = alpha * path + (1.0 - alpha) * prev
        else:
            self._ema = path
        self._last_anchor_log = anchor_log
        smoothed = self._ema

        prices = anchor_price * np.exp(smoothed)
        qp = None
        if "quantile_paths" in out:
            qp = anchor_price * np.exp(out["quantile_paths"][0].float().cpu().numpy())

        return Forecast(
            epoch=self.buffer.last_epoch,
            anchor_price=anchor_price,
            horizon_s=self.horizon,
            prices=prices,
            log_returns=smoothed,
            quantile_prices=qp,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            meta={"variant": self.variant_name},
        )

    def warm_start(self, bars: list[Bar]) -> None:
        """Preload history so the first live tick can predict immediately.

        Without this the notebook would stare at a blank chart for twelve
        hours. Feed it the trailing 12h of klines from the REST API.
        """
        for bar in bars:
            self.buffer.push(bar)
        LOG.info(
            "warm start: %d bars, %d seconds still needed",
            len(bars), self.buffer.warmup_remaining,
        )
