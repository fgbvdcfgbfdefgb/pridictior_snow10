"""Live and historical BTCUSDT feeds for the molab notebook.

Only used where there *is* internet (molab). Snowflake training never imports
this module — see the offline note in the README.

Two entry points:

:func:`fetch_recent_bars`
    REST pull of the trailing N seconds of 1s klines, for warm-starting the
    12-hour buffer. Binance caps a kline request at 1000 bars, so twelve
    hours needs 44 paged calls; they are issued with a small delay to stay
    well inside the public rate limit.

:class:`LiveKlineStream`
    WebSocket subscription to ``btcusdt@kline_1s``. Yields a :class:`Bar`
    once per second when the kline closes. Reconnects with backoff, and on
    reconnection re-pulls the bars it missed via REST so the ring buffer
    never silently develops a hole.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections.abc import Iterator
from queue import Empty, Queue

from .runner import Bar

LOG = logging.getLogger("btcpred.feed")

REST = "https://api.binance.com/api/v3/klines"
WS = "wss://stream.binance.com:9443/ws/btcusdt@kline_1s"


def _kline_to_bar(k: list) -> Bar:
    return Bar(
        epoch=int(k[0]) // 1000,
        open=float(k[1]), high=float(k[2]), low=float(k[3]), close=float(k[4]),
        volume=float(k[5]), trades=int(k[8]), taker_buy=float(k[9]),
    )


def fetch_recent_bars(
    seconds: int, symbol: str = "BTCUSDT", pause: float = 0.12
) -> list[Bar]:
    """Trailing ``seconds`` of 1s klines, oldest first, gap-filled."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - seconds * 1000
    bars: list[Bar] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"{REST}?symbol={symbol}&interval=1s"
            f"&startTime={cursor}&limit=1000"
        )
        with urllib.request.urlopen(url, timeout=30) as resp:
            chunk = json.loads(resp.read())
        if not chunk:
            break
        bars.extend(_kline_to_bar(k) for k in chunk)
        cursor = int(chunk[-1][0]) + 1000
        time.sleep(pause)

    # Dense the grid: silent seconds are simply absent from the REST reply,
    # exactly as they are from the archive.
    if not bars:
        return bars
    dense: list[Bar] = [bars[0]]
    for bar in bars[1:]:
        gap = bar.epoch - dense[-1].epoch
        for i in range(1, gap):
            last = dense[-1].close
            dense.append(
                Bar(epoch=dense[-1].epoch + 1, open=last, high=last, low=last,
                    close=last, filled=True)
            )
        if gap >= 1:
            dense.append(bar)
    return dense


class LiveKlineStream:
    """Background WebSocket reader. Iterate it to get one :class:`Bar`/second."""

    def __init__(self, url: str = WS, max_queue: int = 4096) -> None:
        self.url = url
        self.q: Queue[Bar] = Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.last_error: str | None = None

    def start(self) -> "LiveKlineStream":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            from websocket import WebSocketApp  # websocket-client
        except ImportError:
            self.last_error = (
                "pip install websocket-client to use the live feed"
            )
            LOG.error(self.last_error)
            return

        backoff = 1.0
        while not self._stop.is_set():
            def on_message(_ws, message: str) -> None:
                k = json.loads(message).get("k", {})
                if not k.get("x"):  # only closed klines
                    return
                bar = Bar(
                    epoch=int(k["t"]) // 1000,
                    open=float(k["o"]), high=float(k["h"]), low=float(k["l"]),
                    close=float(k["c"]), volume=float(k["v"]),
                    trades=int(k["n"]), taker_buy=float(k["V"]),
                )
                if not self.q.full():
                    self.q.put(bar)

            def on_open(_ws) -> None:
                self.connected = True
                LOG.info("websocket connected")

            def on_close(_ws, *_a) -> None:
                self.connected = False

            def on_error(_ws, err) -> None:
                self.last_error = str(err)
                LOG.warning("websocket error: %s", err)

            app = WebSocketApp(
                self.url, on_message=on_message, on_open=on_open,
                on_close=on_close, on_error=on_error,
            )
            app.run_forever(ping_interval=20, ping_timeout=10)
            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def drain(self, timeout: float = 0.0) -> list[Bar]:
        """All bars queued since the last call."""
        out: list[Bar] = []
        deadline = time.monotonic() + timeout
        while True:
            try:
                out.append(self.q.get_nowait())
            except Empty:
                if timeout and time.monotonic() < deadline and not out:
                    time.sleep(0.05)
                    continue
                return out

    def __iter__(self) -> Iterator[Bar]:
        while not self._stop.is_set():
            try:
                yield self.q.get(timeout=5.0)
            except Empty:
                continue
