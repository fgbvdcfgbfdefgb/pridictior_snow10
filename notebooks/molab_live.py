"""Live BTC price + 25-minute forecast, CoinMarketCap styling.

marimo notebook for molab (RTX PRO 6000). Run with::

    marimo edit notebooks/molab_live.py

ONE CHART, UPDATED IN PLACE
---------------------------
The usual failure here is a new chart appearing under the old one every tick.
That happens when a cell *appends* to its output (``mo.output.append``), or
when the figure is built inside a loop, or when each tick creates a new cell.

This notebook avoids it structurally:

* The refresh timer (``ticker``) is a ``mo.ui.refresh`` element.
* Exactly one cell reads ``ticker`` and *returns* a single figure.
* marimo's reactive runtime re-executes that one cell on each tick and
  **replaces** its output. A cell has one output, so there is physically only
  ever one chart on screen.
* All mutable state (model, ring buffer, history) lives in cells that do NOT
  depend on ``ticker``, so it is created once and never rebuilt.

Do not add ``mo.output.append`` anywhere in the render cell, and do not move
the figure construction into the setup cell.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full", app_title="BTC 25-minute forecast")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Bitcoin — live price & 25-minute forecast

        Market Analyser (CPU) → Price Predictor (GPU) → a forecast refreshed
        every second. The shaded band is the model's p10–p90 range when the
        loaded checkpoint has quantile heads.
        """
    )
    return (mo,)


@app.cell
def _():
    # --- imports and path setup -----------------------------------------
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go

    REPO = Path(__file__).resolve().parent.parent
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))

    from btcpred.live.binance_feed import LiveKlineStream, fetch_recent_bars
    from btcpred.live.runner import Bar, LivePredictor

    return (
        Bar,
        LiveKlineStream,
        LivePredictor,
        Path,
        REPO,
        fetch_recent_bars,
        go,
        np,
        sys,
        time,
    )


@app.cell
def _(REPO, mo):
    # --- controls --------------------------------------------------------
    import glob

    found = sorted(glob.glob(str(REPO / "runs" / "**" / "best*.pt"), recursive=True))
    checkpoint = mo.ui.dropdown(
        options=found or ["<no checkpoint found>"],
        value=(found or ["<no checkpoint found>"])[0],
        label="checkpoint",
    )
    source = mo.ui.radio(
        options=["live websocket", "replay archive"],
        value="live websocket",
        label="data source",
    )
    lookback = mo.ui.slider(
        5, 120, value=30, step=5, label="minutes of history shown"
    )
    smoothing = mo.ui.slider(
        0, 30, value=5, step=1, label="forecast EMA half-life (s)"
    )
    mo.hstack([checkpoint, source, lookback, smoothing], justify="start", gap=2)
    return checkpoint, found, glob, lookback, smoothing, source


@app.cell
def _(
    Bar,
    LiveKlineStream,
    LivePredictor,
    checkpoint,
    fetch_recent_bars,
    mo,
    np,
    smoothing,
    source,
):
    # --- one-time setup: model, feed, history ----------------------------
    # This cell deliberately does NOT depend on `ticker`, so it runs once.
    # It re-runs only if you change the checkpoint or the data source.
    state = {
        "predictor": None,
        "stream": None,
        "replay": None,
        "times": [],
        "prices": [],
        "forecast": None,
        "error": None,
        "last_update": None,
        "n_ticks": 0,
    }

    try:
        if checkpoint.value and not checkpoint.value.startswith("<"):
            state["predictor"] = LivePredictor(
                checkpoint.value,
                device="cuda",
                ema_halflife=float(smoothing.value),
            )
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"could not load checkpoint: {exc}"

    if source.value == "live websocket":
        try:
            warm = fetch_recent_bars(12 * 3600)
            if state["predictor"] is not None:
                state["predictor"].warm_start(warm)
            state["times"] = [b.epoch for b in warm[-7200:]]
            state["prices"] = [b.close for b in warm[-7200:]]
            state["stream"] = LiveKlineStream().start()
        except Exception as exc:  # noqa: BLE001
            state["error"] = f"feed unavailable: {exc}"
    else:
        from btcpred.data.cache import FeatureCache
        from btcpred.data.replay import ReplayStream

        try:
            cache = FeatureCache("data/cache")
            state["replay"] = ReplayStream(cache, speed=60.0)
        except Exception as exc:  # noqa: BLE001
            state["error"] = f"replay cache unavailable: {exc}"

    mo.md(
        f"**status:** {state['error'] or 'ready'} · "
        f"model: `{getattr(state['predictor'], 'variant_name', 'none')}`"
    )
    return (state,)


@app.cell
def _(mo):
    # --- the heartbeat ----------------------------------------------------
    ticker = mo.ui.refresh(
        options=["1s", "2s", "5s"], default_interval="1s", label="refresh"
    )
    ticker
    return (ticker,)


@app.cell
def _():
    # --- CoinMarketCap palette -------------------------------------------
    CMC = {
        "bg": "#0D1421",
        "panel": "#171924",
        "grid": "rgba(255,255,255,0.06)",
        "text": "#A1A7BB",
        "title": "#FFFFFF",
        "up": "#16C784",
        "down": "#EA3943",
        "up_fill": "rgba(22,199,132,0.18)",
        "down_fill": "rgba(234,57,67,0.18)",
        "forecast": "#3861FB",
        "forecast_fill": "rgba(56,97,251,0.14)",
    }
    return (CMC,)


@app.cell
def _(Bar, CMC, go, lookback, mo, np, state, ticker):
    # --- THE single chart cell -------------------------------------------
    # Depends on `ticker`, so marimo re-runs it every second and REPLACES
    # this cell's output. Returning one figure is what keeps it to one chart.
    ticker  # noqa: B018 - the reactive dependency, intentionally referenced

    import datetime as _dt

    # 1. ingest whatever arrived since the last tick
    new_bars = []
    if state["stream"] is not None:
        new_bars = state["stream"].drain()
    elif state["replay"] is not None:
        for _ in range(60):
            t = state["replay"].step()
            if t is None:
                break
            new_bars.append(
                Bar(epoch=t.epoch, open=float(np.exp(t.log_price)),
                    high=float(np.exp(t.log_price)), low=float(np.exp(t.log_price)),
                    close=float(np.exp(t.log_price)))
            )

    for b in new_bars:
        state["times"].append(b.epoch)
        state["prices"].append(b.close)
        if state["predictor"] is not None:
            state["predictor"].push(b)
    state["n_ticks"] += len(new_bars)

    keep = int(lookback.value) * 60
    if len(state["times"]) > keep:
        state["times"] = state["times"][-keep:]
        state["prices"] = state["prices"][-keep:]

    # 2. refresh the forecast
    if state["predictor"] is not None and new_bars:
        try:
            fc = state["predictor"].predict()
            if fc is not None:
                state["forecast"] = fc
        except Exception as exc:  # noqa: BLE001
            state["error"] = f"inference: {exc}"

    # 3. draw -- one figure, rebuilt from scratch, returned once
    hist_t = [_dt.datetime.fromtimestamp(t, _dt.timezone.utc) for t in state["times"]]
    hist_p = state["prices"]
    fig = go.Figure()

    if hist_p:
        rising = hist_p[-1] >= hist_p[0]
        colour = CMC["up"] if rising else CMC["down"]
        fill = CMC["up_fill"] if rising else CMC["down_fill"]
        fig.add_trace(
            go.Scatter(
                x=hist_t, y=hist_p, mode="lines", name="BTC/USDT",
                line=dict(color=colour, width=2, shape="linear"),
                fill="tozeroy", fillcolor=fill,
                hovertemplate="%{x|%H:%M:%S}  $%{y:,.2f}<extra></extra>",
            )
        )

    fc = state["forecast"]
    if fc is not None:
        f_t = [
            _dt.datetime.fromtimestamp(fc.epoch + 1 + i, _dt.timezone.utc)
            for i in range(fc.horizon_s)
        ]
        if fc.quantile_prices is not None:
            lo, _, hi = fc.quantile_prices
            fig.add_trace(
                go.Scatter(
                    x=f_t + f_t[::-1],
                    y=list(hi) + list(lo[::-1]),
                    fill="toself", fillcolor=CMC["forecast_fill"],
                    line=dict(width=0), hoverinfo="skip",
                    name="p10-p90", showlegend=True,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=f_t, y=fc.prices, mode="lines", name="forecast +25m",
                line=dict(color=CMC["forecast"], width=2, dash="dot"),
                hovertemplate="%{x|%H:%M:%S}  $%{y:,.2f}<extra>forecast</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[f_t[-1]], y=[fc.terminal_price], mode="markers+text",
                marker=dict(color=CMC["forecast"], size=9),
                text=[f"  ${fc.terminal_price:,.0f} ({fc.terminal_bps:+.0f} bp)"],
                textposition="middle right",
                textfont=dict(color=CMC["title"], size=12),
                showlegend=False, hoverinfo="skip",
            )
        )

    # y-range padded around the visible data only, so the line uses the full
    # panel height the way CMC's does instead of hugging the bottom.
    ys = list(hist_p) + (list(fc.prices) if fc is not None else [])
    if ys:
        lo_y, hi_y = min(ys), max(ys)
        pad = max((hi_y - lo_y) * 0.12, hi_y * 0.0005)
        yrange = [lo_y - pad, hi_y + pad]
    else:
        yrange = None

    last = hist_p[-1] if hist_p else float("nan")
    change = (
        (hist_p[-1] / hist_p[0] - 1.0) * 100.0 if len(hist_p) > 1 else 0.0
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CMC["bg"],
        plot_bgcolor=CMC["bg"],
        font=dict(
            family="Inter, SF Pro Display, Helvetica Neue, Arial", color=CMC["text"]
        ),
        title=dict(
            text=(
                f"<b style='color:{CMC['title']};font-size:22px'>"
                f"${last:,.2f}</b>"
                f"<span style='color:{CMC['up'] if change >= 0 else CMC['down']};"
                f"font-size:15px'>  {change:+.2f}%</span>"
                f"<span style='color:{CMC['text']};font-size:12px'>"
                f"   ·  BTC/USDT  ·  {lookback.value}m window</span>"
            ),
            x=0.01, y=0.95,
        ),
        margin=dict(l=10, r=120, t=60, b=30),
        height=560,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showline=False,
            tickformat="%H:%M", color=CMC["text"],
        ),
        yaxis=dict(
            showgrid=True, gridcolor=CMC["grid"], zeroline=False,
            side="right", tickprefix="$", tickformat=",.0f",
            range=yrange, color=CMC["text"],
        ),
        uirevision="btc-live",  # keeps user zoom/pan across refreshes
    )
    if hist_t:
        fig.add_vline(
            x=hist_t[-1], line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot")
        )

    # exactly one output -> exactly one chart on screen
    mo.ui.plotly(fig)
    return CMC, fc, fig, hist_p, hist_t


@app.cell
def _(mo, state):
    # --- telemetry (separate cell, separate single output) ---------------
    fc = state["forecast"]
    rows = [
        ("ticks ingested", f"{state['n_ticks']:,}"),
        ("buffer ready", str(getattr(state["predictor"], "ready", False))),
        ("inference latency", f"{fc.latency_ms:.1f} ms" if fc else "—"),
        ("forecast anchor", f"${fc.anchor_price:,.2f}" if fc else "—"),
        ("25m target", f"${fc.terminal_price:,.2f}" if fc else "—"),
        ("expected move", f"{fc.terminal_bps:+.1f} bp" if fc else "—"),
        ("error", state["error"] or "none"),
    ]
    mo.md(
        "| metric | value |\n|---|---|\n"
        + "\n".join(f"| {k} | {v} |" for k, v in rows)
    )
    return (fc, rows)


if __name__ == "__main__":
    app.run()
