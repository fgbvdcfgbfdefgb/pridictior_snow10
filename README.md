# Bitcoin 25-Minute Price Predictor

Second-by-second BTC/USDT forecasting. A **Market Analyser** on CPU extracts
market structure from a rolling 12-hour window; a **Price Predictor** on GPU
turns that into a smooth forecast of the next **25 minutes**, refreshed every
second. Trained online — no epochs — against the real archive, and scored by a
real-time reward each simulated market second.

Built to train offline on **Snowflake (4× A10)** and serve live on **molab
(RTX PRO 6000)** from a marimo notebook with a CoinMarketCap-style chart.

```
                     12h rolling window, 1-second bars
                                   |
                       multi-resolution tokeniser (660 tokens)
                                   |
         +-------------------------+-------------------------+
         |                                                   |
   Market Analyser  (CPU, 27M params)                        |
   dilated causal CNN, DDP/gloo = ONE shared model           | raw tokens
         |  memory [B,660,512] + state [B,512]               | (skip path)
         +-------------------------+-------------------------+
                                   |
                   Price Predictor (GPU, ~794M params/variant)
                   pre-norm transformer, log-time RoPE
                                   |
                   48 smooth-basis coefficients
                                   |
                     1500-step log-return path  ->  prices
```

---

## Table of contents

1. [What is actually in this repo](#1-what-is-actually-in-this-repo)
2. [Quick start](#2-quick-start)
3. [The dataset](#3-the-dataset)
4. [How the model works](#4-how-the-model-works)
5. [Training: online, epoch-free, distributed](#5-training-online-epoch-free-distributed)
6. [The ensemble and model selection](#6-the-ensemble-and-model-selection)
7. [Running on Snowflake (offline)](#7-running-on-snowflake-offline)
8. [Running live on molab](#8-running-live-on-molab)
9. [Throughput, and the one decision you should revisit](#9-throughput-and-the-one-decision-you-should-revisit)
10. [Honest limitations](#10-honest-limitations)
11. [Repository layout](#11-repository-layout)

---

## 1. What is actually in this repo

Everything is real and runs. Specifically:

| Thing | Status |
|---|---|
| BTCUSDT 1-second dataset, 2022-01-01 → 2026-10-03 | **Downloaded and committed** — 58 monthly Parquet shards, 150M seconds, 1.35 GB |
| Downloader (resumable, handles the ms→µs switch and missing seconds) | Working |
| Memory-mapped feature cache builder | Working |
| Live market simulator (single-stream + N-way parallel replay) | Working |
| Market Analyser (CPU) + Price Predictor (GPU) | Working, 19 tests green |
| Online trainer, DDP across 4 GPUs, shared analyser | Working end-to-end on CPU; **GPU path not executed** — see §10 |
| 4 training-method variants + automated selection | Working, verified on real held-out data |
| marimo live notebook, single in-place-updating chart | Written; needs a GPU + feed to exercise |
| Snowflake offline notebook + shell runner | Written |

**What has not happened: a real training run.** The build sandbox has 2 vCPUs,
2 GB of RAM and no GPU. Every component is verified on real market data with
short smoke runs, but the model in this repo is untrained. §10 is explicit
about what that means.

---

## 2. Quick start

```bash
git clone https://github.com/fgbvdcfgbfdefgb/pridictior_snow10.git
cd pridictior_snow10
pip install -r requirements.txt

# 1. Build the feature cache from the committed shards (no network).
#    Full history is ~9 GB and ~7 min; start smaller to get moving:
python scripts/build_cache.py --shards data/btcusdt_1s --cache data/cache \
    --start 2026-08 --end 2026-10

# 2. See what the sizing planner makes of your machine
PYTHONPATH=src python -m btcpred.utils.hardware

# 3. A short run that actually trains
python scripts/train.py --cache data/cache --out runs/smoke \
    --max-steps 500 --batch 4 --d-model 256 --layers 6

# 4. Tests
PYTHONPATH=src pytest tests -q
```

To refresh the dataset up to today (needs internet, ~5 min):

```bash
PYTHONPATH=src python -m btcpred.data.download_binance \
    --out data/btcusdt_1s --workers 4          # 2022-01 → today
# ...or pull the full archive Binance has, back to 2020-01:
PYTHONPATH=src python -m btcpred.data.download_binance \
    --out data/btcusdt_1s --start 2020-01 --workers 4
```

---

## 3. The dataset

**Source.** `data.binance.vision`, Binance's public S3 archive. No API key, no
account. Spot **1-second klines** for BTCUSDT are real tick-grid data, not
minute bars interpolated down.

**What is committed.** `data/btcusdt_1s/` — one Parquet shard per month,
**2022-01 through 2026-10, 150,076,800 seconds, 1.35 GB**, gap-free.

**Why 2022 and not earlier.** Binance publishes 1s klines back to 2020-01 and
the downloader will fetch them (`--start 2020-01`), but the default window
starts at 2022-01. That span still covers four distinct regimes — the 2022
bear market and the LUNA/FTX dislocations, the 2023 grind back, the 2024-25
bull run, and the current market — while dropping the 2020-21 era whose
microstructure is least like today's: thinner books, a different fee and
market-maker landscape, and no spot-ETF flow. For a 25-minute horizon the
model is learning microstructure dynamics, not macro history, so data whose
microstructure no longer exists is closer to noise than to signal.

**Storage format.** Shards hold no timestamp column. Rows sit on a dense,
gap-free 1-second grid, so row *i* of month *M* is unambiguously
`month_start + i` seconds. Prices are stored as integer cents, with open/high/low
as *deltas against the close* — those deltas are tiny integers and compress
about 10× better than four correlated absolute price series. Volumes are
milli-BTC integers. Everything is `DELTA_BINARY_PACKED` + zstd. Net result:
~24 MB per month instead of ~380 MB of raw CSV, with no loss of precision.

**The honest bit about gaps.** A Binance "1s kline" is only emitted for seconds
in which something traded. Quiet seconds are *absent from the CSV*, not zero.
Silently concatenating the rows would compress away real time and shift every
subsequent timestamp. So the builder reindexes onto the dense grid,
forward-fills price (open = high = low = close = last print, volume = 0), and
sets a **`flags` column to 1 on every synthesised row**. Those rows are
down-weighted in the loss (`ObjectiveConfig.filled_weight`, default 0.25)
rather than being passed off as observations. Across the committed history
coverage is 99.9–100% real for every month.

| Column | Type | Meaning |
|---|---|---|
| `c` | int32 | close, in cents |
| `o_c`, `h_c`, `l_c` | int32 | open/high/low minus close, in cents |
| `v`, `tb` | int64 | volume, taker-buy volume, in milli-BTC |
| `n` | int32 | trade count |
| `flags` | int8 | 1 = this second was synthesised |

**Features.** 14 channels, all scale-free (log-returns, ratios, bounded
imbalances, calendar sin/cos). The model never sees an absolute price level —
BTC traded at \$16k in late 2022, and a network that memorises levels dies on
the first new regime. `tests/test_pipeline.py::test_features_are_finite_and_scale_free`
pins this down by checking that feature statistics are unchanged between a
\$16k and a \$120k price series.

**The cache.** `scripts/build_cache.py` expands shards into three flat
memory-mapped arrays on one contiguous timeline. Mapping, not loading, is the
point: every dataloader worker shares the same physical pages, and a random
12-hour window read becomes a pointer offset instead of a zstd decode. ~9.2 GB
for the full history, which the 100 GB Snowflake box holds in page cache
comfortably. The cache is derived data and is `.gitignore`d.

---

## 4. How the model works

### 4.1 The 12-hour context, at decaying resolution

Twelve hours at 1 Hz is 43,200 steps. Feeding that to a transformer is not an
option, and uniformly downsampling it throws away exactly what a 25-minute
forecast needs most — the last few minutes. So the window is tokenised at
resolution that decays with age:

| Age (seconds before now) | Bin | Tokens |
|---|---|---|
| 0 – 300 | 1 s | 300 |
| 300 – 1,800 | 10 s | 150 |
| 1,800 – 7,200 | 60 s | 90 |
| 7,200 – 43,200 | 300 s | 120 |
| | | **660** |

660 tokens cover the full twelve hours with the last five minutes at native
tick resolution. A uniform 1-second encoding would be 65× larger for no extra
information about the deep past.

Pooling is **per-channel and not uniform**, which is the part that is easy to
get wrong. A log-return is additive — the 10-second return is the *sum* of ten
1-second returns, not their mean. A high–low range is an extent and takes the
max. Volume is additive. Calendar channels take the bin's last value. Pool
them all with a mean and every token's scale silently depends on its bin
width, and the model burns capacity undoing that.

The *same function* (`btcpred.data.pyramid.tokenize`) is used by the trainer,
the evaluator and the live notebook. Tokenisation skew between training and
serving is the most common way a forecaster that looks good offline dies in
production.

### 4.2 Market Analyser — CPU

Dilated causal CNN, gated residual blocks, dilations cycling 1→128. At the
planned size (d=512, 10 layers, ~27M params) its receptive field is 1,033
tokens, comfortably more than the 660 it needs to see the whole window.

Why a CNN and not a transformer: this runs on CPU. Attention over 660 tokens
is an `O(n²d)` matmul that CPUs are bad at; a dilated conv stack reaches the
same receptive field in `O(n·d²·k)` with an access pattern BLAS likes.

**One causality trap worth naming.** The obvious normalisation for a `[B, C, T]`
tensor is `nn.GroupNorm(1, C)`. It is wrong here: on an NCL tensor GroupNorm
normalises over channels **and length**, so every output position is divided by
statistics that include future timesteps. That is a textbook target leak — it
produces no error, just a backtest that cannot be reproduced live. This repo
uses a `ChannelNorm` that normalises each `(batch, time)` position
independently, and `test_analyser_cannot_see_the_future` asserts that
perturbing everything after step 300 leaves outputs before step 300 bit-identical.

The analyser also emits six interpretable auxiliary signals (realised vol at
5m/1h, 15m trend slope, volume burst, order-flow imbalance, range expansion)
supervised against values computed exactly from the input window. They leak
nothing, and they give the CPU stack a dense, well-conditioned gradient of its
own instead of waiting for credit to trickle back through the predictor.

### 4.3 Price Predictor — GPU

Pre-norm transformer, SwiGLU feed-forwards, RMSNorm, RoPE. Two problem-specific
details:

**Dual input path.** It consumes the analyser's output *and* the raw tokens
(the brief asks for both). Without the skip connection the analyser would be an
information bottleneck that early training has no way to widen. With it, the
analyser is strictly additive: worst case it contributes nothing and the
predictor still works.

**Non-uniform positions.** Tokens are not equally spaced in time — 300 of them
span one second each, 120 span five minutes each. Feeding RoPE an ordinal index
would assert that the gap between tokens 0 and 1 equals the gap between 658 and
659, wrong by 300×. We feed it `log(1 + seconds_before_now)` instead, so
geometric distance in time maps to linear phase.

### 4.4 The output: 48 coefficients, not 1500 numbers

The predictor emits 48 coefficients of a fixed half-cycle sine basis; the path
is `coeffs @ B`. This is the mechanism behind the "predictions must be stable"
requirement:

* **Stability is structural.** A head with 1500 free outputs can emit a
  sawtooth. A 48-term band-limited expansion physically cannot. Smoothness is a
  property of the parameterisation, not a penalty we hope the optimiser
  respects. The basis vanishes at the forecast origin, so every path starts at
  the anchor price and cannot open with a jump.
* **Capacity.** 1500 outputs from a 1536-wide model is a 2.3M-parameter head
  that mostly learns to copy its neighbours. 48 outputs is 74k.
* **Conditioning.** Adjacent horizons are ~0.999 correlated; regressing them
  independently spends the gradient budget on a direction the data barely
  distinguishes.

Coefficients are decay-scaled by `1/(k+1)`, so producing a high-frequency
wiggle requires a correspondingly large logit — the prior leans smooth but the
model can still move fast when the data demands it. Measured against real
random-walk paths the basis captures **99.6%** of path variance at K=48 (99.2%
at K=24, 99.8% at K=96), so it is not a meaningful handicap.

---

## 5. Training: online, epoch-free, distributed

### 5.1 One step = one market second

There are no epochs, no shuffling and no `DataLoader`. `ParallelReplay` runs
*B* independent cursors scattered across disjoint spans of the archive, all
stepping forward one second in lockstep. At each step every stream:

1. reads its trailing 12 hours (and **nothing** past its cursor),
2. issues a 1500-step forecast,
3. is scored against the realised next 1500 seconds from the real archive,
4. advances one second.

The batch dimension is *parallel market time*, not shuffled samples. That is
what gives an inherently sequential objective enough arithmetic intensity to
keep four A10s busy, and it also means each batch mixes the 2022 bear with
2024 trend with last week's regime instead of *B* near-identical neighbours.

### 5.2 The real-time reward

Five terms, each earning its place:

| Term | What it buys |
|---|---|
| `path` | Huber on the full predicted vs realised log-return path. Huber, not MSE — one-second crypto returns have tails that make a squared loss chase outliers and flatten everything else. |
| `terminal` | Extra weight on h=1500. The actual question is "where is the price in 25 minutes"; without this the loss is dominated by the 1,499 easy near-horizons. |
| `direction` | Hinge on sign agreement. MAE can look excellent while directional accuracy sits at 50%, and a model that is calibrated about magnitude and wrong about sign loses money. |
| `consistency` | The stability requirement, as a loss: what the model says now about instant *t+k* must agree with what it said one second ago about the same instant. |
| `jump` | Second difference of the path. A light guard rail; the basis already bounds curvature. |

The consistency term needed care. The previous forecast was anchored one second
earlier, so comparing the two directly would punish the model **for the price
having moved**, which is not instability. The term re-anchors by subtracting
the one-step return the market actually printed between the anchors.
`test_oracle_scores_zero_on_every_supervised_term` verifies that a perfect,
perfectly stable forecaster scores **exactly 0.0** on path, terminal, direction
*and* consistency — which it does not if the re-anchoring is wrong.

Measured on synthetic oracle data: oracle total loss `0.001`, sign-flipped
anti-oracle `0.178`.

### 5.3 The device topology, and how "one shared analyser" holds

```
tokens (CPU) → MarketAnalyser (CPU, DDP/gloo)
                     ↓  memory, state  .to(cuda)
               PricePredictor (GPU, a DIFFERENT variant per rank)
                     ↓
               RealTimeObjective (GPU)
                     ↓  loss.backward()
      GPU half of the graph, then CPU half — autograd crosses devices itself
```

PyTorch's `cpu:gloo,cuda:nccl` backend string gives one process group with two
transports, routed by tensor device. The analyser is wrapped in DDP so its
gradients all-reduce across all four ranks — every copy stays bit-identical, so
there is **exactly one analyser**, learning from the combined signal of all
four predictors. The predictors in ensemble mode are deliberately *not*
wrapped; all-reducing four different architectures would be a silent disaster.

That is the whole trick that lets "multiple predictors, different training
methods" and "only one market analyser" both be true at once.

The analyser's learning rate is scaled by `0.5/√world_size`, because it
receives gradient from every rank and its effective batch is `world_size`×
the predictor's.

### 5.4 Preventing leakage

* Splits are **chronological**, never random. Neighbouring seconds are nearly
  identical, so a shuffled validation set measures memorisation, not
  forecasting. Default: 90% train / 5% val / 5% test, in time order.
* The replay hands the model `[t-43199, t]` and the objective `[t+1, t+1500]`;
  they never meet.
* The analyser is causal, and tested for it.

### 5.5 A note on settlement delay

Live, a forecast issued at *t* cannot be scored until *t+1500*. Offline the
future is already on disk, so the trainer scores and updates immediately. This
is deliberate, and it is **not** lookahead: the model input at step *t* contains
nothing after *t*. The 1500-second delay only constrains when a gradient could
physically exist in production, where no gradient is applied anyway. Deferring
the update through a FIFO would change the order of 1500 updates and nothing
about what the model may see.

---

## 6. The ensemble and model selection

Four variants, one per A10. These differ along the axes that actually
decorrelate a forecaster — four seeds of one recipe would average to the same
model:

| Variant | Head | Optimiser | Distinguishing feature |
|---|---|---|---|
| `point-adamw` | point path | AdamW, lr 3e-4, β=(0.9,0.95) | the sharp baseline |
| `quantile-pinball` | p10/p50/p90 | AdamW, lr 2e-4, β=(0.9,0.98) | pinball loss; honest uncertainty band, monotone by construction |
| `distribution-hlgauss` | median + 101-bin categorical | AdamW, lr 2.5e-4 | Gaussian-smoothed cross-entropy; handles the fat tails Huber ignores |
| `point-ema-stable` | point path | AdamW, lr 1.5e-4, EMA teacher 0.9995 | 5× consistency weight, self-distilled; the calm member |

Quantiles cannot cross: the head predicts p50 plus non-negative softplus
spreads. A model free to cross its own quantiles will, and a crossed band is
worse than no band for position sizing.

`scripts/select_best.py` walks the held-out tail second by second and scores
on four axes at once:

```
score = terminal_MAE_bps + 0.5 · stability_bps + 20 · max(0, 0.5 − dir_acc)
```

plus reported skill-vs-naive and a toy long/flat/short PnL at 1 bp cost per
side. **Beating "the price will not move" is the entire job** at a 25-minute
horizon and is a genuinely hard benchmark — `skill_vs_naive` exposes a model
that has learned nothing but the unconditional mean.

Real output from a (deliberately undertrained, 20–30 step) verification run on
the held-out tail of August–October 2026, showing the machinery works:

```
model                          mae_bps/terminal   dir_acc  stability_bps  skill_vs_naive   score
best_distribution-hlgauss.pt            13.4477    0.4875         0.2830         -0.1993  13.8392
best_quantile-pinball.pt                14.1218    0.4500         0.1754         -0.2600  15.2095
best_point-ema-stable.pt                14.0910    0.3563         0.1486         -0.2560  17.0403
```

Negative skill is exactly right for a model trained for 30 steps. Note that
`point-ema-stable` already has the best stability (0.149 vs 0.283) — the design
intent shows up even at this scale.

---

## 7. Running on Snowflake (offline)

**No internet is required or used.** The dataset is in the repo; only
`pip install` reaches outside.

Either run the notebook `notebooks/snowflake_train.ipynb` top to bottom, or:

```bash
source configs/snowflake_4xa10.env
bash scripts/snowflake_train.sh
```

That script: audits hardware → builds the cache → `torchrun --nproc_per_node=4`
→ selects the best model into `models/best.pt`.

What the sizing planner picks for 4× A10 (23 GB), measured:

```
predictor  d_model=1536  layers=28  heads=24   → 794M params
analyser   d_model=512   layers=10             →  27M params (CPU, 11 threads/rank)
batching   micro=4  accum=2  bf16  checkpointing=off
estimated  15.8 GB / 22.5 GB per GPU at 80% headroom
total      3.2B parameters across 4 predictors + 1 shared analyser
```

The planner solves for the widest model whose *training* footprint fits the
smallest visible GPU — weights + grads + **fp32 Adam moments** (the term people
forget: 794M params is 6.4 GB of optimiser state alone) + activations, with
20% headroom for CUDA context, NCCL buffers and fragmentation. If nothing fits
it raises rather than emitting a config that OOMs three hours in.

Cap it with `--max-d-model 1024` if you want something saner (see §10).

---

## 8. Running live on molab

```bash
pip install -r requirements.txt -r requirements-live.txt
marimo edit notebooks/molab_live.py
```

Pick `models/best.pt` in the dropdown. The notebook warm-starts the 12-hour
buffer from the Binance REST API (44 paged calls), then subscribes to
`btcusdt@kline_1s` over WebSocket and refreshes once a second.

**One chart, updated in place.** The usual failure — a new chart appearing
below the old one every tick — is avoided structurally, not by cleanup:

* the refresh timer is a `mo.ui.refresh` element;
* **exactly one cell** depends on it, and that cell *returns* a single figure;
* marimo re-executes that one cell per tick and **replaces** its output. A cell
  has one output, so there is physically only ever one chart;
* all mutable state (model, ring buffer, history) lives in cells that do *not*
  depend on the ticker, so it is created once and never rebuilt;
* `uirevision="btc-live"` preserves your zoom and pan across refreshes.

Do not add `mo.output.append` to the render cell, and do not move figure
construction into the setup cell. Those are the two ways to break it.

Styling follows CoinMarketCap: `#0D1421` background, `#16C784` / `#EA3943`
up/down with matching area fill, right-hand price axis, large price + percent
header, dotted forecast line in `#3861FB` with a shaded p10–p90 band and a
labelled 25-minute target marker.

A `replay archive` source is included so you can demo the notebook with no
internet, driven by `ReplayStream` at 60× market speed.

The live path reuses `LivePredictor`, the same class the backtester uses — if
serving used different code from evaluation, the numbers above would be
fiction. It gap-fills the feed exactly as the archive builder does, so a quiet
second looks identical in training and production, and applies a re-anchored
EMA (default 5 s half-life) as a second layer of tick-noise suppression.

---

## 9. Throughput, and the one decision you should revisit

The brief puts the Market Analyser on CPU. This repo honours that by default,
and you should know what it costs.

At the planned size the analyser is ~3.5 × 10¹⁰ FLOPs per sample forward. On 48
vCPUs (~100 GFLOP/s realistic for this shape) that is roughly **a second per
sample, forward alone**, against single-digit milliseconds for the same stack
on an A10. In an end-to-end step the CPU analyser will be well over 95% of wall
time; the trainer logs the measured split (`time_frac`) every `--log-every`
steps so this is visible rather than theoretical.

```bash
# honours the brief, slow
ANALYSER_DEVICE=cpu bash scripts/snowflake_train.sh

# ~100x faster per step, costs ~1 GB of VRAM
ANALYSER_DEVICE=cuda bash scripts/snowflake_train.sh
```

The analyser is 27M parameters — about 3% of one predictor — so moving it to
GPU is nearly free in memory. **For any serious training run, use
`ANALYSER_DEVICE=cuda`.** The CPU placement is the right call for *live
serving* on molab, where one forward per second is trivially within budget and
it leaves the whole GPU to the predictor.

---

## 10. Honest limitations

Read this before trusting anything.

1. **The model is untrained.** The build environment had 2 vCPUs, 2 GB RAM and
   no GPU. Every component is verified on real data; no meaningful training
   run has occurred. The checkpoints referenced in §6 come from 20–30 step
   smoke runs and have *negative* skill. You must train before using it.

2. **The GPU and multi-GPU paths have not been executed.** DDP topology, bf16
   autocast and the `cpu:gloo,cuda:nccl` process group are written to spec and
   are not covered by the CPU test suite. Budget time for a first-run debug
   pass on Snowflake. Start with `STEPS=2000 NPROC=2` before committing to a
   long run.

3. **794M parameters per predictor is almost certainly too big.** The planner
   maximises size because the brief asked it to. But 150M seconds of
   ~0.999-autocorrelated data contains far fewer independent observations than
   the row count suggests, and at a 25-minute horizon the signal-to-noise ratio
   is brutal. My expectation is that `--max-d-model 768` (~200M params) will
   match or beat the 1536 configuration and train 4× faster. Treat the big
   config as a ceiling to compare against, not a default to trust.

4. **Forecasting 25-minute BTC moves is close to the hardest version of this
   problem.** A naive "no change" forecast is a strong baseline. Expect
   directional accuracy in the 50–53% range if things go well, and be deeply
   suspicious of anything above 55% — it almost always means a leak. The
   `skill_vs_naive` metric exists specifically to keep you honest.

5. **The PnL figure in `select_best.py` is a sanity check, not a backtest.**
   It assumes fills at the mid, a flat 1 bp cost, no slippage, no market
   impact, no funding and no latency. Do not size positions off it.

6. **Spot klines only.** No order book, no futures basis, no funding rate, no
   perp open interest, no cross-exchange flow. These are the features most
   likely to actually add edge at this horizon, and the feature pipeline has
   clean seams for them (`FEATURE_NAMES` + `CHANNEL_REDUCERS`).

7. **The token in the original request is in this chat's history.** It is a
   throwaway test account, but revoke it anyway — GitHub auto-revokes tokens
   found in public text, and you should not rely on that having happened.

---

## 11. Repository layout

```
pridictior_snow10/
├── data/btcusdt_1s/              58 monthly Parquet shards, 2022-01 → 2026-10
├── src/btcpred/
│   ├── data/
│   │   ├── schema.py             on-disk contract, dense-grid arithmetic
│   │   ├── download_binance.py   resumable archive downloader
│   │   ├── features.py           14 scale-free feature channels
│   │   ├── cache.py              memory-mapped cache build/open
│   │   ├── pyramid.py            multi-resolution tokeniser (660 tokens)
│   │   └── replay.py             market simulator, single + parallel
│   ├── models/
│   │   ├── market_analyser.py    CPU dilated causal CNN + ChannelNorm
│   │   ├── price_predictor.py    GPU transformer, 3 heads, log-time RoPE
│   │   └── basis.py              smooth path basis + projection
│   ├── train/
│   │   ├── objective.py          real-time reward, metrics
│   │   ├── variants.py           the 4 training methods
│   │   └── trainer.py            online epoch-free DDP trainer
│   ├── live/
│   │   ├── runner.py             LivePredictor, ring buffer, EMA
│   │   └── binance_feed.py       REST warm-start + WebSocket stream
│   └── utils/
│       ├── hardware.py           detection + capacity planning
│       └── dist.py               cpu:gloo,cuda:nccl process group
├── scripts/
│   ├── build_cache.py            shards → mmap cache
│   ├── train.py                  training entry point
│   ├── select_best.py            held-out evaluation + selection
│   └── snowflake_train.sh        full offline pipeline
├── notebooks/
│   ├── snowflake_train.ipynb     offline training notebook
│   └── molab_live.py             marimo live chart (CMC styling)
├── configs/                      environment presets
└── tests/test_pipeline.py        19 tests: causality, leakage, scaling
```

### Commands worth knowing

```bash
# What would this machine run?
PYTHONPATH=src python -m btcpred.utils.hardware

# Extend the dataset to today (resumable; re-run any time)
PYTHONPATH=src python -m btcpred.data.download_binance --out data/btcusdt_1s

# Train one specific variant
python scripts/train.py --cache data/cache --out runs/q \
    --variant quantile-pinball --max-steps 50000

# Four GPUs, four variants, one shared analyser
torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --cache data/cache --out runs/ensemble --mode ensemble

# Four GPUs, one model, data-parallel
torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --cache data/cache --out runs/ddp --mode ddp --variant point-adamw

# Score every checkpoint and pick a winner
python scripts/select_best.py --runs runs --cache data/cache --out models/best.pt
```

Training is resumable and signal-safe: SIGINT/SIGTERM finishes the current step
and checkpoints rather than losing the run.
