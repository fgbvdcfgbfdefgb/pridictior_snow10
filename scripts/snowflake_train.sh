#!/usr/bin/env bash
# Offline end-to-end run for Snowflake Container Runtime: 4x A10, 48 vCPU, 100 GB RAM.
#
# NOTHING in here touches the network except `pip install`. The dataset is in
# the repo as Parquet shards; the cache is derived from them locally.
#
#   bash scripts/snowflake_train.sh                 # full run
#   STEPS=2000 bash scripts/snowflake_train.sh      # short validation run
#   ANALYSER_DEVICE=cuda bash scripts/snowflake_train.sh   # much faster
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SHARDS="${SHARDS:-data/btcusdt_1s}"
CACHE="${CACHE:-data/cache}"
RUNS="${RUNS:-runs/ensemble}"
STEPS="${STEPS:-400000}"
MODE="${MODE:-ensemble}"
ANALYSER_DEVICE="${ANALYSER_DEVICE:-cpu}"
NPROC="${NPROC:-$(python3 -c 'import torch;print(max(torch.cuda.device_count(),1))')}"
START_MONTH="${START_MONTH:-}"
END_MONTH="${END_MONTH:-}"

echo "=============================================================="
echo " BTC 25-minute predictor -- offline training"
echo " repo      : $REPO"
echo " gpus      : $NPROC"
echo " mode      : $MODE   (ensemble = one variant per GPU, shared analyser)"
echo " analyser  : $ANALYSER_DEVICE"
echo " steps     : $STEPS market-seconds per rank"
echo "=============================================================="

echo
echo "--- [0/4] environment -----------------------------------------"
python3 -c "import sys;print('python', sys.version.split()[0])"
pip install -q -r requirements.txt
PYTHONPATH=src python3 -m btcpred.utils.hardware

echo
echo "--- [1/4] feature cache ---------------------------------------"
CACHE_ARGS=(--shards "$SHARDS" --cache "$CACHE")
[ -n "$START_MONTH" ] && CACHE_ARGS+=(--start "$START_MONTH")
[ -n "$END_MONTH" ] && CACHE_ARGS+=(--end "$END_MONTH")
python3 scripts/build_cache.py "${CACHE_ARGS[@]}"

echo
echo "--- [2/4] distributed training --------------------------------"
# One process per GPU. Rank k trains variant k; the analyser is DDP'd over
# gloo across all ranks so there is exactly one of it.
#
# OMP threads are split across ranks so the four CPU-side analysers do not
# oversubscribe the 48 vCPUs and thrash.
TOTAL_CPU="$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')"
export OMP_NUM_THREADS=$(( TOTAL_CPU / NPROC > 0 ? TOTAL_CPU / NPROC : 1 ))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS per rank"

torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py \
    --cache "$CACHE" \
    --out "$RUNS" \
    --mode "$MODE" \
    --max-steps "$STEPS" \
    --analyser-device "$ANALYSER_DEVICE" \
    --log-every 100 \
    --ckpt-every 5000 \
    --eval-every 10000

echo
echo "--- [3/4] model selection -------------------------------------"
python3 scripts/select_best.py \
    --runs "$RUNS" \
    --cache "$CACHE" \
    --out models/best.pt \
    --steps 3000 --batch 8

echo
echo "--- [4/4] done ------------------------------------------------"
echo "best model : models/best.pt"
echo "scoreboard : models/selection.json"
echo "run logs   : $RUNS/log_rank*.jsonl"
echo
echo "Copy models/best.pt to molab and open notebooks/molab_live.py."
