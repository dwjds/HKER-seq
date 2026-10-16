#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-}"
if [[ "${DATASET}" != "mooccube" && "${DATASET}" != "mooper" ]]; then
  echo "Usage: bash scripts/train.sh {mooccube|mooper}" >&2
  exit 2
fi

GPU_ID="${GPU_ID:-0}"
BATCH_QUOTA="${BATCH_QUOTA:-128}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p outputs/results outputs/checkpoints

python tune.py \
  --model hypsasrec \
  --dataset "${DATASET}" \
  --config_path "grids/hypsasrec_${DATASET}.py" \
  --split_strategy recbole_ls \
  --target_metric NDCG \
  --topn 10 \
  --report_topn 20 \
  --grid_steps 1 \
  --check_best \
  --save_config \
  --config_overrides "{\"device\":\"${DEVICE}\",\"batch_quota\":${BATCH_QUOTA},\"seed\":${SEED},\"checkpoint_dir\":\"outputs/checkpoints/${DATASET}\"}" \
  --result_path "outputs/results/${DATASET}.json"
