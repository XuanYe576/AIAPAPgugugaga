#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
mkdir -p weights/logs

run_step() {
  local label="$1"
  local log_path="$2"
  shift 2
  echo "============================================================"
  echo "${label}"
  echo "Command: $*"
  echo "Log: ${log_path}"
  echo "============================================================"
  "$@" 2>&1 | tee "${log_path}"
}

run_step \
  "Step 1/4 | R6O feed/notch prior (1093 samples, 501-point head)" \
  "weights/logs/R6O_stage1_feed501.log" \
  python3 -u main.py \
    --model r6o \
    --dataset-profile feed1093 \
    --curve-head-points 501,61 \
    --active-curve-points 501 \
    --export-head-points 501 \
    --export-curve-points 501 \
    --epochs 120 \
    --device cuda \
    --results-dir weights/R6O_stage1_feed501

run_step \
  "Step 2/4 | R6O main geometry training (60k samples, 61-point loss, 501-point export)" \
  "weights/logs/R6O_stage2_60k_main_final501.log" \
  python3 -u main.py \
    --model r6o \
    --dataset-profile 60k \
    --curve-head-points 501,61 \
    --active-curve-points 61 \
    --export-head-points 501 \
    --export-curve-points 501 \
    --init-checkpoint-path weights/R6O_stage1_feed501/r6o_best.ckpt \
    --epochs 120 \
    --device cuda \
    --results-dir weights/R6O_stage2_60k_main_final501

run_step \
  "Step 3/4 | R6P feed/notch prior (1093 samples, native 501-point frequency axis)" \
  "weights/logs/R6P_stage1_feed501.log" \
  python3 -u main.py \
    --model r6p \
    --dataset-profile feed1093 \
    --export-curve-points 501 \
    --epochs 120 \
    --device cuda \
    --results-dir weights/R6P_stage1_feed501

run_step \
  "Step 4/4 | R6P main geometry training (60k samples, native 61-point loss, 501-point export)" \
  "weights/logs/R6P_stage2_60k_main_final501.log" \
  python3 -u main.py \
    --model r6p \
    --dataset-profile 60k \
    --init-checkpoint-path weights/R6P_stage1_feed501/r6p_best.ckpt \
    --export-curve-points 501 \
    --epochs 120 \
    --device cuda \
    --results-dir weights/R6P_stage2_60k_main_final501
