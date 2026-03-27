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
  "Step 1/3 | R5O" \
  "weights/logs/R5O_no_PINN.log" \
  python3 -u main.py \
    --csv-path Data/60k61db.csv \
    --output-mode mag_only \
    --seq-len 61 \
    --epochs 80 \
    --device cuda \
    --output-dir weights/R5O_no_PINN

run_step \
  "Step 2/3 | R5.5B-s" \
  "weights/logs/R55BS_no_PINN.log" \
  python3 -u main.py \
    --model r55bs \
    --csv-path Data/60k61db.csv \
    --output-mode mag_only \
    --epochs 50 \
    --device cuda \
    --output-dir weights/R55BS_no_PINN

run_step \
  "Step 3/3 | PINN" \
  "weights/logs/PINN.log" \
  python3 -u main.py --usepinn \
    --processed-csv-path Data/60k61db.csv \
    --processed-meta-path Data/60k61db.meta.json \
    --epochs 80 \
    --batch-size 256 \
    --device cuda \
    --results-dir weights/PINN
