#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${SALA_KAGGLE_OUTPUT:-/kaggle/working/sala_qwen25_g15_nq_bleurt}"

python sala_kaggle.py \
  --output-dir "$OUTPUT_DIR" \
  --settings G15 \
  --samples-per-domain 0 \
  "$@"
