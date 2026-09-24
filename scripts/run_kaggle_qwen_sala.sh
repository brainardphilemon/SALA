#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${SALA_KAGGLE_OUTPUT:-/kaggle/working/sala_qwen25_g3_g5_bleurt}"

python sala_kaggle.py \
  --output-dir "$OUTPUT_DIR" \
  --settings G3 G5 \
  "$@"
