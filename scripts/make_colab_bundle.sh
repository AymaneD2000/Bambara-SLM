#!/usr/bin/env bash
# Make dist/bambara_slm_colab.zip for Colab: code, configs and CPT data only (no raw data, no eval sets, no .env).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f dist/bambara_slm_colab.zip
mkdir -p dist
zip -qr dist/bambara_slm_colab.zip pyproject.toml requirements-gpu.txt src configs scripts data/cpt data/sft/train.jsonl data/sft/valid.jsonl data/sft_v2/train.jsonl data/sft_v2/valid.jsonl data/sft_v2/stats.json docs/GPU_RUNBOOK.md \
  -x "*/__pycache__/*" "*.pyc"
ls -lh dist/bambara_slm_colab.zip
