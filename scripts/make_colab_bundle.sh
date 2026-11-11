#!/usr/bin/env bash
# Make dist/bambara_slm_colab.zip for Colab/Kaggle: code, configs, CPT/SFT data and the v3 generation inputs
# (no raw data, no eval sets, no .env, no generated v3 outputs). SFT data v1, v2 and v3
# (v3: train, valid, chat_valid, stats; the zip is ~115 MB, so v1/v2 stay for the older notebooks).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f dist/bambara_slm_colab.zip
mkdir -p dist
# v3 phase-3 inputs (bslm.v3data.generate). smoltalk_threads.jsonl is left out: not used (Llama/GPT-4 outputs).
V3=()
for f in data/interim/v3/bt_sources.jsonl data/interim/v3/mono_bam.jsonl data/interim/v3/oasst2_threads.jsonl \
         data/interim/v3/dolly_grounded.jsonl data/interim/v3/orthography_map.json data/interim/v3/*.stats.json; do
  [ -f "$f" ] && V3+=("$f")
done
[ -f data/interim/v3/bt_sources.jsonl ] || echo "WARNING: no bt_sources.jsonl: run 'python -m bslm.v3data.generate prep-bt' first"
zip -qr dist/bambara_slm_colab.zip pyproject.toml requirements-gpu.txt src configs scripts data/cpt data/sft/train.jsonl data/sft/valid.jsonl data/sft_v2/train.jsonl data/sft_v2/valid.jsonl data/sft_v2/stats.json data/sft_v3/train.jsonl data/sft_v3/valid.jsonl data/sft_v3/chat_valid.jsonl data/sft_v3/stats.json docs/GPU_RUNBOOK.md \
  "${V3[@]}" -x "*/__pycache__/*" "*.pyc" "data/interim/v3/smoltalk_threads.stats.json"
ls -lh dist/bambara_slm_colab.zip
