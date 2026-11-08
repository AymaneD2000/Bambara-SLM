#!/usr/bin/env bash
# Stage 1 CPT on a rented CUDA GPU. Run from the repository root on the GPU machine:
#   bash scripts/gpu_run_cpt.sh            # both variants: with_nllb, then no_nllb
#   bash scripts/gpu_run_cpt.sh with_nllb  # one variant
# Requires: NVIDIA GPU (A100/H100 class), Python 3.10+, a CUDA build of torch, HF_TOKEN in the environment
# (only to download Qwen/Qwen3-1.7B-Base faster / avoid rate limits; the model is public).
set -euo pipefail

VARIANTS=("${@:-with_nllb no_nllb}")
VARIANTS=(${VARIANTS[@]})

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

python -m pip install -q -r requirements-gpu.txt
python -m pip install -q -e . --no-deps || true
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m pip uninstall -y -q torchao 2>/dev/null || true   # preinstalled torchao versions break peft; unused here
mkdir -p runs

for v in "${VARIANTS[@]}"; do
  echo "=== CPT variant: $v  ($(date))"
  test -f "data/cpt/$v/train.jsonl" || { echo "missing data/cpt/$v (upload it first)"; exit 1; }
  # exit code 2 = no checkpoint met the French/English regression limit (see runs/cpt_$v/selection.json)
  set +e
  PYTHONUNBUFFERED=1 python -m bslm.train.cpt --config "configs/train/cpt_$v.yaml" 2>&1 | tee "runs/cpt_$v.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "=== $v finished with exit code $rc ($(date))"
  # what to bring back to the Mac: the merged model + all reports (checkpoints stay on the GPU box)
  tar -czf "runs/cpt_${v}_results.tgz" --exclude="runs/cpt_$v/checkpoints" "runs/cpt_$v" "runs/cpt_$v.log" || true
done
echo "Done. Download runs/cpt_*_results.tgz"
