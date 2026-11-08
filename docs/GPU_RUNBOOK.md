# Running Stage 1 (continued pretraining) on a rented GPU

Everything is prepared and was dry-run on the Mac. You only rent the machine, copy the files, and start one script.

## 1. Rent a machine

- **GPU:** 1 × NVIDIA **A100 80 GB** (or A100 40 GB, or H100). A single GPU is enough for a 1.7B model with LoRA.
- **Image:** any "PyTorch + CUDA" template (PyTorch ≥ 2.6, Python ≥ 3.10).
- **Disk:** at least 60 GB (model, checkpoints, two runs).
- **Providers** (pick one; prices change, so check before renting): RunPod, Lambda, Vast.ai, Paperspace, or a
  cloud provider you already use.
- **Expected time:** roughly **1–3 hours per variant**, so **3–6 hours for both** including setup and evaluations.
  At typical A100 prices this is a modest cost, but check the hourly rate of the offer you choose.
- **Stop the machine when it is finished:** billing usually continues while it is running.

## 2. Copy the project (from the Mac, in Terminal)

Replace `USER@HOST` and `PORT` with the SSH details the provider gives you.

```bash
cd ~/Documents/bambara-SLM
rsync -avz -e "ssh -p PORT" --relative \
  pyproject.toml requirements-gpu.txt src configs scripts data/cpt \
  USER@HOST:~/bambara-SLM/
```

This uploads about 85 MB (code + both CPT data variants). No raw data, no evaluation sets, no `.env`.

## 3. Start the run (on the GPU machine)

```bash
cd ~/bambara-SLM
export HF_TOKEN=...            # your Hugging Face token (optional but recommended); never commit it
tmux new -s cpt                # keeps running if your connection drops (reattach: tmux attach -t cpt)
bash scripts/gpu_run_cpt.sh    # both variants: with_nllb, then no_nllb
```

Watch progress: `tail -f runs/cpt_with_nllb.log`. Every 50 steps it prints Bambara, French and English validation loss.

## 4. What the script does

1. Measures validation loss on Bambara, French and English **before** training (reference).
2. Trains LoRA (rank 64, all linear layers) for 2 epochs. It evaluates and saves a checkpoint every 50 steps.
3. Picks the checkpoint with the **lowest Bambara loss** among those whose French and English perplexity stay within
   **+10 %** of the start. If none qualifies, it stops with exit code 2. In that case, tell me: we re-run with more replay data.
4. Merges the chosen adapter into the model and saves a normal Hugging Face model in `runs/cpt_<variant>/merged`.
5. Packs results into `runs/cpt_<variant>_results.tgz` (merged model + reports; checkpoints stay on the GPU box).

## 5. Bring the results back (from the Mac)

```bash
cd ~/Documents/bambara-SLM
scp -P PORT USER@HOST:~/bambara-SLM/runs/cpt_*_results.tgz runs/
for f in runs/cpt_*_results.tgz; do tar -xzf "$f"; done
```

Then tell me. I run the same evaluation as the baseline on the Mac (perplexity, translation, and the 67-prompt sheet
for your review) and write the Gate 4 report.

## 6. Then stop or delete the machine

Keep it only if we plan the SFT run right after (it can reuse the same machine).
