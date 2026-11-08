# Stage 0: smoke test on the Mac (Gate 3), 2026-11-05

Purpose (brief §11): validate data format, chat template, loss masking, memory, speed and the evaluation scripts.
**No quality conclusions** are drawn from these runs: they are 150–200 steps on 1,500-example subsets.

## Setup

- Machine: MacBook Pro M3, 16 GB unified memory, mlx-lm 0.32.2.
- Model: Qwen/Qwen3-1.7B-Base.
- LoRA rank 16 (scale 20) on all 7 linear projections of all 28 layers: 17.4 M trainable parameters (1.01%).
- Configs: `configs/train/smoke_cpt.yaml`, `configs/train/smoke_sft.yaml`. Logs: `runs/smoke_*.log`.

| run | data | steps | wall time | peak memory | speed | val loss |
|---|---|---|---|---|---|---|
| CPT smoke | 1,500 packed examples of `data/cpt/with_nllb` (≤ 2,048 tokens) | 200 × (batch 1, grad-accum 4) | 51 min | **8.3 GB** | ≈ 120 tok/s | 4.45 → 4.19 → 3.69 → **3.59** |
| SFT smoke | 1,500 chat examples of `data/sft` (≤ 512 tokens), `mask_prompt` | 150 × batch 4 | 12.5 min | 6.2 GB | ≈ 33 trained tok/s | 5.93 → 4.25 → **3.83** |

## Checks

| item | result |
|---|---|
| CPT data format (`{"text"}`, ≤ 2,048 tokens, EOS between documents) | ✅ loads and trains. EOS is a single special token. 11 of 8,729 examples exceed 2,048 tokens (one very long line) and are truncated by the trainer. |
| SFT data format (`{"messages"}`) | ✅ loads and trains |
| Chat template | ✅ Qwen3 ChatML with **thinking disabled**: every assistant turn is `<think>\n\n</think>\n\n` + answer + `<|im_end|>`, identical at training and inference |
| Loss masking | ✅ system and user tokens are masked (≈ 33 trained tokens per example = answers only). Known detail: mlx-lm computes the mask without `enable_thinking=False`, so the 4 tokens of the empty think block are trained. This is harmless because inference puts the same block in the prompt. |
| Memory | ✅ 8.3 GB peak at 2,048-token sequences with gradient checkpointing (16 GB machine) |
| Adapters save / reload | ✅ saved every 100/150 steps; reloaded by the evaluation scripts |
| Evaluation with adapters | ✅ perplexity, few-shot MT (base format), **new chat-mode MT** (`--chat`), and the human-review sheet all run |

## Signals (subset of 30 texts per set, same subset for all rows; indicative only)

| perplexity | base | CPT smoke (200 steps) | SFT smoke (150 steps, on base) |
|---|---|---|---|
| Bambara spoken val | 293 | **42** | 99 |
| Bambara web val | 37.9 | **23.6** | 61.9 |
| Bambara written val | 193 | **87** | 198 |
| Belebele Bambara | 91.3 | **37.6** | 66.2 |
| Belebele French | 7.5 | 8.6 (+15%) | 16.3 |
| Belebele English | 16.2 | 18.6 (+15%) | 30.7 |

- **CPT works in the right direction:** Bambara perplexity drops 2–7× after only 330k training tokens.
- **Forgetting is already visible** (+15% French/English perplexity) even with 25% replay. The real CPT run must
  watch this closely (brief §11). Options: a lower learning rate, a higher replay share, or picking the checkpoint by the regression limit.
- **SFT directly on the base model collapsed** into repetition loops (`bɔrɔ bɔrɔ…`, `si tu ne fais pas ça…`), and French/English got worse. This was expected for 150 steps at learning rate 1e-4 on 600 examples. It confirms the brief's order: **CPT first, then SFT on the CPT model**. Translation scores of the smoke adapters are near zero and meaningless.

## Speed and the plan for real training

- The Mac trains at about **120 tokens/s** on 2,048-token sequences. One epoch over the CPT mix (14.25 M tokens) would take **about 33 hours**, and 2–3 epochs about 3–4 days. Not practical.
- **Recommendation:** run Stage 1 CPT on a rented single CUDA GPU (A100 or H100 class), as the brief planned. At typical throughput for a 1.7B model with LoRA, 2–3 epochs is a matter of a few hours.
- The repository is portable (same JSONL data, same configs). A CUDA training script (`train/cpt.py` with transformers + peft) is the next code to write.
- SFT (≈ 3.7 M trained tokens per epoch, answers only) would take about 31 hours per epoch on the Mac. It is feasible but slow, and much faster on the same GPU.
