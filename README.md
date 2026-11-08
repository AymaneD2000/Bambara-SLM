# Bambara SLM

Adapting **Qwen3-1.7B-Base** to **Bambara** (bamanankan, `bam_Latn`) for translation (Bambara ↔ French, Bambara ↔ English) and, later, simple question answering and chat. The pipeline covers data extraction, cleaning, alignment scoring, continued pre-training (CPT), supervised fine-tuning (SFT) with LoRA, and evaluation.

## Pipeline

| Stage | Code | Notes |
|---|---|---|
| Sources and licenses | `src/bslm/sources/`, `PROVENANCE.md` | Licenses are recorded before any source is used |
| Cleaning, langid, dedup, PII | `src/bslm/clean.py`, `langid.py`, `dedup.py`, `pii.py` | Reports in `reports/` |
| Alignment scoring | `src/bslm/align.py`, `scripts/laser_embed.py` | LASER plus heuristics, checked by hand per score bucket |
| Decontamination and splits | `src/bslm/decontam.py`, `splits.py` | No overlap with evaluation sets |
| CPT data and training | `src/bslm/build_cpt.py`, `src/bslm/train/cpt.py` | LoRA, with French and English replay |
| SFT data and training | `src/bslm/build_sft.py`, `src/bslm/train/sft.py` | Qwen3 ChatML, thinking disabled, loss on assistant tokens only |
| Evaluation | `src/bslm/eval/` | BLEU and chrF++ (Bayelemabaga, MAFAND, FLORES/SIB-200), perplexity, native-speaker review |

GPU runs use Kaggle or Colab notebooks (`notebooks/`, `docs/GPU_RUNBOOK.md`). The code bundle is built with `scripts/make_colab_bundle.sh`.

## Notes

- **Qwen3 base special tokens.** In Qwen3-*-Base, every special token after `<|endoftext|>` (`<|im_start|>`, `<|im_end|>`, `<think>`, …) shares one identical, untrained embedding row, and the embeddings are tied to the output head. LoRA on the attention/MLP layers cannot change that row, so a fine-tuned model never emits `<|im_end|>`. `sft.py` gives these tokens their own starting rows and trains them with PEFT `trainable_token_indices`.
- **Data policy.** Synthetic data is labelled (`synthetic: true` plus the generator name), capped in the training mix and never used for evaluation. Evaluation-only sources stay out of training; see `PROVENANCE.md`.

## Results

Summaries are in `reports/eval/*/summary.md`. Example, SFT v1 (step 3706) against the CPT model on Bayelemabaga test, chrF++: bam→fr 25.1 vs 21.7, fr→bam 23.0 vs 14.7.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,train]"
cp .env.example .env   # fill in your tokens; never commit .env
python -m bslm.cli --help
```

Datasets, model weights and training runs are not part of this repository.
