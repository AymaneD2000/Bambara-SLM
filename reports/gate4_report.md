# Gate 4: after continued pretraining (CPT, variant `with_nllb`), 2026-11-07

Status: **complete (2026-11-07)**: CPT run finished, fixed evaluation and native-speaker review done.

## Training run

- Qwen3-1.7B-Base + LoRA r=64 on all linear layers, 2 epochs (546 steps of 32 × 2,048-token sequences, 14.25 M tokens per epoch), lr 2e-4 cosine.
- Run on Colab (T4, steps 0–150) then Kaggle (T4, steps 100–546), resumed from checkpoints (about 19 GPU-hours in total). fp16 mixed precision on T4.
- Checkpoint chosen: **step 546** (lowest Bambara validation loss, French/English within +10 %). Merged model: `runs/cpt_with_nllb/merged` (bf16, 3.4 GB).

| step | Bambara val ppl | French val ppl | English val ppl |
|---|---|---|---|
| 0 | 40.00 | 9.59 | 10.26 |
| 50 | 18.03 | 9.32 | 9.96 |
| 100 | 14.73 | 9.26 | 9.91 |
| 200 | 12.55 | 9.26 | 9.93 |
| 300 | 11.67 | 9.27 | 9.96 |
| 400 | 11.23 | 9.28 | 9.98 |
| 500 | 11.08 | 9.28 | 9.98 |
| 546 | 11.07 | 9.28 | 9.98 |

(French/English validation here = held-out replay web text, same source as the replay data.)

## Fixed evaluation (same config, prompts and samples as the baseline)

| perplexity (lower is better) | baseline | after CPT | change |
|---|---|---|---|
| belebele_bam | 104.26 | **26.30** | -75% |
| bam_val_spoken | 295.06 | **20.76** | -93% |
| bam_val_web | 38.91 | **10.96** | -72% |
| bam_val_written | 192.94 | **27.93** | -86% |
| belebele_fr | 7.97 | **8.02** | +1% |
| belebele_en | 15.41 | **15.61** | +1% |

Belebele French/English is the **independent** forgetting check (not replay-like text): +0.6 % / +1.3 %, far inside the +10 % limit.

| translation (5-shot, greedy) | n | baseline BLEU / chrF++ | after CPT BLEU / chrF++ | repetition loops (base → CPT) |
|---|---|---|---|---|
| bayelemabaga_test:bam-fr | 500 | 0.58 / 10.66 | **7.53 / 21.73** | 0% → 0% |
| bayelemabaga_test:fr-bam | 500 | 0.11 / 4.28 | **1.85 / 14.73** | 29% → 2% |
| mafand_test:bam-fr | 500 | 0.43 / 12.35 | **2.67 / 22.27** | 0% → 0% |
| mafand_test:fr-bam | 500 | 0.08 / 6.08 | **1.27 / 12.72** | 22% → 12% |
| sib200_test:bam-fr | 191 | 1.04 / 13.69 | **3.43 / 22.64** | 0% → 0% |
| sib200_test:fr-bam | 191 | 0.19 / 5.96 | **1.01 / 13.01** | 25% → 14% |
| sib200_test:bam-en | 191 | 2.78 / 19.66 | **5.64 / 23.85** | 0% → 0% |
| sib200_test:en-bam | 191 | 0.50 / 8.14 | **1.53 / 13.54** | 6% → 16% |

## Reading

- **CPT worked:** Bambara perplexity is down 72–93 % on every Bambara set, and translation chrF++ roughly doubles into French/English and triples into Bambara. Repetition loops into Bambara drop on French sources (Bayelemabaga fr→bam 29 % → 2 %, MAFAND 22 % → 12 %, FLORES 25 % → 14 %) but **rise for English sources** (FLORES en→bam 6 % → 16 %): the base model mostly copied English, while the CPT model now attempts Bambara on long technical sentences and sometimes loops.
- **No measurable forgetting** of French or English.
- **Still a weak translator in absolute terms:** BLEU 1–8. Into French/English the output is fluent but often only partly right (e.g. `An ka bo ka Siraba in bila` → "Nous allons la laisser à Siraba"; ref. "Éloignons-nous plutôt, et quittons la route"). Into Bambara it now writes real Bambara, but often with the wrong meaning ("La maison est grande" → `Dugu ye kɔrɔ ye`, ref. `So ka bon`), and loops remain on long technical FLORES sentences (16 % en→bam).
- **It does not follow instructions yet:** the 67 prompts are answered by continuation, not as an assistant (e.g. repeating greetings; inventing "la messe" in bf-05; "pig meat" soup in be-01). This is expected before SFT.
- Reference: published fine-tuned encoder-decoder models (Bayelemabaga paper) remain far ahead on translation; we did not run NLLB (non-commercial; optional).

## Native-speaker review: baseline vs after CPT (67 frozen prompts)

Scores AI-drafted and reviewed by the supervisor; same consistency rules for both runs (`human_review_final.csv`).

| | baseline | after CPT |
|---|---|---|
| answers: no_answer | 40 | 16 |
| answers: answer without Bambara | 21 | 24 |
| answers: attempted Bambara | 6 | 27 |
| mean adequacy (all 67) | 1.09 (n=67) | **1.40** (n=67) |
| mean fluency (all 67) | 1.31 (n=67) | **2.00** (n=67) |
| mean orthography (own Bambara only) | 1.17 (n=6) | **3.56** (n=27) |
| mean hallucination (real answers only) | 1.81 (n=27) | **2.00** (n=51) |
| real answers inventing content (H ≤ 2) | 21/27 (78%) | 37/51 (73%) |

| category | baseline adequacy | after-CPT adequacy |
|---|---|---|
| generate | 1.00 | **1.13** |
| qa | 1.50 | **1.00** |
| summarize | 1.00 | **1.17** |
| translate_bam_en | 1.00 | **2.25** |
| translate_bam_fr | 1.00 | **2.33** |
| translate_en_bam | 1.00 | **1.20** |
| translate_fr_bam | 1.00 | **1.15** |

**Reading:** the model now writes Bambara (27 real Bambara answers vs 6; orthography 3.56 vs 1.17) and partly understands it (translation into French/English adequacy 2.3 vs 1.0). **Question answering regressed** (1.50 → 1.00): the base model gave 3 right facts in French/English, while the CPT model answers in Bambara with wrong content (e.g. invents a river "Kɔrɔn"). It still does not follow instructions, and 73 % of real answers invent content. CPT teaches the language, not assistant behaviour: that is the job of SFT.

## Decisions needed

1. ~~Score the after-CPT answers~~ done (§ native-speaker review).
2. Approve moving to **Stage 2 SFT** on top of this CPT model (translation data built: 113.6 k examples; QA/summarization/chat data still to decide).
3. Run the **`no_nllb` ablation** (about 13 h on Kaggle) now, or after SFT?

