# Gate 2: Alignment, splits, contamination, baseline (2026-11-04)

Status: **complete (2026-11-05)**: alignment inspected, thresholds set, prompt set frozen, baseline measured and human-reviewed. Open: optional NLLB re-check (§6.1).

Detail files: `reports/alignment/alignment_report.md`, `reports/alignment/validation.md`, `reports/alignment/inspect/*.tsv`,
`reports/contamination_report.md`, `reports/splits_report.md`, `reports/eval/baseline_qwen3_1.7b_base/` (scores + every generation).

## 1. Alignment scoring (brief §8.2)

Every parallel pair has an `align_score` = P(aligned) (`data/clean/parallel_scored.jsonl`).
- **Signals (all commercial-use compatible):**
  - LASER3 `bam_Latn` / LASER2 margin and cosine (BSD license; SONAR and NLLB were rejected because they are non-commercial).
  - Cross-fitted IBM-1 word-translation scores in both directions.
  - Length ratio, number mismatch, punctuation agreement.
- **Combiner:** a logistic regression trained on trusted pairs against shuffled and off-by-one negatives.
- **Problem found and fixed:** the first version learned two shortcuts, list numbering and lexicons memorized from the very pairs being scored. It gave *correct* FLORES pairs a score of 0. After the fix:

| held-out gold (SIB-200 test = FLORES sentences) | gold median | shuffled median | gold ≥ 0.5 | shuffled ≥ 0.5 |
|---|---|---|---|---|
| bam-fr | 0.45 | 0.06 | 42% | 1% |
| bam-en | 0.47 | 0.04 | 45% | 1% |

- **Reading:** the score **ranks** pairs well and almost never accepts a clearly wrong pair. It is **not calibrated across domains**: correct pairs from unfamiliar domains or very short lines often score low. Thresholds must therefore be set **per source**.

**Score distribution:**

| group | pairs | ≥ 0.8 | < 0.2 | median |
|---|---|---|---|---|
| Bayelemabaga (bam-fr) | 40,226 | 70% | 6% | 0.92 |
| jeli-asr (bam-fr, spoken) | 27,608 | 81% | 3% | 0.96 |
| NLLB mined bam-fr | 156,770 | 37% | 22% | 0.64 |
| NLLB mined bam-en | 98,517 | 47% | 16% | 0.77 |

My spot check (my Bambara is limited, so this needs your confirmation):
- The lowest Bayelemabaga scores are mostly **correct short headings** (`Kuma wɛrɛ` ↔ `Compléments`).
- NLLB ≥ 0.8 is roughly 60% acceptable for French and about 50% for English.

**Thresholds set from the supervisor's inspection (2026-11-05)** (`configs/align.yaml`, details in `reports/alignment/inspection_results.md`):

| source | marked | finding | decision |
|---|---|---|---|
| Bayelemabaga | 156 | 83–97% correct in every score range | keep all: 40,177 pairs, est. 94% correct |
| jeli-asr | 165 | 48–64% below 0.4; 80–97% from 0.4 | keep ≥ 0.4: 25,985 pairs (1,585 dropped), est. 95% |
| NLLB bam-fr | 155 | 0–8% below 0.8; **38% at 0.80–0.87** | **dropped for v1 for now** |
| NLLB bam-en | 152 | 6–27% below 0.8; **30% at 0.81–0.92** | **dropped for v1 for now** |

**Sampling flaw (my error):** the inspection files were sorted by score, and the reviewer marked the first rows, so each range was judged on its **lowest** scores. For Bayelemabaga and jeli-asr this only makes the estimates conservative. For NLLB, the 0.93–1.0 band (median of the top range) was never judged. A random 30-pair sample per language at ≥ 0.95 is in `reports/alignment/inspect_extra/`; NLLB comes back only if it reaches 80% correct. Inspection files are now written in random order.

**Consequence:** there are **no Bambara–English training pairs** left (NLLB was the only source). English directions will rely on the model's existing English and on transfer from French until we find a clean bam-en source.

## 2. Data defects found in this gate

- **Dioula Bible inside NLLB "Bambara":** all 15,984 pairs from `bibles.org/dyu-…` are Dioula, not Bambara. They are excluded (this also follows your decision to defer Bible data to v2).
- **Kikuyu inside FLORES-200 `bam_Latn`:** 26 Belebele passages and 60 SIB-200 sentences on the "Bambara" side are Kikuyu (ũ/ĩ characters, GlotLID `kik_Latn`). They are removed from the eval sets. **FLORES+ itself probably has the same errors:** check it when access arrives.
- **NLLB mined data overlaps Bayelemabaga test:** 319 exact matches (web copies of the same source texts).

## 3. Contamination control (brief §8.4)

The index holds every sentence of the eval sets (Bayelemabaga test, MAFAND, Belebele, SIB-200): exact match for sentences of ≥ 4 words, MinHash Jaccard ≥ 0.7 for sentences of ≥ 6 words.
- Removed: **455 parallel pairs** and **126 monolingual sentences** (rest of the document kept).
- Bayelemabaga's own train split had 46 near or exact copies of its test split; these are removed.

## 4. Splits (brief §8.5): by document, speaker or recording, never random lines

| set | records | Bambara tokens (Qwen3) |
|---|---|---|
| mono train | 177,298 | 5.38 M (spoken 3.5 M, web 1.7 M, written 0.17 M) |
| mono val | 9,275 | 0.28 M (original ASR test splits + 2% of web/wiki documents) |
| parallel train | 60,503 | 1.59 M Bambara side (Bayelemabaga 35.8k, jeli-asr 24.7k pairs; NLLB dropped) |
| parallel val | 5,852 | 0.18 M (Bayelemabaga official valid + jeli test) |
| below threshold / excluded | 188,255 | kept in `data/train/parallel_below_threshold.jsonl`, not deleted |

Religious-register markers: about 12% of Bayelemabaga, 16% of NLLB-fr and 24% of NLLB-en pairs (keyword estimate). The cap will be set at Gate 3 (CPT mix).

## 5. Baseline: untouched Qwen3-1.7B-Base (brief §10)

Fixed config `configs/eval.yaml`: 5-shot prompts, greedy decoding, the same prompts and samples for every later stage.

| perplexity (lower is better) | ppl |
|---|---|
| Belebele passages: **Bambara** / French / English (same content) | **104.3** / 8.0 / 15.4 |
| Bambara val, spoken (500 texts) | 295.1 |
| Bambara val, web (31 docs) | 38.9 (\*) |
| Bambara val, written/Wikipedia (12 docs, 889 tokens: small) | 192.9 |

(\*) MADLAD web text may overlap Qwen's own pretraining data, so this number is likely optimistic.

| translation | n | BLEU | chrF++ |
|---|---|---|---|
| Bayelemabaga test bam→fr | 500 | 0.58 | 10.66 |
| Bayelemabaga test fr→bam | 500 | 0.11 | 4.28 |
| MAFAND test bam→fr | 500 | 0.43 | 12.35 |
| MAFAND test fr→bam | 500 | 0.08 | 6.08 |
| SIB-200 test (FLORES) bam→fr | 191 | 1.04 | 13.69 |
| SIB-200 test (FLORES) fr→bam | 191 | 0.19 | 5.96 |
| SIB-200 test (FLORES) bam→en | 191 | 2.78 | 19.66 |
| SIB-200 test (FLORES) en→bam | 191 | 0.50 | 8.14 |

**Reading:** the base model essentially cannot translate Bambara.
- **Into French/English:** it writes fluent but unrelated sentences, and sometimes copies the Bambara input (24 of 500 cases).
- **Into Bambara:** it loops on repeated words in 6–29% of outputs depending on the set (`fɛ fɛ fɛ…`; 29% on Bayelemabaga fr→bam).
- This gives a clear floor to measure CPT and SFT against. Reference point: fine-tuned NLLB-style models reach far higher scores (Bayelemabaga paper); we did not run NLLB ourselves because it is non-commercial (you approved it as an eval-only reference, so tell me if you want it run).

## 5b. Native-speaker review of the baseline (67 frozen prompts)

Scores AI-drafted, then reviewed and confirmed by the supervisor (2026-11-05). Consistency rules applied in `human_review_final.csv` (originals kept in `human_review_scored.csv`): orthography only for Bambara the model actually wrote; hallucination only when the model produced a real answer (copies, loops and empty headers are "no answer", already captured by adequacy = 1).

| answer type | count |
|---|---|
| no_answer (copy/loop/header) | 40 |
| answer without Bambara | 21 |
| attempted Bambara (gibberish) | 6 |

| measure | mean (1–5) | rows scored |
|---|---|---|
| adequacy | 1.09 | 67 |
| fluency | 1.31 | 67 |
| orthography (own Bambara only) | 1.17 | 6 |
| hallucination (real answers only) | 1.81 | 27 |

**Real answers that invent content (hallucination ≤ 2): 21 of 27 (78%).** The only correct facts (qa-08, qa-09, qa-13; adequacy 3) were given in French or English instead of Bambara.

| category | n | adequacy | no answer | real answers inventing content |
|---|---|---|---|---|
| generate | 15 | 1.00 | 10 | 3/5 |
| qa | 12 | 1.50 | 8 | 0/4 |
| summarize | 6 | 1.00 | 4 | 2/2 |
| translate_bam_en | 4 | 1.00 | 1 | 3/3 |
| translate_bam_fr | 12 | 1.00 | 8 | 4/4 |
| translate_en_bam | 5 | 1.00 | 1 | 4/4 |
| translate_fr_bam | 13 | 1.00 | 8 | 5/5 |

## 6. Decisions needed from the supervisor

1. ~~Alignment inspection~~ **done (2026-11-05).** Optional follow-up: mark the 2 × 30 random NLLB pairs in `reports/alignment/inspect_extra/` (≈ 20 min) to decide whether NLLB's top band comes back.
2. ~~Native-speaker prompt set~~ **done (2026-11-05):** 67 approved prompts frozen in `data/eval/human_prompts.jsonl` (sha256 `860fe0d9…`, from `prompts/human_prompts_corrected.csv`). Much of the Bambara was AI-drafted and then reviewed or rewritten by the supervisor (recorded in the notes). It is added to the contamination index; 4 matching training sentences were removed. Baseline answers scored (see §5b).
3. **Gate 1 defaults still in force unless you object:** encoding fixes, `ʃ`/`ñ` left unmapped, `[?]`/`[cs]` segments dropped, the tokenizer kept as is, no orthographic normalization.
4. **Optional:** run the NLLB-200 reference baseline (non-commercial model, eval only)?
5. **Access still pending:** FLORES+ (accept the terms), `djelia/bambara-texts`.

## 7. Next (Gate 3, after approval)

- Choose French/English **replay data** (commercial-use license, e.g. FineWeb-2 fra / FineWeb eng, verified first).
- Build the CPT mix: Bambara mono + parallel formatted as text + 20–30% replay + religious cap.
- Stage 0 smoke test on the Mac with mlx-lm LoRA (format, loss masking, memory).
