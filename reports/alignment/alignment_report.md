# Alignment scoring (Gate 2, brief §8.2)

Score = P(aligned) from a logistic regression over LASER3/LASER2 margin + cosine, IBM-1 lexical scores, length ratio, number and punctuation agreement. Trained on trusted bam-fr pairs vs shuffled and off-by-one negatives.

Combiner held-out AUC: with lexicon 0.952, without lexicon 0.893.

## Score distribution (records per bucket)

| group | n | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 | median |
|---|---|---|---|---|---|---|---|
| bayelemabaga_fr | 40,226 | 2,345 | 1,952 | 2,631 | 5,096 | 28,202 | 0.92 |
| jeli_asr_fr | 27,608 | 838 | 748 | 1,139 | 2,403 | 22,480 | 0.96 |
| nllb_mined_en | 98,517 | 15,389 | 10,160 | 11,167 | 15,605 | 46,196 | 0.77 |
| nllb_mined_fr | 156,770 | 34,985 | 19,745 | 19,019 | 24,689 | 58,332 | 0.64 |

## Manual inspection files

`reports/alignment/inspect/<group>__<bucket>.tsv`: up to 100 random pairs per bucket. Please fill column `ok` with y / n / ? (partial). Thresholds are chosen from these judgements.

