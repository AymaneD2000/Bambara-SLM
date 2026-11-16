# FLORES-200 devtest: Bambara model vs NLLB baselines

Full FLORES-200 devtest (1012 aligned sentences), directions bam→fr, fr→bam, bam→en, en→bam.
Metrics: sacrebleu corpus BLEU (13a tokenisation) and chrF++ (`word_order=2`), same calls as `bslm.eval.baseline`. Hypotheses: `<system>__<dir>.jsonl`; re-run with `PYTHONPATH=src .venv/bin/python -m bslm.eval.flores_compare --system <name>`.

## Scores (BLEU / chrF++)

| system | bam-fr | fr-bam | bam-en | en-bam | time (s) |
|---|---|---|---|---|---|
| v3 (ours, Qwen3-1.7B SFT, MLX) | 4.93 / 24.96 | 2.66 / 20.88 | 6.16 / 26.46 | 3.85 / 21.49 | 2225 |
| NLLB-200-distilled-600M | 11.64 / 32.33 | 4.51 / 25.99 | 13.79 / 35.21 | 5.95 / 27.94 | 7098 |
| Youssouf0123/bambara-french-nllb | 9.94 / 32.37 | 5.07 / 27.34 | n/a | n/a | 3548 |

## Setup

- **v3** (`runs/sft_v3/merged`): zero-shot chat instruction from `configs/eval.yaml` (`Traduis en français : {x}` / `Traduis en bambara : {x}` / `Translate into English: {x}`), chat template with thinking disabled, greedy, max tokens = 2 × source tokens + 16, batch 16, first output line kept (`mt_prompts` / `mt_generate` in `bslm.eval.baseline`, the `eval-baseline --chat` path).
- **CPT base** (`runs/cpt_with_nllb/merged`, if present): the repo's 5-shot completion prompt with the fixed shots (fr shots: Bayelemabaga train; en shots: FLORES *dev* via SIB-200 train, not devtest).
- **NLLB** (`facebook/nllb-200-distilled-600M`) and **NLLB-FT** (`Youssouf0123/bambara-french-nllb`, root = checkpoint 14500, bam↔fr only): HF transformers on MPS, fp32, codes bam_Latn / fra_Latn / eng_Latn, `forced_bos_token_id` = target code, beam 4, max_length 256, batch 16.

## Caveats

- Decoding differs by design: our model uses the repo's fixed greedy decoding, NLLB uses beam 4. Prompts/decoding were not tuned on devtest for any system.
- NLLB-200 was trained with FLORES-200 as its official evaluation set (dev used for development); no FLORES training contamination is documented, but the NLLB data pipeline (incl. mined data) is not auditable here.
- NLLB-FT: its model card lists training data as jeli-asr transcripts + Bayelemabaga + human-reviewed Bambara corpora and does not mention FLORES; overlap with FLORES cannot be ruled out (not verified).
- The **v2** model is not included: it was trained on FLORES dev+devtest sentences (via the SIB-200 splits), so its FLORES scores would be contaminated.
- The repo's `sib200_test` scores (reports/eval/sft_v3) are on a 191–204 sentence subset of the same devtest; these full-devtest numbers supersede them for FLORES comparisons.
- Single reference, corpus-level scores; Bambara orthography varies, which depresses BLEU into Bambara.
- v3 reproduces the repo's eval up to batch numerics: on the 79 devtest sentences shared with `sib200_test`, 39/79 bam→fr hypotheses are identical to reports/eval/sft_v3 and chrF++ on that overlap is 26.3 vs 26.7 (MLX batched greedy decoding depends on batch composition/padding).
- Times are generation only (MPS / MLX on a 16 GB M-series Mac, under memory pressure); NLLB beam search is slowed by runaway repetitions up to max_length (e.g. 34/1012 fr→bam outputs > 2× source length + 5).
- CPT base (`--system cpt`) is implemented but was not run for this report (time budget).

## Licences

- facebook/nllb-200-distilled-600M: CC-BY-NC-4.0 (used for evaluation only).
- Youssouf0123/bambara-french-nllb: CC-BY-NC-4.0 (fine-tune of NLLB; evaluation only).
- FLORES-200: CC-BY-SA-4.0.

## Contamination check (FLORES devtest vs our training data)

Normalised (NFC, lower-case, punctuation stripped) match. *exact* = a whole message / line equals the sentence; *contained* = the sentence (≥ 6 words) appears word-bounded inside any message / line. Positive control: the same check finds ~305 devtest sentences per language in data/sft_v2/train.jsonl.

| corpus | lang | exact | contained (≥6 words) / checked |
|---|---|---|---|
| sft_v3 | bam | 0 | 0 / 1012 |
| sft_v3 | fr | 0 | 0 / 1012 |
| sft_v3 | en | 0 | 0 / 1012 |
| cpt_with_nllb | bam | 0 | 0 / 1012 |
| cpt_with_nllb | fr | 0 | 0 / 1012 |
| cpt_with_nllb | en | 0 | 0 / 1012 |
