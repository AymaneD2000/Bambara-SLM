# Contamination report (brief §8.4)

Eval index: exact match for sentences ≥ 4 words; MinHash near-duplicate (Jaccard ≥ 0.7) for sentences ≥ 6 words. Eval sets: bayelemabaga_test, belebele_passages, human_prompts, mafand_dev, mafand_test, mafand_train, sib200_dev, sib200_test, sib200_train.

- mono: 128 sentences removed; documents kept 186,572
- parallel: 457 pairs removed; kept 322,664

| data | source | match | eval set | count |
|---|---|---|---|---|
| parallel | nllb_mined | exact | bayelemabaga_test | 319 |
| mono | madlad_bm | exact | bayelemabaga_test | 65 |
| parallel | nllb_mined | near | bayelemabaga_test | 35 |
| parallel | bayelemabaga | near | bayelemabaga_test | 30 |
| parallel | jeli_asr | exact | bayelemabaga_test | 17 |
| mono | madlad_bm | near | bayelemabaga_test | 16 |
| parallel | bayelemabaga | exact | bayelemabaga_test | 16 |
| parallel | jeli_asr | near | bayelemabaga_test | 16 |
| mono | kunkado | exact | mafand_test | 15 |
| parallel | nllb_mined | near | mafand_test | 11 |
| mono | afvoices | near | bayelemabaga_test | 9 |
| mono | afvoices | exact | bayelemabaga_test | 6 |
| mono | kunkado | exact | bayelemabaga_test | 4 |
| parallel | bayelemabaga | exact | mafand_train | 3 |
| parallel | jeli_asr | exact | mafand_train | 3 |
| mono | kunkado | near | mafand_test | 2 |
| mono | afvoices | near | mafand_test | 2 |
| mono | afvoices_notag | exact | bayelemabaga_test | 1 |
| mono | afvoices | near | mafand_dev | 1 |
| mono | afvoices | near | human_prompts | 1 |
| mono | afvoices | exact | mafand_train | 1 |
| mono | afvoices | exact | mafand_test | 1 |
| mono | afvoices | near | mafand_train | 1 |
| mono | afvoices | exact | mafand_dev | 1 |
| mono | asr_bam_only | exact | bayelemabaga_test | 1 |
| mono | madlad_bm | exact | human_prompts | 1 |
| parallel | bayelemabaga | exact | human_prompts | 1 |
| parallel | jeli_asr | near | human_prompts | 1 |
| parallel | jeli_asr | exact | mafand_test | 1 |
| parallel | bam_asr_all_extra_mali_pense | exact | bayelemabaga_test | 1 |
| parallel | nllb_mined | near | mafand_dev | 1 |
| parallel | nllb_mined | exact | mafand_train | 1 |
| parallel | nllb_mined | near | belebele_passages | 1 |

FLORES+ is not yet in the index (access pending); re-run this step when it is added.
