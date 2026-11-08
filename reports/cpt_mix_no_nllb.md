# CPT mix: no_nllb

Config: `configs/cpt_mix.yaml`. Examples ≤ 2048 tokens, packed per stream, shuffled.

- train: 7,241 examples, **11.61 M tokens**; valid: 522 examples
- replay share: 25.0% (target 25%); fr: 1.74 M tokens, 5 eval-matching sentences removed, en: 1.16 M tokens, 1 eval-matching sentences removed
- religious share of Bambara tokens: 4.0% (cap 20%; dropped 0 tokens)

| stream | tokens | share |
|---|---|---|
| mono:afvoices | 2,386,963 | 20.6% |
| par:bayelemabaga-fr | 2,185,607 | 18.8% |
| replay:fr | 1,742,639 | 15.0% |
| mono:madlad_bm | 1,719,694 | 14.8% |
| replay:en | 1,161,994 | 10.0% |
| par:jeli_asr-fr | 1,052,153 | 9.1% |
| mono:kunkado | 814,217 | 7.0% |
| mono:afvoices_notag | 353,431 | 3.0% |
| mono:wikipedia_bm | 174,037 | 1.5% |
| mono:asr_bam_only | 19,523 | 0.2% |
| par:bam_asr_all_extra_mali_pense-fr | 2,159 | 0.0% |
