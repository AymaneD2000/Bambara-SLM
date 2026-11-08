# CPT mix: with_nllb

Config: `configs/cpt_mix.yaml`. Examples ≤ 2048 tokens, packed per stream, shuffled.

- train: 8,729 examples, **14.25 M tokens**; valid: 524 examples
- replay share: 25.0% (target 25%); fr: 2.13 M tokens, 5 eval-matching sentences removed, en: 1.42 M tokens, 1 eval-matching sentences removed
- religious share of Bambara tokens: 8.1% (cap 20%; dropped 0 tokens)

| stream | tokens | share |
|---|---|---|
| mono:afvoices | 2,386,963 | 16.8% |
| par:bayelemabaga-fr | 2,185,607 | 15.3% |
| replay:fr | 2,137,094 | 15.0% |
| mono:madlad_bm | 1,719,694 | 12.1% |
| replay:en | 1,425,099 | 10.0% |
| par:jeli_asr-fr | 1,052,153 | 7.4% |
| par_cpt_only:nllb_mined-en | 1,009,400 | 7.1% |
| par_cpt_only:nllb_mined-fr | 966,527 | 6.8% |
| mono:kunkado | 814,217 | 5.7% |
| mono:afvoices_notag | 353,431 | 2.5% |
| mono:wikipedia_bm | 174,037 | 1.2% |
| mono:asr_bam_only | 19,523 | 0.1% |
| par:bam_asr_all_extra_mali_pense-fr | 2,159 | 0.0% |
