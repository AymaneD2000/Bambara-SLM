# Splits (brief §8.5)

Alignment threshold (PROVISIONAL, config `configs/align.yaml`): `{'bayelemabaga': 0.0, 'jeli_asr': 0.4, 'bam_asr_all_extra_mali_pense': 0.4, 'nllb_mined': 1.01, 'default': 0.5}`

| data | source | train | val | CPT-only train/val | below threshold | excluded (URL rule) |
|---|---|---|---|---|---|---|
| mono | afvoices | 128,485 | 3,472 | 0/0 | 0 | 0 |
| mono | afvoices_notag | 19,473 | 1,042 | 0/0 | 0 | 0 |
| mono | asr_bam_only | 561 | 23 | 0/0 | 0 | 0 |
| mono | kunkado | 26,936 | 4,695 | 0/0 | 0 | 0 |
| mono | madlad_bm | 1,221 | 31 | 0/0 | 0 | 0 |
| mono | wikipedia_bm | 621 | 12 | 0/0 | 0 | 0 |
| parallel | bam_asr_all_extra_mali_pense-fr | 71 | 0 | 0/0 | 19 | 0 |
| parallel | bayelemabaga-fr | 35,759 | 4,417 | 0/0 | 0 | 0 |
| parallel | jeli_asr-fr | 24,671 | 1,242 | 0/0 | 1,566 | 0 |
| parallel | nllb_mined-en | 0 | 0 | 20,817/26 | 70,748 | 6,816 |
| parallel | nllb_mined-fr | 0 | 0 | 21,535/27 | 125,782 | 9,168 |
