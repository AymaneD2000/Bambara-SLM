# Manual inspection results

`y` = 1, `?` = 0.5, `n` = 0. Target precision for a kept bucket: **80%**. Interval = 95% Wilson. Buckets with fewer than 10 marks are shown but not trusted.

## bayelemabaga_fr

| bucket | marked | y | ? | n | precision | 95% interval | pairs in data |
|---|---|---|---|---|---|---|---|
| 0.0-0.2 | 34 | 30 | 0 | 4 | 88% | 73%–95% | 2,345 |
| 0.2-0.4 | 30 | 25 | 0 | 5 | 83% | 66%–93% | 1,949 |
| 0.4-0.6 | 31 | 30 | 0 | 1 | 97% | 84%–99% | 2,632 |
| 0.6-0.8 | 31 | 27 | 0 | 4 | 87% | 71%–95% | 5,091 |
| 0.8-1.0 | 30 | 29 | 0 | 1 | 97% | 83%–99% | 28,160 |

**Proposal:** keep score ≥ 0.0 → 40,177 pairs, estimated 94% correct.

## jeli_asr_fr

| bucket | marked | y | ? | n | precision | 95% interval | pairs in data |
|---|---|---|---|---|---|---|---|
| 0.0-0.2 | 33 | 21 | 0 | 12 | 64% | 47%–78% | 837 |
| 0.2-0.4 | 31 | 15 | 0 | 16 | 48% | 32%–65% | 748 |
| 0.4-0.6 | 30 | 24 | 0 | 6 | 80% | 63%–90% | 1,138 |
| 0.6-0.8 | 33 | 24 | 5 | 4 | 80% | 64%–90% | 2,401 |
| 0.8-1.0 | 38 | 36 | 2 | 0 | 97% | 87%–100% | 22,446 |

**Proposal:** keep score ≥ 0.4 → 25,985 pairs, estimated 95% correct.

## nllb_mined_en

| bucket | marked | y | ? | n | precision | 95% interval | pairs in data |
|---|---|---|---|---|---|---|---|
| 0.0-0.2 | 31 | 0 | 4 | 27 | 6% | 2%–21% | 15,388 |
| 0.2-0.4 | 30 | 3 | 9 | 18 | 25% | 13%–43% | 10,159 |
| 0.4-0.6 | 30 | 2 | 7 | 21 | 18% | 8%–35% | 11,172 |
| 0.6-0.8 | 31 | 6 | 5 | 20 | 27% | 15%–45% | 15,588 |
| 0.8-1.0 | 30 | 6 | 6 | 18 | 30% | 17%–48% | 46,100 |

**Proposal:** no score range reaches 80% → **drop this source** (or lower the target).

## nllb_mined_fr

| bucket | marked | y | ? | n | precision | 95% interval | pairs in data |
|---|---|---|---|---|---|---|---|
| 0.0-0.2 | 37 | 0 | 1 | 36 | 1% | 0%–12% | 34,981 |
| 0.2-0.4 | 32 | 0 | 0 | 32 | 0% | 0%–11% | 19,741 |
| 0.4-0.6 | 33 | 1 | 1 | 31 | 5% | 1%–18% | 19,021 |
| 0.6-0.8 | 32 | 1 | 3 | 28 | 8% | 2%–22% | 24,655 |
| 0.8-1.0 | 21 | 4 | 8 | 9 | 38% | 21%–59% | 58,114 |

**Proposal:** no score range reaches 80% → **drop this source** (or lower the target).

