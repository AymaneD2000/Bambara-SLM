# Alignment scorer validation on held-out gold pairs

Gold file: `data/eval/sib200_test.jsonl`

| lang | n | median score (gold) | median (shuffled) | gold ≥0.5 | shuffled ≥0.5 | gold ≥0.8 | shuffled ≥0.8 |
|---|---|---|---|---|---|---|---|
| bam-en | 191 | 0.47 | 0.04 | 45% | 1% | 9% | 0% |
| bam-fr | 191 | 0.45 | 0.06 | 42% | 1% | 9% | 0% |

Gold ≥ t approximates recall at threshold t; shuffled ≥ t approximates the false-accept rate of clearly wrong pairs. Off-domain caveat: FLORES sentences are longer and more formal than most training pairs.
