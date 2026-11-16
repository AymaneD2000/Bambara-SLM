# Evaluation: sft_v3

Model: `runs/sft_v3/merged`  adapter: `None`

## Perplexity (lower is better)

| set | texts | tokens | ppl |
|---|---|---|---|
| bam_val_spoken | 500 | 11,284 | 20.05 |
| bam_val_web | 31 | 24,090 | 9.94 |
| bam_val_written | 12 | 889 | 22.33 |
| belebele_bam | 462 | 94,792 | 23.96 |
| belebele_fr | 462 | 69,113 | 8.22 |
| belebele_en | 462 | 44,398 | 16.02 |

## Translation (few-shot, greedy)

| set:direction | n | BLEU | chrF++ |
|---|---|---|---|
| bayelemabaga_test:bam-fr | 500 | 10.0 | 26.98 |
| bayelemabaga_test:fr-bam | 500 | 8.22 | 26.16 |
| mafand_test:bam-fr | 500 | 5.58 | 27.83 |
| mafand_test:fr-bam | 500 | 4.7 | 22.39 |
| sib200_test:bam-fr | 191 | 4.76 | 26.02 |
| sib200_test:fr-bam | 191 | 2.34 | 20.82 |
| sib200_test:bam-en | 191 | 6.83 | 27.66 |
| sib200_test:en-bam | 191 | 3.61 | 22.05 |
