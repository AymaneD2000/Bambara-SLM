# Evaluation: sft_v2

Model: `runs/sft_v2/merged`  adapter: `None`

## Perplexity (lower is better)

| set | texts | tokens | ppl |
|---|---|---|---|
| bam_val_spoken | 500 | 11,284 | 21.56 |
| bam_val_web | 31 | 24,090 | 11.33 |
| bam_val_written | 12 | 889 | 26.73 |
| belebele_bam | 462 | 94,792 | 23.8 |
| belebele_fr | 462 | 69,113 | 7.91 |
| belebele_en | 462 | 44,398 | 15.29 |

## Translation (few-shot, greedy)

| set:direction | n | BLEU | chrF++ |
|---|---|---|---|
| bayelemabaga_test:bam-fr | 500 | 8.66 | 24.24 |
| bayelemabaga_test:fr-bam | 500 | 4.5 | 20.83 |
| mafand_test:bam-fr | 500 | 4.21 | 25.34 |
| mafand_test:fr-bam | 500 | 2.54 | 17.1 |
| sib200_test:bam-fr | 191 | 5.88 | 26.83 |
| sib200_test:fr-bam | 191 | 1.35 | 17.11 |
| sib200_test:bam-en | 191 | 6.2 | 25.89 |
| sib200_test:en-bam | 191 | 2.65 | 16.27 |
