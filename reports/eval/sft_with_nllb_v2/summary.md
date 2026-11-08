# Evaluation: sft_with_nllb_v2

Model: `runs/sft_with_nllb_v2/merged`  adapter: `None`

## Perplexity (lower is better)

| set | texts | tokens | ppl |
|---|---|---|---|
| bam_val_spoken | 500 | 11,284 | 24.58 |
| bam_val_web | 31 | 24,090 | 12.35 |
| bam_val_written | 12 | 889 | 33.55 |
| belebele_bam | 462 | 94,792 | 28.6 |
| belebele_fr | 462 | 69,113 | 8.19 |
| belebele_en | 462 | 44,398 | 15.9 |

## Translation (few-shot, greedy)

| set:direction | n | BLEU | chrF++ |
|---|---|---|---|
| bayelemabaga_test:bam-fr | 500 | 8.74 | 25.09 |
| bayelemabaga_test:fr-bam | 500 | 5.53 | 23.03 |
| mafand_test:bam-fr | 500 | 3.81 | 25.03 |
| mafand_test:fr-bam | 500 | 2.72 | 17.68 |
| sib200_test:bam-fr | 191 | 3.75 | 23.56 |
| sib200_test:fr-bam | 191 | 1.5 | 16.51 |
| sib200_test:bam-en | 191 | 1.39 | 16.99 |
| sib200_test:en-bam | 191 | 2.05 | 16.8 |
