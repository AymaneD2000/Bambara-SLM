# Evaluation: cpt_with_nllb

Model: `runs/cpt_with_nllb/merged`  adapter: `None`

## Perplexity (lower is better)

| set | texts | tokens | ppl |
|---|---|---|---|
| bam_val_spoken | 500 | 11,284 | 20.76 |
| bam_val_web | 31 | 24,090 | 10.96 |
| bam_val_written | 12 | 889 | 27.93 |
| belebele_bam | 462 | 94,792 | 26.3 |
| belebele_fr | 462 | 69,113 | 8.02 |
| belebele_en | 462 | 44,398 | 15.61 |

## Translation (few-shot, greedy)

| set:direction | n | BLEU | chrF++ |
|---|---|---|---|
| bayelemabaga_test:bam-fr | 500 | 7.53 | 21.73 |
| bayelemabaga_test:fr-bam | 500 | 1.85 | 14.73 |
| mafand_test:bam-fr | 500 | 2.67 | 22.27 |
| mafand_test:fr-bam | 500 | 1.27 | 12.72 |
| sib200_test:bam-fr | 191 | 3.43 | 22.64 |
| sib200_test:fr-bam | 191 | 1.01 | 13.01 |
| sib200_test:bam-en | 191 | 5.64 | 23.85 |
| sib200_test:en-bam | 191 | 1.53 | 13.54 |
