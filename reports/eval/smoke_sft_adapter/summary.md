# Evaluation: smoke_sft_adapter

Model: `Qwen/Qwen3-1.7B-Base`  adapter: `runs/smoke_sft`

## Perplexity (lower is better)

| set | texts | tokens | ppl |
|---|---|---|---|
| bam_val_spoken | 30 | 808 | 98.61 |
| bam_val_web | 30 | 23,292 | 61.93 |
| bam_val_written | 12 | 889 | 197.89 |
| belebele_bam | 30 | 6,689 | 66.15 |
| belebele_fr | 30 | 4,742 | 16.29 |
| belebele_en | 30 | 2,987 | 30.65 |

## Translation (few-shot, greedy)

| set:direction | n | BLEU | chrF++ |
|---|---|---|---|
| bayelemabaga_test:bam-fr | 30 | 0.13 | 9.16 |
| bayelemabaga_test:fr-bam | 30 | 0.21 | 3.56 |
| mafand_test:bam-fr | 30 | 0.05 | 8.22 |
| mafand_test:fr-bam | 30 | 0.37 | 3.76 |
| sib200_test:bam-fr | 30 | 0.07 | 7.99 |
| sib200_test:fr-bam | 30 | 0.09 | 2.5 |
| sib200_test:bam-en | 30 | 0.04 | 7.08 |
| sib200_test:en-bam | 30 | 0.22 | 3.94 |
