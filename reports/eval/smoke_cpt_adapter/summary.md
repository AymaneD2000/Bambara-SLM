# Evaluation: smoke_cpt_adapter

Model: `Qwen/Qwen3-1.7B-Base`  adapter: `runs/smoke_cpt`

## Perplexity (lower is better)

| set | texts | tokens | ppl |
|---|---|---|---|
| bam_val_spoken | 30 | 808 | 42.3 |
| bam_val_web | 30 | 23,292 | 23.61 |
| bam_val_written | 12 | 889 | 86.56 |
| belebele_bam | 30 | 6,689 | 37.56 |
| belebele_fr | 30 | 4,742 | 8.63 |
| belebele_en | 30 | 2,987 | 18.56 |

## Translation (few-shot, greedy)

| set:direction | n | BLEU | chrF++ |
|---|---|---|---|
| bayelemabaga_test:bam-fr | 30 | 1.37 | 12.13 |
| bayelemabaga_test:fr-bam | 30 | 0.69 | 5.67 |
| mafand_test:bam-fr | 30 | 0.35 | 13.98 |
| mafand_test:fr-bam | 30 | 1.3 | 6.9 |
| sib200_test:bam-fr | 30 | 3.19 | 16.87 |
| sib200_test:fr-bam | 30 | 0.31 | 5.68 |
| sib200_test:bam-en | 30 | 0.46 | 14.98 |
| sib200_test:en-bam | 30 | 0.43 | 6.11 |
