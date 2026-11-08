# Evaluation: baseline_qwen3_1.7b_base

Model: `Qwen/Qwen3-1.7B-Base`  adapter: `None`  config: `configs/eval.yaml`

## Perplexity (lower is better)

| set | texts | tokens | ppl |
|---|---|---|---|
| bam_val_spoken | 500 | 11,284 | 295.06 |
| bam_val_web | 31 | 24,090 | 38.91 |
| bam_val_written | 12 | 889 | 192.94 |
| belebele_bam | 462 | 94,792 | 104.26 |
| belebele_fr | 462 | 69,113 | 7.97 |
| belebele_en | 462 | 44,398 | 15.41 |

## Translation (5-shot, greedy)

| set:direction | n | BLEU | chrF++ |
|---|---|---|---|
| bayelemabaga_test:bam-fr | 500 | 0.58 | 10.66 |
| bayelemabaga_test:fr-bam | 500 | 0.11 | 4.28 |
| mafand_test:bam-fr | 500 | 0.43 | 12.35 |
| mafand_test:fr-bam | 500 | 0.08 | 6.08 |
| sib200_test:bam-fr | 191 | 1.04 | 13.69 |
| sib200_test:fr-bam | 191 | 0.19 | 5.96 |
| sib200_test:bam-en | 191 | 2.78 | 19.66 |
| sib200_test:en-bam | 191 | 0.5 | 8.14 |
