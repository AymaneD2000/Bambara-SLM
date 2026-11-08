# Raw data verification

**Overall: PASS**

## Integrity

- files on disk: 183
- checksum mismatches vs MANIFEST.json: 0
- files missing from manifest: 0
- manifest entries without a file: 0
- partial downloads (.part/.tmp): 0
- empty (0-byte) files: 0

## Row counts vs published

| dataset | rows | expected | empty texts | status |
|---|---|---|---|---|
| afvoices train | 253,290 | 253,290 | 0 | ✅ |
| afvoices test | 6,718 | 6,718 | 0 | ✅ |
| afvoices_notag train | 21,253 | 21,253 | 0 | ✅ |
| afvoices_notag test | 1,128 | 1,128 | 0 | ✅ |
| kunkado train | 33,282 | 33,282 | 9 | ✅ |
| kunkado test | 5,775 | 5,775 | 0 | ✅ |
| jeli_asr bam-asr-oza train | 30,685 | 30,685 | 0 | ✅ |
| jeli_asr bam-asr-oza test | 1,243 | 1,243 | 0 | ✅ |
| jeli_asr jeli-asr-rmai train | 1,495 | 1,495 | 0 | ✅ |
| jeli_asr jeli-asr-rmai test | 220 | 220 | 0 | ✅ |
| bam-asr-all oza-mali-pense | 4,421 | 4,421 | 0 | ✅ |
| bam-asr-all rt-data-collection | 705 | 705 | 0 | ✅ |
| bayelemabaga train | 37,580 | 37,580 |  | ✅ |
| bayelemabaga valid | 4,698 | 4,698 |  | ✅ |
| bayelemabaga test | 4,698 | 4,698 |  | ✅ |
| wikipedia_bm | 1,258 | 1,258 | 0 | ✅ |
| nllb bam-fra | 307,529 | 307,529 |  | ✅ |
| nllb bam-eng | 180,936 | 180,936 |  | ✅ |
| madlad bm clean | 702 | None |  | ✅ |
| madlad bm noisy | 21,939 | None |  | ✅ |
| mafand train (eval only) | 3,013 | None |  | ✅ |
| mafand dev (eval only) | 1,500 | 1,500 |  | ✅ |
| mafand test (eval only) | 1,500 | 1,500 |  | ✅ |
| belebele bam_Latn | 900 | 900 |  | ✅ |
| belebele fra_Latn | 900 | 900 |  | ✅ |
| belebele eng_Latn | 900 | 900 |  | ✅ |
| sib200 bam_Latn train | 701 | 701 |  | ✅ |
| sib200 fra_Latn train | 701 | 701 |  | ✅ |
| sib200 eng_Latn train | 701 | 701 |  | ✅ |
| sib200 bam_Latn dev | 99 | 99 |  | ✅ |
| sib200 fra_Latn dev | 99 | 99 |  | ✅ |
| sib200 eng_Latn dev | 99 | 99 |  | ✅ |
| sib200 bam_Latn test | 204 | 204 |  | ✅ |
| sib200 fra_Latn test | 204 | 204 |  | ✅ |
| sib200 eng_Latn test | 204 | 204 |  | ✅ |

## Not downloaded (access)

- `flores_plus`: gated: supervisor must accept terms on HF
- `djelia/bambara-texts`: gated: access not granted
- `oza75/bambara-tts`: gated: request pending (its mali_pense text is covered by bam-asr-all)
