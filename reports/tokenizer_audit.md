# Tokenizer audit – Qwen3-1.7B-Base

Data: Belebele FLORES passages (parallel content), NFC-normalized.

| lang | passages | words | tokens | tokens/word | chars/token |
|---|---|---|---|---|---|
| bam_Latn | 488 | 39745 | 102966 | 2.59 | 2.17 |
| fra_Latn | 488 | 44448 | 75567 | 1.70 | 3.65 |
| eng_Latn | 488 | 38737 | 48219 | 1.24 | 4.81 |

**bam / fra tokens-per-word ratio: 1.52** (threshold 2.5) → **KEEP original tokenizer** (supervisor approves).
bam / eng ratio: 2.08

## Special characters (tokens when isolated, NFC)

`ɛ`=1 `ɔ`=1 `ɲ`=1 `ŋ`=1 `Ɛ`=1 `Ɔ`=2 `Ɲ`=1 `Ŋ`=1 `à`=1 `á`=1 `è`=1 `é`=1 `ì`=1 `í`=1 `ò`=1 `ó`=1 `ù`=1 `ú`=1 `ʃ`=1

NFD (decomposed tone marks) cost: `à`=1 `á`=1 `è`=1 `é`=1 `ì`=1 `í`=1 `ò`=1 `ó`=1 `ù`=1 `ú`=1

N'Ko sample tokens/word: 5.00

## Most expensive frequent Bambara words (count ≥ 5)

| word | tokens | count |
|---|---|---|
| kɛrɛnkɛrɛniyala | 11 | 25 |
| kɛrɛnkɛrɛnlen | 10 | 6 |
| lakanatigilamɔgɔw | 10 | 5 |
| ɛntɛrɛnɛti | 9 | 22 |
| sɛgɛsɛgɛli | 9 | 18 |
| kɛrɛnkɛrɛni | 9 | 7 |
| yɛrɛmahɔrɔnya | 9 | 7 |
| farikoloɲɛnɛjɛ | 9 | 6 |
| sɔnnikɛyɔrɔ | 9 | 6 |
| ɲɛmɔgɔw | 8 | 13 |
| fɔlɔfɔlɔ | 8 | 11 |
| farikoloɲɛnajɛ | 8 | 8 |
| ɲɛnɛbɔba | 8 | 8 |
| yɛrɛyɛrɛ | 8 | 7 |
| baarakɛyɔrɔw | 8 | 5 |
| sɛgɛsɛgɛ | 8 | 5 |
| faraɲɔgɔnkan | 8 | 5 |
| ɲɛmɔgɔya | 8 | 5 |
| ɲɛmɔgɔ | 7 | 28 |
| ɲɔgɔndan | 7 | 19 |
| ɲɔgɔncɛ | 7 | 13 |
| baarakɛyɔrɔ | 7 | 12 |
| kilomɛtɛrɛ | 7 | 11 |
| kɔrɔnyanfan | 7 | 10 |
| ɔridinatɛri | 7 | 7 |

## Segmentation examples

- `kɛrɛnkɛrɛniyala` → Ġk | ÉĽ | r | ÉĽ | nk | ÉĽ | r | ÉĽ | ni | y | ala
- `kɛrɛnkɛrɛnlen` → Ġk | ÉĽ | r | ÉĽ | nk | ÉĽ | r | ÉĽ | n | len
- `lakanatigilamɔgɔw` → Ġl | akan | at | ig | il | am | ÉĶ | g | ÉĶ | w
- `ɛntɛrɛnɛti` → Ġ | ÉĽ | nt | ÉĽ | r | ÉĽ | n | ÉĽ | ti
- `sɛgɛsɛgɛli` → Ġs | ÉĽ | g | ÉĽ | s | ÉĽ | g | ÉĽ | li
- `kɛrɛnkɛrɛni` → Ġk | ÉĽ | r | ÉĽ | nk | ÉĽ | r | ÉĽ | ni
- `yɛrɛmahɔrɔnya` → Ġy | ÉĽ | r | ÉĽ | mah | ÉĶ | r | ÉĶ | nya
- `farikoloɲɛnɛjɛ` → Ġfar | ik | olo | É² | ÉĽ | n | ÉĽ | j | ÉĽ
- `sɔnnikɛyɔrɔ` → Ġs | ÉĶ | nn | ik | ÉĽ | y | ÉĶ | r | ÉĶ
- `ɲɛmɔgɔw` → Ġ | É² | ÉĽ | m | ÉĶ | g | ÉĶ | w
