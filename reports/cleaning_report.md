# Cleaning report (Gate 1)

Config: `configs/clean.yaml` (seed 13).

## Counts per source and step

| source | step → count |
|---|---|
| wikipedia_bm | read: 1,258 → normalize+markers: 1,258 → boilerplate+para_langid: 716 → langid: 706 → quality: 697 → dedup_exact: 697 → dedup_near: 633 |
| kunkado | read: 39,057 → normalize+markers: 31,878 → langid: 31,878 → quality: 31,854 → dedup_exact: 31,718 → dedup_near: 31,637 |
| afvoices_notag | read: 22,381 → normalize+markers: 20,802 → langid: 20,802 → quality: 20,791 → dedup_exact: 20,616 → dedup_near: 20,516 |
| afvoices | read: 260,008 → normalize+markers: 155,274 → langid: 155,274 → quality: 155,138 → dedup_exact: 133,036 → dedup_near: 131,975 |
| asr_bam_only | read: 1,612 → normalize+markers: 1,427 → langid: 1,427 → quality: 1,427 → dedup_exact: 587 → dedup_near: 585 |
| madlad_bm | read: 22,641 → normalize+markers: 22,641 → boilerplate+para_langid: 17,489 → langid: 2,080 → quality: 1,988 → dedup_exact: 1,316 → dedup_near: 1,252 |
| bayelemabaga | read: 46,976 → normalize+markers: 46,976 → minus_test_split: 42,278 → length_ratio: 41,793 → langid: 41,793 → dedup_exact_pair: 40,226 |
| jeli_asr | read: 33,643 → normalize+markers: 33,039 → length_ratio: 32,678 → langid: 32,678 → dedup_exact_pair: 27,517 |
| bam_asr_all_extra_mali_pense | read: 4,421 → normalize+markers: 4,118 → length_ratio: 4,075 → langid: 4,075 → dedup_exact_pair: 91 |
| bam_asr_all_extra_rt | read: 705 → normalize+markers: 0 → length_ratio: 0 → langid: 0 |
| nllb_mined | read: 488,465 → normalize+markers: 488,465 → length_ratio: 486,500 → langid: 271,303 → dedup_exact_pair: 255,287 |

## Rejections by step and reason

| step | reason | count |
|---|---|---|
| quality | doc_too_few_words | 815 |
| quality | too_short | 178 |
| quality | repetitive_ngrams | 59 |
| quality | too_long | 34 |
| quality | low_letter_ratio | 1 |
| doc_no_bambara_paragraphs | doc_no_bambara_paragraphs | 4,879 |
| langid | eng_Latn | 6,510 |
| langid | dag_Latn | 787 |
| langid | kiu_Latn | 403 |
| langid | dan_Latn | 388 |
| langid | por_Latn | 312 |
| langid | fra_Latn | 287 |
| langid | ind_Latn | 249 |
| langid | nld_Latn | 244 |
| langid | pol_Latn | 235 |
| langid | sco_Latn | 197 |
| langid | bam_Latn | 193 |
| langid | srd_Latn | 193 |
| transcript_short | transcript_short | 18,720 |
| transcript_marker | marker | 94,957 |
| dedup_exact | dedup_exact | 23,925 |
| dedup_near | dedup_near | 1,372 |
| par_length | len_ratio | 2,848 |
| par_length | too_few_words | 6 |
| empty_side | empty_side | 1,612 |
| par_langid | bam_side | 196,067 |
| par_langid | bam_side_is_fr | 15,671 |
| par_langid | bam_side_is_en | 1,640 |
| par_langid | both_sides | 1,082 |
| par_langid | other_side | 675 |
| par_langid | both_sides_is_fr | 49 |
| par_langid | both_sides_is_en | 13 |
| par_dedup_exact | par_dedup_exact | 26,728 |

## Clean output (Qwen3 tokens)

| file | source | records | Bambara tokens | other-side tokens |
|---|---|---|---|---|
| mono | wikipedia_bm | 633 | 174,304 | |
| mono | kunkado | 31,637 | 941,318 | |
| mono | afvoices_notag | 20,516 | 362,527 | |
| mono | afvoices | 131,975 | 2,387,956 | |
| mono | asr_bam_only | 585 | 19,991 | |
| mono | madlad_bm | 1,252 | 1,782,889 | |
| parallel | bayelemabaga (bam-fr) | 40,226 | 1,238,719 | 1,008,121 |
| parallel | jeli_asr (bam-fr) | 27,517 | 553,453 | 414,637 |
| parallel | bam_asr_all_extra_mali_pense (bam-fr) | 91 | 968 | 832 |
| parallel | nllb_mined (bam-fr) | 156,770 | 4,098,150 | 2,702,179 |
| parallel | nllb_mined (bam-en) | 98,517 | 2,720,824 | 1,313,853 |
| **total** | | | **14,281,099** | |

N'Ko-script records kept apart: 0. Eval records: 13,695 (bayelemabaga_test: 4,698, mafand_dev: 1,500, mafand_test: 1,500, mafand_train: 3,013, belebele_passages: 976, sib200_train: 1,402, sib200_dev: 198, sib200_test: 408).
Langid borderline (kept, sent to review): 9,645.

## PII replacements

bayelemabaga:<EMAIL>=2, kunkado:<PHONE>=139, madlad_bm:<EMAIL>=7, madlad_bm:<ID>=1, madlad_bm:<PHONE>=23, nllb_mined:<EMAIL>=1, nllb_mined:<ID>=2, nllb_mined:<PHONE>=12, wikipedia_bm:<PHONE>=4

## Character inventory (non-basic characters, Bambara side)

Full table: `reports/char_inventory.tsv`. Confusables applied: `ε`→`ɛ` (U+03B5→U+025B), `ὲ`→`ɛ̀` (U+1F72→U+025B+U+0300), `є`→`ɛ` (U+0454→U+025B), `ԑ`→`ɛ` (U+0511→U+025B), `ͻ`→`ɔ` (U+037B→U+0254), `ᴐ`→`ɔ` (U+1D10→U+0254), `כ`→`ɔ` (U+05DB→U+0254), `ว`→`ɔ` (U+0E27→U+0254), `ɳ`→`ɲ` (U+0273→U+0272), `π`→`ɲ` (U+03C0→U+0272), `Л`→`ɲ` (U+041B→U+0272). **Pending supervisor decision (not mapped):** `ʃ` (U+0283), `ñ` (U+00F1).

| char | codepoint | count | name |
|---|---|---|---|
| `è` | U+00E8 | 108,878 | LATIN SMALL LETTER E WITH GRAVE |
| `é` | U+00E9 | 72,833 | LATIN SMALL LETTER E WITH ACUTE |
| `ò` | U+00F2 | 46,702 | LATIN SMALL LETTER O WITH GRAVE |
| `à` | U+00E0 | 42,956 | LATIN SMALL LETTER A WITH GRAVE |
| `ô` | U+00F4 | 18,738 | LATIN SMALL LETTER O WITH CIRCUMFLEX |
| `̀` | U+0300 | 17,417 | COMBINING GRAVE ACCENT |
| `ù` | U+00F9 | 13,180 | LATIN SMALL LETTER U WITH GRAVE |
| `ì` | U+00EC | 9,454 | LATIN SMALL LETTER I WITH GRAVE |
| `É` | U+00C9 | 5,092 | LATIN CAPITAL LETTER E WITH ACUTE |
| `/` | U+002F | 3,911 | SOLIDUS |
| `À` | U+00C0 | 3,581 | LATIN CAPITAL LETTER A WITH GRAVE |
| `_` | U+005F | 2,217 | LOW LINE |
| `ʃ` | U+0283 | 2,106 | LATIN SMALL LETTER ESH |
| `›` | U+203A | 1,753 | SINGLE RIGHT-POINTING ANGLE QUOTATION MARK |
| `ç` | U+00E7 | 1,483 | LATIN SMALL LETTER C WITH CEDILLA |
| `ê` | U+00EA | 1,283 | LATIN SMALL LETTER E WITH CIRCUMFLEX |
| `á` | U+00E1 | 1,246 | LATIN SMALL LETTER A WITH ACUTE |
| `=` | U+003D | 1,220 | EQUALS SIGN |
| `%` | U+0025 | 1,199 | PERCENT SIGN |
| `[` | U+005B | 1,135 | LEFT SQUARE BRACKET |
| `]` | U+005D | 1,101 | RIGHT SQUARE BRACKET |
| `&` | U+0026 | 1,033 | AMPERSAND |
| `>` | U+003E | 979 | GREATER-THAN SIGN |
| `â` | U+00E2 | 878 | LATIN SMALL LETTER A WITH CIRCUMFLEX |
| `́` | U+0301 | 839 | COMBINING ACUTE ACCENT |
| `í` | U+00ED | 766 | LATIN SMALL LETTER I WITH ACUTE |
| `*` | U+002A | 760 | ASTERISK |
| `î` | U+00EE | 676 | LATIN SMALL LETTER I WITH CIRCUMFLEX |
| `ń` | U+0144 | 652 | LATIN SMALL LETTER N WITH ACUTE |
| `ú` | U+00FA | 610 | LATIN SMALL LETTER U WITH ACUTE |
| `→` | U+2192 | 600 | RIGHTWARDS ARROW |
| `Ò` | U+00D2 | 583 | LATIN CAPITAL LETTER O WITH GRAVE |
| `…` | U+2026 | 582 | HORIZONTAL ELLIPSIS |
| `ó` | U+00F3 | 561 | LATIN SMALL LETTER O WITH ACUTE |
| `<` | U+003C | 493 | LESS-THAN SIGN |
| `|` | U+007C | 482 | VERTICAL LINE |
| `·` | U+00B7 | 465 | MIDDLE DOT |
| `Ù` | U+00D9 | 460 | LATIN CAPITAL LETTER U WITH GRAVE |
| `+` | U+002B | 426 | PLUS SIGN |
| `ï` | U+00EF | 403 | LATIN SMALL LETTER I WITH DIAERESIS |
| `–` | U+2013 | 403 | EN DASH |
| `•` | U+2022 | 381 | BULLET |
| `▁` | U+2581 | 370 | LOWER ONE EIGHTH BLOCK |
| `ñ` | U+00F1 | 367 | LATIN SMALL LETTER N WITH TILDE |
| `ẁ` | U+1E81 | 367 | LATIN SMALL LETTER W WITH GRAVE |
| `̂` | U+0302 | 327 | COMBINING CIRCUMFLEX ACCENT |
| `°` | U+00B0 | 261 | DEGREE SIGN |
| `^` | U+005E | 255 | CIRCUMFLEX ACCENT |
| `#` | U+0023 | 255 | NUMBER SIGN |
| `û` | U+00FB | 202 | LATIN SMALL LETTER U WITH CIRCUMFLEX |
| `‹` | U+2039 | 199 | SINGLE LEFT-POINTING ANGLE QUOTATION MARK |
| `€` | U+20AC | 189 | EURO SIGN |
| `ꞃ` | U+A783 | 177 | LATIN SMALL LETTER INSULAR R |
| `\` | U+005C | 174 | REVERSE SOLIDUS |
| `©` | U+00A9 | 161 | COPYRIGHT SIGN |
| `̌` | U+030C | 161 | COMBINING CARON |
| `ö` | U+00F6 | 155 | LATIN SMALL LETTER O WITH DIAERESIS |
| `Ô` | U+00D4 | 127 | LATIN CAPITAL LETTER O WITH CIRCUMFLEX |
| `ë` | U+00EB | 120 | LATIN SMALL LETTER E WITH DIAERESIS |
| `²` | U+00B2 | 118 | SUPERSCRIPT TWO |

## Examples of near-duplicates removed

- `wikipedia_bm-0000057` dup of `wikipedia_bm-0000052`: 'Mopti ye Mali ka dugu ye. A be Jeliba dala.'
- `wikipedia_bm-0000386` dup of `wikipedia_bm-0000326`: "Nha Trang (Thành phố Nha Trang) bɛ Khanh Hoa, Vietnam kogo dugu nyogo fa. Fôlô fôlô donso cè dô tun bé y'in na tôgô tun "
- `wikipedia_bm-0000496` dup of `wikipedia_bm-0000494`: 'Bagneux ye Faransi ka dugu ye. Dugumogo be taa jon yooro.\nBagneux ɛntɛrinɛti siti'
- `wikipedia_bm-0000504` dup of `wikipedia_bm-0000497`: 'Colombes ye Faransi ka dugu ye. Dugumogo be taa jon yooro.\nColombes ɛntɛrinɛti siti'
- `wikipedia_bm-0000505` dup of `wikipedia_bm-0000500`: 'Courbevoie ye Faransi ka dugu ye. Dugumogo be taa jon yooro.\nCourbevoie ɛntɛrinɛti siti'
- `wikipedia_bm-0000507` dup of `wikipedia_bm-0000494`: 'Garches ye Faransi ka dugu ye. Dugumogo be taa jon yooro.\nGarches ɛntɛrinɛti siti'
- `wikipedia_bm-0000513` dup of `wikipedia_bm-0000494`: 'Malakoff ye Faransi ka dugu ye. Dugumogo be taa jon yooro.\nMalakoff ɛntɛrinɛti siti'
- `wikipedia_bm-0000519` dup of `wikipedia_bm-0000494`: 'Puteaux ye Faransi ka dugu ye. Dugumogo be taa jon yooro (2015).\nPuteaux ɛntɛrinɛti siti'
- `wikipedia_bm-0000529` dup of `wikipedia_bm-0000517`: 'Nantes ye Faransi ka dugu ye. Dugumogo be taa jon yooro (2015).\nNantes ɛntɛrinɛti siti'
- `wikipedia_bm-0000543` dup of `wikipedia_bm-0000494`: 'Coutras ye Faransi ka dugu ye. Dugumogo be taa jon yooro (2015).\nCoutras ɛntɛrinɛti siti'
