# Gate 1 – Clean data (2026-11-04)

Status: **waiting for supervisor approval**. Detailed files:
`reports/cleaning_report.md` (counts per step), `reports/char_inventory.tsv`, `reports/orthography_report.md`,
`reports/tokenizer_audit.md`, `reports/raw_verification.md` (download check: PASS), `reports/samples/*.txt` (100 random items per source).
Rejected items are kept with their reason in `data/clean/rejected/`; borderline language-ID items are in `data/clean/review/`.

## 1. What we have after cleaning

| set | records | Bambara tokens (Qwen3) | trust |
|---|---|---|---|
| Mono, spoken (afvoices, afvoices-notag, kunkado, untranslated ASR) | 184,713 | 3.71 M | high (human-corrected transcripts) |
| Mono, written (Wikipedia, MADLAD) | 1,885 | 1.96 M | medium (web, language-ID filtered) |
| Parallel bam–fr, Bayelemabaga (train+valid) | 40,226 | 1.24 M | high (some misalignment; Gate 2 scoring) |
| Parallel bam–fr, jeli-asr / Mali Pense (spoken) | 27,608 | 0.55 M | high |
| Parallel bam–fr / bam–en, NLLB mined | 255,287 | 6.82 M | **low**: many misaligned pairs (see §4) |
| **Total** | | **14.28 M** | about **7.5 M** if NLLB is mostly removed at Gate 2 |

Evaluation sets, never trained on: Bayelemabaga test (4,698), MAFAND dev/test/train (6,013; non-commercial, test only), Belebele passages (488 × fr/en), SIB-200 (1,004 × fr/en). FLORES+ is still blocked (accept its terms on HF).

**Honest assessment:** this is small. About 7–8 M reliable Bambara tokens is enough for a useful LoRA continued-pretraining run on a 1.7B model, but not for fluency across domains. More than half of the reliable monolingual text is **spoken** (transcripts), so the model will lean to a conversational register. Access to `djelia/bambara-texts` (written text) would help the most.

## 2. Decisions needed from the supervisor

**D1. Character fixes (encoding errors, applied; please confirm).** Some web and NLLB text was typed with legacy fonts that use letters from other scripts in place of Bambara letters. I mapped them on the Bambara side only:

| found | mapped to | example (before → after) | where |
|---|---|---|---|
| `ε` Greek, `є` `ԑ` Cyrillic | `ɛ` | `bε` → `bɛ`; `Kԑnyԑrԑye` → `Kɛnyɛrɛye` | MADLAD, Wikipedia, kunkado |
| `ὲ` Greek | `ɛ̀` | `kὲlɛ` → `kɛ̀lɛ` | NLLB, MADLAD, Wikipedia |
| `ͻ` Greek, `ᴐ`, `כ` Hebrew, `ว` Thai | `ɔ` | `kכnכ` → `kɔnɔ`; `dวnnibagaw` → `dɔnnibagaw` | NLLB |
| `ɳ`, `π` Greek, `Л` Cyrillic | `ɲ` | `damankɛπɛni` → `damankɛɲɛni`; `Лini` → `ɲini` | NLLB, MADLAD, Wikipedia |

`Л`→`ɲ` and `π`→`ɲ` are inferred from context, so please check them.

**D2. Characters left unchanged, your decision:**
- `ʃ` (2,106 occurrences, "sh" sound: `Ayiʃa`, `ʃu`, `ʃankɔrɔ`): keep, map to `sh`, or map to `s`?
- `ñ` (old spelling of `ɲ`, also Spanish names): keep as is? (my proposal)

**D3. Transcript annotation policy.** Segments where words are missing or unclear are dropped:
- afvoices: **35% dropped** (91,638 of 260,008). `[?]` (unclear word) accounts for 57,285 and `[cs]` (code-switched word replaced by the tag, so the word is missing) for 34,353. `[noise]`, `[um]` and `[pause]` are removed and the segment kept.
- kunkado: 2,717 dropped (`<INCOMPRÉHENSIBLE>`, `<CHEVAUCHEMENT>`, `<INVOCATION>`, or unknown tags). Noise and laughter tags are removed. `__French word__` code-switch markers are unwrapped and the French word is kept.
- Alternative: keep `[?]`/`[cs]` segments for CPT only, with the tag removed. That adds up to 1.7 M tokens of text with missing words (before deduplication). **My proposal: keep them dropped.**

**D4. Language identification.** GlotLID mixes Bambara with Dioula and other Manding languages, and it is unreliable on short lines: with a Bambara-only threshold, 34% of Bayelemabaga, which we know is Bambara, would be wrongly rejected. Rule used, calibrated on Bayelemabaga:
- Score the Manding cluster (bam + dyu + mnk + emk + …), not only `bam`.
- Lines of 9+ words need a Manding score of at least 0.5. Shorter lines are dropped only if they are confidently another language (p ≥ 0.9). On Bayelemabaga this wrongly rejects 4%.
- Noisy web sources (NLLB, MADLAD) use the strict rule at every length.
- Curated sources (transcripts, Bayelemabaga) are never dropped by language ID. 9,645 borderline items were sent to `data/clean/review/` for a native-speaker look.

Note: Dioula is very close to Bambara, so some Dioula text may remain in the web data.

**D5. Tokenizer.** Bambara needs 2.59 tokens per word, against 1.70 for French, a **ratio of 1.52**, below the 2.5 threshold. **Proposal: keep the Qwen3 tokenizer.** Caveat: `ɛ ɔ ɲ` are single tokens that split words (`sɛgɛsɛgɛli` becomes 9 tokens), so vocabulary extension remains a good **later ablation**.

**D6. Orthography: measured, not normalized.** Reformed spelling dominates in the transcripts (≈90% of records use ɛ/ɔ/ɲ/ŋ). Bayelemabaga and NLLB contain about 25–30% old or French-influenced spelling (`be`/`bè` for `bɛ`, `mogo` for `mɔgɔ`). Tone marks are rare except in MADLAD, Wikipedia and NLLB. **Proposal: no orthographic normalization now.** Revisit with `bambara-normalizer` (MIT) as an ablation, and use it for number normalization if you approve.

## 3. Other findings

- Exact duplicates: afvoices has 22k repeated prompts (several speakers reading the same sentence), and the Mali Pense ASR set is almost entirely contained in jeli-asr (4,075 → 91 after dedup).
- MADLAD noisy split: mostly English and Hindi web spam; 22,641 → 1,252 documents kept, which still give 1.78 M good tokens.
- Personal data: 191 replacements, mostly phone numbers in kunkado radio transcripts (`<PHONE>`), plus a few e-mails.
- No N'Ko-script text was found in these sources.

## 4. Known issue carried into Gate 2

NLLB mined pairs are **mostly misaligned** (e.g. `fanta bè taa` ↔ `La force aujourd'hui`) and contain many Bible and Quran website pairs. Language ID removed 44%. The rest needs alignment scoring with an encoder that covers Bambara, plus manual inspection of score buckets (brief §8.2). I expect a large share to be removed there. The contamination check against the evaluation sets (§8.4) also runs at Gate 2.

## 5. Still blocked on access

- FLORES+: accept the terms on Hugging Face (approval is automatic).
- `djelia/bambara-texts`: request access; check its license for commercial use.
- `oza75/bambara-tts`: pending review (low priority; its text is covered elsewhere).
