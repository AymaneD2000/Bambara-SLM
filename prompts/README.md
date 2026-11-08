# Native-speaker prompt set (brief §5.3, §10)

A fixed set of 50–100 prompts. It is used to judge the model **by humans** at every stage: baseline, after CPT, after SFT.
It is written or approved by native speakers, **frozen before any training**, and never used for training.

## How to fill it

1. Open `prompts/human_prompts_draft.csv` in Excel, LibreOffice or Google Sheets. Keep it as **CSV, UTF-8** when you save.
2. Rows marked `[À ÉCRIRE EN BAMBARA]` / `[WRITE IN BAMBARA]` need Bambara text from a native speaker.
   Rows I drafted in French or English only need checking: correct them if anything is unnatural or not relevant to Mali.
3. For each row, fill:
   - `prompt`: the exact text given to the model.
   - `reference`: a good answer, written by a native speaker. It is optional for free generation, but strongly recommended for translation and questions.
   - `approved`: `y` once a native speaker has checked the row.
4. Add, delete or change rows freely. Keep these rough shares:

| category | target | what it tests |
|---|---|---|
| `translate_bam_fr` | 12–15 | understanding Bambara |
| `translate_fr_bam` | 12–15 | writing correct Bambara |
| `translate_bam_en` / `translate_en_bam` | 8–10 | English direction |
| `qa` | 12–15 | answering simple questions in Bambara |
| `summarize` | 8–10 | summarizing a short Bambara text |
| `generate` | 12–15 | free writing: short story, message, advice, dialogue |

## Rules that make the set useful

- **Write new text.** Do not copy sentences from Bayelemabaga, MAFAND, Wikipedia, radio transcripts, the Bible or websites the model may have seen. Your own sentences are best.
- **Use everyday Mali topics:** health, farming, market and money, family, school, weather, transport, mobile phones, local administration, culture and proverbs.
- **Vary the length.** Use some very short prompts and some of 2–3 sentences. Summarization texts should be about 80–150 words.
- **Vary the instruction language:** instructions in Bambara, in French and in English.
- **Spelling:** use the standard orthography (ɛ, ɔ, ɲ, ŋ). Mark tones only if you would normally write them.
- **Avoid sensitive data:** no real names of private people, no phone numbers.
- Leave `id` as it is for drafted rows. New rows can use any unique id (e.g. `gen-16`).

## Freezing

When at least 50 rows have `approved = y`:

```
.venv/bin/bslm freeze-prompts
```

This checks the file:
- every approved row has a prompt;
- no placeholder text is left;
- categories are valid;
- there are no duplicates;
- it warns if a prompt already appears in the training data.

It then writes `data/eval/human_prompts.jsonl` and `prompts/human_prompts.FROZEN.sha256`. From then on, the training
pipeline removes any matching text from training (it is part of the contamination index), and the file must not change.

## How it will be scored (after each stage)

For every output, a native speaker gives 1–5 on each of:
- **adequacy:** is the meaning correct and complete?
- **fluency:** is it natural Bambara, French or English?
- **orthography:** is the Bambara spelling correct?
- **hallucination:** 5 = nothing invented, 1 = mostly invented.

Add a free comment where useful. I will generate the review sheet (`bslm` will produce a CSV with the model outputs side by side).
