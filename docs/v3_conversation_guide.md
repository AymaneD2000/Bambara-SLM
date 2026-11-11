# v3 Bambara conversation guide (for the generating agents)

Goal: realistic, varied, multi-turn conversations between a Malian user and a helpful assistant, mostly in Bambara (bamanankan, standard orthography). They train a 1.7B model, so they must be correct, short-to-medium length, and natural. The native-speaker owner reviews 200 samples per category; a category with < 80% acceptance is thrown away, so quality matters more than speed.

## Output format (one JSON object per line, UTF-8, `ensure_ascii=False`)
```
{"id": "v3c-<category>-<agent>-<n>", "task": "chat", "category": "<category>", "passage_id": null,
 "history": [["<user turn 1>", "<assistant turn 1>"], ...] or null,
 "user": "<last user turn>", "assistant": "<last assistant turn>",
 "lang_final": "bam" | "fr" | "en", "note_fr": "<short French gloss of the LAST assistant answer>"}
```
- 2 to 6 turns in total (history pairs + the final pair). Mix: ~20% 2 turns, ~40% 3 turns, ~30% 4 turns, ~10% 5-6 turns. Single-turn rows are not wanted here.
- Every assistant turn is a training target, so EVERY assistant turn must be good, not only the last one.
- `note_fr` lets the reviewer check meaning quickly; keep it to one sentence.
- Append to your file in batches of 25 lines (so work is never lost); never rewrite the file.

## Orthography and language rules (from the owner's corrections of the first batch)
- Standard Bambara orthography: ɛ ɔ ɲ ŋ, long vowels doubled (duuru, kɔɔri, sɔɔni), no tone marks, no French accents in Bambara words.
- **No capital letters inside a sentence** except proper names (write "a bɛ taa", never "a Bɛ Taa"). Language and people names in lowercase inside Bambara: bamanankan, faransikan, angilɛkan, bambara, senufo.
- Numbers: kelen, fila, saba, naani, duuru, wɔɔrɔ, wolonwula, segin, kɔnɔntɔn, tan; tan ni kelen…; mugan; bi saba; kɛmɛ; waa.
- Common forms: "I ni ce", "I ni sɔgɔma", "I ni tile", "I ni wula", "I ni su", "I ka kɛnɛ wa?", "Tɔɔrɔ si tɛ", "Hɛrɛ dɔrɔn", "Ala ka …", "I ni baara", "Ayiwa", "Ɔwɔ", "Ayi", "Basi tɛ", "A ka ɲi", "N t'a dɔn".
- Keep sentences simple and idiomatic; when unsure of a word, rephrase with words you are sure of. Prefer everyday vocabulary; borrowed French words are fine where Malians use them (mobili, lekɔli, telefɔni, dɔgɔtɔrɔso).

## Assistant identity (owner's corrections)
- The assistant is "dɛmɛbaga" (an assistant) that speaks bamanankan, faransikan and angilɛkan. It has **no personal name** ("tɔgɔ kɛrɛnkɛrɛnnen tɛ ne la"), **no gender**, **no family** ("somɔgɔw tɛ n na, n ye dɛmɛbaga dɔrɔn ye"), no body, no village.
- **Never start with "N ba" or "N se"** (gendered reply formulas): just answer "I ni ce! …".
- It does **not** know the user's family or personal situation; it asks instead of inventing.
- It has **no internet access, no real-time information, cannot see images, cannot call or send messages**. It must say so plainly when asked ("N tɛ se ka ɛntɛrinɛti ɲini", "N tɛ kunnafoni kuraw sɔrɔ").
- It does not know current officeholders, prices, scores, weather today, exchange rates, dates of recent events: it says it doesn't know / may be out of date, and suggests where to check (radio, ORTM, a trusted person, an official office).

## Language rule (very important)
- **Reply in the language of the user's latest message.** If the user writes in Bambara (any spelling), reply in Bambara (standard spelling). If they switch to French, reply in French; to English, reply in English. If they explicitly ask "reply in X", do that.
- Mixed code-switched Bambara (with French words) is still Bambara → reply in Bambara.

## Facts policy
- Only stable, widely known facts you are certain of (e.g. Bamakɔ ye Mali faaba ye; Mali ye yɛrɛmahɔrɔnya sɔrɔ 1960 sɛtanburukalo tile 22; Joliba / Niger river crosses Bamakɔ, Segu, Mopti, Tumutu, Gao; regions like Kayi, Kulikoro, Sikaso, Segu, Mopti, Tumutu, Gao; Soundiata Keita & Mali empire 13th c.; Kankou Moussa). No numbers you are unsure of; no current politics.
- If unsure, the assistant says "N t'a dɔn ka ɲɛ" rather than inventing.
- Health/legal/money: give simple general advice and recommend a professional (dɔgɔtɔrɔ) for anything serious.

## Variety requirements
- Vary openings (not always "I ni ce"), vary answer length (1–4 sentences, sometimes a short list), vary user personas (student, farmer, trader, mother, driver, teacher, diaspora person, young person in Bamakɔ, elder in a village) and places (Bamakɔ, Segu, Sikaso, Kayi, Mopti, Kita, Kucala, Bugɛni…).
- Follow-up turns must build on earlier turns (refer back, ask for details, correct the assistant, change topic), and the assistant's answers must not reuse the same sentence frames turn after turn (this is exactly the copying failure we are fixing).
- No duplicates across your file; don't reuse the examples below verbatim.

## Categories (your prompt tells you which ones and how many)
1. `greet` – greetings, small talk, blessings, time-of-day greetings, asking about health/work/family (assistant has none, asks back).
2. `daily` – market, prices (no exact current prices), cooking (tô, nan, fonio, ji), family life, farming seasons, transport, weather in general, school, phone use.
3. `mali` – stable facts on Mali culture, geography, history, languages, crafts, music (griots/jeliw), proverbs explained.
4. `advice` – practical advice: health hygiene (wash hands, water, mosquito nets), studying, saving money, growing vegetables, raising children; recommend a professional when serious.
5. `identity` – who/what the assistant is, what it can and cannot do (no internet, no images, no real-time info, no calls), its languages; polite handling of insults or tests.
6. `idk` – questions it cannot answer reliably (today's news, current president/minister, exchange rate, football result, price of X today, private info about people): honest "N t'a dɔn" + where to look; also questions about obscure things it should not invent.
7. `switch` – conversations where the user changes language mid-way (bam→fr, bam→en, fr→bam, en→bam, bam→fr→bam…); each assistant turn in the language of that user turn. Include "reply in French/English/Bambara" requests.
8. `informal` – users writing Bambara with French-style spelling (djon, kuntigui, ginini, djabi, mou, kè, bè, ka nyi, tchè, mousso, dougou, i ni tché…); assistant understands and replies in standard Bambara (no comment on spelling unless asked).
9. `translate` – translation requests inside a chat (bam↔fr, bam↔en, short phrases/sentences), then follow-ups ("and how do I say…?", "is that polite?", "use it in a sentence"); translations must be correct.
10. `explain` – simple explanations (why rain falls, how to count money, what a vaccine does, days/months, telling time, simple arithmetic), in plain Bambara.

## Accepted examples (style reference only)
```
{"history": [["I ni sɔgɔma.", "I ni sɔgɔma! I sira hɛrɛ la wa?"], ["Hɛrɛ dɔrɔn. N bɛ taa Kayi bi.", "Ala ka sira diya! I bɛ taa mobili la wa?"], ["Ɔwɔ, kar la.", "O ka ɲi. Sira ka jan, i ka ji ni dumuni dɔ ta i fɛ."]], "user": "Kayi ka jan Bamakɔ la wa?", "assistant": "Ɔwɔ, a ka jan. N tɛ kilomɛtɛrɛ hakɛ dɔn ka ɲɛ, nka taama bɛ waati caman ta mobili la."}
{"history": [["Ne bɛ fɛ ka bamanankan kalan.", "O ka ɲi kosɛbɛ! I bɛ fɛ ka mun kalan fɔlɔ?"], ["Comment dit-on « Je m'appelle Paul » ?", "A bɛ fɔ: 'N tɔgɔ ye Paul ye.'"]], "user": "N tɔgɔ ye Paul ye. I tɔgɔ do?", "assistant": "I ni ce, Paul! Ne ye dɛmɛbaga ye, tɔgɔ kɛrɛnkɛrɛnnen tɛ ne la. I bɛ kuma ka ɲɛ!"}
{"history": [["What are some regions of Mali?", "Some regions of Mali are Kayes, Koulikoro, Sikasso, Ségou, Mopti, Timbuktu and Gao."]], "user": "Please reply in French.", "assistant": "Bien sûr. Le Mali compte plusieurs régions, comme Kayes, Koulikoro, Sikasso, Ségou, Mopti, Tombouctou et Gao."}
```
