# PROVENANCE

Every source is listed here **before** use. Verified 2026-11-04 via the HF Hub API and the upstream sites
(raw metadata: `reports/gate0_inventory.json`, cached cards/scripts: `data/raw/_meta/`).
Gate 0 approved by the supervisor on 2026-11-04.

## Policy (supervisor decisions, 2026-11-04)
1. **Training data must allow commercial use.** Non-commercial (NC) data is never used for training. It may be used for evaluation only.
2. When a source states conflicting licenses, apply the **less restrictive** one that allows commercial use, and record it here.
3. Share-alike (CC-BY-SA) and attribution (CC-BY, ODC-BY) sources are allowed. Their obligations go into the model card.
4. Bible data is deferred to v2. Corpus Bambara de Référence is skipped.

| Source | Role | License (verified) | Access | Revision | Status |
|---|---|---|---|---|---|
| `RobotsMaliAI/bayelemabaga` | parallel bam-fr (train/valid; **test = eval only**) | **CONFLICT:** README says CC-BY-SA-4.0; `LICENSE` file and GitHub repo say CC-BY-4.0 | open (script + parquet conversion; tarball 46.6 MB on github.io) | `70b44b26f0` | **approved** – train/valid for training; test = eval only. Decision 2026-11-04: apply the less restrictive license (CC-BY-4.0, per `LICENSE` file); attribution required |
| `masakhane/mafand` (fr-bam) | parallel fr-bam news | **CC-BY-NC-4.0** (data); code GPL-3.0 | open | `b215cd3c70` | **approved – TEST ONLY** (non-commercial, never in training) |
| `allenai/nllb` (`bam_Latn-fra_Latn`, `bam_Latn-eng_Latn`) | parallel (mined, noisy) | ODC-BY + terms of original sources | open | `c36967abb4` | **approved** – after alignment scoring (§8.2); upstream source terms noted |
| `djelia/bambara-texts` | mono (84,328 rows, `text`, `source_dataset`) | not stated on card metadata | **gated, no access yet** | `fefe46acbb` | blocked – no access; license must be checked for commercial use before any use |
| `allenai/MADLAD-400` (`bm`) | mono web | ODC-BY | open; `bm_clean` 1.2 MB gz, `bm_noisy` 15.1 MB gz | `9d886a76bd` | **approved** |
| bmwiki dump (`bmwiki-latest-pages-articles.xml.bz2`, 0.86 MB, 2026-11-01) | mono | CC-BY-SA-4.0 / GFDL (Wikimedia) | open | dump date 2026-11-01 | **approved** |
| `RobotsMali/afvoices` (`human-corrected`, `text`) | mono spoken | CC-BY-4.0 | open | `68f1d6873a` | **approved** |
| `RobotsMali/afvoices-notag` (`top20-speakers`) | mono spoken | CC-BY-4.0 | open | `b119321af4` | **approved** – dedup vs afvoices |
| `RobotsMali/kunkado` (`human-reviewed`, `corrected-label`) | mono spoken | CC-BY-SA-4.0 | open | `be43c51b54` | **approved** |
| `RobotsMali/bam-asr-all` (`bam`, `french` columns) | mono spoken + bam-fr pairs | CC-BY-4.0 | open | `a413c21896` | **approved** – dedup vs jeli-asr |
| `RobotsMali/jeli-asr` | mono spoken + bam-fr pairs | CC-BY-4.0 | open | `e0afc02875` | **approved** – dedup vs bam-asr-all |
| `oza75/bambara-tts` (`bambara`, `french`) | mono + pairs (transcripts only) | CC-BY-SA-4.0 | gated, **request pending review** (README readable, data 403) | `b96642a1fc` | approved, **blocked until access granted**; its mali_pense text is also in `bam-asr-all/oza-mali-pense` |
| Bible Brain API (`4.dbt.io`) | mono + verse-aligned parallel | per-fileset copyright, **not redistributable** | needs `BIBLE_BRAIN_API_KEY` (API reachable, 422 without key) | – | **deferred to v2** (supervisor decision 2026-11-04) |
| Corpus Bambara de Référence (`cormand.huma-num.fr`) | mono (upstream of Bayelemabaga) | **unknown** – site gives no terms | site up | – | **skipped** (supervisor decision 2026-11-04) |
| `openlanguagedata/flores_plus` (`bam_Latn`) | **eval only** | CC-BY-SA-4.0 | gated (auto-approve), **not yet requested** (data 403) | see json | approved, **blocked: supervisor must accept terms on HF** |
| `facebook/belebele` (`bam_Latn`) | **eval only** | CC-BY-SA-4.0 | open | see json | **approved** |
| `Davlan/sib200` (`bam_Latn`) | **eval only** | CC-BY-SA-4.0 | open | see json | **approved** |
| `Qwen/Qwen3-1.7B-Base` | base model | Apache-2.0 (LICENSE file checked) | open | see json | **approved** |
| `facebook/nllb-200-distilled-600M` | MT reference baseline | CC-BY-NC-4.0 | open | see json | **approved – eval reference only** (NC model: used only to produce comparison scores, never to generate training data) |
| `cis-lmu/glotlid` | language ID (`bam`, `bm` labels present) | Apache-2.0 plus notices | open | `baa20d1391` | **approved** |
| `bambara-normalizer` (PyPI; `diarray-hub/bambara-normalizer`) | text normalization | MIT | open | – | **approved** |

## Replay data (Gate 3, added 2026-11-05)

French/English replay to limit forgetting during CPT (brief §9.1). Only a small sample of each source is read
(first row groups of one shard, via column-projected range reads).

| Source | Role | License (verified) | Access | Revision | Status |
|---|---|---|---|---|---|
| `HuggingFaceFW/fineweb-2` (`fra_Latn`) | French replay | ODC-By 1.0 + Common Crawl terms of use | open | `af9c13333e` | **approved** (supervisor, 2026-11-05) |
| `HuggingFaceFW/fineweb-edu` (`sample/10BT`) | English replay | ODC-By 1.0 + Common Crawl terms of use | open | `87f09149ef` | **approved** (supervisor, 2026-11-05) |

## SFT instruction replay (Stage 2, added 2026-11-07)

| Source | Role | License (verified) | Access | Revision | Status |
|---|---|---|---|---|---|
| `CohereLabs/aya_dataset` (French 1,422 + English 3,944 rows, train split) | French/English general instruction replay for SFT; human-written (original + re-annotations) | Apache-2.0 | open | `f9ea04583f` | **approved** (supervisor chose "translation + fr/en replay", 2026-11-07) |

## v3 sources (prepared 2026-11-09)

Code: `src/bslm/v3data/` (one module per source; `python -m bslm.v3data.<module>`). Outputs: `data/interim/v3/`
(with a `*.stats.json` of the counts after each filter). Raw files: `data/raw/<source>/`. Licences checked on the
dataset cards and HF metadata at the pinned revision. Every output is decontaminated with `bslm.decontam.EvalIndex`
(data/eval: Bayelemabaga test, MAFAND train/dev/test, Belebele, SIB-200, human prompts) **plus FLORES-200 dev+devtest**
(bam/fra/eng, from the public release `dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`, stored in
`data/raw/flores200/`, used for the index only, because `openlanguagedata/flores_plus` is still gated).
Downloaded files are only parsed (parquet/jsonl/csv); no dataset code is run and `trust_remote_code` is never used.

| Source | Role | License (verified) | Access | Revision | Filters | Status |
|---|---|---|---|---|---|---|
| `HPLT/HPLT2.0_cleaned` (`bam_Latn`) | mono Bambara segments, back-translation targets (`mono_bam.jsonl`) | CC0-1.0 (packaging) + HPLT terms of use; text under the original sites' rights | open | `d1324a5283` | normalize + confusables; drop domains jw.org, bible.is (copyrighted religious / Bible text) and chatgpt4.win, aivideo-app.com, zhitov.ru (machine-translated); boilerplate lines, URL lines; line + segment GlotLID (Manding ≥ 0.5, strict); segments of 1–3 sentences, 10–60 words; quality filters of `clean.py`; drop if PII found; standard orthography (≥ 1 of ɛ ɔ ɲ ŋ, no French accents); exact + near dedup; dedup vs `parallel_train` and `data/cpt`; decontam | prepared, **awaiting supervisor approval** |
| `cis-lmu/GlotCC-V1` (`bam-Latn`) | same as HPLT (merged into `mono_bam.jsonl`, HPLT has priority in dedup) | CC0-1.0 (packaging) + Common Crawl terms of use | open | `9ad140b6be` | same as HPLT | prepared, **awaiting supervisor approval** |
| `OpenAssistant/oasst2` (`ready.trees`) | en/fr multi-turn chat threads (`oasst2_threads.jsonl`) | Apache-2.0 | open | `179dd21fc5` | en/fr trees; not deleted/spam/toxic/synthetic; best-ranked usable reply at each step; 2–8 messages, ends with assistant; no code, heavy math, URLs, turns > 800 chars, Open Assistant/LAION identity mentions; decontam | prepared, **awaiting supervisor approval** |
| `HuggingFaceTB/smol-smoltalk` | short en multi-turn chats (`smoltalk_threads.jsonl`) | Apache-2.0 on the card; **but** the rows are synthetic: smol-magpie-ultra and everyday-conversations were generated with Llama-3.1 (405B/70B) Instruct, openhermes with GPT-4 — the Llama 3.1 licence asks that models trained on its outputs carry "Llama" in their name | open | `f73fe857d5` | sub-sources magpie-ultra-short, everyday-conversations, openhermes-50k only; ≥ 2 user turns; every turn < 600 chars; no code, math, URLs, rewriting/formatting/summary/translation requests; first-turn dedup; decontam; cap 60k; rows labelled `synthetic` by source | prepared, **licence question for the supervisor** |
| `databricks/databricks-dolly-15k` | grounded QA (`dolly_grounded.jsonl`) | CC-BY-SA-3.0 (contexts from Wikipedia) | open | `bdd27f4d94` | closed_qa + information_extraction; context non-empty and < 1,200 chars; citation markers removed; dedup; decontam | prepared, **awaiting supervisor approval** |
| `sudoping01/bambara-orthography` | standard → informal spelling map (`orthography_map.json`) for input augmentation | CC-BY-4.0 (data); MIT (code, not used) | open | `c51e340fe3` | reverse of `normalization.csv` (10 generic + 251 lexical rows); `bambara_normalize.py` not run | prepared, **awaiting supervisor approval** |
| `djelia/bambara-mt-dataset` (`clean`) | bam-fr-en pairs (EGAFE books, *La Guerre des Griots de Kita*) | CC-BY-SA-4.0 (card) | gated (auto-approve), **not yet accepted** (data 403) | `8ba79cb953` | normalize; pair dedup; dedup vs `parallel_train` + `data/cpt`; decontam | **blocked: the user must click "Agree" on the dataset page** |
| FLORES-200 public release (`bam_Latn`, `fra_Latn`, `eng_Latn` dev/devtest) | **decontamination index only** | CC-BY-SA-4.0 | open | tarball of 2026-11-09 (sha256 bam dev `82118a2f…`, devtest `af2e337a…`) | – | **approved – eval only** (never in training) |

Not used (licence or contamination problems): `MALIBA-AI/bambara-mt-dataset` (contains MAFAND), oza75 bambara-lm-qa
(no licence, contains FLORES), saillab alpaca bambara (non-commercial), xP3x `bam` (it is FLORES).
