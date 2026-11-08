# Gate 0 – Inventory report (2026-11-04)

Metadata only; no data downloaded yet. Full per-source table: `PROVENANCE.md`. Raw metadata: `reports/gate0_inventory.json`.

## Summary

- **Parallel bam–fr:** Bayelemabaga (46,976 pairs; splits 37,580/4,698/4,698), MAFAND-MT fr-bam (≈3k train / 1.5k dev / 1.5k test, news), NLLB mined (307,529 bam–fra, 180,936 bam–eng; noisy, needs scoring), ASR/TTS corpora with French translation columns (bam-asr-all, jeli-asr, oza75/bambara-tts).
- **Monolingual Bambara:** djelia/bambara-texts (84,328 rows, **no access**), MADLAD-400 bm (≈16 MB gz, mostly the noisy split), bmwiki (0.86 MB bz2: small), spoken transcripts (afvoices human-corrected 253k segments, kunkado 33k human-reviewed, etc.), Bible via Bible Brain (key needed).
- **Evaluation:** FLORES+, Belebele and SIB-200 all have `bam_Latn` and are accessible.
- **Tools:** Qwen3-1.7B-Base is Apache-2.0. GlotLID covers `bam`. bambara-normalizer is on PyPI (MIT).

## Findings to report honestly

1. **Bayelemabaga license conflict.** The dataset card says CC-BY-SA-4.0. The `LICENSE` file in the same repo and the GitHub `robotsmali-ai/datasets` repo say CC-BY-4.0.
2. **MAFAND-MT is CC-BY-NC-4.0** (non-commercial). Using it would make the derived model non-commercial in spirit. The NLLB-200 reference model is also CC-BY-NC, but it is used for evaluation only.
3. **MAFAND test overlap risk:** MAFAND and NLLB mined data could contain news sentences close to FLORES. Contamination control (§8.4) applies.
4. **The speech corpora overlap heavily** (bam-asr-all ⊇ jeli-asr and oza-mali-pense; oza75/bambara-tts reuses jeli_asr/mali_pense; afvoices-notag is likely a subset of afvoices). They will be deduplicated, not double counted.
5. **The monolingual written-text volume is small** without djelia/bambara-texts and the Bible. MADLAD bm clean is only about 1.2 MB gz.
6. Corpus Bambara de Référence publishes **no terms**, so I did not touch it.

## Supervisor decisions (2026-11-04) — Gate 0 APPROVED

- [x] Sources approved (see PROVENANCE.md policy: commercial-use data only for training).
- [x] Bayelemabaga: apply the less restrictive license (CC-BY-4.0).
- [x] MAFAND-MT: test only.
- [x] NLLB mined pairs: included, subject to scoring.
- [ ] Request access to `djelia/bambara-texts` with your HF account.
- [x] Bible: deferred to v2.
- [x] Corpus Bambara de Référence: skipped.
- [ ] Use bambara-normalizer as the normalization starting point?
- [ ] Use `.venv` + `uv` with Python 3.11 as the environment (already set up)?
