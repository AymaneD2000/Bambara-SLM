"""HPLT/HPLT2.0_cleaned bam_Latn (+ GlotCC-V1 bam-Latn) -> data/interim/v3/mono_bam.jsonl.

Monolingual Bambara segments (1-3 sentences, 10-60 words) used as targets for back-translation.
Pipeline (counts after each step go to data/interim/v3/mono_bam.stats.json):
  1. normalize (text_norm.normalize + the confusables of configs/clean.yaml); drop N'Ko-script documents;
     drop documents from DROP_DOMAINS (copyrighted religious publishers / Bible text, and machine-translated spam)
  2. remove boilerplate lines (repeated in >= 3 documents) and lines with URLs/e-mails
  3. line-level GlotLID, strict Manding rule (as for madlad_bm/nllb_mined: score >= manding_min at any length)
  4. segment into 1-3 sentences with 10-60 words (sentences longer than 60 words are dropped)
  5. segment-level filters: clean._quality_reason, strict langid again, PII (drop if pii.scrub finds anything)
  6. standard orthography: >= 1 of ɛ ɔ ɲ ŋ and no French-style accented vowel (è é ê ô ò ...): BT targets must be standard
  7. exact + near dedup inside the set (dedup.py settings of configs/clean.yaml), HPLT first
  8. dedup against training text (data/train/parallel_train.jsonl, data/cpt/*/*.jsonl): any sentence matching
  9. decontamination against eval sets (data/eval/* + FLORES-200 dev/devtest): any sentence matching
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter

import pyarrow.parquet as pq
import yaml

from bslm import langid
from bslm.clean import _quality_reason
from bslm.decontam import sentences
from bslm.dedup import boilerplate_paragraphs, exact_dedup, hash_key, near_dedup
from bslm.pii import scrub
from bslm.text_norm import normalize, script_of
from bslm.v3data import OUT, TrainMatcher, download, eval_index, iter_train_texts, write_jsonl
from bslm.v3data import glotcc

REPO, FILE, SUBDIR = "HPLT/HPLT2.0_cleaned", "bam_Latn/train-00000-of-00001.parquet", "hplt2"
URL = re.compile(r"https?://|www\.|\.(com|org|net|fr|ml)\b|@\w", re.I)
STD = re.compile(r"[ɛɔɲŋƐƆƝŊ]")
FR_ACCENT = re.compile(r"[èéêëòóôöàâùûçîï]", re.I)
MIN_W, MAX_W, MAX_S = 10, 60, 3
LIST_NUM = re.compile(r"^\(?\d{1,3}[.)]\s+")
# Inspected 2026-11-09 on the first pass (38% of segments came from jw.org, 8% from bible.is):
#   jw.org (Watch Tower, all rights reserved; religious register), bible.is (Bible text: Bible data is handled
#   separately, under its own licence), chatgpt4.win / aivideo-app.com / zhitov.ru (machine-translated pages).
DROP_DOMAINS = {"jw.org", "bible.is", "chatgpt4.win", "aivideo-app.com", "zhitov.ru"}


def _domain(url: str) -> str:
    from urllib.parse import urlparse
    h = urlparse(url or "").netloc.lower()
    return h[4:] if h.startswith("www.") else h


def read_hplt():
    download(REPO, "README.md", SUBDIR)
    f = pq.ParquetFile(download(REPO, FILE, SUBDIR))
    for g in range(f.num_row_groups):
        for r in f.read_row_group(g, columns=["id", "text", "u"]).to_pylist():
            yield {"doc_id": r["id"], "text": r["text"] or "", "url": r["u"], "source": "hplt2_cleaned"}


def segments(para: str) -> list[str]:
    out, buf = [], []
    for s in sentences(para):
        if len(s.split()) > MAX_W:
            buf = []
            continue
        buf.append(s)
        while len(buf) > MAX_S or sum(len(x.split()) for x in buf) > MAX_W:
            buf.pop(0)
        if sum(len(x.split()) for x in buf) >= MIN_W:
            out.append(" ".join(buf))
            buf = []
    return out


def run(config: str = "configs/clean.yaml", seed: int = 13) -> dict:
    cfg = yaml.safe_load(open(config))
    conf, lc, q = cfg["normalize"]["confusables"], cfg["langid"], cfg["quality"]
    manding, ms_min = lc["manding_labels"], lc["manding_min"]
    stats: dict[str, Counter] = {}

    def log(step, rows, key="source"):
        stats[step] = Counter(r[key] for r in rows)
        print(f"{step:32s} {dict(stats[step])}")

    docs = list(read_hplt()) + list(glotcc.read())
    log("1_read_docs", docs)
    docs = [{**d, "text": normalize(d["text"], conf)} for d in docs]
    docs = [d for d in docs if script_of(d["text"]) != "nko" and d["text"].strip()]
    log("1_normalize_latin_docs", docs)
    docs = [d for d in docs if not any(_domain(d["url"]) == x or _domain(d["url"]).endswith("." + x)
                                       for x in DROP_DOMAINS)]
    log("1b_domain_filter", docs)

    # 2-3: lines
    bp = boilerplate_paragraphs([d["text"] for d in docs], cfg["dedup"]["boilerplate_paragraph_min_count"])
    lines = []
    for d in docs:
        for j, ln in enumerate(d["text"].split("\n")):
            if ln.strip() and hash_key(ln) not in bp and not URL.search(ln):
                lines.append({"source": d["source"], "doc_id": d["doc_id"], "line": j,
                              "text": LIST_NUM.sub("", ln.strip())})
    log("2_lines_after_boilerplate_url", lines)
    preds = langid.predict([x["text"] for x in lines])
    lines = [x for x, p in zip(lines, preds) if langid.manding_score(p, manding) >= ms_min]
    log("3_lines_langid_manding", lines)

    # 4: segments
    segs = []
    for x in lines:
        for k, s in enumerate(segments(x["text"])):
            segs.append({"source": x["source"], "doc_id": x["doc_id"], "text": s,
                         "id": f"{x['source']}-{x['doc_id']}-{x['line']}-{k}"})
    log("4_segments_1-3sent_10-60w", segs)

    # 5: quality, langid, PII
    segs = [s for s in segs if _quality_reason(s["text"], q) is None and not URL.search(s["text"])]
    log("5a_quality", segs)
    preds = langid.predict([s["text"] for s in segs])
    segs = [s for s, p in zip(segs, preds) if langid.manding_score(p, manding) >= ms_min]
    log("5b_segment_langid", segs)
    segs = [s for s in segs if not scrub(s["text"])[1]]
    log("5c_pii", segs)

    # 6: standard orthography
    segs = [s for s in segs if STD.search(s["text"]) and not FR_ACCENT.search(s["text"])]
    log("6_standard_orthography", segs)

    # 7: internal dedup (HPLT first: list order)
    segs, _ = exact_dedup(segs)
    log("7a_dedup_exact", segs)
    d = cfg["dedup"]
    segs, _ = near_dedup(segs, d["minhash_threshold"], d["minhash_num_perm"], d["shingle_chars"])
    log("7b_dedup_near", segs)

    # 8: against training text (any sentence of the segment)
    sent_of = [(i, s) for i, r in enumerate(segs) for s in sentences(r["text"])]
    tm = TrainMatcher([s for _, s in sent_of])
    hit = tm.scan(iter_train_texts())
    bad = {sent_of[j][0] for j in hit}
    segs = [r for i, r in enumerate(segs) if i not in bad]
    log("8_dedup_vs_train", segs)

    # 9: eval sets
    idx = eval_index()
    from bslm.v3data import eval_hit
    rej = Counter()
    keep = []
    for r in segs:
        h = eval_hit(idx, r["text"])
        if h:
            rej[f"{h[0]}:{h[1]}"] += 1
        else:
            keep.append(r)
    segs = keep
    log("9_decontam_eval", segs)
    print("eval matches removed:", dict(rej))

    out = [{"id": r["id"], "text": r["text"], "source": r["source"]} for r in segs]
    n = write_jsonl(OUT / "mono_bam.jsonl", out)
    st = {k: dict(v) for k, v in stats.items()} | {"eval_matches_removed": dict(rej), "written": n}
    (OUT / "mono_bam.stats.json").write_text(json.dumps(st, indent=1, ensure_ascii=False))
    for r in random.Random(seed).sample(out, min(5, len(out))):
        print("SAMPLE", json.dumps(r, ensure_ascii=False))
    return st


if __name__ == "__main__":
    run()
