"""`bslm clean`: raw -> interim -> clean, with a full report (brief §7, Gate 1).

Outputs
  data/interim/<source>.jsonl            shaped, unfiltered records
  data/clean/mono.jsonl                  monolingual Bambara (Latin script)
  data/clean/mono_nko.jsonl              N'Ko script, kept apart
  data/clean/parallel.jsonl              bam<->fr/en pairs (Bayelemabaga test excluded)
  data/eval/*.jsonl                      evaluation sets (never trained on)
  data/clean/rejected/<step>.jsonl       every dropped item with its reason
  data/clean/review/langid_borderline.jsonl
  reports/cleaning_report.md, reports/char_inventory.tsv, reports/samples/<source>.txt
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from rich.console import Console

from bslm import langid
from bslm.dedup import boilerplate_paragraphs, exact_dedup, hash_key, near_dedup
from bslm.pii import scrub
from bslm.sources import readers
from bslm.text_norm import normalize, script_of

console = Console()
INTERIM, CLEAN, EVAL = Path("data/interim"), Path("data/clean"), Path("data/eval")
WORD = re.compile(r"\w+", re.UNICODE)
TRANSCRIPT_SOURCES = {"asr_bam_only", "afvoices", "afvoices_notag", "kunkado", "jeli_asr",
                      "bam_asr_all_extra_mali_pense", "bam_asr_all_extra_rt"}
DOC_SOURCES = {"wikipedia_bm", "madlad_bm"}
# Priority for dedup: earlier sources win (human-curated before web).
MONO_PRIORITY = ["wikipedia_bm", "kunkado", "afvoices_notag", "afvoices", "asr_bam_only", "madlad_bm"]
PAR_PRIORITY = ["bayelemabaga", "jeli_asr", "bam_asr_all_extra_mali_pense", "bam_asr_all_extra_rt", "nllb_mined"]


def _write(path: Path, recs) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


class Stats:
    def __init__(self):
        self.steps: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.rejected: dict[str, list[dict]] = defaultdict(list)
        self.pii: Counter = Counter()

    def log(self, source: str, step: str, n: int):
        self.steps[source].append((step, n))

    def reject(self, step: str, rec: dict, reason: str = ""):
        self.rejected[step].append({**rec, "_reason": reason or step})


def _quality_reason(text: str, q: dict) -> str | None:
    if len(text) < q["min_chars"]:
        return "too_short"
    if len(text) > q["max_chars"]:
        return "too_long"
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return "empty"
    letters = sum(c.isalpha() for c in nonspace)
    if letters / len(nonspace) < q["min_letter_ratio"]:
        return "low_letter_ratio"
    if sum(c.isdigit() for c in nonspace) / len(nonspace) > q["max_digit_ratio"]:
        return "mostly_digits"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 4 and 1 - len(set(lines)) / len(lines) > q["max_dup_line_ratio"]:
        return "repeated_lines"
    words = WORD.findall(text.lower())
    if len(words) >= 30:
        tri = Counter(zip(words, words[1:], words[2:]))
        if tri.most_common(1)[0][1] * 3 / len(words) > q["max_top_trigram_ratio"]:
            return "repetitive_ngrams"
    return None


def _transcript(text: str, cfg: dict) -> tuple[str | None, str]:
    """Apply the transcript annotation rules from configs/clean.yaml. Returns (text|None, reason)."""
    for pat in cfg["drop_regex"]:
        if re.search(pat, text):
            return None, f"marker:{pat}"
    for pat in cfg["strip_regex"]:
        text = re.sub(pat, " ", text)
    text = re.sub(cfg["unwrap_regex"], r"\1", text)
    if re.search(cfg["leftover_regex"], text):
        return None, "marker:other_bracket"
    return re.sub(r"\s{2,}", " ", text).strip(), ""


# ---------------------------------------------------------------- mono

def clean_mono(cfg: dict, st: Stats, review: list[dict]) -> tuple[list[dict], list[dict]]:
    ncfg, tcfg, lcfg, q = cfg["normalize"], cfg["transcripts"], cfg["langid"], cfg["quality"]
    manding = lcfg["manding_labels"]

    def _ok(text, pred, strict=False):
        if strict:
            return langid.manding_score(pred, manding) >= lcfg["manding_min"]
        return langid.is_manding(text, pred, manding, lcfg["manding_min"], lcfg["short_words"], lcfg["short_reject_conf"])
    out, nko = [], []
    for src in MONO_PRIORITY:
        recs = list(readers.MONO[src]())
        _write(INTERIM / f"{src}.jsonl", recs)
        st.log(src, "read", len(recs))
        mode = lcfg["mode"].get(src, lcfg["mode"]["default"])

        # 1. normalize + transcript markers
        stage = []
        for r in recs:
            t = normalize(r["text"], ncfg["confusables"])
            if src in TRANSCRIPT_SOURCES:
                t, why = _transcript(t, tcfg)
                if t is None:
                    st.reject("transcript_marker", r, why)
                    continue
                if len(WORD.findall(t)) < tcfg["min_words"]:
                    st.reject("transcript_short", {**r, "text": t})
                    continue
            stage.append({**r, "text": t})
        st.log(src, "normalize+markers", len(stage))

        # 2. documents: boilerplate paragraphs + paragraph-level langid
        if src in DOC_SOURCES:
            kept = []
            for r in stage:
                if len(WORD.findall(r["text"])) < q["min_doc_words"]:
                    st.reject("quality", r, "doc_too_few_words")
                else:
                    kept.append(r)
            stage = kept
            bp = boilerplate_paragraphs([r["text"] for r in stage], cfg["dedup"]["boilerplate_paragraph_min_count"])
            kept = []
            for r in stage:
                paras = [p for p in r["text"].split("\n") if p.strip() and hash_key(p) not in bp]
                if paras:
                    preds = langid.predict(paras)
                    good = [p for p, pr in zip(paras, preds) if _ok(p, pr)]
                else:
                    good = []
                if not good:
                    st.reject("doc_no_bambara_paragraphs", r)
                    continue
                kept.append({**r, "text": "\n".join(good), "_paras_kept": f"{len(good)}/{len(paras)}"})
            stage = kept
            st.log(src, "boilerplate+para_langid", len(stage))

        # 3. script split
        lat = []
        for r in stage:
            (nko if script_of(r["text"]) == "nko" else lat).append({**r, "script": script_of(r["text"])})
        stage = lat

        # 4. record-level langid
        preds = langid.predict([r["text"] for r in stage])
        kept = []
        for r, pr in zip(stage, preds):
            ms = langid.manding_score(pr, manding)
            lab, p = langid.top(pr)
            r = {**r, "langid_top": lab, "langid_p": round(p, 4), "manding_score": round(ms, 4)}
            if not _ok(r["text"], pr, strict=mode == "filter_strict"):
                if mode.startswith("filter"):
                    st.reject("langid", r, f"{lab}:{p:.2f}")
                    continue
                review.append(r)
            kept.append(r)
        stage = kept
        st.log(src, "langid", len(stage))

        # 5. quality filters
        kept = []
        for r in stage:
            why = _quality_reason(r["text"], q) if src not in TRANSCRIPT_SOURCES else (
                "too_short" if len(r["text"]) < q["min_chars"] else None)
            if why:
                st.reject("quality", r, why)
            else:
                kept.append(r)
        stage = kept
        st.log(src, "quality", len(stage))

        # 6. PII
        for r in stage:
            r["text"], c = scrub(r["text"])
            st.pii.update({f"{src}:{k}": v for k, v in c.items()})
        out.extend(stage)

    # 7. dedup across all mono sources (priority order preserved)
    before = Counter(r["source"] for r in out)
    out, drop = exact_dedup(out)
    for r in drop:
        st.reject("dedup_exact", r)
    after_exact = Counter(r["source"] for r in out)
    d = cfg["dedup"]
    out, drop = near_dedup(out, d["minhash_threshold"], d["minhash_num_perm"], d["shingle_chars"])
    for r in drop:
        st.reject("dedup_near", r)
    after_near = Counter(r["source"] for r in out)
    for s in before:
        st.log(s, "dedup_exact", after_exact[s])
        st.log(s, "dedup_near", after_near[s])
    return out, nko


# ---------------------------------------------------------------- parallel

def clean_parallel(cfg: dict, st: Stats, review: list[dict]) -> tuple[list[dict], list[dict]]:
    ncfg, tcfg, lcfg, pcfg = cfg["normalize"], cfg["transcripts"], cfg["langid"], cfg["parallel"]
    manding = lcfg["manding_labels"]
    want = {"fr": "fra_Latn", "en": "eng_Latn"}

    def _ok(text, pred, strict=False):
        if strict:
            return langid.manding_score(pred, manding) >= lcfg["manding_min"]
        return langid.is_manding(text, pred, manding, lcfg["manding_min"], lcfg["short_words"], lcfg["short_reject_conf"])
    out, eval_out = [], []
    for src in PAR_PRIORITY:
        recs = list(readers.PARALLEL[src]())
        _write(INTERIM / f"{src}.jsonl", recs)
        st.log(src, "read", len(recs))
        mode = lcfg["mode"].get(src, lcfg["mode"]["default"])
        stage = []
        for r in recs:
            b = normalize(r["src_text"], ncfg["confusables"])
            o = normalize(r["tgt_text"])  # confusables are Bambara-specific: not applied to fr/en
            if src in TRANSCRIPT_SOURCES:
                b, why = _transcript(b, tcfg)
                if b is None:
                    st.reject("transcript_marker", r, why)
                    continue
                for pat in tcfg["strip_regex"]:
                    o = re.sub(pat, " ", o)
                o = re.sub(tcfg["unwrap_regex"], r"\1", o)
                o = re.sub(r"\s{2,}", " ", o).strip()
            if not b.strip() or not o.strip():
                st.reject("empty_side", r)
                continue
            stage.append({**r, "src_text": b, "tgt_text": o})
        st.log(src, "normalize+markers", len(stage))

        # Bayelemabaga official test split -> eval only
        if src == "bayelemabaga":
            eval_out += [{**r, "eval_set": "bayelemabaga_test"} for r in stage if r["orig_split"] == "test"]
            stage = [r for r in stage if r["orig_split"] != "test"]
            st.log(src, "minus_test_split", len(stage))

        # length ratio
        kept = []
        for r in stage:
            nb, no = len(WORD.findall(r["src_text"])), len(WORD.findall(r["tgt_text"]))
            ratio = max(nb, no) / max(1, min(nb, no))
            if min(nb, no) < pcfg["min_words"] or ratio > pcfg["max_len_ratio"]:
                st.reject("par_length", {**r, "_words": f"{nb}/{no}"}, "too_few_words" if min(nb, no) < pcfg["min_words"] else "len_ratio")
                continue
            kept.append({**r, "len_ratio": round(ratio, 3)})
        stage = kept
        st.log(src, "length_ratio", len(stage))

        # langid on both sides
        pb = langid.predict([r["src_text"] for r in stage])
        po = langid.predict([r["tgt_text"] for r in stage])
        kept = []
        for r, x, y in zip(stage, pb, po):
            ms = langid.manding_score(x, manding)
            op = y.get(want[r["tgt_lang"]], 0.0)
            r = {**r, "manding_score": round(ms, 4), "tgt_lang_p": round(op, 4),
                 "langid_top_src": langid.top(x)[0], "langid_top_tgt": langid.top(y)[0]}
            short_other = len(r["tgt_text"].split()) < lcfg["short_words"]
            other_ok = op >= pcfg["other_lang_min"] or (short_other and langid.top(y)[1] < lcfg["short_reject_conf"])
            bam_ok = _ok(r["src_text"], x, strict=mode == "filter_strict")
            if not bam_ok or not other_ok:
                why = "both_sides" if not (bam_ok or other_ok) else ("bam_side" if not bam_ok else "other_side")
                if bam_ok is False and r["langid_top_src"] == want[r["tgt_lang"]]:
                    why += "_is_" + r["tgt_lang"]
                if mode.startswith("filter"):
                    st.reject("par_langid", r, why)
                    continue
                review.append(r)
            kept.append(r)
        stage = kept
        st.log(src, "langid", len(stage))

        for r in stage:
            r["src_text"], c1 = scrub(r["src_text"])
            r["tgt_text"], c2 = scrub(r["tgt_text"])
            st.pii.update({f"{src}:{k}": v for k, v in (Counter(c1) + Counter(c2)).items()})
        out.extend(stage)

    before = Counter(r["source"] for r in out)
    out, drop = exact_dedup(out, key=lambda r: f"{r['tgt_lang']}\t{r['src_text']}\t{r['tgt_text']}")
    for r in drop:
        st.reject("par_dedup_exact", r)
    after = Counter(r["source"] for r in out)
    for s in before:
        st.log(s, "dedup_exact_pair", after[s])
    return out, eval_out


# ---------------------------------------------------------------- report

def _cps(s: str) -> str:
    return "+".join(f"U+{ord(c):04X}" for c in s)


def _char_inventory(texts) -> list[tuple[str, int, str]]:
    c = Counter()
    for t in texts:
        c.update(t)
    rows = []
    for ch, n in c.most_common():
        name = unicodedata.name(ch, f"U+{ord(ch):04X}")
        rows.append((ch, n, name))
    return rows


STANDARD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZɛɔɲŋƐƆƝŊ0123456789 \n.,;:!?'\"-()")


def report(cfg, st: Stats, mono, nko, par, ev, review):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")

    def ntok(texts):
        n = 0
        batch = []
        for t in texts:
            batch.append(t)
            if len(batch) == 2000:
                n += sum(len(x) for x in tok(batch, add_special_tokens=False)["input_ids"]); batch = []
        if batch:
            n += sum(len(x) for x in tok(batch, add_special_tokens=False)["input_ids"])
        return n

    L = ["# Cleaning report (Gate 1)", "", f"Config: `configs/clean.yaml` (seed {cfg['seed']}).", ""]
    L += ["## Counts per source and step", "", "| source | " + " | ".join(["step → count"]) + " |", "|---|---|"]
    for s, steps in st.steps.items():
        L.append(f"| {s} | " + " → ".join(f"{k}: {v:,}" for k, v in steps) + " |")

    L += ["", "## Rejections by step and reason", "", "| step | reason | count |", "|---|---|---|"]
    for step, recs in st.rejected.items():
        for why, n in Counter(r["_reason"].split(":")[0] for r in recs).most_common(12):
            L.append(f"| {step} | {why} | {n:,} |")

    by_src_mono = defaultdict(list)
    for r in mono:
        by_src_mono[r["source"]].append(r["text"])
    by_src_par = defaultdict(list)
    for r in par:
        by_src_par[(r["source"], r["tgt_lang"])].append(r)
    L += ["", "## Clean output (Qwen3 tokens)", "", "| file | source | records | Bambara tokens | other-side tokens |", "|---|---|---|---|---|"]
    tot_b = 0
    for s, texts in by_src_mono.items():
        n = ntok(texts); tot_b += n
        L.append(f"| mono | {s} | {len(texts):,} | {n:,} | |")
    for (s, lg), recs in by_src_par.items():
        nb, no = ntok(r["src_text"] for r in recs), ntok(r["tgt_text"] for r in recs)
        tot_b += nb
        L.append(f"| parallel | {s} (bam-{lg}) | {len(recs):,} | {nb:,} | {no:,} |")
    L.append(f"| **total** | | | **{tot_b:,}** | |")
    L += ["", f"N'Ko-script records kept apart: {len(nko):,}. Eval records: {len(ev):,} "
          f"({', '.join(f'{k}: {v:,}' for k, v in Counter(r['eval_set'] for r in ev).items())}).",
          f"Langid borderline (kept, sent to review): {len(review):,}.", ""]

    L += ["## PII replacements", "", ", ".join(f"{k}={v}" for k, v in sorted(st.pii.items())) or "none", ""]

    # character inventory over all Bambara text
    inv = _char_inventory([r["text"] for r in mono] + [r["src_text"] for r in par])
    with open("reports/char_inventory.tsv", "w") as f:
        f.write("char\tcodepoint\tcount\tname\tstandard\n")
        for ch, n, name in inv:
            f.write(f"{ch!r}\tU+{ord(ch):04X}\t{n}\t{name}\t{ch in STANDARD}\n")
    rare = [(ch, n, name) for ch, n, name in inv if ch not in STANDARD]
    L += ["## Character inventory (non-basic characters, Bambara side)", "",
          "Full table: `reports/char_inventory.tsv`. Confusables applied: "
          + ", ".join(f"`{a}`→`{b}` ({_cps(a)}→{_cps(b)})" for a, b in cfg["normalize"]["confusables"].items())
          + ". **Pending supervisor decision (not mapped):** "
          + ", ".join(f"`{c}` (U+{ord(c):04X})" for c in cfg["normalize"]["pending_decision"]) + ".", "",
          "| char | codepoint | count | name |", "|---|---|---|---|"]
    for ch, n, name in rare[:60]:
        L.append(f"| `{ch}` | U+{ord(ch):04X} | {n:,} | {name} |")

    # top near-duplicates removed
    L += ["", "## Examples of near-duplicates removed", ""]
    for r in st.rejected.get("dedup_near", [])[:10]:
        L.append(f"- `{r['id']}` dup of `{r['_dup_of']}`: {r['text'][:120]!r}")
    Path("reports/cleaning_report.md").write_text("\n".join(L) + "\n")

    # samples
    rng = random.Random(cfg["seed"])
    sd = Path("reports/samples"); sd.mkdir(parents=True, exist_ok=True)
    for s, texts in by_src_mono.items():
        pick = rng.sample(texts, min(cfg["samples_per_source"], len(texts)))
        (sd / f"mono_{s}.txt").write_text("\n\n---\n\n".join(t[:600] for t in pick) + "\n")
    for (s, lg), recs in by_src_par.items():
        pick = rng.sample(recs, min(cfg["samples_per_source"], len(recs)))
        (sd / f"parallel_{s}_{lg}.txt").write_text("\n\n".join(
            f"[bam] {r['src_text']}\n[{lg}]  {r['tgt_text']}" for r in pick) + "\n")


KIKUYU_CHARS = re.compile("[\u0169\u0129\u0168\u0128]")  # ũ ĩ: used by Kikuyu, never by Bambara


def eval_sets_checked(st: Stats | None = None) -> list[dict]:
    """Eval sets from readers.eval_sets(), normalized, minus items whose Bambara side is not Bambara.
    Upstream defect: FLORES-200 bam_Latn (used by Belebele and SIB-200) contains some Kikuyu sentences."""
    recs = [{**r, "src_text": normalize(r["src_text"]), "tgt_text": normalize(r["tgt_text"])} for r in readers.eval_sets()]
    preds = langid.predict([r["src_text"] for r in recs])
    keep = []
    for r, p in zip(recs, preds):
        lab, prob = langid.top(p)
        if KIKUYU_CHARS.search(r["src_text"]) or (lab == "kik_Latn" and prob >= 0.5):
            if st is not None:
                st.reject("eval_not_bambara", r, f"{r['eval_set']}:{lab}")
            continue
        keep.append(r)
    return keep


def write_eval_sets() -> dict:
    """Regenerate data/eval files that come from readers.eval_sets() (Bayelemabaga test is written by run())."""
    st = Stats()
    ev = eval_sets_checked(st)
    for name in sorted({r["eval_set"] for r in ev}):
        _write(EVAL / f"{name}.jsonl", (r for r in ev if r["eval_set"] == name))
    _write(CLEAN / "rejected" / "eval_not_bambara.jsonl", st.rejected["eval_not_bambara"])
    return dict(Counter(r["_reason"].split(":")[0] for r in st.rejected["eval_not_bambara"]))


def run(config: str = "configs/clean.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    random.seed(cfg["seed"])
    st, review = Stats(), []
    console.print("[bold]mono[/bold]")
    mono, nko = clean_mono(cfg, st, review)
    console.print("[bold]parallel[/bold]")
    par, ev = clean_parallel(cfg, st, review)
    console.print("[bold]eval sets[/bold]")
    ev += eval_sets_checked(st)

    _write(CLEAN / "mono.jsonl", mono)
    _write(CLEAN / "mono_nko.jsonl", nko)
    _write(CLEAN / "parallel.jsonl", par)
    for name in sorted({r["eval_set"] for r in ev}):
        _write(EVAL / f"{name}.jsonl", (r for r in ev if r["eval_set"] == name))
    for step, recs in st.rejected.items():
        _write(CLEAN / "rejected" / f"{step}.jsonl", recs)
    _write(CLEAN / "review" / "langid_borderline.jsonl", review)
    console.print("[bold]report[/bold]")
    report(cfg, st, mono, nko, par, ev, review)
    from bslm.ortho_report import run as ortho
    ortho()
    console.print("wrote reports/cleaning_report.md, reports/orthography_report.md")
