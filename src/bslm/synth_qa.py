"""Synthetic Bambara QA / summary / chat data with human review (brief §5.4; user decision 2026-11-08).

Generator: Claude (claude-opus-5-5), grounded on native Bambara passages from Bayelemabaga *train* (with their French
translation so the content is understood exactly). Every row is synthetic=true with the generator name, and is only
used for training after a native speaker accepts it.

  python -m bslm.synth_qa passages --n 160        -> data/interim/qa_passages.jsonl (grouped consecutive sentences)
  (generation)                                    -> data/interim/qa_raw/*.jsonl
  python -m bslm.synth_qa sheet                   -> automatic filters + reports/review/qa_review.csv (for the reviewer)
  python -m bslm.synth_qa apply                   -> reviewed CSV -> data/interim/qa_synthetic.jsonl (sft_v2 `tasks`)

Raw row format: {"id", "task": "qa"|"summary"|"chat", "passage_id"|null, "history": [[user, assistant], ...]|null,
                 "user", "assistant", "note_fr"}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

GENERATOR = "claude-opus-5-5"
PASSAGES = Path("data/interim/qa_passages.jsonl")
RAW_DIR = Path("data/interim/qa_raw")
SHEET = Path("reports/review/qa_review.csv")
OUT = Path("data/interim/qa_synthetic.jsonl")
_SPECIAL = re.compile(r"[ɛɔɲŋƐƆƝŊ]")
_NUM = re.compile(r"(\d+)$")
_PASSAGE_ROW = re.compile(r"^b\d+-(.+-\d+)-\d+$")     # b<batch>-<source>-<split>-<num>-<k> -> <source>-<split>-<num>


def split_key(r: dict) -> str:
    """Train/val split unit: the passage, so questions/summaries on one passage never straddle train and val.
    Uses `passage_id` when the row has one, else derives it from the id (the review CSV has no passage_id column;
    the derived form equals the raw passage_id, e.g. b2-bayelemabaga-train-0014779-3 -> bayelemabaga-train-0014779).
    Chat rows (c<batch>-<n>) fall back to their own id: each one is an independent conversation."""
    if r.get("passage_id"):
        return r["passage_id"]
    m = _PASSAGE_ROW.match(r["id"])
    return m.group(1) if m else r["id"]


def passages(n: int, min_sent: int = 3, max_sent: int = 6, max_chars: int = 700) -> None:
    rows = [json.loads(l) for l in open("data/train/parallel_train.jsonl")]
    rows = [r for r in rows if r["source"] == "bayelemabaga" and r["tgt_lang"] == "fr"]
    blocks, cur, last = [], [], None
    for r in rows:                                   # consecutive ids = same running text
        k = int(_NUM.search(r["id"]).group(1))
        if last is not None and k != last + 1:
            blocks.append(cur); cur = []
        cur.append(r); last = k
    blocks.append(cur)
    cands = []
    for b in blocks:
        for i in range(0, len(b) - min_sent + 1, max_sent):
            chunk = b[i:i + max_sent]
            bam = " ".join(x["src_text"] for x in chunk)
            if len(chunk) < min_sent or len(bam) > max_chars or len(bam) < 180:
                continue
            special = len(_SPECIAL.findall(bam)) / max(len(bam), 1)
            score = sum((x.get("align_score") or 0) for x in chunk) / len(chunk)
            if len(re.findall(r"\d+\)", bam)) >= 2:     # numbered proverb lists: no running text
                continue
            if special < 0.01 or score < 0.6:            # standard orthography + well-aligned only
                continue
            cands.append({"passage_id": chunk[0]["id"], "bam": bam, "fr": " ".join(x["tgt_text"] for x in chunk),
                          "n_sent": len(chunk)})
    step = max(1, len(cands) // n)                     # spread over the whole corpus (many different texts)
    picked = cands[::step][:n]
    PASSAGES.parent.mkdir(parents=True, exist_ok=True)
    with PASSAGES.open("w") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(cands)} candidate passages, {len(picked)} written to {PASSAGES}")


def _auto_checks(r: dict, pas: dict | None) -> list[str]:
    flags = []
    if not r.get("user", "").strip() or not r.get("assistant", "").strip():
        flags.append("empty")
    if len(r.get("assistant", "")) > 1200:
        flags.append("too_long")
    if r["task"] in ("qa", "summary") and pas is None:
        flags.append("no_passage")
    if r["task"] == "qa" and pas is not None:          # grounded: most answer words should appear in the passage
        words = [w for w in re.findall(r"\w+", r["assistant"].lower()) if len(w) > 2]
        overlap = sum(w in pas["bam"].lower() for w in words) / max(len(words), 1)
        if overlap < 0.5:
            flags.append(f"low_overlap_{overlap:.2f}")
    return flags


def sheet(pattern: str = "*.jsonl", out: Path = SHEET) -> None:
    pas = {p["passage_id"]: p for p in map(json.loads, open(PASSAGES))} if PASSAGES.exists() else {}
    rows, seen = [], set()
    for fn in sorted(RAW_DIR.glob(pattern)):
        for line in open(fn):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r.get("user", "").strip().lower(), r.get("assistant", "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            r["auto_flags"] = ";".join(_auto_checks(r, pas.get(r.get("passage_id"))))
            rows.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "task", "history", "user", "assistant", "note_fr", "auto_flags", "passage_bam", "passage_fr",
                    "decision (ok / fix / no)", "corrected_assistant", "comment"])
        for r in rows:
            p = pas.get(r.get("passage_id")) or {}
            hist = "\n".join(f"USER: {u}\nASSISTANT: {a}" for u, a in r.get("history") or [])
            w.writerow([r["id"], r["task"], hist, r["user"], r["assistant"], r.get("note_fr", ""), r["auto_flags"],
                        p.get("bam", ""), p.get("fr", ""), "", "", ""])
    flagged = sum(bool(r["auto_flags"]) for r in rows)
    print(f"{len(rows)} rows -> {out} ({flagged} with automatic flags)")


def apply() -> None:
    n = {"ok": 0, "fix": 0, "no": 0, "pending": 0, "fix_without_answer": 0}
    with OUT.open("w") as f:
        for r in (r for sh in sorted(SHEET.parent.glob("qa_review*.csv")) for r in csv.DictReader(sh.open())):
            d = (r["decision (ok / fix / no)"] or "").strip().lower()
            if d == "fix" and not r["corrected_assistant"].strip():   # flagged wrong but no correction: keep out
                n["fix_without_answer"] += 1
                continue
            if d in ("ok", "fix"):
                ans = r["corrected_assistant"].strip() if d == "fix" else r["assistant"]
                split = "val" if int(hashlib.md5(split_key(r).encode()).hexdigest(), 16) % 20 == 0 else "train"
                hist = [[t.split("\nASSISTANT: ")[0].removeprefix("USER: "), t.split("\nASSISTANT: ")[1]]
                        for t in re.split(r"\n(?=USER: )", r["history"])] if r.get("history", "").strip() else None
                f.write(json.dumps({"id": r["id"], "split": split, "task": r["task"], "user": r["user"], "history": hist,
                                    "assistant": ans, "synthetic": True, "generator": GENERATOR,
                                    "review": "accepted", "review_decision": d}, ensure_ascii=False) + "\n")
            n[d if d in n else "pending"] += 1
    print(f"review applied: {n} -> {OUT}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["passages", "sheet", "apply"])
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--out", default=str(SHEET))
    a = ap.parse_args(argv)
    {"passages": lambda: passages(a.n), "sheet": lambda: sheet(a.pattern, Path(a.out)), "apply": apply}[a.cmd]()


if __name__ == "__main__":
    main()
