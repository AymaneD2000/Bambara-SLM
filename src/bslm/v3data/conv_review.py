"""Claude-written v3 Bambara conversations: automatic checks, review sheets, and applying the review.

  python -m bslm.v3data.conv_review qc       -> data/interim/v3/conv_all.jsonl (+ auto_flags), stats on stdout
  python -m bslm.v3data.conv_review sheet    -> reports/review/v3_conv_<category>.csv (200 random rows per category)
  python -m bslm.v3data.conv_review apply    -> data/interim/v3/conv_accepted.jsonl

Review policy (user decision 2026-11-09): the owner reviews a 200-row sample per category. A category whose
acceptance rate (ok + fix) in the sample is below 80% is dropped entirely; otherwise every unreviewed row of the
category is kept, reviewed rows follow their decision. `fix` rows carry corrections as lines "A<n>: <new text>"
(n = assistant turn number, 1-based, as numbered in the sheet).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("data/interim/qa_raw/v3conv")
ALL = Path("data/interim/v3/conv_all.jsonl")
OUT = Path("data/interim/v3/conv_accepted.jsonl")
SHEETS = Path("reports/review")
DECISION = "decision (ok / fix / no)"
MIN_ACCEPT = 0.80
GENERATOR = "claude-opus-5-5"

# capitalised Bambara function words in mid-sentence (owner's most frequent correction)
_MIDCAP = re.compile(r"(?<=[a-zɛɔɲŋ,;] )(Bɛ|Tɛ|Ka|Ye|Ni|Ma|Kɛ|Don|Ko|Fɔ|Taa|Na|Sɔrɔ|Kɔni|Fana|Dɔrɔn)\b")
_BA_SE = re.compile(r"^\s*N (ba|se)\b[,!]?", re.I)


def _turns(r: dict) -> list[tuple[str, str]]:
    return [tuple(t) for t in (r.get("history") or [])] + [(r["user"], r["assistant"])]


def _lang(texts: list[str]) -> list[str]:
    from bslm import langid
    out = []
    for p in langid.predict(texts):
        top = max(p, key=p.get) if p else "?"
        out.append({"fra_Latn": "fr", "eng_Latn": "en"}.get(top, "bam" if top.startswith("bam") or top in ("dyu_Latn", "mnk_Latn") else top))
    return out


def qc() -> None:
    rows, seen, st = [], set(), Counter()
    for fn in sorted(RAW.glob("*.jsonl")):
        for line in open(fn):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                st["bad_json"] += 1
                continue
            if not all(k in r for k in ("id", "category", "user", "assistant")):
                st["missing_keys"] += 1
                continue
            key = tuple(x.strip().lower() for t in _turns(r) for x in t)
            if key in seen:
                st["duplicate"] += 1
                continue
            seen.add(key)
            rows.append(r)
    finals = _lang([r["assistant"] for r in rows])
    for r, lf in zip(rows, finals):
        flags = []
        for i, (u, a) in enumerate(_turns(r), 1):
            if not a.strip():
                flags.append(f"A{i}_empty")
            if _BA_SE.match(a):
                flags.append(f"A{i}_N_ba_se")
            if _MIDCAP.search(a):
                flags.append(f"A{i}_midcap")
            if re.search(r"\bduru\b", a):
                flags.append(f"A{i}_duru")
        want = r.get("lang_final")
        # language hint only (quoted translations are legitimate): skip `translate`, ignore short-text noise labels
        if want and lf != want and lf in ("bam", "fr", "en") and r["category"] != "translate":
            flags.append(f"lang_final_{lf}_vs_{want}")
        r["auto_flags"] = ";".join(flags)
        st["flagged" if flags else "clean"] += 1
    ALL.parent.mkdir(parents=True, exist_ok=True)
    with ALL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_cat = Counter(r["category"] for r in rows)
    nturns = Counter(len(_turns(r)) for r in rows)
    flag_kinds = Counter(re.sub(r"^A\d+_", "", x) for r in rows for x in r["auto_flags"].split(";") if x)
    print(json.dumps({"rows": len(rows), **st, "by_category": by_cat, "turns": dict(sorted(nturns.items())),
                      "flag_kinds": flag_kinds}, ensure_ascii=False, indent=1))


def sheet(n: int = 200, seed: int = 13) -> None:
    rows = [json.loads(l) for l in open(ALL)]
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    rng = random.Random(seed)
    SHEETS.mkdir(parents=True, exist_ok=True)
    for cat, rs in sorted(by.items()):
        pick = rng.sample(rs, min(n, len(rs)))
        path = SHEETS / f"v3_conv_{cat}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "category", "conversation", "note_fr", "auto_flags", DECISION, "corrected (A<n>: text)", "comment"])
            for r in pick:
                conv = "\n".join(f"U{i}: {u}\nA{i}: {a}" for i, (u, a) in enumerate(_turns(r), 1))
                w.writerow([r["id"], cat, conv, r.get("note_fr", ""), r.get("auto_flags", ""), "", "", ""])
        print(f"{cat}: {len(pick)} of {len(rs)} -> {path}")


def apply() -> None:
    rows = {r["id"]: r for r in map(json.loads, open(ALL))}
    decided, stats = {}, defaultdict(Counter)
    for path in sorted(SHEETS.glob("v3_conv_*.csv")):
        for s in csv.DictReader(path.open(encoding="utf-8-sig")):
            d = (s[DECISION] or "").strip().lower()
            stats[s["category"]][d or "pending"] += 1
            if d in ("ok", "fix", "no"):
                decided[s["id"]] = (d, s["corrected (A<n>: text)"])
    keep_cat = {}
    for cat, c in stats.items():
        done = c["ok"] + c["fix"] + c["no"]
        rate = (c["ok"] + c["fix"]) / done if done else None
        keep_cat[cat] = rate is not None and rate >= MIN_ACCEPT and c["pending"] == 0
        print(f"{cat}: {dict(c)} acceptance={rate if rate is None else round(rate, 3)} -> {'KEEP' if keep_cat[cat] else 'DROP/PENDING'}")
    n = Counter()
    with OUT.open("w") as f:
        for rid, r in rows.items():
            if not keep_cat.get(r["category"]):
                continue
            d, corr = decided.get(rid, ("unreviewed", ""))
            if d == "no" or (d == "fix" and not corr.strip()):
                n["dropped"] += 1
                continue
            turns = [list(t) for t in _turns(r)]
            for m in re.finditer(r"^A(\d+):\s*(.+?)\s*$", corr, re.M):
                i = int(m.group(1)) - 1
                if 0 <= i < len(turns):
                    turns[i][1] = m.group(2)
            f.write(json.dumps({"id": rid, "task": "chat", "category": r["category"], "history": turns[:-1] or None,
                                "user": turns[-1][0], "assistant": turns[-1][1], "synthetic": True, "generator": GENERATOR,
                                "review": "accepted", "review_decision": d}, ensure_ascii=False) + "\n")
            n[d] += 1
    print(f"written {OUT}: {dict(n)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qc", "sheet", "apply"])
    a = ap.parse_args(argv)
    {"qc": qc, "sheet": sheet, "apply": apply}[a.cmd]()


if __name__ == "__main__":
    main()
