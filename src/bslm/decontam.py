"""Contamination control (brief §8.4): remove training text that matches evaluation text.

Index: every sentence (both sides) of every file in data/eval/ (Bayelemabaga test, MAFAND, Belebele, SIB-200,
and FLORES when available).
  - exact match of normalized sentences with >= MIN_EXACT words
  - near-duplicate (MinHash, char 5-grams, Jaccard >= NEAR_T) for sentences with >= MIN_NEAR words
Mono documents: matching sentences are removed, the rest of the document is kept.
Parallel pairs: the pair is removed if either side (or any sentence in it) matches.
Removed items are written to data/clean/rejected/contamination_*.jsonl with the matched eval set.
"""

from __future__ import annotations

import glob
import json
import re
from collections import Counter
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from rich.console import Console

from bslm.dedup import hash_key, norm_key

console = Console()
SENT = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_EXACT, MIN_NEAR, NEAR_T, NUM_PERM, K = 4, 6, 0.7, 64, 5


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.split(text) if s and s.strip()]


def _mh(s: str) -> MinHash:
    k = norm_key(s)
    m = MinHash(num_perm=NUM_PERM, seed=7)
    m.update_batch([k[i:i + K].encode() for i in range(max(1, len(k) - K + 1))])
    return m


class EvalIndex:
    def __init__(self, eval_dir: str = "data/eval"):
        self.exact: dict[str, str] = {}
        self.lsh = MinHashLSH(threshold=NEAR_T, num_perm=NUM_PERM)
        self.near_owner: dict[str, str] = {}
        n = 0
        for f in sorted(glob.glob(f"{eval_dir}/*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                for side in ("src_text", "tgt_text", "text"):
                    for s in sentences(r.get(side) or ""):
                        nw = len(norm_key(s).split())
                        if nw >= MIN_EXACT:
                            self.exact.setdefault(hash_key(s), r["eval_set"])
                        if nw >= MIN_NEAR:
                            key = f"e{n}"
                            self.lsh.insert(key, _mh(s))
                            self.near_owner[key] = r["eval_set"]
                            n += 1
        console.print(f"eval index: {len(self.exact):,} exact keys, {n:,} near-dup sentences")

    def match(self, s: str) -> tuple[str, str] | None:
        nw = len(norm_key(s).split())
        if nw >= MIN_EXACT and (es := self.exact.get(hash_key(s))):
            return "exact", es
        if nw >= MIN_NEAR:
            hits = self.lsh.query(_mh(s))
            if hits:
                return "near", self.near_owner[hits[0]]
        return None


def _write(p: Path, recs) -> None:
    with p.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(mono_in="data/clean/mono.jsonl", par_in="data/clean/parallel_scored.jsonl") -> dict:
    idx = EvalIndex()
    stats = Counter()
    # mono: remove matching sentences, keep the rest of the document
    mono_out, mono_rej = [], []
    for line in open(mono_in):
        r = json.loads(line)
        sents = sentences(r["text"])
        hits = [(s, idx.match(s)) for s in sents]
        bad = [(s, h) for s, h in hits if h]
        if not bad:
            mono_out.append(r)
            continue
        for s, (kind, es) in bad:
            stats[("mono", r["source"], kind, es)] += 1
            mono_rej.append({"id": r["id"], "source": r["source"], "sentence": s, "match": kind, "eval_set": es})
        keep_text = r["text"]
        for s, _ in bad:
            keep_text = keep_text.replace(s, " ")
        keep_text = re.sub(r"[ \t]{2,}", " ", keep_text).strip()
        if len(keep_text.split()) >= 4:
            mono_out.append({**r, "text": keep_text, "decontam_removed_sentences": len(bad)})
    # parallel: drop the pair if any side matches
    par_out, par_rej = [], []
    for line in open(par_in):
        r = json.loads(line)
        hit = None
        for side in ("src_text", "tgt_text"):
            for s in [r[side]] + sentences(r[side]):
                if (hit := idx.match(s)):
                    break
            if hit:
                break
        if hit:
            stats[("parallel", r["source"], hit[0], hit[1])] += 1
            par_rej.append({**r, "_match": hit[0], "_eval_set": hit[1]})
        else:
            par_out.append(r)
    _write(Path("data/clean/mono.decontam.jsonl"), mono_out)
    _write(Path("data/clean/parallel_scored.decontam.jsonl"), par_out)
    _write(Path("data/clean/rejected/contamination_mono_sentences.jsonl"), mono_rej)
    _write(Path("data/clean/rejected/contamination_parallel.jsonl"), par_rej)

    L = ["# Contamination report (brief §8.4)", "",
         f"Eval index: exact match for sentences ≥ {MIN_EXACT} words; MinHash near-duplicate (Jaccard ≥ {NEAR_T}) "
         f"for sentences ≥ {MIN_NEAR} words. Eval sets: " + ", ".join(sorted({Path(f).stem for f in glob.glob('data/eval/*.jsonl')})) + ".", "",
         f"- mono: {len(mono_rej):,} sentences removed; documents kept {len(mono_out):,}",
         f"- parallel: {len(par_rej):,} pairs removed; kept {len(par_out):,}", "",
         "| data | source | match | eval set | count |", "|---|---|---|---|---|"]
    for (d, s, kind, es), n in sorted(stats.items(), key=lambda x: -x[1]):
        L.append(f"| {d} | {s} | {kind} | {es} | {n:,} |")
    L += ["", "FLORES+ is not yet in the index (access pending); re-run this step when it is added."]
    Path("reports/contamination_report.md").write_text("\n".join(L) + "\n")
    console.print("wrote reports/contamination_report.md")
    return dict(stats)
