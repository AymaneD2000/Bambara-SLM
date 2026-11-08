"""Turn the supervisor's manual inspection marks into per-source alignment thresholds (brief §8.2).

Reads reports/alignment/inspect/<group>__<bucket>.tsv (column `ok`: y / n / ?), computes the share of correct pairs
per score bucket (with a 95% Wilson interval), and proposes, for each group, the lowest bucket boundary such that
every bucket at or above it reaches the target precision. `?` (partial) counts as half correct.
The estimate of the kept set's precision weights each bucket by its real size in parallel_scored.decontam.jsonl.
"""

from __future__ import annotations

import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

INSPECT = "reports/alignment/inspect"
BUCKETS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _group_of(r: dict) -> str:
    if r["source"] == "bam_asr_all_extra_mali_pense":
        return "jeli_asr_fr"
    return f"{r['source']}_{r['tgt_lang']}"


def _bucket(s: float) -> str:
    return BUCKETS[min(int(s / 0.2), 4)]


def run(target: float = 0.8, out_md: str = "reports/alignment/inspection_results.md") -> dict:
    marks: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for f in sorted(glob.glob(f"{INSPECT}/*.tsv")):
        group, bucket = Path(f).stem.split("__")
        with open(f, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                m = (row.get("ok") or "").strip().lower()
                if m in ("y", "n", "?"):
                    marks[group][bucket][m] += 1

    sizes: dict[str, Counter] = defaultdict(Counter)
    for line in open("data/clean/parallel_scored.decontam.jsonl"):
        r = json.loads(line)
        sizes[_group_of(r)][_bucket(r["align_score"])] += 1

    L = ["# Manual inspection results", "",
         f"`y` = 1, `?` = 0.5, `n` = 0. Target precision for a kept bucket: **{target:.0%}**. "
         "Interval = 95% Wilson. Buckets with fewer than 10 marks are shown but not trusted.", ""]
    proposal = {}
    for group in sorted(marks):
        L += [f"## {group}", "", "| bucket | marked | y | ? | n | precision | 95% interval | pairs in data |",
              "|---|---|---|---|---|---|---|---|"]
        prec = {}
        for b in BUCKETS:
            c = marks[group][b]
            n = sum(c.values())
            p = (c["y"] + 0.5 * c["?"]) / n if n else float("nan")
            lo, hi = _wilson(p, n) if n else (float("nan"), float("nan"))
            prec[b] = (p, n)
            L.append(f"| {b} | {n} | {c['y']} | {c['?']} | {c['n']} | "
                     + (f"{p:.0%} | {lo:.0%}–{hi:.0%}" if n else "– | –") + f" | {sizes[group][b]:,} |")
        # lowest boundary such that all buckets from it upward reach the target (and have >= 10 marks)
        thr = None
        for i, b in enumerate(BUCKETS):
            above = [prec[x] for x in BUCKETS[i:]]
            if all(n >= 10 and p >= target for p, n in above):
                thr = float(b.split("-")[0])
                break
        kept = [b for b in BUCKETS if thr is not None and float(b.split("-")[0]) >= thr]
        n_kept = sum(sizes[group][b] for b in kept)
        est = (sum(prec[b][0] * sizes[group][b] for b in kept) / n_kept) if n_kept else float("nan")
        proposal[group] = {"threshold": thr, "pairs_kept": n_kept, "est_precision": est}
        L += ["", ("**Proposal:** " + (f"keep score ≥ {thr:.1f} → {n_kept:,} pairs, estimated {est:.0%} correct."
                                         if thr is not None else
                                         f"no score range reaches {target:.0%} → **drop this source** (or lower the target)."))
              , ""]
    Path(out_md).write_text("\n".join(L) + "\n")
    return proposal
