"""Orthography variation report (brief §7.3): measure, do not normalize.

Indicators per source:
  - share of records using reformed letters (ɛ ɔ ɲ ŋ)
  - competing spellings of very frequent words (reformed vs old/French-influenced)
  - tone marks (combining or precomposed grave/acute on vowels)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VARIANTS = {  # reformed form: alternative spellings seen in older or informal writing
    "bɛ": ["be", "bè", "bé"],
    "tɛ": ["te", "tè", "té"],
    "kɛ": ["ke", "kè", "ké"],
    "mɔgɔ": ["mogo", "mɔkɔ", "moko", "mòkò", "mɔgo"],
    "ɲɛ": ["nyɛ", "nye", "nyè", "ɲe"],
    "dɔ": ["do", "dò"],
    "fɔ": ["fo", "fò"],
    "sɔrɔ": ["soro", "sòrò"],
    "ɲɔgɔn": ["nyogon", "ɲogon", "nyɔgɔn"],
}
WORD = re.compile(r"\w+(?:'\w+)?", re.UNICODE)
TONE = re.compile("[\u00e0\u00e1\u00e8\u00e9\u00ec\u00ed\u00f2\u00f3\u00f9\u00fa\u01ce\u011b\u01d0\u01d2\u01d4\u0144\u01f9]|[\u0300\u0301\u030c]")  # tone marks only; ɛ/ɔ alone do not match


def _texts(path: Path, field: str):
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            yield r["source"], r[field]


def run(out_md: str = "reports/orthography_report.md") -> None:
    per_src_words: dict[str, Counter] = defaultdict(Counter)
    per_src_n, per_src_reform, per_src_tone = Counter(), Counter(), Counter()
    feeds = [(Path("data/clean/mono.jsonl"), "text"), (Path("data/clean/parallel.jsonl"), "src_text")]
    for path, field in feeds:
        for src, t in _texts(path, field):
            per_src_n[src] += 1
            if re.search("[ɛɔɲŋƐƆƝŊ]", t):
                per_src_reform[src] += 1
            if TONE.search(t):
                per_src_tone[src] += 1
            per_src_words[src].update(w.lower() for w in WORD.findall(t))
    L = ["# Orthography variation (measured, not normalized)", "",
         "| source | records | % with ɛ/ɔ/ɲ/ŋ | % with tone marks (grave/acute/caron) |", "|---|---|---|---|"]
    for s, n in per_src_n.most_common():
        L.append(f"| {s} | {n:,} | {100*per_src_reform[s]/n:.1f} | {100*per_src_tone[s]/n:.1f} |")
    L += ["", "## Competing spellings of frequent words (share of the reformed form)", "",
          "| source | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for s in per_src_n:
        c = per_src_words[s]
        cells = []
        for ref, alts in VARIANTS.items():
            tot = c[ref] + sum(c[a] for a in alts)
            cells.append(f"{100*c[ref]/tot:.0f}% (n={tot})" if tot else "–")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "Note: `be`/`te`/`do`/`fo` can also be French or other words; read the shares as indicators, not exact rates."]
    Path(out_md).write_text("\n".join(L) + "\n")
