"""sudoping01/bambara-orthography (data: CC-BY-4.0; code: MIT) -> data/interim/v3/orthography_map.json.

The dataset maps spellings "seen in the wild" to the standard orthography (normalization.csv: 10 generic rules,
251 lexical rows). We build the reverse map, standard -> informal spellings, for input-side augmentation:
  char_rules  generic rules (from the dataset, plus the French-style rules the dataset does not list, marked
              origin "project")
  words       standard word -> informal forms (dataset lexical rows; place/first names kept apart in `names`)
Only normalization.csv and examples.jsonl are parsed. bambara_normalize.py / test_normalize.py are never run.

The map is compared with bslm.build_sft.informal_spelling (imported read-only): for every lexical pair we check
whether any informal_spelling style (many seeds) can produce the wild form.
"""

from __future__ import annotations

import csv
import json
import random
import unicodedata
from collections import defaultdict

from bslm.v3data import OUT, REVISIONS, download

REPO, SUBDIR = "sudoping01/bambara-orthography", "bambara_orthography"

# Reverse of the dataset's generic rules (standard -> informal). context: where the rule applies.
CHAR_RULES = [
    {"standard": "ɛ", "informal": ["è", "e"], "context": "any", "origin": "dataset (è→ɛ; bare e only lexically)"},
    {"standard": "ɔ", "informal": ["ò", "o"], "context": "any", "origin": "dataset (ò→ɔ; bare o only lexically)"},
    {"standard": "ɲ", "informal": ["ny"], "context": "any", "origin": "dataset (pre-1982 digraph)"},
    {"standard": "ŋ", "informal": ["ng"], "context": "word-initial", "origin": "dataset (pre-1982 digraph)"},
    {"standard": "c", "informal": ["ty"], "context": "word-initial", "origin": "dataset (pre-1982 orthography)"},
    {"standard": "j", "informal": ["dy"], "context": "word-initial", "origin": "dataset (pre-1982 orthography)"},
    {"standard": "'", "informal": ["’", "‘", "ʼ", "`"], "context": "elision", "origin": "dataset"},
    {"standard": "k'a", "informal": ["k' a", "ka a"], "context": "elision", "origin": "dataset (k' a spaced); "
     "ka a = unelided form"},
    {"standard": "(no tone)", "informal": ["à á ǎ â on vowels"], "context": "optional tone marks",
     "origin": "dataset (tone marks removed by the normalizer)"},
    # French-style habits not in the dataset (common in user messages; already in build_sft._INFORMAL)
    {"standard": "ɛ", "informal": ["é", "ê"], "context": "any", "origin": "project"},
    {"standard": "ɔ", "informal": ["ô"], "context": "any", "origin": "project"},
    {"standard": "ɲ", "informal": ["gn"], "context": "any", "origin": "project"},
    {"standard": "j", "informal": ["dj"], "context": "word-initial or after a vowel", "origin": "project"},
    {"standard": "c", "informal": ["tch", "ti"], "context": "before a vowel", "origin": "project"},
    {"standard": "u", "informal": ["ou"], "context": "any", "origin": "project"},
    {"standard": "g", "informal": ["gu"], "context": "before i/e", "origin": "project"},
]


def _strip_tones(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))


def run(seeds: int = 400) -> dict:
    for fn in ("README.md", "LICENSE", "normalization.csv", "examples.jsonl"):
        download(REPO, fn, SUBDIR)
    rows = list(csv.DictReader(open(f"data/raw/{SUBDIR}/normalization.csv", encoding="utf-8")))
    words, names = defaultdict(set), defaultdict(set)
    for r in rows:
        if r["rule"] != "lexical":
            continue
        target = names if "spelling of" in r["note"] else words
        target[r["standard"].strip()].add(r["wild"].strip())
    examples = [json.loads(x) for x in open(f"data/raw/{SUBDIR}/examples.jsonl", encoding="utf-8")]

    # coverage by build_sft.informal_spelling (read-only import)
    from bslm.build_sft import informal_spelling
    reach: dict[str, set[str]] = {}
    for std in list(words) + list(names):
        outs = set()
        for s in range(seeds):
            outs.add(informal_spelling(std, random.Random(s)))
            outs.add(informal_spelling(std, random.Random(s)).lower())
        reach[std] = outs
    covered, new = [], []
    for d in (words, names):
        for std, wild in d.items():
            for w in sorted(wild):
                (covered if w in reach[std] or w.lower() in {x.lower() for x in reach[std]} else new).append(
                    {"standard": std, "informal": w})

    def pattern(p):
        s, w = p["standard"].lower(), p["informal"].lower()
        if s in names or p["standard"] in names:
            return "name (French spelling)"
        if "ò" in w:
            return "ò for ɔ"
        if w.startswith("dy") or w.startswith("ty"):
            return "dy/ty word-initial"
        if _strip_tones(w) != w:
            return "accent pattern"
        return "other"
    new_by_pattern = defaultdict(list)
    for p in new:
        new_by_pattern[pattern(p)].append(f"{p['standard']}→{p['informal']}")

    out = {
        "source": REPO, "revision": REVISIONS[REPO], "license": "CC-BY-4.0 (data), MIT (code, not used)",
        "direction": "standard -> informal (reverse of normalization.csv)",
        "char_rules": CHAR_RULES,
        "words": {k: sorted(v) for k, v in sorted(words.items())},
        "names": {k: sorted(v) for k, v in sorted(names.items())},
        "examples_informal_inputs": sorted({e["input"] for e in examples if e["input"] != e["expected"]}),
        "comparison_with_build_sft_informal_spelling": {
            "lexical_pairs": len(covered) + len(new),
            "reachable_by_informal_spelling": len(covered),
            "not_reachable": len(new),
            "not_reachable_by_pattern": {k: v for k, v in sorted(new_by_pattern.items())},
        },
    }
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "orthography_map.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    c = out["comparison_with_build_sft_informal_spelling"]
    print({k: v for k, v in c.items() if k != "not_reachable_by_pattern"},
          {k: (len(v), v[:6]) for k, v in c["not_reachable_by_pattern"].items()})
    print(len(out["words"]), "words,", len(out["names"]), "names")
    return out


if __name__ == "__main__":
    run()
