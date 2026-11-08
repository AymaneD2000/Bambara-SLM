"""Tokenizer audit (brief §4): tokens-per-word of the Qwen3 tokenizer on Bambara vs French vs English.

Uses Belebele FLORES passages (same content in all three languages) so the comparison is parallel.
Decision rule: if bam tokens/word > 2.5 x fra tokens/word, propose vocabulary extension.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B-Base"
WORD_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
SPECIAL_CHARS = ["ɛ", "ɔ", "ɲ", "ŋ", "Ɛ", "Ɔ", "Ɲ", "Ŋ", "à", "á", "è", "é", "ì", "í", "ò", "ó", "ù", "ú", "ʃ"]


def _passages(lang: str) -> list[str]:
    seen, out = set(), []
    for line in Path(f"data/raw/belebele/data__{lang}.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = r["link"]
        if key not in seen:
            seen.add(key)
            out.append(unicodedata.normalize("NFC", r["flores_passage"]))
    return out


def audit(out_md: str = "reports/tokenizer_audit.md", extra_bam: list[str] | None = None) -> dict:
    tok = AutoTokenizer.from_pretrained(MODEL)
    res = {}
    for lang in ["bam_Latn", "fra_Latn", "eng_Latn"]:
        texts = _passages(lang)
        n_tok = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
        n_word = sum(len(WORD_RE.findall(t)) for t in texts)
        n_char = sum(len(t) for t in texts)
        res[lang] = {"passages": len(texts), "tokens": n_tok, "words": n_word,
                     "tok_per_word": n_tok / n_word, "chars_per_tok": n_char / n_tok}
    ratio = res["bam_Latn"]["tok_per_word"] / res["fra_Latn"]["tok_per_word"]

    # Per-character cost: how many tokens a special letter takes when isolated.
    char_cost = {c: len(tok(c, add_special_tokens=False)["input_ids"]) for c in SPECIAL_CHARS}
    char_cost_nfd = {c: len(tok(unicodedata.normalize("NFD", c), add_special_tokens=False)["input_ids"])
                     for c in SPECIAL_CHARS if unicodedata.normalize("NFD", c) != c}

    # Worst Bambara words by tokens (frequent words only).
    words = Counter(w for t in _passages("bam_Latn") + (extra_bam or []) for w in WORD_RE.findall(t.lower()))
    costs = [(w, len(tok(" " + w, add_special_tokens=False)["input_ids"]), n) for w, n in words.items() if n >= 5]
    worst = sorted(costs, key=lambda x: (-x[1], -x[2]))[:25]
    seg = {w: tok.convert_ids_to_tokens(tok(" " + w, add_special_tokens=False)["input_ids"]) for w, _, _ in worst[:10]}

    nko = "ߒ ߓߊ߯ߙߊ߲ ߞߊ߲"  # sample N'Ko string
    nko_cost = len(tok(nko, add_special_tokens=False)["input_ids"]) / len(nko.split())

    decision = "PROPOSE vocabulary extension" if ratio > 2.5 else "KEEP original tokenizer"
    lines = [
        "# Tokenizer audit – Qwen3-1.7B-Base", "",
        "Data: Belebele FLORES passages (parallel content), NFC-normalized.", "",
        "| lang | passages | words | tokens | tokens/word | chars/token |", "|---|---|---|---|---|---|",
    ]
    for k, v in res.items():
        lines.append(f"| {k} | {v['passages']} | {v['words']} | {v['tokens']} | {v['tok_per_word']:.2f} | {v['chars_per_tok']:.2f} |")
    lines += [
        "", f"**bam / fra tokens-per-word ratio: {ratio:.2f}** (threshold 2.5) → **{decision}** (supervisor approves).",
        f"bam / eng ratio: {res['bam_Latn']['tok_per_word'] / res['eng_Latn']['tok_per_word']:.2f}", "",
        "## Special characters (tokens when isolated, NFC)", "",
        " ".join(f"`{c}`={n}" for c, n in char_cost.items()), "",
        "NFD (decomposed tone marks) cost: " + " ".join(f"`{c}`={n}" for c, n in char_cost_nfd.items()), "",
        f"N'Ko sample tokens/word: {nko_cost:.2f}", "",
        "## Most expensive frequent Bambara words (count ≥ 5)", "",
        "| word | tokens | count |", "|---|---|---|",
        *[f"| {w} | {n} | {c} |" for w, n, c in worst], "",
        "## Segmentation examples", "",
        *[f"- `{w}` → {' | '.join(s)}" for w, s in seg.items()],
    ]
    Path(out_md).write_text("\n".join(lines) + "\n")
    out = {"results": res, "ratio_bam_fra": ratio, "decision": decision, "char_cost": char_cost}
    Path(out_md).with_suffix(".json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out
