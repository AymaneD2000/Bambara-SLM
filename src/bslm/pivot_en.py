"""Synthetic Bambara-English pairs by pivoting through French (brief §5.4 / §9.2, approved by the user 2026-11-08).

The French side of clean bam-fr pairs (Bayelemabaga + Jeli-ASR with align_score >= min_align_score;
MAFAND is excluded: CC-BY-NC, test-only per PROVENANCE.md)
is machine-translated into English with a Marian fr->en model. Every output row carries synthetic=true and the
generator name, keeps its original split (train/val), and is never used for evaluation.

Resumable: rows already in the output file are skipped.
Usage: python -m bslm.pivot_en [--limit N] [--device mps|cuda|cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GENERATOR = "Helsinki-NLP/opus-mt-fr-en"
OUT = Path("data/interim/pivot_en.jsonl")
INPUTS = [  # (file, split)
    ("data/train/parallel_train.jsonl", "train"), ("data/train/parallel_val.jsonl", "val"),
]


def rows(min_align: float = 0.4):
    for fn, split in INPUTS:
        for line in open(fn):
            r = json.loads(line)
            if r["tgt_lang"] != "fr" or not r["tgt_text"].strip():
                continue
            if (r.get("align_score") or 0) < min_align:
                continue
            yield split, r


def main(argv=None) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    a = ap.parse_args(argv)
    done = {json.loads(l)["id"] for l in OUT.open()} if OUT.exists() else set()
    todo = [(s, r) for s, r in rows() if f"pivot-{r['id']}" not in done][: a.limit]
    print(f"pivot fr->en: {len(done)} done, {len(todo)} to translate", flush=True)
    tok = AutoTokenizer.from_pretrained(GENERATOR)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR).to(a.device).eval()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    todo.sort(key=lambda x: len(x[1]["tgt_text"]))  # length-sorted batches = less padding
    with OUT.open("a") as f, torch.no_grad():
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            enc = tok([r["tgt_text"] for _, r in chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(a.device)
            out = model.generate(**enc, num_beams=2, max_new_tokens=256)
            for (split, r), en in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
                f.write(json.dumps({"id": f"pivot-{r['id']}", "split": split, "src_lang": "bam", "tgt_lang": "en",
                                    "src_text": r["src_text"], "tgt_text": en.strip(), "pivot_fr": r["tgt_text"],
                                    "source": f"pivot:{r['source']}", "register": r.get("register"),
                                    "synthetic": True, "generator": GENERATOR}, ensure_ascii=False) + "\n")
            if (i // a.batch) % 50 == 0:
                f.flush()
                print(f"  {i + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
