"""Freeze the native-speaker prompt set (brief §5.3): validate, write data/eval/human_prompts.jsonl, hash it."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from rich.console import Console

from bslm.text_norm import normalize

console = Console()
CATEGORIES = {"translate_bam_fr", "translate_fr_bam", "translate_bam_en", "translate_en_bam", "qa", "summarize", "generate"}
PLACEHOLDERS = ("[À ÉCRIRE", "[A ECRIRE", "[WRITE IN")
OUT = Path("data/eval/human_prompts.jsonl")
SHA = Path("prompts/human_prompts.FROZEN.sha256")


def _training_overlap(recs: list[dict]) -> list[tuple[str, str, str]]:
    """Return (prompt id, match kind, training source) for prompts/references found in the training data."""
    from bslm.decontam import EvalIndex, sentences
    with tempfile.TemporaryDirectory() as d:
        with open(Path(d) / "p.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps({"eval_set": r["id"], "text": r["text"], "tgt_text": r["tgt_text"]},
                                   ensure_ascii=False) + "\n")
        idx = EvalIndex(d)
    hits = []
    for fn, fields in (("data/train/mono_train.jsonl", ("text",)),
                       ("data/train/parallel_train.jsonl", ("src_text", "tgt_text"))):
        if not Path(fn).exists():
            continue
        for line in open(fn):
            r = json.loads(line)
            for fld in fields:
                for s in sentences(r[fld]):
                    if (m := idx.match(s)):
                        hits.append((m[1], m[0], r["source"]))
    return hits


def freeze(csv_path: str = "prompts/human_prompts_draft.csv", min_rows: int = 50, force: bool = False) -> bool:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig", newline="")))
    errors, recs = [], []
    approved = [r for r in rows if (r.get("approved") or "").strip().lower() in ("y", "yes", "o", "oui", "ɔwɔ")]
    ids = Counter(r["id"].strip() for r in approved)
    for r in approved:
        rid, cat, prompt = r["id"].strip(), r["category"].strip(), (r["prompt"] or "").strip()
        if ids[rid] > 1:
            errors.append(f"{rid}: duplicate id")
        if cat not in CATEGORIES:
            errors.append(f"{rid}: unknown category '{cat}'")
        if not prompt:
            errors.append(f"{rid}: empty prompt")
        if any(p in prompt.upper() or p in prompt for p in PLACEHOLDERS) or any(p in (r.get("reference") or "") for p in PLACEHOLDERS):
            errors.append(f"{rid}: placeholder text still present")
        recs.append({"id": rid, "eval_set": "human_prompts", "category": cat,
                     "instruction_lang": r.get("instruction_lang", "").strip(), "domain": r.get("domain", "").strip(),
                     "text": normalize(prompt), "tgt_text": normalize(r.get("reference") or ""),
                     "notes": r.get("notes", "").strip()})
    dup_prompts = [k for k, v in Counter(x["text"] for x in recs).items() if v > 1]
    errors += [f"duplicate prompt text: {p[:60]}" for p in dup_prompts]

    console.print(f"rows: {len(rows)}, approved: {len(approved)}; by category: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(Counter(x['category'] for x in recs).items())))
    if len(approved) < min_rows:
        errors.append(f"only {len(approved)} approved rows (minimum {min_rows})")
    missing_ref = [x["id"] for x in recs if x["category"] != "generate" and not x["tgt_text"]]
    if missing_ref:
        console.print(f"[yellow]warning: {len(missing_ref)} non-generation prompts have no reference: {missing_ref[:10]}[/yellow]")
    if errors:
        for e in errors:
            console.print(f"[red]error:[/red] {e}")
        if not force:
            console.print("[red]not frozen[/red]")
            return False

    if SHA.exists() and OUT.exists() and not force:
        console.print(f"[red]already frozen ({SHA}); refusing to overwrite. Use --force only with supervisor approval.[/red]")
        return False

    hits = _training_overlap(recs)
    if hits:
        console.print(f"[yellow]warning: {len(hits)} training sentences match prompts/references "
                      f"(they will be removed from training by `bslm decontam`): "
                      f"{Counter(h[0] for h in hits).most_common(10)}[/yellow]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for x in sorted(recs, key=lambda x: x["id"]):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    SHA.write_text(f"{digest}  {OUT}\n")
    console.print(f"[green]frozen {len(recs)} prompts -> {OUT} (sha256 {digest[:16]}…)[/green]")
    console.print("Next: re-run `bslm decontam` and `bslm splits` so training excludes any matching text.")
    return True


def check_frozen() -> bool:
    """True if data/eval/human_prompts.jsonl still matches its frozen hash."""
    if not (SHA.exists() and OUT.exists()):
        return False
    return SHA.read_text().split()[0] == hashlib.sha256(OUT.read_bytes()).hexdigest()
