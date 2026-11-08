"""Verify data/raw against MANIFEST.json (checksums) and published row counts. Read-only."""

from __future__ import annotations

import csv
import glob
import gzip
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path("data/raw")

# (label, glob, text column or None, expected rows or None if not published)
PARQUET = [
    ("afvoices train", "afvoices/*train*", "text", 253290),
    ("afvoices test", "afvoices/*test*", "text", 6718),
    ("afvoices_notag train", "afvoices_notag/*train*", "text", 21253),
    ("afvoices_notag test", "afvoices_notag/*test*", "text", 1128),
    ("kunkado train", "kunkado/*train*", "corrected-label", 33282),
    ("kunkado test", "kunkado/*test*", "corrected-label", 5775),
    ("jeli_asr bam-asr-oza train", "jeli_asr/bam-asr-oza__train*", "bam", 30685),
    ("jeli_asr bam-asr-oza test", "jeli_asr/bam-asr-oza__test*", "bam", 1243),
    ("jeli_asr jeli-asr-rmai train", "jeli_asr/jeli-asr-rmai__train*", "bam", 1495),
    ("jeli_asr jeli-asr-rmai test", "jeli_asr/jeli-asr-rmai__test*", "bam", 220),
    ("bam-asr-all oza-mali-pense", "bam_asr_all_extra_mali_pense/*", "bam", 4421),
    ("bam-asr-all rt-data-collection", "bam_asr_all_extra_rt/*", "bam", 705),
    ("bayelemabaga train", "bayelemabaga/*train*", None, 37580),
    ("bayelemabaga valid", "bayelemabaga/*validation*", None, 4698),
    ("bayelemabaga test", "bayelemabaga/*test*", None, 4698),
    ("wikipedia_bm", "wikipedia_bm/*", "text", 1258),
]
GZ = [
    ("nllb bam-fra", "nllb_mined/bam_Latn-fra_Latn.gz", 307529),
    ("nllb bam-eng", "nllb_mined/bam_Latn-eng_Latn.gz", 180936),
    ("madlad bm clean", "madlad_bm/bm_clean_0000.jsonl.gz", None),
    ("madlad bm noisy", "madlad_bm/bm_noisy_0000.jsonl.gz", None),
]
LINES = [
    ("mafand train (eval only)", "mafand_fr_bam/train.json", None),
    ("mafand dev (eval only)", "mafand_fr_bam/dev.json", 1500),
    ("mafand test (eval only)", "mafand_fr_bam/test.json", 1500),
    *[(f"belebele {lg}", f"belebele/data__{lg}.jsonl", 900) for lg in ("bam_Latn", "fra_Latn", "eng_Latn")],
]
SIB = {"train": 701, "dev": 99, "test": 204}
NOT_AVAILABLE = {
    "flores_plus": "gated: supervisor must accept terms on HF",
    "djelia/bambara-texts": "gated: access not granted",
    "oza75/bambara-tts": "gated: request pending (its mali_pense text is covered by bam-asr-all)",
}


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(out_md: str = "reports/raw_verification.md") -> bool:
    man = json.loads((RAW / "MANIFEST.json").read_text())
    files = sorted(p for p in glob.glob(str(RAW / "*/*")) if "/_meta/" not in p)
    partial = [p for p in files if p.endswith((".part", ".tmp"))]
    empty = [p for p in files if Path(p).stat().st_size == 0 and not p.endswith(".gitkeep")]
    files = [p for p in files if p not in partial and not p.endswith(".gitkeep")]
    not_in_man = [p for p in files if p not in man]
    bad_sha = [p for p in files if p in man and _sha(p) != man[p]["sha256"]]
    ghosts = [k for k in man if not Path(k).exists()]

    rows, ok = [], True

    def add(label, n, exp, empty_text=""):
        nonlocal ok
        good = exp is None or n == exp
        ok &= good
        rows.append(f"| {label} | {n:,} | {exp if exp is None else f'{exp:,}'} | {empty_text} | {'✅' if good else '❌'} |")

    for label, pat, col, exp in PARQUET:
        n = e = 0
        fl = glob.glob(str(RAW / pat))
        for f in fl:
            t = pq.read_table(f)
            n += t.num_rows
            if col:
                e += sum(1 for x in t.column(col).to_pylist() if not x or not str(x).strip())
        add(label + ("" if fl else " (NO FILES)"), n, exp, str(e) if col else "")
    for label, rel, exp in GZ:
        with gzip.open(RAW / rel, "rt", errors="replace") as f:
            add(label, sum(1 for _ in f), exp)
    for label, rel, exp in LINES:
        add(label, len((RAW / rel).read_text().splitlines()), exp)
    for split, exp in SIB.items():
        for lg in ("bam_Latn", "fra_Latn", "eng_Latn"):
            with open(RAW / f"sib200/data__{lg}__{split}.tsv", newline="") as f:
                add(f"sib200 {lg} {split}", sum(1 for _ in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)), exp)

    ok &= not (not_in_man or bad_sha or ghosts or partial or empty)
    L = ["# Raw data verification", "",
         f"**Overall: {'PASS' if ok else 'FAIL'}**", "",
         "## Integrity", "",
         f"- files on disk: {len(files)}",
         f"- checksum mismatches vs MANIFEST.json: {len(bad_sha)}",
         f"- files missing from manifest: {len(not_in_man)}",
         f"- manifest entries without a file: {len(ghosts)}",
         f"- partial downloads (.part/.tmp): {len(partial)}",
         f"- empty (0-byte) files: {len(empty)}",
         *[f"  - {p}" for p in (bad_sha + not_in_man + ghosts + partial + empty)[:20]],
         "", "## Row counts vs published", "",
         "| dataset | rows | expected | empty texts | status |", "|---|---|---|---|---|", *rows,
         "", "## Not downloaded (access)", "",
         *[f"- `{k}`: {v}" for k, v in NOT_AVAILABLE.items()], ""]
    Path(out_md).write_text("\n".join(L))
    return ok
