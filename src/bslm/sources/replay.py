"""Fetch a small French/English replay sample (brief §9.1) with row-group range reads (no full shard download)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from rich.console import Console

from bslm.pii import scrub
from bslm.text_norm import normalize

console = Console()
OUT = Path("data/raw/replay")
SPECS = {
    # lang: (repo, shard, filter column, min value, docs wanted)
    "fr": ("HuggingFaceFW/fineweb-2", "data/fra_Latn/train/000_00000.parquet", "language_score", 0.9, 6000),
    "en": ("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", "int_score", 3, 4500),
}
MIN_CHARS, MAX_CHARS = 300, 8000


def fetch(seed: int = 13) -> dict:
    fs, api = HfFileSystem(), HfApi()
    OUT.mkdir(parents=True, exist_ok=True)
    rng, stats = random.Random(seed), {}
    for lang, (repo, shard, col, minv, want) in SPECS.items():
        out = OUT / f"replay_{lang}.jsonl"
        if out.exists():
            stats[lang] = "exists"
            continue
        sha = api.dataset_info(repo).sha
        pf = pq.ParquetFile(fs.open(f"datasets/{repo}@{sha}/{shard}", block_size=8 << 20))
        docs, rg = [], 0
        while len(docs) < want * 2 and rg < pf.num_row_groups:  # read row groups until we have 2x the need
            t = pf.read_row_group(rg, columns=["text", "id", "url", col]).to_pylist()
            docs += [d for d in t if (d[col] or 0) >= minv and MIN_CHARS <= len(d["text"]) <= MAX_CHARS]
            rg += 1
        pick = rng.sample(docs, min(want, len(docs)))
        with out.open("w") as f:
            for d in pick:
                text, _ = scrub(normalize(d["text"]))
                f.write(json.dumps({"id": f"replay_{lang}-{d['id']}", "text": text, "lang": lang,
                                    "source": f"replay_{lang}", "license": "ODC-By-1.0",
                                    "register": "web", "synthetic": False, "url_or_ref": d["url"],
                                    "split_group": f"replay_{lang}:{d['id']}", "repo": repo,
                                    "revision_sha": sha, "shard": shard}, ensure_ascii=False) + "\n")
        stats[lang] = f"{len(pick)} docs from {rg} row groups ({len(docs)} passed filters)"
        console.print(f"replay {lang}: {stats[lang]}")
    return stats
