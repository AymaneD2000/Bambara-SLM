"""v3 SFT data sources (permissive licences only). One module per source; outputs go to data/interim/v3/.

Run each with `PYTHONPATH=src .venv/bin/python -m bslm.v3data.<module>`:
  hplt (+ glotcc)  -> mono_bam.jsonl          monolingual Bambara segments, back-translation targets
  oasst2           -> oasst2_threads.jsonl    en/fr conversation threads
  smoltalk         -> smoltalk_threads.jsonl  short everyday multi-turn chats
  dolly            -> dolly_grounded.jsonl    closed_qa / information_extraction with short context
  orthography      -> orthography_map.json    standard -> informal Bambara spellings
  djelia           -> djelia_mt.jsonl         bam-fr-en pairs (gated)

Shared helpers here: pinned revisions, download, the evaluation index (bslm.decontam.EvalIndex over data/eval/
plus FLORES-200 dev+devtest), and an index of the existing training text (exact + near duplicate).
Downloaded data is untrusted: it is only parsed (parquet/jsonl/csv), never executed; no trust_remote_code.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from datasketch import MinHashLSH

from bslm import decontam
from bslm.decontam import EvalIndex, sentences
from bslm.dedup import hash_key, norm_key

RAW, OUT = Path("data/raw"), Path("data/interim/v3")

# Every HF download is pinned to these commits (checked 2026-11-09 with HfApi.dataset_info).
REVISIONS = {
    "HPLT/HPLT2.0_cleaned": "d1324a5283f762ee62c2a5c81de08fc6450ea540",
    "cis-lmu/GlotCC-V1": "9ad140b6be3ac7b539606a2b4809b49d122823de",
    "OpenAssistant/oasst2": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
    "HuggingFaceTB/smol-smoltalk": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
    "databricks/databricks-dolly-15k": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
    "sudoping01/bambara-orthography": "c51e340fe3fe609603c0bbab889a05f59c978b56",
    "djelia/bambara-mt-dataset": "8ba79cb953f4d889ed0f22b381976302bdbb5195",
}
# FLORES-200 public release (CC-BY-SA-4.0), used ONLY to extend the decontamination index, because
# openlanguagedata/flores_plus is still gated for this account. Only bam/fra/eng dev+devtest files were extracted.
FLORES_DIR = RAW / "flores200/flores200_dataset"


def _load_env() -> None:
    """Put HF_TOKEN from .env into the environment if it is not set (huggingface_hub falls back to the cached login)."""
    p = Path(".env")
    if "HF_TOKEN" in os.environ or not p.exists():
        return
    for line in p.read_text().splitlines():
        if line.startswith("HF_TOKEN=") and line.split("=", 1)[1].strip():
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def download(repo: str, filename: str, subdir: str) -> Path:
    from huggingface_hub import hf_hub_download
    _load_env()
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo],
                                local_dir=str(RAW / subdir)))


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def eval_index() -> EvalIndex:
    """decontam.EvalIndex over data/eval/ (Bayelemabaga test, MAFAND train/dev/test, Belebele, SIB-200, human
    prompts), extended with all FLORES-200 dev+devtest sentences (bam, fra, eng) using the same thresholds."""
    idx = EvalIndex()
    n = len(idx.near_owner)
    for f in sorted(glob.glob(str(FLORES_DIR / "*/*_Latn.*"))):
        es = "flores200_" + Path(f).suffix.lstrip(".")
        for line in open(f):
            for s in sentences(line.strip()):
                nw = len(norm_key(s).split())
                if nw >= decontam.MIN_EXACT:
                    idx.exact.setdefault(hash_key(s), es)
                if nw >= decontam.MIN_NEAR:
                    key = f"f{n}"
                    idx.lsh.insert(key, decontam._mh(s))
                    idx.near_owner[key] = es
                    n += 1
    print(f"eval index + FLORES-200: {len(idx.exact):,} exact keys, {len(idx.near_owner):,} near-dup sentences")
    return idx


def eval_hit(idx: EvalIndex, text: str):
    """First eval match for the text or any of its sentences, or None."""
    for s in [text] + sentences(text):
        if (h := idx.match(s)):
            return h
    return None


def train_files() -> list[str]:
    fs = ["data/train/parallel_train.jsonl"] + sorted(glob.glob("data/cpt/*/*.jsonl"))
    return [f for f in fs if Path(f).exists()]


def iter_train_texts(fields=("src_text", "tgt_text", "text")):
    for f in train_files():
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                for k in fields:
                    if r.get(k):
                        yield r[k]


class TrainMatcher:
    """Marks candidate sentences that already occur in the training text.

    The candidates (small) go into an LSH index; the training text (large) is streamed once and queried, so memory
    stays small. Exact = same normalized sentence (>= decontam.MIN_EXACT words); near = MinHash Jaccard >= NEAR_T on
    char 5-grams (>= MIN_NEAR words), the same settings as decontam.py.
    """

    def __init__(self, candidates: list[str]):
        self.cand = candidates
        self.exact = {}
        self.lsh = MinHashLSH(threshold=decontam.NEAR_T, num_perm=decontam.NUM_PERM)
        for i, s in enumerate(candidates):
            nw = len(norm_key(s).split())
            if nw >= decontam.MIN_EXACT:
                self.exact.setdefault(hash_key(s), []).append(i)
            if nw >= decontam.MIN_NEAR:
                self.lsh.insert(i, decontam._mh(s))
        self.hit: dict[int, str] = {}

    def scan(self, texts) -> dict[int, str]:
        n = 0
        for t in texts:
            for s in sentences(t):
                n += 1
                nw = len(norm_key(s).split())
                if nw >= decontam.MIN_EXACT and (ids := self.exact.get(hash_key(s))):
                    for i in ids:
                        self.hit.setdefault(i, "exact")
                if nw >= decontam.MIN_NEAR:
                    for i in self.lsh.query(decontam._mh(s)):
                        self.hit.setdefault(i, "near")
        print(f"train scan: {n:,} training sentences, {len(self.hit):,} candidate sentences matched")
        return self.hit
