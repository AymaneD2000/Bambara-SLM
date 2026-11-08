"""Fetch approved sources into data/raw/ (immutable, text columns only).

- HF parquet: column-projected range reads via HfFileSystem (audio bytes are never downloaded).
- HF arrow (save_to_disk) shards: downloaded to a temp dir, text columns extracted, audio shard deleted.
- Direct URLs (MADLAD, NLLB, MAFAND): downloaded as-is.
Every output file is listed in data/raw/MANIFEST.json with sha256, source revision and timestamp.
Idempotent: existing outputs are skipped.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from rich.console import Console

console = Console()
RAW = Path("data/raw")
CONVERT = "refs/convert/parquet"


@dataclass
class HFParquet:
    name: str
    repo: str
    globs: list[str]            # relative to repo root (at revision)
    columns: list[str] | None   # None = all columns
    revision: str = "main"


@dataclass
class HFArrow:
    name: str
    repo: str
    prefix: str                 # e.g. "rt-data-collection/"
    columns: list[str]


@dataclass
class URL:
    name: str
    urls: dict[str, str] = field(default_factory=dict)  # filename -> url


SOURCES = [
    HFParquet("bayelemabaga", "RobotsMaliAI/bayelemabaga", ["bam-fr/*/*.parquet"], None, CONVERT),
    HFParquet("afvoices", "RobotsMali/afvoices", ["human-corrected/*.parquet"], ["text", "duration"]),
    HFParquet("afvoices_notag", "RobotsMali/afvoices-notag", ["top20-speakers/*.parquet"], ["text", "participantId", "split"]),
    HFParquet("kunkado", "RobotsMali/kunkado", ["human-reviewed/*.parquet"], ["corrected-label", "duration"]),
    HFParquet("jeli_asr", "RobotsMali/jeli-asr", ["bam-asr-oza/*/*.parquet", "jeli-asr-rmai/*/*.parquet"], ["bam", "french", "duration"], CONVERT),
    HFArrow("bam_asr_all_extra_mali_pense", "RobotsMali/bam-asr-all", "oza-mali-pense/", ["bam", "french", "duration"]),
    HFArrow("bam_asr_all_extra_rt", "RobotsMali/bam-asr-all", "rt-data-collection/", ["bam", "french", "duration"]),
    HFParquet("wikipedia_bm", "wikimedia/wikipedia", ["20231101.bm/*.parquet"], ["id", "url", "title", "text"]),
    HFParquet("belebele", "facebook/belebele", ["data/bam_Latn.jsonl", "data/fra_Latn.jsonl", "data/eng_Latn.jsonl"], None),
    HFParquet("sib200", "Davlan/sib200", ["data/bam_Latn/*.tsv", "data/fra_Latn/*.tsv", "data/eng_Latn/*.tsv"], None),
    HFParquet("flores_plus", "openlanguagedata/flores_plus", ["*/bam_Latn.*", "*/fra_Latn.*", "*/eng_Latn.*"], None),
    URL("madlad_bm", {
        "bm_clean_0000.jsonl.gz": "https://huggingface.co/datasets/allenai/MADLAD-400/resolve/main/data/bm/bm_clean_0000.jsonl.gz",
        "bm_noisy_0000.jsonl.gz": "https://huggingface.co/datasets/allenai/MADLAD-400/resolve/main/data/bm/bm_noisy_0000.jsonl.gz",
    }),
    URL("nllb_mined", {
        "bam_Latn-fra_Latn.gz": "https://storage.googleapis.com/allennlp-data-bucket/nllb/bam_Latn-fra_Latn.gz",
        "bam_Latn-eng_Latn.gz": "https://storage.googleapis.com/allennlp-data-bucket/nllb/bam_Latn-eng_Latn.gz",
    }),
    URL("mafand_fr_bam", {  # TEST ONLY (CC-BY-NC-4.0)
        f"{s}.json": f"https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/fr-bam/{s}.json"
        for s in ("train", "dev", "test")
    }),
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_add(entries: list[dict]) -> None:
    mf = RAW / "MANIFEST.json"
    data = json.loads(mf.read_text()) if mf.exists() else {}
    for e in entries:
        data[e["file"]] = e
    mf.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _in_manifest(path: Path) -> bool:
    mf = RAW / "MANIFEST.json"
    return mf.exists() and str(path) in json.loads(mf.read_text())


def _entry(path: Path, **kw) -> dict:
    return {"file": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **kw}


def _fetch_hf_parquet(s: HFParquet, fs: HfFileSystem, api: HfApi) -> list[dict]:
    rev_sha = api.dataset_info(s.repo, revision=s.revision).sha
    base = f"datasets/{s.repo}@{s.revision.replace('/', '%2F')}"
    out_dir = RAW / s.name
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    files = sorted({f for g in s.globs for f in fs.glob(f"{base}/{g}")})
    if not files:
        raise RuntimeError(f"{s.name}: no files matched {s.globs}")
    for f in files:
        rel = f.split(f"{base}/", 1)[1]
        is_parquet = rel.endswith(".parquet")
        out = out_dir / rel.replace("/", "__")
        meta = dict(source=s.name, repo=s.repo, revision=s.revision,
                    revision_sha=rev_sha, remote=rel, columns=s.columns)
        if out.exists():
            if not _in_manifest(out):
                _manifest_add([_entry(out, **meta)])
            continue
        if is_parquet:
            with fs.open(f, "rb", block_size=4 << 20) as fh:
                table = pq.read_table(fh, columns=s.columns)
            tmp = out.with_suffix(".tmp")
            pq.write_table(table, tmp)
            tmp.rename(out)
        else:
            tmp = out.with_name(out.name + ".part")
            try:
                fs.get(f, str(tmp))
                tmp.rename(out)
            finally:
                tmp.unlink(missing_ok=True)
        entries.append(_entry(out, **meta))
        _manifest_add(entries[-1:])
        console.print(f"  {s.name}: {rel} -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return entries


def _fetch_hf_arrow(s: HFArrow, api: HfApi) -> list[dict]:
    info = api.dataset_info(s.repo)
    out_dir = RAW / s.name
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    shards = sorted(x.rfilename for x in info.siblings
                    if x.rfilename.startswith(s.prefix) and x.rfilename.endswith(".arrow"))
    for rel in shards:
        out = out_dir / (rel.replace("/", "__").removesuffix(".arrow") + ".parquet")
        meta = dict(source=s.name, repo=s.repo, revision_sha=info.sha, remote=rel, columns=s.columns)
        if out.exists():
            if not _in_manifest(out):
                _manifest_add([_entry(out, **meta)])
            continue
        tmpd = Path(tempfile.mkdtemp(prefix="bslm_arrow_"))
        try:
            local = hf_hub_download(s.repo, rel, repo_type="dataset", revision=info.sha, local_dir=tmpd)
            with pa.memory_map(local) as src:
                table = pa.ipc.open_stream(src).read_all().select(s.columns)
            pq.write_table(table, out)
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)  # audio is not kept
        entries.append(_entry(out, **meta))
        _manifest_add(entries[-1:])
        console.print(f"  {s.name}: {rel} -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return entries


def _fetch_urls(s: URL) -> list[dict]:
    out_dir = RAW / s.name
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for fname, url in s.urls.items():
        out = out_dir / fname
        if out.exists():
            if not _in_manifest(out):
                _manifest_add([_entry(out, source=s.name, url=url)])
            continue
        tmp = out.with_suffix(out.suffix + ".part")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.rename(out)
        entries.append(_entry(out, source=s.name, url=url))
        _manifest_add(entries[-1:])
        console.print(f"  {s.name}: {fname} ({out.stat().st_size/1e6:.1f} MB)")
    return entries


def fetch(only: list[str] | None = None) -> dict[str, str]:
    fs, api = HfFileSystem(), HfApi()
    status = {}
    for s in SOURCES:
        if only and s.name not in only:
            continue
        console.print(f"[bold]{s.name}[/bold]")
        try:
            if isinstance(s, HFParquet):
                e = _fetch_hf_parquet(s, fs, api)
            elif isinstance(s, HFArrow):
                e = _fetch_hf_arrow(s, api)
            else:
                e = _fetch_urls(s)
            status[s.name] = f"ok ({len(e)} new files)"
        except Exception as ex:  # report, do not hide
            status[s.name] = f"FAILED: {type(ex).__name__}: {str(ex)[:200]}"
            console.print(f"  [red]{status[s.name]}[/red]")
    return status
