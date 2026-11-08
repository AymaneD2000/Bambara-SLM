"""Gate 0 inventory: query the Hugging Face Hub for each candidate source.

Read-only: fetches repo metadata (card data, file list, sizes, gating, revision).
Never downloads gated content and never bypasses a gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from rich.console import Console
from rich.table import Table

console = Console()


def _check(api: HfApi, repo_id: str, repo_type: str) -> dict:
    rec: dict = {"id": repo_id, "repo_type": repo_type}
    try:
        info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    except RepositoryNotFoundError:
        rec["status"] = "not_found_or_private"
        return rec
    except Exception as e:  # network or other
        rec["status"] = f"error: {type(e).__name__}: {e}"[:300]
        return rec
    card = getattr(info, "card_data", None) or {}
    card = card.to_dict() if hasattr(card, "to_dict") else dict(card)
    siblings = info.siblings or []
    rec.update(
        status="ok",
        sha=info.sha,
        gated=getattr(info, "gated", False),
        private=getattr(info, "private", False),
        license=card.get("license"),
        languages=card.get("language"),
        configs=[c.get("config_name") for c in card.get("configs", []) or []],
        dataset_info=card.get("dataset_info"),
        n_files=len(siblings),
        total_bytes=sum((s.size or 0) for s in siblings),
        files_sample=[s.rfilename for s in siblings][:40],
        last_modified=str(getattr(info, "last_modified", "")),
    )
    # Can the current token read files? (only metadata probe, no data download)
    if rec["gated"]:
        try:
            api.list_repo_tree(repo_id, repo_type=repo_type, recursive=False)
            readme = api.hf_hub_download  # noqa: F841 (not called)
            rec["access"] = "metadata-visible (gated: download needs approved request)"
        except GatedRepoError:
            rec["access"] = "gated: no access"
    return rec


def run_inventory(config: str, out: str) -> list[dict]:
    cfg = yaml.safe_load(Path(config).read_text())
    api = HfApi()
    results = []
    for item in cfg.get("hf_datasets", []):
        r = _check(api, item["id"], "dataset")
        r.update(role=item.get("role"), note=item.get("note"))
        results.append(r)
    for item in cfg.get("hf_models", []):
        r = _check(api, item["id"], "model")
        r.update(role=item.get("role"), note=item.get("note"))
        results.append(r)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False))

    t = Table(title="Gate 0 inventory (HF Hub)")
    for col in ["id", "status", "license", "gated", "size MB", "configs"]:
        t.add_column(col)
    for r in results:
        t.add_row(
            r["id"], r["status"], str(r.get("license")), str(r.get("gated")),
            f"{r.get('total_bytes', 0) / 1e6:.1f}", str(len(r.get("configs") or [])),
        )
    console.print(t)
    console.print(f"wrote {out}")
    return results
