"""Train/validation splits by document group, never by random line (brief §8.5).

Mono:     original test splits of afvoices / afvoices-notag / kunkado / ASR -> val (no speaker or recording
          leakage); Wikipedia and MADLAD: ~2% of documents (hash of split_group) -> val.
Parallel: Bayelemabaga official validation split -> val; jeli-asr original test split -> val;
          NLLB: ~1% of website domains (split_group) -> val.
Parallel pairs below the alignment threshold go to data/train/parallel_below_threshold.jsonl (kept, not deleted).
The threshold is PROVISIONAL until the supervisor finishes the manual inspection.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

OUT = Path("data/train")


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 1000


def mono_split(r: dict) -> str:
    if r.get("orig_split") == "test":
        return "val"
    if r["source"] in ("wikipedia_bm", "madlad_bm"):
        return "val" if _h(r["split_group"]) < 20 else "train"
    return "train"


def par_split(r: dict) -> str:
    if r["source"] == "bayelemabaga":
        return "val" if r.get("orig_split") == "validation" else "train"
    if r.get("orig_split") == "test":
        return "val"
    if r["source"] == "nllb_mined":
        return "val" if _h(r["split_group"]) < 10 else "train"
    return "train"


def run(config: str = "configs/align.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    thr = cfg["threshold"]
    import re
    excl = [re.compile(x) for x in cfg.get("exclude_url_regex", [])]
    OUT.mkdir(parents=True, exist_ok=True)
    cpt_only = cfg.get("cpt_only_threshold", {})
    files = {k: (OUT / f"{k}.jsonl").open("w") for k in
             ("mono_train", "mono_val", "parallel_train", "parallel_val", "parallel_below_threshold",
              "parallel_cpt_only_train", "parallel_cpt_only_val")}
    c = Counter()
    for line in open("data/clean/mono.decontam.jsonl"):
        r = json.loads(line)
        sp = mono_split(r)
        r["split"] = sp
        files[f"mono_{sp}"].write(json.dumps(r, ensure_ascii=False) + "\n")
        c[("mono", r["source"], sp)] += 1
    for line in open("data/clean/parallel_scored.decontam.jsonl"):
        r = json.loads(line)
        t = thr.get(r["source"], thr["default"]) if isinstance(thr, dict) else thr
        sp = par_split(r)
        r["split"] = sp
        if any(x.search(r.get("url_or_ref") or "") for x in excl):
            r["excluded_by"] = "exclude_url_regex"
            files["parallel_below_threshold"].write(json.dumps(r, ensure_ascii=False) + "\n")
            c[("parallel", f"{r['source']}-{r['tgt_lang']}", "excluded_url")] += 1
            continue
        if r["align_score"] < t and r["source"] in cpt_only and r["align_score"] >= cpt_only[r["source"]]:
            r["use"] = "cpt_only"
            files[f"parallel_cpt_only_{sp}"].write(json.dumps(r, ensure_ascii=False) + "\n")
            c[("parallel", f"{r['source']}-{r['tgt_lang']}", f"cpt_only_{sp}")] += 1
            continue
        if r["align_score"] < t:
            files["parallel_below_threshold"].write(json.dumps(r, ensure_ascii=False) + "\n")
            c[("parallel", f"{r['source']}-{r['tgt_lang']}", "below_threshold")] += 1
            continue
        files[f"parallel_{sp}"].write(json.dumps(r, ensure_ascii=False) + "\n")
        c[("parallel", f"{r['source']}-{r['tgt_lang']}", sp)] += 1
    for f in files.values():
        f.close()
    L = ["# Splits (brief §8.5)", "", f"Alignment threshold (PROVISIONAL, config `{config}`): `{thr}`", "",
         "| data | source | train | val | CPT-only train/val | below threshold | excluded (URL rule) |",
         "|---|---|---|---|---|---|---|"]
    for d, s in sorted({(d, s) for d, s, _ in c}):
        L.append(f"| {d} | {s} | {c[(d, s, 'train')]:,} | {c[(d, s, 'val')]:,} | {c[(d, s, 'cpt_only_train')]:,}/{c[(d, s, 'cpt_only_val')]:,} | {c[(d, s, 'below_threshold')]:,} | {c[(d, s, 'excluded_url')]:,} |")
    Path("reports/splits_report.md").write_text("\n".join(L) + "\n")
