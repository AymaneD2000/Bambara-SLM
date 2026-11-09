"""databricks/databricks-dolly-15k (CC-BY-SA-3.0) -> data/interim/v3/dolly_grounded.jsonl.

Grounded QA: closed_qa and information_extraction rows with a non-empty context under 1,200 characters and a
non-empty response. Wikipedia citation markers ([1], [citation needed]) are removed. Exact duplicates (same context
and instruction) and rows overlapping an eval set (any sentence of context/instruction/response) are dropped.
Rows: {id, lang, category, context, instruction, response, source}.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from bslm.dedup import hash_key
from bslm.v3data import OUT, download, eval_hit, eval_index, write_jsonl

REPO, FILE, SUBDIR = "databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", "dolly"
CATS, MAX_CTX = {"closed_qa", "information_extraction"}, 1200
CITE = re.compile(r"\[(\d+|[a-z]|citation needed|note \d+)\]", re.I)


def _clean(t: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", CITE.sub("", t or "")).strip()


def run() -> dict:
    download(REPO, "README.md", SUBDIR)
    st, rows, seen = Counter(), [], set()
    for i, line in enumerate(open(download(REPO, FILE, SUBDIR))):
        r = json.loads(line)
        st["read"] += 1
        if r["category"] not in CATS:
            continue
        st[f"cat_{r['category']}"] += 1
        ctx, ins, resp = _clean(r["context"]), _clean(r["instruction"]), _clean(r["response"])
        if not ctx or len(ctx) >= MAX_CTX or not ins or not resp:
            st["dropped_context_len_or_empty"] += 1
            continue
        h = hash_key(ctx + " || " + ins)
        if h in seen:
            st["dropped_dup"] += 1
            continue
        seen.add(h)
        rows.append({"id": f"dolly-{i:05d}", "lang": "en", "category": r["category"], "context": ctx,
                     "instruction": ins, "response": resp, "source": "databricks-dolly-15k"})
    idx = eval_index()
    keep = []
    for r in rows:
        if any(eval_hit(idx, r[k]) for k in ("context", "instruction", "response")):
            st["dropped_eval_overlap"] += 1
        else:
            keep.append(r)
    n = write_jsonl(OUT / "dolly_grounded.jsonl", keep)
    out = {"stats": dict(st), "written": n, "by_category": dict(Counter(r["category"] for r in keep))}
    (OUT / "dolly_grounded.stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
