"""HuggingFaceTB/smol-smoltalk (Apache-2.0 on the card) -> data/interim/v3/smoltalk_threads.jsonl.

Short everyday multi-turn chats (English).
  - sub-sources kept: smol-magpie-ultra-short, everyday-conversations, openhermes-50k. Dropped whole: code
    (self-oss-instruct), rewriting (smollm-rewrite-30k, explore-instruct-rewrite), formatting constraints
    (smol-contraints), summarization (smol-summarize-*), long documents (longalign)
  - >= 2 user turns; system messages removed; every turn < 600 characters; ends with an assistant message
  - content filters on every turn: code, math, URLs; on user turns: rewriting/formatting/translation requests
  - dedup on the user turns; rows carry synthetic=true and the generator; decontamination against eval sets; cap 60,000 conversations
Both the train and test splits of the dataset are read (its test split is not one of our evaluation sets).
Parquet shards (~230 MB) are read in record batches, one shard at a time.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter

import pyarrow.parquet as pq

from bslm.dedup import hash_key
from bslm.v3data import OUT, download, eval_hit, eval_index, write_jsonl

REPO, SUBDIR = "HuggingFaceTB/smol-smoltalk", "smol_smoltalk"
FILES = ["data/test-00000-of-00001.parquet"] + [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]
KEEP_SOURCES = {"smol-magpie-ultra-short", "everyday-conversations", "openhermes-50k"}
# Generators as documented on the HuggingFaceTB/smoltalk card (the rows are model-written).
GENERATOR = {"smol-magpie-ultra-short": "Llama-3.1-405B-Instruct (Magpie)",
             "everyday-conversations": "Llama-3.1-70B-Instruct", "openhermes-50k": "OpenHermes-2.5 (mostly GPT-4)"}
MAX_CHARS, CAP, SEED = 600, 60_000, 13
CODE = re.compile(r"```|`[^`\n]+`|^( {4}|\t)\S|^\s*(def \w+\(|import \w|from \w+ import|#include|class \w+[:(])"
                  r"|console\.log\(|printf?\(|;\s*$|\b(python|javascript|java|c\+\+|sql|html|css|regex|function|"
                  r"code|script|programming|algorithm)\b", re.M | re.I)
MATH = re.compile(r"\$[^$]+\$|\\(frac|sqrt|sum|int|begin|cdot|times)\b|\^\d|[=<>≤≥±∑∫√]|\d\s*[*/+x×÷-]\s*\d|"
                  r"\b(equation|integral|derivative|probability|solve for|calculate|polynomial|theorem|"
                  r"matrix|algebra|geometry|arithmetic)\b", re.I)
URL = re.compile(r"https?://|www\.|\b\w+\.(com|org|net|io)\b", re.I)
TASK = re.compile(r"\b(rewrite|rephrase|paraphrase|reword|proofread|edit (this|my|the)|improve (this|my|the)|"
                  r"make (this|it) (more|sound|shorter|concise)|format|bullet|table|json|csv|markdown|list of \d+|"
                  r"translate|summari[sz]e|in \d+ words|word count|essay|poem|haiku|"
                  r"limerick|acrostic|title|headline|tweet|slogan)\b", re.I)


def _why_bad(msgs: list[dict]) -> str | None:
    if sum(m["role"] == "user" for m in msgs) < 2:
        return "single_turn"
    if msgs[-1]["role"] != "assistant":
        return "not_ending_assistant"
    for m in msgs:
        t = m["content"]
        if len(t) >= MAX_CHARS:
            return "long_turn"
        if CODE.search(t):
            return "code"
        if URL.search(t):
            return "url"
        if MATH.search(t):
            return "math"
        if m["role"] == "user" and TASK.search(t):
            return "rewrite_format_task"
    return None


def run() -> dict:
    download(REPO, "README.md", SUBDIR)
    st, why = Counter(), Counter()
    seen, cand = set(), []
    for fn in FILES:
        f = pq.ParquetFile(download(REPO, fn, SUBDIR))
        split = fn.split("/")[1].split("-")[0]
        for b in f.iter_batches(batch_size=2000, columns=["messages", "source"]):
            for k, r in enumerate(b.to_pylist()):
                st["read"] += 1
                if r["source"] not in KEEP_SOURCES:
                    st["dropped_subsource"] += 1
                    continue
                msgs = [{"role": m["role"], "content": (m["content"] or "").strip()} for m in r["messages"]
                        if m["role"] in ("user", "assistant")]
                if (w := _why_bad(msgs)):
                    why[w] += 1
                    continue
                h = hash_key(" || ".join(m["content"] for m in msgs if m["role"] == "user"))
                if h in seen:
                    why["dup_user_turns"] += 1
                    continue
                seen.add(h)
                cand.append({"id": f"smoltalk-{split}-{st['read']:07d}", "lang": "en", "messages": msgs,
                             "source": r["source"], "synthetic": True, "generator": GENERATOR[r["source"]]})
        print(fn, dict(st), len(cand))
    idx = eval_index()
    keep = []
    for r in cand:
        if any(eval_hit(idx, m["content"]) for m in r["messages"]):
            why["eval_overlap"] += 1
        else:
            keep.append(r)
    if len(keep) > CAP:
        keep = random.Random(SEED).sample(keep, CAP)
    n = write_jsonl(OUT / "smoltalk_threads.jsonl", keep)
    out = {"stats": dict(st), "dropped": dict(why), "candidates": len(cand), "written": n,
           "by_source": dict(Counter(r["source"] for r in keep)),
           "by_messages": dict(sorted(Counter(len(r["messages"]) for r in keep).items()))}
    (OUT / "smoltalk_threads.stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
