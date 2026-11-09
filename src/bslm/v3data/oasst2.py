"""OpenAssistant/oasst2 (Apache-2.0) -> data/interim/v3/oasst2_threads.jsonl.

English and French conversation threads: the best-ranked path through each message tree.
  - only trees whose prompt is en/fr; every message on the path must be the same language, not deleted,
    review_result true, not synthetic, spam label < 0.5, detoxify toxicity < 0.5
  - usable message: no code block, no heavy math, no URL, at most 800 characters, no mention of Open Assistant/LAION
    (assistant identity); a tree whose prompt is not usable is dropped
  - path: from the prompt, always follow the best-ranked USABLE reply (rank 0 first; unranked replies after ranked
    ones, then by +1 votes); stop at 8 messages or when no usable reply is left, then cut trailing user messages so
    it ends on an assistant message; threads shorter than 2 messages are dropped
  - decontamination: any message sentence matching an eval set (bslm.v3data.eval_index) drops the thread
"""

from __future__ import annotations

import ast
import gzip
import json
import re
from collections import Counter

from bslm.v3data import OUT, download, eval_hit, eval_index, write_jsonl

REPO, FILE, SUBDIR = "OpenAssistant/oasst2", "2023-11-05_oasst2_ready.trees.jsonl.gz", "oasst2"
LANGS = {"en", "fr"}
MAX_CHARS, MAX_MSG = 800, 8
CODE = re.compile(r"```|`[^`\n]+`|^( {4}|\t)\S|^\s*(def \w+\(|import \w|from \w+ import|#include|class \w+[:(])"
                  r"|console\.log\(|printf?\(|;\s*$", re.M)
URL = re.compile(r"https?://|www\.|\b\w+\.(com|org|net|io|fr)\b", re.I)
MATH = re.compile(r"\$[^$]+\$|\\(frac|sqrt|sum|int|begin|cdot|times|left|right)\b|\^\d|[=<>≤≥±∑∫√]")
IDENT = re.compile(r"open[ -]?assistant|laion", re.I)


def _f(x, default=None):
    """oasst2 jsonl stores some fields as native JSON; tolerate strings too (never eval: literal_eval only)."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return default
    return default if x is None else x


def _ok_msg(m: dict, lang: str) -> bool:
    if m.get("lang") != lang or _f(m.get("deleted"), False) or _f(m.get("synthetic"), False):
        return False
    if _f(m.get("review_result"), True) is False:
        return False
    labels = _f(m.get("labels"), {}) or {}
    if (labels.get("spam") or {}).get("value", 0) >= 0.5:
        return False
    tox = _f(m.get("detoxify"), {}) or {}
    return (tox.get("toxicity") or 0) < 0.5


def _bad_content(t: str) -> str | None:
    if len(t) > MAX_CHARS:
        return "long_turn"
    if CODE.search(t):
        return "code"
    if URL.search(t):
        return "url"
    if len(MATH.findall(t)) >= 3:
        return "math"
    if IDENT.search(t):
        return "identity"
    return None


def _best(replies: list[dict], lang: str) -> dict | None:
    ok = [r for r in replies if _ok_msg(r, lang) and _bad_content(r["text"]) is None]
    if not ok:
        return None
    def key(r):
        rank = _f(r.get("rank"))
        votes = (_f(r.get("emojis"), {}) or {}).get("+1", 0)
        return (rank is None, rank if rank is not None else 0, -votes)
    return min(ok, key=key)


def run() -> dict:
    download(REPO, "README.md", SUBDIR)
    path = download(REPO, FILE, SUBDIR)
    st, cut_reasons = Counter(), Counter()
    threads = []
    with gzip.open(path, "rt") as f:
        for line in f:
            t = json.loads(line)
            p = t["prompt"]
            st["trees"] += 1
            lang = p.get("lang")
            if lang not in LANGS:
                continue
            st[f"trees_{lang}"] += 1
            if not _ok_msg(p, lang):
                continue
            if (why := _bad_content(p["text"])):
                cut_reasons[f"prompt_{why}"] += 1
                continue
            path_msgs, node = [p], p
            while len(path_msgs) < MAX_MSG and (node := _best(node.get("replies") or [], lang)):
                path_msgs.append(node)
            while path_msgs and path_msgs[-1]["role"] != "assistant":
                path_msgs.pop()
            if len(path_msgs) < 2:
                st[f"dropped_short_{lang}"] += 1
                continue
            msgs = [{"role": "user" if m["role"] == "prompter" else "assistant", "content": m["text"].strip()}
                    for m in path_msgs]
            threads.append({"id": f"oasst2-{t['message_tree_id']}", "lang": lang, "messages": msgs})
    st["candidate_threads"] = len(threads)
    idx = eval_index()
    keep = []
    for r in threads:
        if any(eval_hit(idx, m["content"]) for m in r["messages"]):
            st["dropped_eval_overlap"] += 1
        else:
            keep.append(r)
    n = write_jsonl(OUT / "oasst2_threads.jsonl", keep)
    by_lang = Counter(r["lang"] for r in keep)
    by_turns = Counter(f"{r['lang']}:{len(r['messages'])}" for r in keep)
    out = {"stats": dict(st), "cut_reasons": dict(cut_reasons), "written": n, "by_lang": dict(by_lang),
           "by_lang_messages": dict(sorted(by_turns.items()))}
    (OUT / "oasst2_threads.stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
