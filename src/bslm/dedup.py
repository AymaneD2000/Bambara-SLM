"""Exact and near-duplicate removal (brief §7.8)."""

from __future__ import annotations

import hashlib
import re
from collections import Counter

from datasketch import MinHash, MinHashLSH

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm_key(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def hash_key(text: str) -> str:
    return hashlib.sha1(norm_key(text).encode()).hexdigest()


def exact_dedup(records: list[dict], key=lambda r: r["text"]) -> tuple[list[dict], list[dict]]:
    seen, keep, drop = set(), [], []
    for r in records:
        h = hash_key(key(r))
        (drop if h in seen else keep).append(r)
        seen.add(h)
    return keep, drop


def _minhash(text: str, k: int, num_perm: int) -> MinHash:
    s = norm_key(text)
    m = MinHash(num_perm=num_perm, seed=1)
    shingles = {s[i:i + k] for i in range(max(1, len(s) - k + 1))}
    m.update_batch([x.encode() for x in shingles])
    return m


def near_dedup(records: list[dict], threshold: float, num_perm: int, k: int,
               key=lambda r: r["text"]) -> tuple[list[dict], list[dict]]:
    """Keep the first occurrence of each near-duplicate cluster (records should be priority-ordered)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep, drop = [], []
    for i, r in enumerate(records):
        m = _minhash(key(r), k, num_perm)
        hits = lsh.query(m)
        if hits:
            r = {**r, "_dup_of": hits[0]}
            drop.append(r)
        else:
            lsh.insert(r["id"], m)
            keep.append(r)
    return keep, drop


def boilerplate_paragraphs(docs: list[str], min_count: int) -> set[str]:
    c = Counter()
    for d in docs:
        c.update({hash_key(p) for p in d.split("\n") if p.strip()})
    return {h for h, n in c.items() if n >= min_count}
