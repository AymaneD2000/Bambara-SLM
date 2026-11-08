"""`bslm align`: score every parallel pair (brief §8.2) and prepare manual inspection.

Signals (all license-clean for commercial use):
  - LASER3 (bam_Latn) / LASER2 (fr, en) cosine and margin score (BSD license). Embeddings are produced
    by scripts/laser_embed.py in .venv-laser.
  - IBM Model 1 lexical translation scores in both directions, trained on the trusted pairs
    (Bayelemabaga train/valid + jeli-asr) for bam-fr, and bootstrapped from the top-LASER NLLB pairs for bam-en.
  - char length ratio, number agreement, question/exclamation agreement.
Combination: logistic regression trained on positives (trusted pairs) vs negatives (shuffled pairs and
off-by-one neighbours, a realistic misalignment). align_score = P(aligned).
NOTE: thresholds are NOT final: they are chosen by the supervisor after inspecting 100 pairs per bucket.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

console = Console()
LASER = Path("data/interim/laser")
OUT = Path("data/clean")
REP = Path("reports/alignment")
TOK = re.compile(r"\w+", re.UNICODE)
NUM = re.compile(r"\d+")
TRUSTED = {"bayelemabaga", "jeli_asr", "bam_asr_all_extra_mali_pense"}
BUCKETS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
FEATS_LEX = ["margin", "cos", "lex_fwd", "lex_bwd", "len_ratio", "num_bad", "punct"]
FEATS_NOLEX = ["margin", "cos", "len_ratio", "num_bad", "punct"]


# ---------------------------------------------------------------- LASER margin

def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def knn_mean(a: np.ndarray, b: np.ndarray, k: int = 4, chunk: int = 1024) -> np.ndarray:
    """For each row of a, mean cosine of its k nearest rows in b (embeddings are L2-normalized)."""
    dev = _device()
    B = torch.from_numpy(b.astype(np.float16)).to(dev)
    out = np.empty(len(a), np.float32)
    for i in range(0, len(a), chunk):
        A = torch.from_numpy(a[i:i + chunk].astype(np.float16)).to(dev)
        out[i:i + chunk] = torch.topk(A @ B.T, k, dim=1).values.float().mean(1).cpu().numpy()
    return out


def margin_scores(src: np.ndarray, tgt: np.ndarray, k: int = 4):
    """Ratio margin (Artetxe & Schwenk 2019). Returns cos, margin, kNN mean of each src row, of each tgt row."""
    cos = (src.astype(np.float32) * tgt.astype(np.float32)).sum(1)
    ks, kt = knn_mean(src, tgt, k), knn_mean(tgt, src, k)
    return cos, cos / ((ks + kt) / 2), ks, kt


# ---------------------------------------------------------------- IBM Model 1

def _toks(t: str) -> list[str]:
    return [w.lower() for w in TOK.findall(t) if not w.isdigit()]


class IBM1:
    """t(f|e) with a NULL source word. Integer-keyed tables to keep memory small."""

    def __init__(self, iters: int = 5):
        self.iters, self.t, self.ve, self.vf = iters, {}, {"<NULL>": 0}, {}

    def _ids(self, ws, vocab, grow):
        out = []
        for w in ws:
            if w not in vocab:
                if not grow:
                    out.append(-1)
                    continue
                vocab[w] = len(vocab)
            out.append(vocab[w])
        return out

    def fit(self, pairs: list[tuple[list[str], list[str]]]) -> "IBM1":
        data = [([0] + self._ids(e, self.ve, True), self._ids(f, self.vf, True)) for e, f in pairs]
        V = len(self.vf) + 1
        t: dict[int, float] = {}
        init = 1.0 / max(1, len(self.vf))
        for _ in range(self.iters):
            count: dict[int, float] = defaultdict(float)
            total: dict[int, float] = defaultdict(float)
            for e, f in data:
                for fw in f:
                    keys = [ew * V + fw for ew in e]
                    ps = [t.get(k, init) if t else init for k in keys]
                    z = sum(ps)
                    for k, ew, p in zip(keys, e, ps):
                        c = p / z
                        count[k] += c
                        total[ew] += c
            t = {k: v / total[k // V] for k, v in count.items() if v / total[k // V] > 1e-4}
            init = 1e-7
        self.t, self.V = t, V
        return self

    def score(self, e: list[str], f: list[str]) -> float:
        """Mean per-target-word log P(f_j | e) under IBM-1; floor 1e-7 for unseen pairs/words."""
        if not f:
            return math.log(1e-7)
        e_ids = [0] + [i for i in self._ids(e, self.ve, False) if i >= 0]
        f_ids = self._ids(f, self.vf, False)
        t, V, n = self.t, self.V, len(e_ids)
        tot = 0.0
        for fw in f_ids:
            p = 0.0 if fw < 0 else sum(t.get(ew * V + fw, 0.0) for ew in e_ids) / n
            tot += math.log(max(p, 1e-7))
        return tot / len(f_ids)


# ---------------------------------------------------------------- features

ENUM = re.compile(r"^\s*\(?\d+(?::\d+)?[).:]?\s+")  # list / verse enumerations: "2878) ", "21 ", "3:2 "


def _strip_enum(t: str) -> str:
    return ENUM.sub("", t, count=1)


def _num_bad(a: str, b: str) -> float:
    """Share of the Bambara side's numbers missing from the other side; 0 when Bambara has none.
    (Bambara often writes numbers in words where French/English use digits, so only mismatches count.)"""
    na, nb = set(NUM.findall(a)), set(NUM.findall(b))
    if not na:
        return 0.0
    return 1.0 - len(na & nb) / len(na)


def _punct(a: str, b: str) -> int:
    return int(("?" in a) == ("?" in b)) + int(("!" in a) == ("!" in b))


class Lex:
    """Pair of IBM-1 models: fwd = P(other | bam), bwd = P(bam | other)."""

    def __init__(self, pairs: list[tuple[list[str], list[str]]]):
        self.f = IBM1().fit(pairs)
        self.b = IBM1().fit([(y, x) for x, y in pairs])

    def scores(self, a: list[str], b: list[str]) -> tuple[float, float]:
        return self.f.score(a, b), self.b.score(b, a)


class CrossFitLex:
    """Two lexicons trained on disjoint halves of the reference pairs (cross-fitting).
    Items of fold k are scored with the lexicon trained on the other fold, so their lexical features are
    out-of-sample; any other pair is scored with the average of both lexicons."""

    def __init__(self, pairs: list[tuple[list[str], list[str]]], folds: list[int]):
        self.lex = [Lex([p for p, f in zip(pairs, folds) if f != k]) for k in (0, 1)]

    def scores(self, a, b, fold: int | None = None) -> tuple[float, float]:
        if fold is not None:
            return self.lex[fold].scores(a, b)
        s0, s1 = self.lex[0].scores(a, b), self.lex[1].scores(a, b)
        return (s0[0] + s1[0]) / 2, (s0[1] + s1[1]) / 2


def _fold(rid: str) -> int:
    return int(hashlib.md5(rid.encode()).hexdigest()[:4], 16) % 2


def features(src: list[str], tgt: list[str], cos, margin, lex: CrossFitLex | None,
             folds: list[int | None] | None = None) -> dict[str, np.ndarray]:
    src = [_strip_enum(x) for x in src]
    tgt = [_strip_enum(x) for x in tgt]
    F = {
        "margin": np.asarray(margin, np.float32), "cos": np.asarray(cos, np.float32),
        "len_ratio": np.array([abs(math.log((len(a) + 5) / (len(b) + 5))) for a, b in zip(src, tgt)], np.float32),
        "num_bad": np.array([_num_bad(a, b) for a, b in zip(src, tgt)], np.float32),
        "punct": np.array([_punct(a, b) for a, b in zip(src, tgt)], np.float32),
    }
    if lex is not None:
        folds = folds or [None] * len(src)
        sc = [lex.scores(_toks(a), _toks(b), f) for a, b, f in zip(src, tgt, folds)]
        F["lex_fwd"] = np.array([x[0] for x in sc], np.float32)
        F["lex_bwd"] = np.array([x[1] for x in sc], np.float32)
    return F


def _X(F: dict, names: list[str]) -> np.ndarray:
    return np.stack([F[n] for n in names], 1)


# ---------------------------------------------------------------- main

def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def run(seed: int = 13, k: int = 4) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)
    REP.mkdir(parents=True, exist_ok=True)
    recs = _load_jsonl(OUT / "parallel.jsonl")
    E_src = np.load(LASER / "parallel_src_bam.npy")
    E_tgt = np.load(LASER / "parallel_tgt_laser2.npy")
    assert len(recs) == len(E_src) == len(E_tgt), "embeddings out of sync with parallel.jsonl: re-run laser_embed"
    fold_of = [_fold(r["id"]) for r in recs]

    by_lang = defaultdict(list)
    for i, r in enumerate(recs):
        by_lang[r["tgt_lang"]].append(i)

    # 1. margin per language pool (neighbours searched in the whole bam-fr or bam-en pool)
    cos, margin = np.zeros(len(recs), np.float32), np.zeros(len(recs), np.float32)
    knn_s, knn_t = np.zeros(len(recs), np.float32), np.zeros(len(recs), np.float32)
    for lg, idx in by_lang.items():
        idx = np.array(idx)
        c, m, ks, kt = margin_scores(E_src[idx], E_tgt[idx], k)
        cos[idx], margin[idx], knn_s[idx], knn_t[idx] = c, m, ks, kt
        console.print(f"margin {lg}: {len(idx):,} pairs")

    # 2. cross-fitted lexicons. fr: trusted pairs. en: bootstrapped from the top 25% NLLB bam-en by LASER margin.
    ref = {"fr": [i for i in by_lang["fr"] if recs[i]["source"] in TRUSTED]}
    if by_lang.get("en"):
        en_idx = np.array(by_lang["en"])
        ref["en"] = list(en_idx[margin[en_idx] >= np.quantile(margin[en_idx], 0.75)])
    lex, in_ref = {}, set()
    for lg, ids in ref.items():
        pairs = [(_toks(_strip_enum(recs[i]["src_text"])), _toks(_strip_enum(recs[i]["tgt_text"]))) for i in ids]
        lex[lg] = CrossFitLex(pairs, [fold_of[i] for i in ids])
        in_ref |= set(ids)
        console.print(f"IBM1 cross-fitted for {lg} on {len(ids):,} pairs")

    # 3. combiner training: trusted bam-fr positives vs negatives built WITHIN a fold (shuffled + off-by-one),
    #    all lexical features out-of-fold.
    pos = np.array(ref["fr"])
    neg = []
    for f in (0, 1):
        pf = [i for i in pos if fold_of[i] == f]
        sh = list(np.random.permutation(pf))
        neg += list(zip(pf, sh))[: len(pf) // 2]
        order = sorted(pf, key=lambda i: recs[i]["id"])
        adj = [(order[j], order[j + 1]) for j in range(len(order) - 1)
               if recs[order[j]]["split_group"] == recs[order[j + 1]]["split_group"]]
        neg += rng.sample(adj, min(len(adj), len(pf) // 2))

    def feat_for(pairs_ij, lg):
        a = np.array([i for i, _ in pairs_ij]); b = np.array([j for _, j in pairs_ij])
        c = (E_src[a].astype(np.float32) * E_tgt[b].astype(np.float32)).sum(1)
        m = c / ((knn_s[a] + knn_t[b]) / 2)  # src a's and tgt b's own k-NN means
        return features([recs[i]["src_text"] for i in a], [recs[j]["tgt_text"] for j in b], c, m, lex[lg],
                        [fold_of[i] for i in a])

    Fp, Fn = feat_for([(i, i) for i in pos], "fr"), feat_for(neg, "fr")
    models, evals = {}, {}
    for name, cols in (("lex", FEATS_LEX), ("nolex", FEATS_NOLEX)):
        X = np.concatenate([_X(Fp, cols), _X(Fn, cols)])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        perm = np.random.permutation(len(y)); cut = int(0.8 * len(y))
        clf = LogisticRegression(max_iter=2000).fit(X[perm[:cut]], y[perm[:cut]])
        evals[name] = roc_auc_score(y[perm[cut:]], clf.predict_proba(X[perm[cut:]])[:, 1])
        models[name] = (clf.fit(X, y), cols)
        console.print(f"combiner {name}: held-out AUC {evals[name]:.3f}; coef "
                      + ", ".join(f"{c}={w:.2f}" for c, w in zip(cols, clf.coef_[0])))

    import pickle
    with open("data/interim/align_model.pkl", "wb") as fh:
        pickle.dump({"lex": lex, "models": models, "k": k}, fh)

    # 4. score every pair; reference pairs get out-of-fold lexical scores, the rest the 2-fold average
    scores = np.zeros(len(recs), np.float32)
    scores_nolex = np.zeros(len(recs), np.float32)
    for lg, idx in by_lang.items():
        idx = np.array(idx)
        F = features([recs[i]["src_text"] for i in idx], [recs[i]["tgt_text"] for i in idx],
                     cos[idx], margin[idx], lex[lg], [fold_of[i] if i in in_ref else None for i in idx])
        clf, cols = models["lex"]
        scores[idx] = clf.predict_proba(_X(F, cols))[:, 1]
        clf, cols = models["nolex"]
        scores_nolex[idx] = clf.predict_proba(_X(F, cols))[:, 1]
        for n, v in F.items():
            for j, i in enumerate(idx):
                recs[i].setdefault("align_feats", {})[n] = round(float(v[j]), 4)

    for i, r in enumerate(recs):
        r["align_method"] = r["align_method"] + "+laser3_margin+ibm1_crossfit+logreg"
        r["align_score"] = round(float(scores[i]), 4)
        r["align_score_nolex"] = round(float(scores_nolex[i]), 4)
    with (OUT / "parallel_scored.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _report(recs, evals, rng)


def _bucket(s: float) -> str:
    for lo, hi in zip(BUCKETS, BUCKETS[1:]):
        if lo <= s < hi:
            return f"{lo:.1f}-{min(hi, 1.0):.1f}"
    return "?"


def _report(recs: list[dict], evals: dict, rng: random.Random) -> None:
    groups = defaultdict(list)
    for r in recs:
        g = f"{r['source']}_{r['tgt_lang']}" if r["source"] != "bam_asr_all_extra_mali_pense" else "jeli_asr_fr"
        groups[g].append(r)
    L = ["# Alignment scoring (Gate 2, brief §8.2)", "",
         "Score = P(aligned) from a logistic regression over LASER3/LASER2 margin + cosine, IBM-1 lexical scores, "
         "length ratio, number and punctuation agreement. Trained on trusted bam-fr pairs vs shuffled and "
         "off-by-one negatives.", "",
         f"Combiner held-out AUC: with lexicon {evals['lex']:.3f}, without lexicon {evals['nolex']:.3f}.", "",
         "## Score distribution (records per bucket)", "",
         "| group | n | " + " | ".join(_bucket(b) for b in BUCKETS[:-1]) + " | median |", "|---|---|" + "---|" * (len(BUCKETS))]
    for g, rs in sorted(groups.items()):
        c = defaultdict(int)
        for r in rs:
            c[_bucket(r["align_score"])] += 1
        med = float(np.median([r["align_score"] for r in rs]))
        L.append(f"| {g} | {len(rs):,} | " + " | ".join(f"{c[_bucket(b)]:,}" for b in BUCKETS[:-1]) + f" | {med:.2f} |")
    L += ["", "## Manual inspection files", "",
          "`reports/alignment/inspect/<group>__<bucket>.tsv`: up to 100 random pairs per bucket. "
          "Please fill column `ok` with y / n / ? (partial). Thresholds are chosen from these judgements.", ""]
    d = REP / "inspect"
    d.mkdir(parents=True, exist_ok=True)
    for g, rs in groups.items():
        bk = defaultdict(list)
        for r in rs:
            bk[_bucket(r["align_score"])].append(r)
        for b, items in bk.items():
            pick = rng.sample(items, min(100, len(items)))
            with (d / f"{g}__{b}.tsv").open("w") as f:
                f.write("ok\tscore\tid\tbambara\tother\n")
                # random order on purpose: a reviewer who marks only the first rows must still get an unbiased sample
                for r in pick:
                    f.write(f"\t{r['align_score']:.3f}\t{r['id']}\t{r['src_text'][:400]}\t{r['tgt_text'][:400]}\n")
    (REP / "alignment_report.md").write_text("\n".join(L) + "\n")
    console.print(f"wrote {REP/'alignment_report.md'} and inspection files")


# ---------------------------------------------------------------- held-out validation on gold pairs

def validate(gold_jsonl: str, src_npy: str, tgt_npy: str, out_md: str = "reports/alignment/validation.md",
             seed: int = 13) -> dict:
    """Score gold-aligned pairs never used in training (e.g. SIB-200 test = FLORES sentences) and shuffled
    versions of them. Margins use k-NN against the full training pool, exactly as in run()."""
    import pickle
    art = pickle.load(open("data/interim/align_model.pkl", "rb"))
    lex, models, k = art["lex"], art["models"], art["k"]
    gold = _load_jsonl(Path(gold_jsonl))
    Gs, Gt = np.load(src_npy), np.load(tgt_npy)
    recs = _load_jsonl(OUT / "parallel.jsonl")
    E_src = np.load(LASER / "parallel_src_bam.npy", mmap_mode="r")
    E_tgt = np.load(LASER / "parallel_tgt_laser2.npy", mmap_mode="r")
    rng = np.random.RandomState(seed)
    L = ["# Alignment scorer validation on held-out gold pairs", "", f"Gold file: `{gold_jsonl}`", "",
         "| lang | n | median score (gold) | median (shuffled) | gold ≥0.5 | shuffled ≥0.5 | gold ≥0.8 | shuffled ≥0.8 |",
         "|---|---|---|---|---|---|---|---|"]
    res = {}
    for lg in sorted({r["tgt_lang"] for r in gold}):
        gi = np.array([i for i, r in enumerate(gold) if r["tgt_lang"] == lg])
        pool = np.array([i for i, r in enumerate(recs) if r["tgt_lang"] == lg])
        ps, pt = np.asarray(E_src[pool]), np.asarray(E_tgt[pool])
        ks, kt = knn_mean(Gs[gi], pt, k), knn_mean(Gt[gi], ps, k)
        out = {}
        for name, perm in (("gold", np.arange(len(gi))), ("shuf", rng.permutation(len(gi)))):
            c = (Gs[gi].astype(np.float32) * Gt[gi][perm].astype(np.float32)).sum(1)
            m = c / ((ks + kt[perm]) / 2)
            F = features([gold[i]["src_text"] for i in gi], [gold[gi[j]]["tgt_text"] for j in perm], c, m, lex[lg])
            clf, cols = models["lex"]
            out[name] = clf.predict_proba(_X(F, cols))[:, 1]
        g, s = out["gold"], out["shuf"]
        res[lg] = {"n": len(gi), "gold_ge_0.5": float((g >= .5).mean()), "shuf_ge_0.5": float((s >= .5).mean()),
                   "gold_ge_0.8": float((g >= .8).mean()), "shuf_ge_0.8": float((s >= .8).mean())}
        L.append(f"| bam-{lg} | {len(gi)} | {np.median(g):.2f} | {np.median(s):.2f} | {100*(g>=.5).mean():.0f}% | "
                 f"{100*(s>=.5).mean():.0f}% | {100*(g>=.8).mean():.0f}% | {100*(s>=.8).mean():.0f}% |")
    L += ["", "Gold ≥ t approximates recall at threshold t; shuffled ≥ t approximates the false-accept rate of "
          "clearly wrong pairs. Off-domain caveat: FLORES sentences are longer and more formal than most training pairs."]
    Path(out_md).write_text("\n".join(L) + "\n")
    return res
