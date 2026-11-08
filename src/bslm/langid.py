"""Language identification with GlotLID (fastText). Scores the Manding cluster, not only bam_Latn."""

from __future__ import annotations

from functools import lru_cache

import fasttext
from huggingface_hub import hf_hub_download


@lru_cache(maxsize=1)
def _model(repo: str = "cis-lmu/glotlid", fname: str = "model_v3.bin"):
    return fasttext.load_model(hf_hub_download(repo, fname))


def predict(texts: list[str], k: int = 5) -> list[dict[str, float]]:
    m = _model()
    clean = [t.replace("\n", " ") for t in texts]
    labels, probs = m.predict(clean, k=k)
    return [{lab.removeprefix("__label__"): float(p) for lab, p in zip(ls, ps)} for ls, ps in zip(labels, probs)]


def manding_score(pred: dict[str, float], manding: list[str]) -> float:
    return sum(pred.get(lab, 0.0) for lab in manding)


def top(pred: dict[str, float]) -> tuple[str, float]:
    lab = max(pred, key=pred.get)
    return lab, pred[lab]


def is_manding(text: str, pred: dict[str, float], manding: list[str], ms_min: float,
               short_words: int, short_reject_conf: float) -> bool:
    """Length-aware rule calibrated on Bayelemabaga (see reports/cleaning_report.md).

    Long lines (>= short_words): Manding-cluster probability must be >= ms_min.
    Short lines: GlotLID is unreliable, so reject only if confidently another (non-Manding) language.
    """
    if len(text.split()) >= short_words:
        return manding_score(pred, manding) >= ms_min
    lab, p = top(pred)
    return lab in manding or p < short_reject_conf
