"""djelia/bambara-mt-dataset (CC-BY-SA-4.0, gated with auto-approval) -> data/interim/v3/djelia_mt.jsonl.

Config `clean` (train + test splits; source_lang/target_lang/source_text/target_text). Rows are oriented bam -> fr/en
and written in the data/train/parallel_train.jsonl format. Filters:
  normalize (configs/clean.yaml confusables); drop empty sides; exact pair dedup;
  dedup against training text: the pair is dropped if its Bambara side (or any of its sentences) matches
  data/train/parallel_train.jsonl or data/cpt/*/*.jsonl (exact or MinHash near-dup, decontam.py settings);
  decontamination: dropped if either side matches an eval set (data/eval/* + FLORES-200).
If access is denied, the user must open https://huggingface.co/datasets/djelia/bambara-mt-dataset and click "Agree".
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter

import pyarrow.parquet as pq
import yaml

from bslm import langid
from bslm.decontam import sentences
from bslm.dedup import hash_key
from bslm.text_norm import normalize
from bslm.v3data import OUT, TrainMatcher, download, eval_hit, eval_index, iter_train_texts, write_jsonl

REPO, SUBDIR = "djelia/bambara-mt-dataset", "djelia_mt"
FILES = {"train": "clean/train-00000-of-00001.parquet", "test": "clean/test-00000-of-00001.parquet"}
LANG = {"bm": "bam", "bam": "bam", "bambara": "bam", "fr": "fr", "fra": "fr", "french": "fr",
        "en": "en", "eng": "en", "english": "en"}
LID = {"fr": "fra_Latn", "en": "eng_Latn"}


def _fetch() -> dict:
    from huggingface_hub.errors import GatedRepoError
    try:
        download(REPO, "README.md", SUBDIR)
        return {k: download(REPO, v, SUBDIR) for k, v in FILES.items()}
    except GatedRepoError:
        sys.exit("djelia/bambara-mt-dataset: access denied (403). Open "
                 "https://huggingface.co/datasets/djelia/bambara-mt-dataset while logged in and click 'Agree' "
                 "(auto-approved), then re-run.")


def run(config: str = "configs/clean.yaml", seed: int = 13) -> dict:
    paths = _fetch()
    cfg = yaml.safe_load(open(config))
    conf, manding = cfg["normalize"]["confusables"], cfg["langid"]["manding_labels"]
    st, rows, seen = Counter(), [], set()
    for split, p in paths.items():
        for i, r in enumerate(pq.read_table(p).to_pylist()):
            st["read"] += 1
            a, b = LANG.get(str(r["source_lang"]).lower()), LANG.get(str(r["target_lang"]).lower())
            sa, sb = normalize(r["source_text"] or "", conf), normalize(r["target_text"] or "", conf)
            if b == "bam":
                a, b, sa, sb = b, a, sb, sa
            if a != "bam" or b not in ("fr", "en") or not sa or not sb:
                st["dropped_lang_or_empty"] += 1
                continue
            h = hash_key(sa + " || " + sb + " || " + b)
            if h in seen:
                st["dropped_dup_pair"] += 1
                continue
            seen.add(h)
            rows.append({"id": f"djelia_mt-{b}-{split}-{i:06d}", "src_lang": "bam", "tgt_lang": b, "src_text": sa,
                         "tgt_text": sb, "source": "djelia_mt", "license": "CC-BY-SA-4.0", "register": "written",
                         "synthetic": False, "url_or_ref": FILES[split], "split_group": f"djelia_mt:{split}",
                         "align_method": "given", "align_score": None, "orig_split": split,
                         "len_ratio": round(max(len(sa.split()), 1) / max(len(sb.split()), 1), 2)})
    st["after_pair_dedup"] = len(rows)

    sent_of = [(i, s) for i, r in enumerate(rows) for s in [r["src_text"]] + sentences(r["src_text"])]
    hit = TrainMatcher([s for _, s in sent_of]).scan(iter_train_texts())
    bad = {sent_of[j][0] for j in hit}
    st["dropped_in_train"] = len(bad)
    rows = [r for i, r in enumerate(rows) if i not in bad]

    idx = eval_index()
    keep = []
    for r in rows:
        if eval_hit(idx, r["src_text"]) or eval_hit(idx, r["tgt_text"]):
            st["dropped_eval_overlap"] += 1
        else:
            keep.append(r)
    pa, pb = langid.predict([r["src_text"] for r in keep]), langid.predict([r["tgt_text"] for r in keep])
    for r, x, y in zip(keep, pa, pb):
        r["manding_score"] = round(langid.manding_score(x, manding), 4)
        r["tgt_lang_p"] = round(y.get(LID[r["tgt_lang"]], 0.0), 4)
        r["langid_top_src"], r["langid_top_tgt"] = langid.top(x)[0], langid.top(y)[0]
        r["split"] = "train"
    n = write_jsonl(OUT / "djelia_mt.jsonl", keep)
    out = {"stats": dict(st), "written_new_pairs": n, "by_tgt_lang": dict(Counter(r["tgt_lang"] for r in keep))}
    (OUT / "djelia_mt.stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    for r in random.Random(seed).sample(keep, min(5, len(keep))):
        print("SAMPLE", r["tgt_lang"], "|", r["src_text"], "|", r["tgt_text"])
    return out


if __name__ == "__main__":
    run()
