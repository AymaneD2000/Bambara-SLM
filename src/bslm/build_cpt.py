"""`bslm build-cpt`: build the continued-pretraining mix (brief §9.1).

Outputs, per variant (data/cpt/<variant>/):
  train.jsonl, valid.jsonl          {"text": ...} examples of <= seq_len tokens (mlx-lm and HF compatible)
  valid_bam.jsonl, valid_fr.jsonl, valid_en.jsonl   per-language validation for perplexity monitoring
  stats.json + reports/cpt_mix_<variant>.md
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from rich.console import Console
from transformers import AutoTokenizer

from bslm.decontam import EvalIndex, sentences

console = Console()
TRAIN = Path("data/train")
# Religious-register markers. "Ala"/"Dieu"/"God" alone are NOT counted: blessings ("Ala k'an dɛmɛ") are everyday
# speech in Bambara. A text counts as religious only with >= 2 strong markers and >= 1 marker per 100 words.
REL = re.compile(r"\b(Yesu|Yezu|Krista|Krisita|Kira|Kuran|Kurana|Bibulu|Isirayɛli|Isirayeli|Yahutu\w*|Jésus|Christ|"
                 r"Seigneur|Lord|Israël|Israel|prophète|prophet|Coran|Quran|Bible|Évangile|Gospel|apôtres?|apostles?|"
                 r"disciples?|kalandenw|Musa|Ibrahima|Dawuda|sourate|surah)\b", re.I)


def _religious(text: str) -> bool:
    hits = len(REL.findall(text))
    return hits >= 2 and hits / max(1, len(text.split())) >= 0.01


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()] if p.exists() else []


class Tok:
    def __init__(self):
        self.t = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")

    def count(self, texts: list[str]) -> list[int]:
        out = []
        for i in range(0, len(texts), 2000):
            out += [len(x) for x in self.t(texts[i:i + 2000], add_special_tokens=False)["input_ids"]]
        return out


def _mono_docs(recs: list[dict]) -> list[dict]:
    """Join records of the same split_group in original order into pseudo-documents (transcripts, brief §6.2)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        groups[r["split_group"]].append(r)
    docs = []
    for g, rs in groups.items():
        if rs[0]["register"] == "spoken":
            for i in range(0, len(rs), 40):  # ~40 segments per pseudo-document
                part = rs[i:i + 40]
                docs.append({"stream": f"mono:{part[0]['source']}", "text": "\n".join(r["text"] for r in part),
                             "bam_text": "\n".join(r["text"] for r in part),
                             "religious": _religious("\n".join(r["text"] for r in part)), "group": g})
        else:
            for r in rs:
                docs.append({"stream": f"mono:{r['source']}", "text": r["text"], "bam_text": r["text"],
                             "religious": _religious(r["text"]), "group": g})
    return docs


def _par_docs(recs: list[dict], cfg: dict, rng: random.Random, tag: str) -> list[dict]:
    tpl, k = cfg["parallel_templates"], cfg["parallel_templates"]["pairs_per_block"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["split_group"], r["tgt_lang"])].append(r)
    docs = []
    for (g, lg), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["id"])
        for i in range(0, len(rs), k):
            block = rs[i:i + k]
            text = "\n\n".join(rng.choice(tpl[lg]).format(b=r["src_text"], o=r["tgt_text"]) for r in block)
            docs.append({"stream": f"{tag}:{block[0]['source']}-{lg}", "text": text,
                         "bam_text": "\n".join(r["src_text"] for r in block),
                         "religious": _religious(" ".join(r["src_text"] + " " + r["tgt_text"] for r in block)), "group": g})
    return docs


def _decontam_replay(docs: list[dict], idx: EvalIndex) -> tuple[list[dict], int]:
    removed = 0
    for d in docs:
        sents = sentences(d["text"])
        bad = [s for s in sents if idx.match(s)]
        for s in bad:
            d["text"] = d["text"].replace(s, " ")
        removed += len(bad)
    return [d for d in docs if len(d["text"].split()) >= 30], removed


def _pack(docs: list[dict], tok: Tok, seq_len: int, eos: str) -> list[dict]:
    """Greedy packing per stream: documents joined with EOS into examples of <= seq_len tokens.
    Over-long documents are split on line boundaries."""
    by_stream: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_stream[d["stream"]].append(d)
    out = []
    budget = seq_len - 8
    for stream, ds in by_stream.items():
        pieces: list[str] = []
        for d in ds:
            lines, cur = d["text"].split("\n"), []
            lens = tok.count(lines) if len(lines) > 1 else [None]
            if len(lines) == 1:
                pieces.append(d["text"])
                continue
            n = 0
            for ln, ll in zip(lines, lens):
                if cur and n + ll + 1 > budget:
                    pieces.append("\n".join(cur)); cur, n = [], 0
                cur.append(ln); n += ll + 1
            if cur:
                pieces.append("\n".join(cur))
        plens = tok.count(pieces)
        cur, n = [], 0
        for p, pl in zip(pieces, plens):
            if pl > budget:  # single very long line: keep as its own (truncated by the trainer)
                out.append({"text": p, "stream": stream, "tokens": min(pl, seq_len)})
                continue
            if cur and n + pl + 1 > budget:
                out.append({"text": f"\n{eos}\n".join(cur), "stream": stream, "tokens": n}); cur, n = [], 0
            cur.append(p); n += pl + 1
        if cur:
            out.append({"text": f"\n{eos}\n".join(cur), "stream": stream, "tokens": n})
    return out


def build(variant: str, cfg: dict, tok: Tok, idx: EvalIndex) -> dict:
    rng = random.Random(cfg["seed"])
    vcfg = cfg["variants"][variant]
    out_dir = Path("data/cpt") / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bambara-side streams
    train = _mono_docs(_jsonl(TRAIN / "mono_train.jsonl")) + _par_docs(_jsonl(TRAIN / "parallel_train.jsonl"), cfg, rng, "par")
    val = _mono_docs(_jsonl(TRAIN / "mono_val.jsonl")) + _par_docs(_jsonl(TRAIN / "parallel_val.jsonl"), cfg, rng, "par")
    if vcfg["include_cpt_only"]:
        train += _par_docs(_jsonl(TRAIN / "parallel_cpt_only_train.jsonl"), cfg, rng, "par_cpt_only")
        val += _par_docs(_jsonl(TRAIN / "parallel_cpt_only_val.jsonl"), cfg, rng, "par_cpt_only")

    # religious cap on Bambara tokens
    bt = tok.count([d["bam_text"] for d in train])
    for d, n in zip(train, bt):
        d["bam_tokens"] = n
    tot_b = sum(bt)
    rel_b = sum(d["bam_tokens"] for d in train if d["religious"])
    cap = cfg["religious"]["cap"]
    dropped_rel = 0
    if rel_b > cap * tot_b:
        rel_docs = [d for d in train if d["religious"]]
        rng.shuffle(rel_docs)
        excess = rel_b - cap * tot_b
        drop_ids = set()
        for d in rel_docs:
            if excess <= 0:
                break
            drop_ids.add(id(d)); excess -= d["bam_tokens"]; dropped_rel += d["bam_tokens"]
        train = [d for d in train if id(d) not in drop_ids]
    rel_share = (rel_b - dropped_rel) / max(1, tot_b - dropped_rel)

    # replay to budget
    tt = tok.count([d["text"] for d in train])
    core_tokens = sum(tt)
    r = cfg["replay"]
    replay_budget = r["share"] / (1 - r["share"]) * core_tokens
    replay_train, replay_val, replay_stats = [], [], {}
    for lang, frac in (("fr", r["fr_fraction"]), ("en", 1 - r["fr_fraction"])):
        docs = [{"stream": f"replay:{lang}", "text": d["text"], "bam_text": "", "religious": False, "group": d["split_group"]}
                for d in _jsonl(Path(f"data/raw/replay/replay_{lang}.jsonl"))]
        docs, removed = _decontam_replay(docs, idx)
        rng.shuffle(docs)
        nval = max(20, int(r["val_fraction"] * len(docs)))
        replay_val += docs[:nval]
        lens = tok.count([d["text"] for d in docs[nval:]])
        budget, used = frac * replay_budget, 0
        for d, n in zip(docs[nval:], lens):
            if used >= budget:
                break
            replay_train.append(d); used += n
        replay_stats[lang] = {"tokens": used, "budget": round(budget), "eval_sentences_removed": removed,
                              "short_of_budget": used < 0.95 * budget}
    train += replay_train

    packed_train = _pack(train, tok, cfg["seq_len"], cfg["eos"])
    packed_val = _pack(val + replay_val, tok, cfg["seq_len"], cfg["eos"])
    rng.shuffle(packed_train)

    def dump(p: Path, rows):
        with p.open("w") as f:
            for x in rows:
                f.write(json.dumps({"text": x["text"]}, ensure_ascii=False) + "\n")

    dump(out_dir / "train.jsonl", packed_train)
    dump(out_dir / "valid.jsonl", packed_val)
    for lang, pref in (("bam", ("mono:", "par:", "par_cpt_only:")), ("fr", ("replay:fr",)), ("en", ("replay:en",))):
        dump(out_dir / f"valid_{lang}.jsonl", [x for x in packed_val if x["stream"].startswith(pref)])

    by = Counter()
    for x in packed_train:
        by[x["stream"]] += x["tokens"]
    total = sum(by.values())
    stats = {"variant": variant, "train_examples": len(packed_train), "valid_examples": len(packed_val),
             "train_tokens": total, "by_stream_tokens": dict(by.most_common()),
             "replay_share": sum(v for k, v in by.items() if k.startswith("replay:")) / total,
             "religious_share_of_bambara_tokens": rel_share, "religious_tokens_dropped_by_cap": dropped_rel,
             "replay": replay_stats}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    L = [f"# CPT mix: {variant}", "", f"Config: `configs/cpt_mix.yaml`. Examples ≤ {cfg['seq_len']} tokens, packed per stream, shuffled.", "",
         f"- train: {len(packed_train):,} examples, **{total/1e6:.2f} M tokens**; valid: {len(packed_val):,} examples",
         f"- replay share: {stats['replay_share']:.1%} (target {r['share']:.0%}); "
         + ", ".join(f"{k}: {v['tokens']/1e6:.2f} M tokens, {v['eval_sentences_removed']} eval-matching sentences removed"
                     for k, v in replay_stats.items()),
         f"- religious share of Bambara tokens: {rel_share:.1%} (cap {cap:.0%}; dropped {dropped_rel:,} tokens)", "",
         "| stream | tokens | share |", "|---|---|---|"]
    L += [f"| {k} | {v:,} | {v/total:.1%} |" for k, v in by.most_common()]
    Path(f"reports/cpt_mix_{variant}.md").write_text("\n".join(L) + "\n")
    console.print(f"{variant}: {len(packed_train):,} train examples, {total/1e6:.2f} M tokens, replay {stats['replay_share']:.1%}, "
                  f"religious {rel_share:.1%}")
    return stats


def run(config: str = "configs/cpt_mix.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    tok, idx = Tok(), EvalIndex()
    for v in cfg["variants"]:
        build(v, cfg, tok, idx)
