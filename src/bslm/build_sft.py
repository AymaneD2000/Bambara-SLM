"""`bslm build-sft`: chat-format SFT data (brief §9.2), {"messages": [...]} JSONL for mlx-lm (--mask-prompt)
and TRL alike. Uses the Qwen3 chat template with thinking disabled (see reports/stage0_report.md).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

OUT = Path("data/sft")


def _examples(path: Path, cfg: dict, rng: random.Random) -> list[dict]:
    out = []
    for line in path.open():
        r = json.loads(line)
        if r["source"] not in cfg["sources"] or (r.get("align_score") or 0) < cfg["min_align_score"]:
            continue
        if r["tgt_lang"] != "fr":
            continue
        for d in cfg["directions"]:
            src, tgt = (r["src_text"], r["tgt_text"]) if d == "bam-fr" else (r["tgt_text"], r["src_text"])
            msgs = []
            if rng.random() < cfg["system_prompt_share"]:
                msgs.append({"role": "system", "content": cfg["system_prompt"]})
            msgs += [{"role": "user", "content": rng.choice(cfg["instructions"][d]).format(x=src)},
                     {"role": "assistant", "content": tgt}]
            out.append({"messages": msgs, "meta": {"id": r["id"], "direction": d, "source": r["source"],
                                                   "synthetic": False}})
    return out


def _replay(cfg: dict, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Human-written French/English instructions (Aya, Apache-2.0), decontaminated against the eval sets."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from bslm.decontam import EvalIndex, sentences
    from bslm.text_norm import normalize

    rc = cfg["replay"]
    fs = HfFileSystem()
    path = f"datasets/{rc['source']}@{rc['revision']}/data/train-00000-of-00001.parquet"
    rows = [r for r in pq.read_table(fs.open(path)).to_pylist()
            if r["language"] in rc["languages"] and len(r["inputs"]) + len(r["targets"]) <= rc["max_chars"]]
    idx = EvalIndex()
    ex, removed = [], 0
    for r in rows:
        q, a = normalize(r["inputs"]), normalize(r["targets"])
        if not q or not a or any(idx.match(s) for s in sentences(q) + sentences(a)):
            removed += 1
            continue
        msgs = []
        if rng.random() < cfg["system_prompt_share"]:
            msgs.append({"role": "system", "content": cfg["system_prompt"]})
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        ex.append({"messages": msgs, "meta": {"id": f"aya-{r['language']}-{len(ex)}", "direction": f"instr-{r['language'][:2].lower()}",
                                              "source": "aya_dataset", "synthetic": False}})
    rng.shuffle(ex)
    nval = int(rc["val_fraction"] * len(ex))
    print(f"replay: {len(ex)} Aya examples kept ({removed} removed: empty or matching eval sets)")
    return ex[nval:], ex[:nval]


def run(config: str = "configs/sft.yaml") -> dict:
    cfg = yaml.safe_load(Path(config).read_text())
    if "parallel" in cfg:
        return run_v2(cfg)
    rng = random.Random(cfg["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    rep_train, rep_val = _replay(cfg, rng) if cfg.get("replay") else ([], [])
    for split, fn in (("train", "parallel_train.jsonl"), ("valid", "parallel_val.jsonl")):
        ex = _examples(Path("data/train") / fn, cfg, rng) + (rep_train if split == "train" else rep_val)
        rng.shuffle(ex)
        with (OUT / f"{split}.jsonl").open("w") as f:
            for e in ex:
                f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
        with (OUT / f"{split}.meta.jsonl").open("w") as f:
            for e in ex:
                f.write(json.dumps(e["meta"], ensure_ascii=False) + "\n")
        stats[split] = {"examples": len(ex), "by_direction": dict(Counter(e["meta"]["direction"] for e in ex)),
                        "by_source": dict(Counter(e["meta"]["source"] for e in ex)), "synthetic_share": 0.0}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------------------------------------------
# v2 (configs/sft_v2.yaml): several parallel sources (bam-fr AND bam-en), per-source filters and caps, synthetic
# pivot / QA data (labelled, capped), and raw-text replay rows {"text": ...} against forgetting.
# ---------------------------------------------------------------------------------------------------------------
_VERSE = re.compile(r"^\s*\d+[:.]?\s+")
_DIRS = {"fr": ("bam-fr", "fr-bam"), "en": ("bam-en", "en-bam")}


def _keep_pair(r: dict, sc: dict) -> tuple[str, str] | None:
    src, tgt = r["src_text"].strip(), r["tgt_text"].strip()
    if sc.get("strip_verse_numbers"):
        src, tgt = _VERSE.sub("", src), _VERSE.sub("", tgt)
    if not src or not tgt:
        return None
    if (r.get("align_score") if r.get("align_score") is not None else 1.0) < sc.get("min_align_score", 0):
        return None
    ws, wt = len(src.split()), len(tgt.split())
    if min(ws, wt) < sc.get("min_words", 1):
        return None
    if max(ws, wt) / max(min(ws, wt), 1) > sc.get("max_len_ratio", 1e9):
        return None
    return src, tgt


def _source_rows(sc: dict, split: str) -> list[dict]:
    spec = sc["files"][split]
    fn, want_split = (spec, None) if isinstance(spec, str) else (spec["file"], spec.get("split"))
    if not Path(fn).exists():
        print(f"  [{sc['name']}] {fn} missing: skipped")
        return []
    langs = sc["tgt_lang"] if isinstance(sc["tgt_lang"], list) else [sc["tgt_lang"]]
    rows, seen = [], set()
    for line in open(fn):
        r = json.loads(line)
        if want_split and r.get("split") != want_split:
            continue
        if r["tgt_lang"] not in langs or ("sources" in sc and r["source"] not in sc["sources"]):
            continue
        kept = _keep_pair(r, sc)
        if kept is None or (kept[0].lower(), r["tgt_lang"]) in seen:
            continue
        seen.add((kept[0].lower(), r["tgt_lang"]))
        rows.append({**r, "src_text": kept[0], "tgt_text": kept[1]})
    return rows


def _chat(cfg, rng, user, assistant, meta, history=None):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}] if rng.random() < cfg["system_prompt_share"] else []
    for u, a in history or []:   # earlier turns are context only (loss is on the last assistant turn)
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": msgs, "meta": meta}


def _parallel_v2(cfg: dict, split: str, rng: random.Random) -> list[dict]:
    out = []
    for sc in cfg["parallel"]:
        rows = _source_rows(sc, split)
        rng.shuffle(rows)
        cap = sc.get("max_pairs") if split == "train" else sc.get("max_val_pairs", sc.get("max_pairs"))
        rows = rows[:cap] if cap else rows
        for r in rows:
            for d in _DIRS[r["tgt_lang"]]:
                src, tgt = (r["src_text"], r["tgt_text"]) if d.startswith("bam-") else (r["tgt_text"], r["src_text"])
                instr = rng.choice(cfg["instructions"][d])
                e = _chat(cfg, rng, instr.format(x=src), tgt,
                          {"id": r["id"], "direction": d, "source": sc["name"],
                           "synthetic": bool(sc.get("synthetic") or r.get("synthetic")),
                           "generator": r.get("generator")})
                if d.startswith("bam-"):
                    e["_bam_input"] = (instr, src)   # informal spelling goes on {x} only, never on the instruction
                out.append(e)
        print(f"  [{sc['name']}] {split}: {len(rows)} pairs")
    return out


def _tasks_v2(cfg: dict, split: str, rng: random.Random) -> list[dict]:
    """Ready-made chat tasks (e.g. synthetic Bambara QA/summaries after filtering + human check):
    rows {"id","split","task","user","assistant","synthetic","generator","review": "accepted"|...}."""
    out = []
    for tc in cfg.get("tasks", []):
        if not Path(tc["file"]).exists():
            print(f"  [{tc['name']}] {tc['file']} missing: skipped")
            continue
        rows = [json.loads(l) for l in open(tc["file"])]
        rows = [r for r in rows if r.get("split", "train") == split and r.get("review", "accepted") in tc.get("accept", ["accepted"])]
        reps = tc.get("repeat", 1) if split == "train" else 1   # small, human-reviewed sets are seen several times
        for _ in range(reps):
            for r in rows:
                out.append(_chat(cfg, rng, r["user"], r["assistant"],
                                 {"id": r["id"], "direction": f"task-{r['task']}", "source": tc["name"], "reviewed": True,
                                  "synthetic": bool(r.get("synthetic", tc.get("synthetic"))), "generator": r.get("generator")},
                                 history=r.get("history")))
        print(f"  [{tc['name']}] {split}: {len(rows)} examples x{reps}")
    return out


def _raw_replay_v2(cfg: dict, n_chat: int, rng: random.Random) -> list[dict]:
    rc = cfg["raw_replay"]
    pieces = []
    for line in open(rc["file"]):
        buf = ""
        for para in json.loads(line)["text"].split("\n"):
            if len(buf) + len(para) > rc["max_chars"] and buf:
                pieces.append(buf.strip()); buf = ""
            buf += para + "\n"
        if len(buf) > rc["min_chars"]:
            pieces.append(buf.strip())
    rng.shuffle(pieces)
    n = int(rc["share"] / (1 - rc["share"]) * n_chat)
    print(f"  [raw_replay] {min(n, len(pieces))} text pieces (share {rc['share']:.0%}) from {len(pieces)}")
    return [{"text": t, "meta": {"id": f"raw-{i}", "direction": "raw", "source": "cpt_replay", "synthetic": False}}
            for i, t in enumerate(pieces[:n])]


# Real users often write Bambara with French-style spelling (djon, kuntigui, ginini, djabi, mou, tché, è).
# Applied only to user messages whose Bambara is an INPUT (bam->X translation, Bambara QA/chat); targets stay standard.
_INFORMAL = [("ɛ", ("è", "e", "é")), ("Ɛ", ("È", "E")), ("ɔ", ("o", "ô")), ("Ɔ", ("O",)), ("ɲ", ("gn", "ny")),
             ("Ɲ", ("Gn", "Ny")), ("ŋ", ("ng",)), ("Ŋ", ("Ng",))]


def _informal_style(rng: random.Random):
    """Draw one writer's informal spelling habits; returns text -> text (same habits for every turn of a row).
    Draws from rng in a fixed order that does not depend on the text (deterministic with the seed)."""
    choice = {k: rng.choice(v) for k, v in _INFORMAL}
    dj, ou, gu = rng.random() < 0.7, rng.random() < 0.7, rng.random() < 0.6
    tch = rng.choice(["tch", "ty", "ti"]) if rng.random() < 0.5 else None

    def apply(text: str) -> str:
        for k, v in choice.items():
            text = text.replace(k, v)
        if dj:
            text = re.sub(r"(?<![a-zA-Z])j|(?<=[aeiouèéô ])j", "dj", text)          # jɔn -> djon
        if ou:
            text = re.sub(r"(?<!o)u", "ou", text)                                    # mun -> moun
        if gu:
            text = re.sub(r"g(?=[ie])", "gu", text)                                  # kuntigi -> kuntigui
        if tch:
            text = re.sub(r"c(?=[aeiouèéô])", tch, text)                            # ce -> tche
        return text
    return apply


def informal_spelling(text: str, rng: random.Random) -> str:
    return _informal_style(rng)(text)


# Task rows mix languages inside one user message ("Lis ce texte en bambara et réponds en bambara : <passage>\n\n
# Question : Où ... ?"), and some chat rows are fully French/English. Only segments that look Bambara are changed.
_SEG = re.compile(r"(\n+|:\s+|\?\s+|(?<=[.!])\s+)")
_BAM_CH = re.compile(r"[ɛɔɲŋƐƆƝŊ]")
_FR_CH = re.compile(r"[éèêàâçùûôîœ]", re.I)
_FR_EN_WORDS = set("""le les des du une un est sont et ou que qui quoi quel quelle quels dans pour avec sur par pas
ce cette ces il elle ils elles nous vous je tu mon ton son leur texte lis réponds résume traduis question réponse
comment pourquoi où quand phrase en français anglais bambara peux peut fais dis the and is are of to what this that
in read answer question summarize translate does how why where when you your can it with from text sentence english
french""".split())
_BAM_WORDS = set("""ye bɛ be tɛ te ka ko ni ma min don kan la na nin o a n i u aw an k' b' t' y' tun kɔ kɔnɔ fɛ ani
walima mun jɔn di cogo yɔrɔ tuma bi sini kunun dɔ dɔw bɛɛ""".split())


def _looks_bambara(seg: str) -> bool:
    words = re.findall(r"[^\W\d_]+'?", seg.lower())
    if not words or _FR_CH.search(seg):
        return False
    fr_en = sum(w in _FR_EN_WORDS for w in words)
    bam = sum(w in _BAM_WORDS for w in words) + len(_BAM_CH.findall(seg))
    return bam > 0 and fr_en == 0 or bam >= 3 * fr_en > 0


def _informal_bambara_only(text: str, style) -> str:
    return "".join(style(p) if i % 2 == 0 and _looks_bambara(p) else p for i, p in enumerate(_SEG.split(text)))


def _augment_informal(ex: list[dict], cfg: dict, rng: random.Random) -> int:
    """French-style spelling on `share` of the eligible rows (bam->X translations and Bambara tasks), Bambara text only:
    translation rows: the source {x} (re-formatted into the same instruction); task rows: Bambara segments of every
    user turn (history included). French/English text, instructions and assistant targets are never changed."""
    ic = cfg.get("informal_spelling")
    if not ic:
        return 0
    n = 0
    for e in ex:
        d = e["meta"]["direction"]
        if "messages" not in e or not (d.startswith("bam-") or d.startswith("task-")) or rng.random() >= ic["share"]:
            continue
        style = _informal_style(rng)
        if "_bam_input" in e:
            instr, x = e["_bam_input"]
            e["messages"][-2]["content"] = instr.format(x=style(x))      # last user turn
            changed = True
        else:
            changed = False
            for m in e["messages"]:
                if m["role"] == "user":
                    new = _informal_bambara_only(m["content"], style)
                    changed |= new != m["content"]
                    m["content"] = new
        if changed:
            e["meta"]["informal"] = True
            n += 1
    return n


def run_v2(cfg: dict) -> dict:
    rng = random.Random(cfg["seed"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    rep_train, rep_val = _replay(cfg, rng) if cfg.get("replay") else ([], [])
    stats = {}
    for split in ("train", "valid"):
        sp = "train" if split == "train" else "val"
        ex = _parallel_v2(cfg, sp, rng) + _tasks_v2(cfg, sp, rng) + (rep_train if split == "train" else rep_val)
        syn = [e for e in ex if e["meta"]["synthetic"] and not e["meta"].get("reviewed")]   # reviewed rows are never dropped
        n_rev = sum(1 for e in ex if e["meta"]["synthetic"] and e["meta"].get("reviewed"))
        cap = cfg.get("synthetic_cap")
        if cap and split == "train" and len(syn) + n_rev > cap * len(ex):   # drop random unreviewed synthetic rows
            real = len(ex) - len(syn) - n_rev; keep = max(0, int(cap * real / (1 - cap)) - n_rev)
            rng.shuffle(syn); drop = {id(e) for e in syn[keep:]}
            ex = [e for e in ex if id(e) not in drop]
            print(f"  synthetic capped at {cap:.0%}: kept {keep} of {len(syn)}")
        if split == "train":
            print(f"  informal spelling applied to {_augment_informal(ex, cfg, rng)} user inputs")
        if split == "train" and cfg.get("raw_replay"):
            ex += _raw_replay_v2(cfg, len(ex), rng)
        rng.shuffle(ex)
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for e in ex:
                f.write(json.dumps({k: e[k] for k in ("messages", "text") if k in e}, ensure_ascii=False) + "\n")
        with (out_dir / f"{split}.meta.jsonl").open("w") as f:
            for e in ex:
                f.write(json.dumps(e["meta"], ensure_ascii=False) + "\n")
        chat = [e for e in ex if "messages" in e]
        stats[split] = {"examples": len(ex), "by_direction": dict(Counter(e["meta"]["direction"] for e in ex)),
                        "by_source": dict(Counter(e["meta"]["source"] for e in ex)),
                        "synthetic_share": round(sum(e["meta"]["synthetic"] for e in chat) / max(len(chat), 1), 4)}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
