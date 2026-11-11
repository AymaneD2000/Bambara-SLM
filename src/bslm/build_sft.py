"""`bslm build-sft`: chat-format SFT data (brief §9.2), {"messages": [...]} JSONL for mlx-lm (--mask-prompt)
and TRL alike. Uses the Qwen3 chat template with thinking disabled (see reports/stage0_report.md).
"""

from __future__ import annotations

import copy
import hashlib
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
    if cfg.get("version") == 3:
        return run_v3(cfg)
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
_LIST_MARK = re.compile(r"^\s*(?:[-*•]\s+)?(?:\[[ xX]?\]\s*)?(?:[A-D]\.\s+)?")   # "- [ ] ", "- B. " (djelia)
_DIRS = {"fr": ("bam-fr", "fr-bam"), "en": ("bam-en", "en-bam")}


def _keep_pair(r: dict, sc: dict) -> tuple[str, str] | None:
    src, tgt = r["src_text"].strip(), r["tgt_text"].strip()
    if sc.get("strip_verse_numbers"):
        src, tgt = _VERSE.sub("", src), _VERSE.sub("", tgt)
    if sc.get("strip_list_markers"):
        src, tgt = _LIST_MARK.sub("", src).strip(), _LIST_MARK.sub("", tgt).strip()
    if sc.get("min_tgt_lang_p") is not None and (r.get("tgt_lang_p") or 0) < sc["min_tgt_lang_p"]:
        return None
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
        if isinstance(spec, dict) and spec.get("md5_split") and \
                _md5_split(r["src_text"], spec.get("md5_val_fraction", 0.03)) != spec["md5_split"]:
            continue
        if r["tgt_lang"] not in langs or ("sources" in sc and r["source"] not in sc["sources"]):
            continue
        kept = _keep_pair(r, sc)
        if kept is None or (kept[0].lower(), r["tgt_lang"]) in seen:
            continue
        seen.add((kept[0].lower(), r["tgt_lang"]))
        rows.append({**r, "src_text": kept[0], "tgt_text": kept[1]})
    return rows


def _md5_split(key: str, val_fraction: float) -> str:
    """Deterministic split for sources without one; keyed on the (Bambara) source text, so the fr and en pairs of
    one Bambara sentence always land in the same split."""
    h = int(hashlib.md5(key.strip().lower().encode()).hexdigest()[:8], 16) / 2**32
    return "val" if h < val_fraction else "train"


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
                e["_texts"] = (src, tgt)             # raw sides, for decontamination (v3)
                if sc.get("synthetic"):
                    e["meta"]["cap_droppable"] = True
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


def _load_ortho(path: str | None) -> dict | None:
    """data/interim/v3/orthography_map.json (reverse of sudoping01/bambara-orthography normalization.csv):
    standard -> informal variants for words and names, compiled once."""
    if not path:
        return None
    m = json.loads(Path(path).read_text())
    words = {k.lower(): v for k, v in m["words"].items() if v}
    names = {k: v for k, v in m["names"].items() if v}
    alt = lambda keys: re.compile(r"(?<![^\W\d_])(" + "|".join(sorted(map(re.escape, keys), key=len, reverse=True)) + r")(?![^\W\d_])")
    return {"words": words, "names": names, "word_re": alt(words), "name_re": alt(names),
            "word_re_i": re.compile(alt(words).pattern, re.I)}


def _informal_style(rng: random.Random, ortho: dict | None = None):
    """Draw one writer's informal spelling habits; returns text -> text (same habits for every turn of a row).
    Draws from rng in a fixed order that does not depend on the text (deterministic with the seed).
    ortho (v3, `informal_spelling.orthography_map`): extra habits from the reverse orthography table, drawn AFTER
    the v2 ones (so v2 draws are unchanged): ò for ɔ, word-initial dy for j / ty for c, French spellings of place
    and person names (Sikasso, Ségou), lexical variants (bamako, mògò), typographic apostrophes."""
    choice = {k: rng.choice(v) for k, v in _INFORMAL}
    dj, ou, gu = rng.random() < 0.7, rng.random() < 0.7, rng.random() < 0.6
    tch = rng.choice(["tch", "ty", "ti"]) if rng.random() < 0.5 else None
    names = words = dy = ty = apos = None
    if ortho:
        if rng.random() < 0.4:                                           # ò / Ò for ɔ (missing from the v2 table)
            choice["ɔ"], choice["Ɔ"] = "ò", "Ò"
        dy, ty = rng.random() < 0.3, rng.random() < 0.3
        names, words = rng.random() < 0.7, rng.random() < 0.5
        apos = rng.choice(["’", "‘", "ʼ"]) if rng.random() < 0.2 else None
        pick = random.Random(rng.getrandbits(32))                        # per-word variant choices

    def apply(text: str) -> str:
        if names:
            text = ortho["name_re"].sub(lambda m: pick.choice(ortho["names"][m.group(1)]), text)
        if words:
            def w(m):
                v = pick.choice(ortho["words"][m.group(1).lower()])
                return v[:1].upper() + v[1:] if m.group(1)[:1].isupper() else v
            text = ortho["word_re_i"].sub(w, text)
        if apos:
            text = text.replace("'", apos)
        if dy:
            text = re.sub(r"(?<![^\W\d_])j", "dy", text)                       # jɔn -> dyon
        if ty:
            text = re.sub(r"(?<![^\W\d_])c(?=[aeiouɛɔèéò])", "ty", text)        # cɛ -> tyè
        for k, v in choice.items():
            text = text.replace(k, v)
        if dj:
            text = re.sub(r"(?<![a-zA-Z])j|(?<=[aeiouèéô ])j", "dj", text)          # jɔn -> djon
        if ou:
            text = re.sub(r"(?<![oO])u", "ou", text)                                 # mun -> moun
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


_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*'?")


def _informal_bambara_only(text: str, style) -> str:
    """Style on segments that look Bambara; inside them, French/English words ("Traduis « ... »") stay as they are."""
    def seg(p):
        return _WORD.sub(lambda m: m.group(0) if m.group(0).lower() in _FR_EN_WORDS else style(m.group(0)), p)
    return "".join(seg(p) if i % 2 == 0 and _looks_bambara(p) else p for i, p in enumerate(_SEG.split(text)))


def _augment_informal(ex: list[dict], cfg: dict, rng: random.Random) -> int:
    """French-style spelling on `share` of the eligible rows (bam->X translations and Bambara tasks), Bambara text only:
    translation rows: the source {x} (re-formatted into the same instruction); task rows: Bambara segments of every
    user turn (history included). French/English text, instructions and assistant targets are never changed."""
    ic = cfg.get("informal_spelling")
    if not ic:
        return 0
    ortho = _load_ortho(ic.get("orthography_map"))
    n = 0
    for e in ex:
        d = e["meta"]["direction"]
        if "messages" not in e or not (d.startswith("bam-") or d.startswith("task-") or d == "stitch") \
                or rng.random() >= ic["share"]:
            continue
        style = _informal_style(rng, ortho)
        if "_turns" in e:                       # stitched row: per user turn, the same rule as its source row
            changed = False
            for i, kind, instr, x in e["_turns"]:
                m = e["messages"][i]
                new = instr.format(x=style(x)) if kind == "fmt" else \
                    _informal_bambara_only(m["content"], style) if kind == "seg" else m["content"]
                changed |= new != m["content"]
                m["content"] = new
        elif "_bam_input" in e:
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


# ---------------------------------------------------------------------------------------------------------------
# v3 (configs/sft_v3.yaml): v2 sources without the 45k cap + djelia, back-translation (X->bam only), Claude-written
# Bambara conversations (5% held out as chat_valid), stitched multi-turn conversations from existing single-turn rows,
# synthetic cap that only drops pivot/BT rows, and a final decontamination pass (eval sets + FLORES-200).
# ---------------------------------------------------------------------------------------------------------------
_REPLY_LANG = {"bam-fr": "fr", "fr-bam": "bam", "bam-en": "en", "en-bam": "bam", "instr-fr": "fr", "instr-en": "en"}


def _bt_rows(cfg: dict, rng: random.Random) -> list[dict]:
    """Back-translation: the Bambara side is real web text, fr/en is model output -> only X->bam (Bambara target)."""
    bc = cfg.get("back_translation")
    if not bc or not Path(bc["file"]).exists():
        return []
    rows = [json.loads(l) for l in open(bc["file"])]
    rng.shuffle(rows)
    out = Counter()
    ex = []
    for r in rows:
        for lg in bc.get("langs", ["fr", "en"]):
            if not (r.get("keep") or {}).get(lg) or not r.get(lg):
                continue
            d = f"{lg}-bam"
            src, tgt = r[lg].strip(), r["bam"].strip()
            e = _chat(cfg, rng, rng.choice(cfg["instructions"][d]).format(x=src), tgt,
                      {"id": f"{r['id']}-{lg}", "direction": d, "source": bc["name"], "synthetic": True, "bt": True,
                       "cap_droppable": True, "generator": r.get("generator"), "bt_source": r.get("source")})
            e["_texts"] = (src, tgt)
            ex.append(e)
            out[d] += 1
    if bc.get("max_rows"):
        ex = ex[: bc["max_rows"]]
    print(f"  [{bc['name']}] train: {dict(out)} (X->bam only)")
    return ex


def _conversations(cfg: dict, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Claude-written multi-turn Bambara conversations: hold out `val_fraction` per category (md5 of the id) as the
    chat eval set; the rest is repeated `repeat` times in train."""
    cc = cfg["conversations"]
    rows = [json.loads(l) for l in open(cc["file"])]
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    val_ids = set()
    for cat, rs in by_cat.items():   # stratified: the round(f * n) rows with the smallest md5(id) of each category
        rs = sorted(rs, key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
        val_ids |= {r["id"] for r in rs[: round(cc["val_fraction"] * len(rs))]}

    def mk(r):
        e = _chat(cfg, rng, r["user"], r["assistant"],
                  {"id": r["id"], "direction": "task-conv", "source": cc["name"], "category": r["category"],
                   "lang_final": r.get("lang_final"), "synthetic": True, "reviewed": True,
                   "review": cc.get("review", "owner-approved"), "generator": cc.get("generator")},
                  history=r.get("history"))
        return e
    train = [mk(r) for _ in range(cc.get("repeat", 1)) for r in rows if r["id"] not in val_ids]
    val = [mk(r) for r in rows if r["id"] in val_ids]
    print(f"  [{cc['name']}] {len(rows) - len(val_ids)} train x{cc.get('repeat', 1)}, {len(val_ids)} held out (chat_valid)")
    return train, val


def _turn(e: dict, pool: str) -> dict | None:
    """One stitchable turn from a single-turn row: (user, assistant) plus what informal spelling may change."""
    msgs = [m for m in e["messages"] if m["role"] != "system"]
    if len(msgs) != 2:
        return None
    d = e["meta"]["direction"]
    if "_bam_input" in e:
        kind, instr, x = "fmt", *e["_bam_input"]
    elif d.startswith("task-"):
        kind, instr, x = "seg", None, None
    else:
        kind, instr, x = None, None, None
    return {"user": msgs[0]["content"], "assistant": msgs[1]["content"], "kind": kind, "instr": instr, "x": x,
            "lang": _REPLY_LANG.get(d, "bam"), "id": e["meta"]["id"], "direction": d,
            "synthetic": e["meta"]["synthetic"], "texts": e.get("_texts", ())}


def stitch(cfg: dict, rows: list[dict], n: int, rng: random.Random, split: str) -> list[dict]:
    """Chain 2-4 existing single-turn rows (human translations in every direction, Aya fr/en instructions, reviewed
    QA) into multi-turn conversations. Each assistant turn keeps its original target, so the reply language follows
    each user turn; a conversation always mixes at least two reply languages."""
    sc = cfg["stitch"]
    pools: dict[str, list] = {k: [] for k in sc["weights"]}
    for e in rows:
        if "messages" not in e:
            continue
        src = e["meta"]["source"]
        pool = next((k for k, srcs in sc["pools"].items() if src in srcs), None)
        if pool is None or (pool == "qa" and e["meta"]["direction"] == "task-conv"):
            continue
        t = _turn(e, pool)
        if t and len(t["user"]) + len(t["assistant"]) <= sc["max_turn_chars"]:
            pools[pool].append(t)
    for p in pools.values():   # QA rows are repeated in train: keep one copy per id
        uniq = {t["id"]: t for t in p}
        p[:] = list(uniq.values())
        rng.shuffle(p)
    names = [k for k in sc["weights"] if pools[k]]
    weights = [sc["weights"][k] for k in names]
    ptr = {k: 0 for k in names}

    def draw(k):
        if ptr[k] >= len(pools[k]):
            rng.shuffle(pools[k]); ptr[k] = 0
        ptr[k] += 1
        return pools[k][ptr[k] - 1]

    out = []
    for i in range(n):
        k = rng.randint(sc["min_turns"], sc["max_turns"])
        turns, chars = [], 0
        for j in range(k):
            for _ in range(20):
                t = draw(rng.choices(names, weights)[0])
                if any(t["id"] == u["id"] for u in turns) or chars + len(t["user"]) + len(t["assistant"]) > sc["max_chars"]:
                    continue
                if j == k - 1 and len({u["lang"] for u in turns}) == 1 and t["lang"] == turns[0]["lang"]:
                    continue
                break
            else:
                continue
            turns.append(t); chars += len(t["user"]) + len(t["assistant"])
        if len(turns) < 2:
            continue
        e = _chat(cfg, rng, turns[-1]["user"], turns[-1]["assistant"],
                  {"id": f"stitch-{split}-{i:05d}", "direction": "stitch", "source": "stitched",
                   "synthetic": any(t["synthetic"] for t in turns), "stitched": True, "n_turns": len(turns),
                   "parts": [t["id"] for t in turns], "part_directions": [t["direction"] for t in turns]},
                  history=[(t["user"], t["assistant"]) for t in turns[:-1]])
        off = 1 if e["messages"][0]["role"] == "system" else 0
        e["_turns"] = [(off + 2 * j, t["kind"], t["instr"], t["x"]) for j, t in enumerate(turns) if t["kind"]]
        e["_texts"] = tuple(x for t in turns for x in t["texts"])
        out.append(e)
    print(f"  [stitch] {split}: {len(out)} conversations from pools " + ", ".join(f"{k}={len(pools[k])}" for k in names))
    return out


def _eval_hit(idx, text: str, min_sent_words: int = 6):
    """Eval-set match of the whole text (decontam thresholds: exact >= 4 words, near >= 6), or of one of its
    sentences with >= min_sent_words words. Short formulaic sentences inside a longer turn ("N y'a faamu.",
    "O kɔrɔ ye mun ye?") also occur in test sets but are not contamination; a whole short segment still is."""
    from bslm.decontam import sentences
    from bslm.dedup import norm_key
    if (h := idx.match(text)):
        return h
    sents = sentences(text)
    if len(sents) < 2:
        return None
    for s in sents:
        if len(norm_key(s).split()) >= min_sent_words and (h := idx.match(s)):
            return h
    return None


def _decontam(ex: list[dict], idx, memo: dict) -> tuple[list[dict], Counter]:
    kept, hits = [], Counter()
    for e in ex:
        texts = [e["text"]] if "text" in e else [m["content"] for m in e["messages"] if m["role"] != "system"]
        texts += list(e.get("_texts", ()))
        hit = None
        for t in texts:
            if t not in memo:
                memo[t] = _eval_hit(idx, t)
            if (hit := memo[t]):
                break
        if hit:
            hits[f"{e['meta']['source']}:{hit[1]}"] += 1
        else:
            kept.append(e)
    return kept, hits


def _syn_cap(ex: list[dict], cap: float, rng: random.Random) -> dict:
    """Keep synthetic chat rows <= cap by dropping random `cap_droppable` rows (pivot, BT) only; conversations, QA
    and stitched rows are never dropped."""
    chat = [e for e in ex if "messages" in e]
    drop_pool = [e for e in chat if e["meta"].get("cap_droppable")]
    n_fixed = sum(1 for e in chat if e["meta"]["synthetic"] and not e["meta"].get("cap_droppable"))
    real = len(chat) - len(drop_pool) - n_fixed
    keep = max(0, int(cap * real / (1 - cap)) - n_fixed)
    if len(drop_pool) <= keep:
        return {"dropped": 0, "droppable": len(drop_pool)}
    rng.shuffle(drop_pool)
    drop = {id(e) for e in drop_pool[keep:]}
    ex[:] = [e for e in ex if id(e) not in drop]
    print(f"  synthetic capped at {cap:.0%}: kept {keep} of {len(drop_pool)} pivot/BT rows")
    return {"dropped": len(drop), "droppable": len(drop_pool),
            "dropped_by_source": dict(Counter(e["meta"]["source"] for e in drop_pool[keep:]))}


def _split_stats(ex: list[dict]) -> dict:
    chat = [e for e in ex if "messages" in e]
    multi = sum(1 for e in chat if sum(m["role"] == "user" for m in e["messages"]) > 1)
    return {"examples": len(ex), "chat_examples": len(chat),
            "by_direction": dict(Counter(e["meta"]["direction"] for e in ex).most_common()),
            "by_source": dict(Counter(e["meta"]["source"] for e in ex).most_common()),
            "by_category": dict(Counter(e["meta"]["category"] for e in ex if e["meta"].get("category")).most_common()),
            "synthetic_share": round(sum(e["meta"]["synthetic"] for e in chat) / max(len(chat), 1), 4),
            "multi_turn_share": round(multi / max(len(chat), 1), 4),
            "informal_rows": sum(1 for e in chat if e["meta"].get("informal"))}


def _write(out_dir: Path, name: str, ex: list[dict]) -> None:
    with (out_dir / f"{name}.jsonl").open("w") as f:
        for e in ex:
            f.write(json.dumps({k: e[k] for k in ("messages", "text") if k in e}, ensure_ascii=False) + "\n")
    with (out_dir / f"{name}.meta.jsonl").open("w") as f:
        for e in ex:
            f.write(json.dumps(e["meta"], ensure_ascii=False) + "\n")


def run_v3(cfg: dict) -> dict:
    from bslm.v3data import eval_index

    rng = random.Random(cfg["seed"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    rep_train, rep_val = _replay(cfg, rng) if cfg.get("replay") else ([], [])
    rep_train = [copy.deepcopy(e) for _ in range(cfg["replay"].get("repeat", 1)) for e in rep_train] if rep_train else []
    conv_train, conv_val = _conversations(cfg, rng)
    stats: dict = {}

    # train
    ex = _parallel_v2(cfg, "train", rng) + _bt_rows(cfg, rng) + _tasks_v2(cfg, "train", rng) + rep_train + conv_train
    idx, memo = eval_index(), {}
    ex, hits = _decontam(ex, idx, memo)
    print(f"  decontamination: {sum(hits.values())} train rows dropped")
    ex += stitch(cfg, ex, cfg["stitch"]["n_train"], rng, "train")
    cap = _syn_cap(ex, cfg["synthetic_cap"], rng) if cfg.get("synthetic_cap") else {}
    n_inf = _augment_informal(ex, cfg, rng)
    print(f"  informal spelling applied to {n_inf} rows")
    if cfg.get("raw_replay"):
        raw, raw_hits = _decontam(_raw_replay_v2(cfg, len(ex), rng), idx, memo)
        hits += raw_hits
        ex += raw
    rng.shuffle(ex)
    _write(out_dir, "train", ex)
    stats["train"] = {**_split_stats(ex), "synthetic_cap": {"cap": cfg.get("synthetic_cap"), **cap},
                      "decontam_dropped": sum(hits.values()), "decontam_by_source_eval": dict(hits.most_common())}

    # valid (eval_sft_loss): single-turn held-out rows, as in v2
    val = _parallel_v2(cfg, "val", rng) + _tasks_v2(cfg, "val", rng) + rep_val
    # chat_valid (eval_chat_loss): held-out Claude conversations + stitched conversations from VAL rows only
    chat_val = conv_val + stitch(cfg, val, cfg["stitch"]["n_val"], rng, "val")
    for name, rows in (("valid", val), ("chat_valid", chat_val)):
        rng.shuffle(rows)
        _write(out_dir, name, rows)
        stats[name] = _split_stats(rows)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats
