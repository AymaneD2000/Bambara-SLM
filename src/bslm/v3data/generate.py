"""Phase 3 synthetic data for v3 with the SFT v2 model (Qwen3 chat template, thinking off).

Subcommands (`PYTHONPATH=src python -m bslm.v3data.generate <cmd> --help`):
  prep-bt   (CPU, local) Bambara back-translation sources -> bt_sources.jsonl (+ .stats.json)
            mono_bam.jsonl (HPLT/GlotCC, already filtered) + new segments from MADLAD (ODC-BY) and Kunkado train
            (CC-BY-SA-4.0, spoken). Same filters as v3data.hplt: normalize, quality, URL, PII, strict GlotLID,
            standard orthography, exact + near dedup, dedup vs training pairs, eval decontamination (+ FLORES-200);
            plus religious / Bible-like and French code-switch filters. 8-60 words, about 60k segments.
  bt        Bambara -> French and -> English (sampling, Edunov et al. 2018), then a greedy round trip of the
            French output back to Bambara, chrF(round trip, original). -> bt_out[.sIofN].jsonl
  chat      oasst2 threads + dolly grounded QA, every message / field en|fr -> Bambara (greedy), and a greedy round
            trip Bambara -> source language with chrF per message. -> chat_bam[.sIofN].jsonl  (smoltalk is NOT used)
  filter    thresholds of configs/v3_generate.yaml -> bt_out_filtered.jsonl, chat_bam_filtered.jsonl,
            v3_filter.stats.json and review sheets reports/review/v3_<category>_sample.csv
  selftest  translate 4 sentences with a backend (used by the Kaggle notebook to choose vLLM or HF)
  plan      item counts and a rough time estimate

The model translates sentence by sentence (v2 is a sentence-level model): every text is split into sentences
(split_text), keeping paragraph breaks, list markers and whitespace verbatim, and joined back after translation.
Generation outputs are append-only jsonl; ids already in any output file are skipped, so every job resumes.
Backends: vllm (Kaggle, fp16, one process per GPU), hf (transformers, fp16, left padding, length-sorted batches),
mlx (local smoke tests only).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IN_DIR = Path("data/interim/v3")
GENERATOR = "bslm-sft-v2"
LICENCES = {"hplt2_cleaned": "CC0-1.0 (packaging) + HPLT terms of use", "glotcc_v1": "CC0-1.0 (packaging) + Common Crawl ToU",
            "madlad_bm": "ODC-BY", "kunkado": "CC-BY-SA-4.0", "oasst2": "Apache-2.0", "dolly": "CC-BY-SA-3.0"}


# ----------------------------------------------------------------------------------------------- sentence splitting
MARKER = re.compile(r"^[ \t]*(?:[-*•–·]|#{1,6}|\d{1,3}[.)]|[A-Za-z][.)]|\(\d{1,3}\)|>)[ \t]+")
BOUNDARY = re.compile(r"[.!?…]+[\"'»”’)\]]*(\s+)")
ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "no", "fig", "approx", "mme", "mlle", "ex", "cf",
          "jr", "sr", "inc", "ltd", "co", "u.s", "n°", "p", "pp", "vol", "ch", "art", "av", "env", "min", "max", "nb",
          "prof", "gen", "col", "lt", "sgt", "mt", "ft", "oz", "lb", "lbs", "approx", "dept", "est", "m", "mm", "km"}
LETTER = re.compile(r"[^\W\d_]")


def _sentences(line: str, lower_ok: bool) -> list[tuple[str, str]]:
    """(sentence, whitespace after it) for one line without leading/trailing whitespace."""
    out, start = [], 0
    for m in BOUNDARY.finditer(line):
        nxt = line[m.end():m.end() + 1]
        if not nxt:
            continue
        if line[m.start()] == ".":
            w = re.search(r"(\S*)$", line[start:m.start()]).group(1).lower().lstrip("(\"'«“")
            if w in ABBREV or re.fullmatch(r"[a-z]", w) or (w.isdigit() and nxt.isdigit()):
                continue
        if nxt.islower() and not lower_ok:
            continue
        out.append((line[start:m.start(1)], m.group(1)))
        start = m.end()
    out.append((line[start:], ""))
    return out


def _cut_long(sent: str, max_words: int) -> list[tuple[str, str]]:
    """Cut a sentence longer than max_words at the ; : or , nearest the middle (else at a word boundary)."""
    if len(sent.split()) <= max_words:
        return [(sent, "")]
    mid = len(sent) // 2
    for pat in (r"[;:]\s+", r",\s+", r"\s+"):
        cuts = [m for m in re.finditer(pat, sent) if 0 < m.start() < len(sent) - 1]
        if cuts:
            m = min(cuts, key=lambda m: abs(m.start() - mid))
            left = sent[:m.start() + (0 if pat == r"\s+" else 1)]
            ws = sent[len(left):m.end()]
            right = sent[m.end():]
            a, b = _cut_long(left, max_words), _cut_long(right, max_words)
            a[-1] = (a[-1][0], ws)
            return a + b
    return [(sent, "")]


def split_text(text: str, lower_ok: bool = False, max_words: int = 60) -> list[tuple[str, str]]:
    """Text -> [(kind, piece)]: kind "t" = sentence to translate, "k" = kept verbatim (newlines, whitespace, list
    markers, pieces without letters). "".join(piece) == text, and join_text(parts, identity) == text."""
    parts: list[tuple[str, str]] = []
    for line in re.split(r"(\n+)", text):
        if not line:
            continue
        if line.startswith("\n"):
            parts.append(("k", line))
            continue
        if (m := MARKER.match(line)):
            parts.append(("k", m.group(0)))
            line = line[m.end():]
        lead = line[:len(line) - len(line.lstrip())]
        body = line.strip()
        trail = line[len(lead) + len(body):]
        if lead:
            parts.append(("k", lead))
        if body:
            for sent, ws in _sentences(body, lower_ok):
                for piece, pws in _cut_long(sent, max_words):
                    if piece:
                        parts.append(("t" if LETTER.search(piece) else "k", piece))
                    if pws:
                        parts.append(("k", pws))
                if ws:
                    parts.append(("k", ws))
        if trail:
            parts.append(("k", trail))
    return parts


def join_text(parts: list[tuple[str, str]], tr) -> str:
    """Rebuild a text; tr maps each "t" piece to its translation (dict or callable)."""
    f = tr if callable(tr) else tr.__getitem__
    return "".join(f(p) if k == "t" else p for k, p in parts)


# ----------------------------------------------------------------------------------------------- io / resume
def read_jsonl(path) -> list[dict]:
    """Rows of a jsonl file; a torn last line (crash during a write) is skipped."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def output_files(out_dir: Path, stem: str) -> list[Path]:
    return sorted({Path(p) for p in [str(out_dir / f"{stem}.jsonl")] + glob.glob(str(out_dir / f"{stem}.s*of*.jsonl"))
                   if Path(p).exists()})


def done_ids(out_dir: Path, stem: str) -> set[str]:
    return {r["id"] for f in output_files(out_dir, stem) for r in read_jsonl(f) if "id" in r}


class Appender:
    """Append-only jsonl writer, flushed after each chunk; repairs a torn last line first."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size:
            with path.open("rb") as f:
                f.seek(-1, 2)
                torn = f.read(1) != b"\n"
            if torn:
                with path.open("a") as f:
                    f.write("\n")
        self.f = path.open("a")

    def write(self, rows):
        for r in rows:
            self.f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def shard_of(id_: str, n: int) -> int:
    return zlib.crc32(id_.encode()) % n


def parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 0, 1
    i, n = (int(x) for x in s.split("/"))
    assert 0 <= i < n, s
    return i, n


def pending(items: list[dict], out_dir: Path, stem: str, shard: tuple[int, int], limit: int | None) -> list[dict]:
    done = done_ids(out_dir, stem)
    i, n = shard
    todo = [r for r in items if shard_of(r["id"], n) == i and r["id"] not in done]
    print(f"{stem}: {len(items):,} items, shard {i}/{n}: {len(todo):,} to do ({len(done):,} ids already done)")
    return todo[:limit] if limit else todo


# ----------------------------------------------------------------------------------------------- language id
_MANDING: list[str] | None = None


def _clean_cfg() -> dict:
    p = Path("configs/clean.yaml")
    return yaml.safe_load(p.open()) if p.exists() else {}


def manding_labels() -> list[str]:
    global _MANDING
    if _MANDING is None:
        _MANDING = _clean_cfg().get("langid", {}).get("manding_labels") or \
            ["bam_Latn", "dyu_Latn", "mnk_Latn", "emk_Latn", "mlq_Latn", "mku_Latn", "kao_Latn"]
    return _MANDING


def lid(texts: list[str], k: int = 5) -> list[dict[str, float]]:
    """GlotLID top-k per text. Uses the low-level fastText call (no numpy, so it also works with numpy 2)."""
    from bslm import langid
    m = langid._model()
    out = []
    for t in texts:
        t1 = " ".join((t or "").split())
        out.append({lab.removeprefix("__label__"): float(p) for p, lab in m.f.predict(t1 + "\n", k, 0.0, "strict")}
                   if t1 else {})
    return out


def lid_scores(pred: dict[str, float]) -> dict[str, float]:
    return {"bam": round(sum(pred.get(x, 0.0) for x in manding_labels()), 4),
            "fr": round(pred.get("fra_Latn", 0.0), 4), "en": round(pred.get("eng_Latn", 0.0), 4),
            "top": max(pred, key=pred.get) if pred else ""}


def try_lid(texts: list[str]) -> list[dict] | None:
    """lid_scores per text, or None when fastText/GlotLID is unavailable (filter computes them later)."""
    try:
        return [lid_scores(p) for p in lid(texts)]
    except Exception as e:  # noqa: BLE001 - langid is optional at generation time
        print(f"langid unavailable here ({type(e).__name__}: {e}); `filter` will compute it")
        return None


# ----------------------------------------------------------------------------------------------- backends
def _strip_output(t: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).replace("<think>", "").replace("</think>", "")
    for line in t.split("\n"):
        if line.strip():
            return line.strip()
    return ""


class Backend:
    """generate(user_messages, max_new, sample) -> [(text, hit_max_tokens)]; counts generated tokens."""
    name = "?"

    def __init__(self, model_dir: str, gcfg: dict, seed: int = 0):
        self.model_dir, self.gcfg, self.seed = model_dir, gcfg, seed
        self.eos = list(gcfg["eos_token_ids"])
        self.n_gen = 0
        self.tok = None

    def ids(self, user: str) -> list[int]:
        s = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return self.tok.encode(s, add_special_tokens=False)

    def n_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def generate(self, prompts: list[str], max_new: list[int], sample: bool,
                 rep: float = 1.0) -> list[tuple[str, bool]]:
        raise NotImplementedError


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_dir, gcfg, seed=0, dtype="float16"):
        super().__init__(model_dir, gcfg, seed)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        self.dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype)).to(self.dev).eval()
        self.pad = self.eos[-1]
        h = gcfg.get("hf", {})
        self.bs, self.btok = int(h.get("batch_size", 64)), int(h.get("batch_tokens", 40000))
        torch.manual_seed(seed)
        print(f"hf backend: {model_dir} on {self.dev} ({dtype}), batch {self.bs}, batch_tokens {self.btok}")

    def generate(self, prompts, max_new, sample, rep=1.0):
        from transformers import GenerationConfig
        torch = self.torch
        ids = [self.ids(p) for p in prompts]
        order = sorted(range(len(ids)), key=lambda j: -(len(ids[j]) + max_new[j]))
        res: list = [None] * len(ids)
        s = self.gcfg["sample"]
        gc = GenerationConfig(do_sample=sample, temperature=s["temperature"] if sample else None,
                              top_p=s["top_p"] if sample else None, top_k=0 if sample else None,
                              repetition_penalty=rep if rep != 1.0 else None,
                              eos_token_id=self.eos, pad_token_id=self.pad)
        i = 0
        while i < len(order):
            longest = len(ids[order[i]]) + max_new[order[i]]
            b = order[i:i + max(1, min(self.bs, self.btok // longest))]
            i += len(b)
            L, mn = max(len(ids[j]) for j in b), max(max_new[j] for j in b)
            inp = torch.full((len(b), L), self.pad, dtype=torch.long)
            att = torch.zeros((len(b), L), dtype=torch.long)
            for r, j in enumerate(b):
                inp[r, L - len(ids[j]):] = torch.tensor(ids[j])
                att[r, L - len(ids[j]):] = 1
            with torch.inference_mode():
                out = self.model.generate(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev),
                                          generation_config=gc, max_new_tokens=mn)
            gen = out[:, L:].tolist()
            for r, j in enumerate(b):
                toks = gen[r][:max_new[j]]
                cut = next((k for k, t in enumerate(toks) if t in self.eos), None)
                toks = toks if cut is None else toks[:cut]
                self.n_gen += len(toks)
                res[j] = (self.tok.decode(toks, skip_special_tokens=True), cut is None)
        return res


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model_dir, gcfg, seed=0, dtype="half"):
        super().__init__(model_dir, gcfg, seed)
        from vllm import LLM
        v = gcfg.get("vllm", {})
        self.llm = LLM(model=model_dir, dtype=dtype, tensor_parallel_size=1, seed=seed,
                       gpu_memory_utilization=float(v.get("gpu_memory_utilization", 0.88)),
                       max_model_len=int(v.get("max_model_len", 2048)))
        self.tok = self.llm.get_tokenizer()
        print(f"vllm backend: {model_dir} ({dtype})")

    def generate(self, prompts, max_new, sample, rep=1.0):
        from vllm import SamplingParams
        s = self.gcfg["sample"]
        sps = [SamplingParams(temperature=s["temperature"] if sample else 0.0, top_p=s["top_p"] if sample else 1.0,
                              max_tokens=m, stop_token_ids=self.eos, skip_special_tokens=True,
                              repetition_penalty=rep) for m in max_new]
        outs = self.llm.generate([{"prompt_token_ids": self.ids(p)} for p in prompts], sps, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            self.n_gen += len(c.token_ids)
            res.append((c.text, c.finish_reason == "length"))
        return res


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model_dir, gcfg, seed=0, dtype=None):
        super().__init__(model_dir, gcfg, seed)
        import mlx.core as mx
        from mlx_lm import load
        mx.random.seed(seed)
        self.model, self.tok = load(model_dir)
        self.bs = int(gcfg.get("mlx", {}).get("batch_size", 16))
        print(f"mlx backend: {model_dir}, batch {self.bs}")

    def generate(self, prompts, max_new, sample, rep=1.0):
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler
        s = self.gcfg["sample"]
        from mlx_lm.sample_utils import make_logits_processors
        kw = {"sampler": make_sampler(temp=s["temperature"], top_p=s["top_p"])} if sample else {}
        if rep != 1.0:
            kw["logits_processors"] = make_logits_processors(repetition_penalty=rep, repetition_context_size=64)
        ids = [self.ids(p) for p in prompts]
        order = sorted(range(len(ids)), key=lambda j: -len(ids[j]))
        res: list = [None] * len(ids)
        for i in range(0, len(order), self.bs):
            b = order[i:i + self.bs]
            r = batch_generate(self.model, self.tok, [ids[j] for j in b], max_tokens=[max_new[j] for j in b],
                               return_token_ids=True, **kw)
            for j, text, toks in zip(b, r.texts, r.token_ids):
                toks = [t for t in toks if t not in self.eos]
                self.n_gen += len(toks)
                res[j] = (text, len(toks) >= max_new[j])
        return res


def make_backend(name: str, model_dir: str, gcfg: dict, seed: int = 0, dtype: str | None = None) -> Backend:
    if name == "hf":
        return HFBackend(model_dir, gcfg, seed, dtype or "float16")
    if name == "vllm":
        return VLLMBackend(model_dir, gcfg, seed, dtype or "half")
    if name == "mlx":
        return MLXBackend(model_dir, gcfg, seed)
    raise ValueError(name)


# ----------------------------------------------------------------------------------------------- translation
class Translator:
    def __init__(self, backend: Backend, cfg: dict):
        self.b, self.cfg, self.g = backend, cfg, cfg["generation"]
        self.stats: Counter = Counter()

    def translate(self, texts: list[str], src: str, tgt: str, sample: bool) -> list[dict]:
        """Sentence-by-sentence translation. Returns per text {text, truncated, n_sent}."""
        instr = self.cfg["prompts"][f"{src}-{tgt}"]
        splits = [split_text(t, lower_ok=(src == "bam"), max_words=self.g["max_sentence_words"]) for t in texts]
        uniq = list(dict.fromkeys(p for parts in splits for k, p in parts if k == "t"))
        f, cap = self.g["max_new_factor"][tgt], self.g["max_new_cap"]
        max_new = [min(cap, int(f * self.b.n_tokens(s)) + 16) for s in uniq]
        prompts = [instr.format(x=s) for s in uniq]
        outs = self.b.generate(prompts, max_new, sample, self.g.get("repetition_penalty", 1.0)) if uniq else []
        # degenerate outputs (hit max tokens, or a word n-gram looping): retry with sampling + repetition penalty
        bad = [i for i, (t, cut) in enumerate(outs) if cut or looping(t)]
        self.stats["sentences"] += len(uniq)
        self.stats["retried"] += len(bad)
        for _ in range(int(self.g.get("retries", 0))):
            if not bad:
                break
            again = self.b.generate([prompts[i] for i in bad], [max_new[i] for i in bad], True,
                                    self.g.get("retry_repetition_penalty", 1.0))
            still = []
            for i, (t, cut) in zip(bad, again):
                if cut or looping(t):
                    still.append(i)
                else:
                    outs[i] = (t, False)
            bad = still
        for i in bad:
            outs[i] = (outs[i][0], True)
        self.stats["failed"] += len(bad)
        tr = {s: _strip_output(o[0]) for s, o in zip(uniq, outs)}
        tr_cut = {s: o[1] for s, o in zip(uniq, outs)}
        res = []
        for parts in splits:
            ts = [p for k, p in parts if k == "t"]
            res.append({"text": join_text(parts, tr).strip(), "truncated": any(tr_cut[p] for p in ts),
                        "n_sent": len(ts)})
        return res


LOOP = re.compile(r"(?<!\S)(\S+(?:\s+\S+){0,3})(?:\s+\1(?!\S)){3,}")


def looping(text: str) -> bool:
    """A word or word n-gram (n <= 4) repeated 4+ times in a row ("bɛɛ bɛɛ bɛɛ bɛɛ", "ka a kɛ ka a kɛ ...")."""
    return bool(LOOP.search(text))


def chrf(hyp: str, ref: str) -> float:
    import sacrebleu
    return round(sacrebleu.sentence_chrf(hyp, [ref]).score, 2)


def ratio(tgt: str, src: str) -> float:
    return round(len(tgt) / max(1, len(src)), 3)


class Meter:
    """Throughput and time-left log for one job."""

    def __init__(self, name: str, total: int, backend: Backend, every: float = 60.0):
        self.name, self.total, self.b, self.every = name, total, backend, every
        self.t0 = self.last = time.time()
        self.done, self.tok0, self.first = 0, backend.n_gen, True

    def update(self, n: int, force: bool = False):
        self.done += n
        now = time.time()
        if not (force or self.first or now - self.last >= self.every):
            return
        el = now - self.t0
        ips, tps = self.done / max(el, 1e-9), (self.b.n_gen - self.tok0) / max(el, 1e-9)
        left = (self.total - self.done) / max(ips, 1e-9)
        tag = "ESTIMATE" if self.first else "progress"
        print(f"[{self.name}] {tag}: {self.done:,}/{self.total:,} items | {ips:.2f} items/s | {tps:,.0f} gen tok/s"
              f" | elapsed {el / 3600:.2f} h | time left ≈ {left / 3600:.2f} h", flush=True)
        self.first, self.last = False, now


def _deadline(a) -> float:
    if a.stop_at:
        return float(a.stop_at)
    return time.time() + a.max_hours * 3600 if a.max_hours else float("inf")


def _shard_suffix(shard):
    return "" if shard == (0, 1) else f".s{shard[0]}of{shard[1]}"


def run_bt(a, cfg) -> int:
    in_dir, out_dir = Path(a.in_dir), Path(a.out_dir)
    shard = parse_shard(a.shard)
    todo = pending(read_jsonl(in_dir / "bt_sources.jsonl"), out_dir, "bt_out", shard, a.limit)
    if not todo:
        return 0
    b = make_backend(a.backend, a.model or cfg["model"], cfg["generation"], a.seed, a.dtype)
    T, sample = Translator(b, cfg), a.decode == "sample"
    w = Appender(out_dir / f"bt_out{_shard_suffix(shard)}.jsonl")
    meter, dl, last = Meter(f"bt s{shard[0]}of{shard[1]}", len(todo), b), _deadline(a), 0.0
    for i in range(0, len(todo), a.chunk):
        if time.time() + 1.2 * last > dl:
            print(f"bt: stopping before the deadline ({i:,}/{len(todo):,} done); rerun to resume")
            w.close()
            return 3
        t0, ch = time.time(), todo[i:i + a.chunk]
        bam = [r["text"] for r in ch]
        fr = T.translate(bam, "bam", "fr", sample)
        en = T.translate(bam, "bam", "en", sample)
        rt = T.translate([x["text"] for x in fr], "fr", "bam", False)
        L = try_lid([x for r, f, e, t in zip(bam, fr, en, rt) for x in (r, f["text"], e["text"], t["text"])])
        rows = []
        for k, (r, f, e, t) in enumerate(zip(ch, fr, en, rt)):
            rows.append({"id": r["id"], "source": r["source"], "license": r["license"],
                         "register": r.get("register"), "bam": r["text"], "fr": f["text"], "en": e["text"],
                         "rt_bam": t["text"], "rt_chrf": chrf(t["text"], r["text"]),
                         "lid": None if L is None else {"bam": L[4 * k]["bam"], "fr": L[4 * k + 1]["fr"],
                                                        "en": L[4 * k + 2]["en"], "rt_bam": L[4 * k + 3]["bam"]},
                         "len_ratio": {"fr": ratio(f["text"], r["text"]), "en": ratio(e["text"], r["text"])},
                         "truncated": {"fr": f["truncated"], "en": e["truncated"], "rt": t["truncated"]},
                         "n_sent": f["n_sent"], "decode": a.decode, "backend": b.name,
                         "generator": GENERATOR, "synthetic": True})
        w.write(rows)
        last = time.time() - t0
        if i == 0:
            for r in rows[:2]:
                print("SAMPLE", json.dumps({k: r[k] for k in ("id", "bam", "fr", "en", "rt_bam", "rt_chrf")},
                                           ensure_ascii=False))
        meter.update(len(ch))
    meter.update(0, force=True)
    print(f"translator: {dict(T.stats)} (retried = degenerate first output; failed = still bad, marked truncated)")
    w.close()
    return 0


def chat_items(in_dir: Path) -> list[dict]:
    items = []
    for r in read_jsonl(in_dir / "oasst2_threads.jsonl"):
        items.append({"id": r["id"], "source": "oasst2", "license": LICENCES["oasst2"], "src_lang": r["lang"],
                      "kind": "thread", "parts": [{"name": f"m{i}", "role": m["role"], "src": m["content"]}
                                                  for i, m in enumerate(r["messages"])]})
    for r in read_jsonl(in_dir / "dolly_grounded.jsonl"):
        items.append({"id": r["id"], "source": "dolly", "license": LICENCES["dolly"], "src_lang": r["lang"],
                      "kind": "grounded", "category": r["category"],
                      "parts": [{"name": "context", "role": "user", "src": r["context"]},
                                {"name": "instruction", "role": "user", "src": r["instruction"]},
                                {"name": "response", "role": "assistant", "src": r["response"]}]})
    return items


def run_chat(a, cfg) -> int:
    in_dir, out_dir = Path(a.in_dir), Path(a.out_dir)
    shard = parse_shard(a.shard)
    items = chat_items(in_dir)
    if a.sources:
        items = [r for r in items if r["source"] in a.sources.split(",")]
    todo = pending(items, out_dir, "chat_bam", shard, a.limit)
    if not todo:
        return 0
    b = make_backend(a.backend, a.model or cfg["model"], cfg["generation"], a.seed, a.dtype)
    T, sample = Translator(b, cfg), a.decode == "sample"
    w = Appender(out_dir / f"chat_bam{_shard_suffix(shard)}.jsonl")
    meter, dl, last = Meter(f"chat s{shard[0]}of{shard[1]}", len(todo), b), _deadline(a), 0.0
    chunks, cur, n = [], [], 0              # chunks of about a.chunk_chars source characters
    for r in todo:
        cur.append(r)
        n += sum(len(p["src"]) for p in r["parts"])
        if n >= a.chunk_chars:
            chunks.append(cur)
            cur, n = [], 0
    chunks += [cur] if cur else []
    done = 0
    for ch in chunks:
        if time.time() + 1.2 * last > dl:
            print(f"chat: stopping before the deadline ({done:,}/{len(todo):,} done); rerun to resume")
            w.close()
            return 3
        t0 = time.time()
        out: dict[tuple[int, int], dict] = {}
        for lang in sorted({r["src_lang"] for r in ch}):
            keys = [(i, j) for i, r in enumerate(ch) if r["src_lang"] == lang for j in range(len(r["parts"]))]
            src = [ch[i]["parts"][j]["src"] for i, j in keys]
            fw = T.translate(src, lang, "bam", sample)
            bw = T.translate([x["text"] for x in fw], "bam", lang, False)
            for key, s, f, bk in zip(keys, src, fw, bw):
                out[key] = {"bam": f["text"], "rt": bk["text"], "rt_chrf": chrf(bk["text"], s),
                            "len_ratio": ratio(f["text"], s), "truncated": f["truncated"] or bk["truncated"],
                            "n_sent": f["n_sent"]}
        keys = sorted(out)
        L = try_lid([out[k]["bam"] for k in keys])
        for k, sc in zip(keys, L or [None] * len(keys)):
            out[k]["lid_bam"] = None if sc is None else sc["bam"]
        rows = []
        for i, r in enumerate(ch):
            rows.append({**{k: v for k, v in r.items() if k != "parts"},
                         "parts": [{**p, **out[(i, j)]} for j, p in enumerate(r["parts"])],
                         "decode": a.decode, "backend": b.name, "generator": GENERATOR, "synthetic": True})
        w.write(rows)
        if done == 0:
            print("SAMPLE", json.dumps({"id": rows[0]["id"], "parts": [{k: p[k] for k in ("role", "src", "bam",
                                        "rt_chrf")} for p in rows[0]["parts"][:2]]}, ensure_ascii=False))
        last = time.time() - t0
        done += len(ch)
        meter.update(len(ch))
    meter.update(0, force=True)
    print(f"translator: {dict(T.stats)} (retried = degenerate first output; failed = still bad, marked truncated)")
    w.close()
    return 0


# ----------------------------------------------------------------------------------------------- prep-bt
STD = re.compile(r"[ɛɔɲŋƐƆƝŊ]")
FR_ACCENT = re.compile(r"[èéêëòóôöàâùûçîï]", re.I)
URL = re.compile(r"https?://|www\.|\.(com|org|net|fr|ml)\b|@\w", re.I)
LIST_NUM = re.compile(r"^\(?\d{1,3}[.)]\s+")
# Single strong religious markers (Christian / Islamic scripture register). Common first names (Musa, Ibrahima,
# Mariyamu, Dawuda) and everyday blessings with "Ala" are NOT markers; build_cpt._religious is applied as well.
REL_STRONG = re.compile(r"\b(Yesu|Yezu|Krista|Krisita|Kirisita|Jehova|Jehowa|Masaba|Alakitabu|Bibulu|Kurana|Kuran|"
                        r"Isirayɛli\w*|Isirayeli\w*|Farisi\w*|apotiri\w*|kerecɛn\w*|Ala\s+ka\s+kuma|"
                        r"Ala\s+ka\s+masaya|Kibaru\s+duman|Hadisi|surat\w*|Zezi|Kris|alakira\w*|kiraya)\b", re.I)
VERSE = re.compile(r"\b\d{1,3}\s?:\s?\d{1,3}(?:\s?[-–]\s?\d{1,3})?\b")
ALA = re.compile(r"\bAla\b")
CODESWITCH = re.compile(r"\b(c'est|est|les|des|une|pour|dans|qui|avec|pas|vous|nous|mais|sont|cette|sur|par|il|"
                        r"elle|oui|alors|voilà|parce|the|and|of|is|that|this|with|you)\b", re.I)


TONE = re.compile(r"(?<=[aiuAIUɛɔƐƆ])[\u0300\u0301\u0302\u030c]")


def strip_tones(text: str) -> str:
    """Remove tone marks from a, i, u, ɛ, ɔ (MADLAD has tonal texts such as "Fàkan", "ɛ̀"). Accents on e / o
    (è é ò ó: the old orthography, or French) and ç stay, so the orthography filter still rejects them."""
    import unicodedata
    return unicodedata.normalize("NFC", TONE.sub("", unicodedata.normalize("NFD", text)))


def religious(text: str) -> str | None:
    from bslm.build_cpt import _religious
    if REL_STRONG.search(text):
        return "strong_marker"
    if VERSE.search(text):
        return "verse_ref"
    if len(ALA.findall(text)) >= 3:
        return "ala_x3"
    if _religious(text):
        return "density"
    return None


KK_TAG = re.compile(r"<[A-ZÉÈ_ -]+>")
KK_FR = re.compile(r"__\s*([^_]+?)\s*__")


def kunkado_clean(text: str) -> str | None:
    """Kunkado transcripts mark noises (<RIRES>, <CHEVAUCHEMENT>, ...) and French code-switching (__mais__).
    Noise tags are removed; <INVOCATION> (prayer) drops the utterance; one short French span (<= 2 words, a
    discourse marker such as "__Mais__") is kept unmarked; more French drops the utterance."""
    if "<INVOCATION>" in text:
        return None
    spans = KK_FR.findall(text)
    if len(spans) > 1 or any(len(x.split()) > 2 for x in spans) or text.count("__") > 2 * len(spans):
        return None
    text = KK_FR.sub(lambda m: m.group(1), KK_TAG.sub(" ", text))
    return re.sub(r"\s{2,}", " ", text).strip()


def codeswitch(text: str) -> bool:
    h = len(CODESWITCH.findall(text))
    return h >= 2 or h / max(1, len(text.split())) >= 0.1


def bt_segments(para: str, min_w: int, max_w: int, max_s: int) -> list[str]:
    """Same as v3data.hplt.segments (1-max_s sentences, min_w-max_w words), with the word limits as parameters."""
    from bslm.decontam import sentences
    out, buf = [], []
    for s in sentences(para):
        if len(s.split()) > max_w:
            buf = []
            continue
        buf.append(s)
        while len(buf) > max_s or sum(len(x.split()) for x in buf) > max_w:
            buf.pop(0)
        if sum(len(x.split()) for x in buf) >= min_w:
            out.append(" ".join(buf))
            buf = []
    return out


def _train_pair_texts():
    """Supervised training text (parallel_train + SFT v2 messages). CPT monolingual text is deliberately NOT used:
    MADLAD and Kunkado are in the CPT mix, and back-translating text the model has read is the point of BT."""
    for r in read_jsonl_iter("data/train/parallel_train.jsonl"):
        for k in ("src_text", "tgt_text"):
            if r.get(k):
                yield r[k]
    for r in read_jsonl_iter("data/sft_v2/train.jsonl"):
        for m in r.get("messages", []):
            yield m["content"]


def read_jsonl_iter(path):
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def run_prep_bt(a, cfg) -> int:
    from bslm import langid
    from bslm.clean import _quality_reason
    from bslm.dedup import boilerplate_paragraphs, exact_dedup, hash_key, near_dedup
    from bslm.pii import scrub
    from bslm.text_norm import normalize, script_of
    from bslm.decontam import sentences
    from bslm.v3data import TrainMatcher, eval_hit, eval_index

    pc, cc = cfg["prep_bt"], _clean_cfg()
    conf, q, ms_min = cc["normalize"]["confusables"], cc["quality"], cc["langid"]["manding_min"]
    man = manding_labels()
    min_w, max_w, max_s = pc["min_words"], pc["max_words"], pc["max_sentences"]
    rng = random.Random(a.seed)
    in_dir, interim = Path(a.in_dir), Path(a.interim)
    stats: dict[str, dict] = {}

    def log(step, rows):
        stats[step] = dict(Counter(r["source"] for r in rows))
        print(f"{step:34s} {stats[step]}", flush=True)

    segs = []
    for r in read_jsonl_iter(in_dir / "mono_bam.jsonl"):
        if min_w <= len(r["text"].split()) <= max_w:
            segs.append({"id": r["id"], "text": r["text"], "source": r["source"], "register": "web", "pre": True})
    for r in read_jsonl_iter(interim / "kunkado.jsonl"):
        if r.get("orig_split") != "train":          # the Kunkado test split stays held out
            continue
        t = kunkado_clean(r["text"])
        if t is None:
            continue
        for k, s in enumerate(bt_segments(normalize(t, conf), min_w, max_w, max_s)):
            segs.append({"id": f"{r['id']}-{k}", "text": s, "source": "kunkado", "register": "spoken", "pre": False})
    docs = [d for d in read_jsonl_iter(interim / "madlad_bm.jsonl")]
    docs = [{**d, "text": normalize(d["text"], conf)} for d in docs]
    docs = [d for d in docs if d["text"].strip() and script_of(d["text"]) != "nko"]
    bp = boilerplate_paragraphs([d["text"] for d in docs], cc["dedup"]["boilerplate_paragraph_min_count"])
    lines = [(d["id"], j, LIST_NUM.sub("", ln.strip())) for d in docs for j, ln in enumerate(d["text"].split("\n"))
             if ln.strip() and hash_key(ln) not in bp and not URL.search(ln)]
    print(f"madlad: {len(docs):,} docs, {len(lines):,} lines after boilerplate/URL")
    preds = langid.predict([x[2] for x in lines])
    for (did, j, ln), p in zip(lines, preds):
        if langid.manding_score(p, man) >= ms_min:
            for k, s in enumerate(bt_segments(ln, min_w, max_w, max_s)):
                segs.append({"id": f"{did}-{j}-{k}", "text": s, "source": "madlad_bm", "register": "web",
                             "doc": did, "pre": False})
    log("1_segments_8-60w", segs)

    new = [s for s in segs if not s["pre"]]
    old = [s for s in segs if s["pre"]]
    new = [s for s in new if _quality_reason(s["text"], q) is None and not URL.search(s["text"])]
    preds = langid.predict([s["text"] for s in new])
    new = [s for s, p in zip(new, preds) if langid.manding_score(p, man) >= ms_min]
    new = [s for s in new if not scrub(s["text"])[1]]
    new = [{**s, "text": strip_tones(s["text"])} for s in new]
    segs = old + new
    log("2_quality_url_langid_pii_tones", segs)
    segs = [s for s in segs if STD.search(s["text"]) and not FR_ACCENT.search(s["text"])]
    log("3_standard_orthography", segs)
    rel = Counter()
    keep = []
    for s in segs:
        why = religious(s["text"])
        if why:
            rel[f"{s['source']}:{why}"] += 1
        else:
            keep.append(s)
    segs = keep
    log("4_religious", segs)
    stats["4_religious_dropped"] = dict(rel)
    segs = [s for s in segs if not codeswitch(s["text"]) and not re.search(r"__|[<>]", s["text"])]  # + blanks, tags
    log("5_french_english_codeswitch", segs)

    prio = {"hplt2_cleaned": 0, "glotcc_v1": 1, "kunkado": 2, "madlad_bm": 3}
    segs.sort(key=lambda s: prio.get(s["source"], 9))
    segs, _ = exact_dedup(segs)
    d = cc["dedup"]
    segs, _ = near_dedup(segs, d["minhash_threshold"], d["minhash_num_perm"], d["shingle_chars"])
    log("6_dedup_exact_near", segs)

    # diversity cap per MADLAD document, then pre-sample (x1.3) before the costly train/eval matching
    per_doc: dict[str, int] = defaultdict(int)
    shuffled = segs[:]
    rng.shuffle(shuffled)
    capped = []
    for s in shuffled:
        if s["source"] == "madlad_bm":
            per_doc[s["doc"]] += 1
            if per_doc[s["doc"]] > pc["madlad_max_per_doc"]:
                continue
        capped.append(s)
    pre = [s for s in capped if s["pre"]]
    kun = [s for s in capped if s["source"] == "kunkado"][:int(pc["kunkado_max"] * 1.3)]
    mad = [s for s in capped if s["source"] == "madlad_bm"]
    mad = mad[:max(0, int((pc["target"] - len(pre) - min(len(kun), pc["kunkado_max"])) * 1.3))]
    segs = pre + kun + mad
    log("7_cap_per_doc_presample", segs)

    sent_of = [(i, s) for i, r in enumerate(segs) for s in sentences(r["text"])]
    tm = TrainMatcher([s for _, s in sent_of])
    bad = {sent_of[j][0] for j in tm.scan(_train_pair_texts())}
    segs = [r for i, r in enumerate(segs) if i not in bad]
    log("8_dedup_vs_train_pairs", segs)
    idx = eval_index()
    ev = Counter()
    keep = []
    for r in segs:
        h = eval_hit(idx, r["text"])
        if h:
            ev[f"{h[0]}:{h[1]}"] += 1
        else:
            keep.append(r)
    segs = keep
    log("9_decontam_eval_flores", segs)
    stats["9_eval_matches_removed"] = dict(ev)

    pre = [s for s in segs if s["pre"]]
    kun = [s for s in segs if s["source"] == "kunkado"][:pc["kunkado_max"]]
    mad = [s for s in segs if s["source"] == "madlad_bm"][:max(0, pc["target"] - len(pre) - len(kun))]
    out = [{"id": s["id"], "text": s["text"], "source": s["source"], "license": LICENCES[s["source"]],
            "register": s["register"]} for s in pre + kun + mad]
    rng.shuffle(out)
    n = _write_jsonl(Path(a.out_dir) / "bt_sources.jsonl", out)
    log("10_written", out)
    stats["written"] = n
    (Path(a.out_dir) / "bt_sources.stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    for r in rng.sample(out, min(5, len(out))):
        print("SAMPLE", json.dumps(r, ensure_ascii=False))
    return 0


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ----------------------------------------------------------------------------------------------- filter
def _drop_bottom(rows: list[dict], key, frac: float, group) -> tuple[list[dict], dict]:
    """Drop the lowest `frac` of rows by key(row), separately in each group(row). Returns (kept, thresholds)."""
    by = defaultdict(list)
    for r in rows:
        by[group(r)].append(r)
    keep, thr = [], {}
    for g, rs in by.items():
        rs = sorted(rs, key=key)
        k = int(len(rs) * frac)
        thr[g] = round(key(rs[k]), 2) if k < len(rs) else None
        keep += rs[k:]
    return keep, thr


def _uniq(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def run_filter(a, cfg) -> int:
    fc, cc = cfg["filter"], _clean_cfg()
    out_dir, rev_dir = Path(a.out_dir), Path(a.review_dir)
    lo, hi = fc["len_ratio"]
    stats: dict = {}
    rng = random.Random(fc["seed"])

    # --- back-translation
    bt = _uniq([r for f in output_files(out_dir, "bt_out") for r in read_jsonl(f)])
    need = [r for r in bt if not r.get("lid")]
    if need:
        L = [lid_scores(p) for p in lid([x for r in need for x in (r["bam"], r["fr"], r["en"], r["rt_bam"])])]
        for k, r in enumerate(need):
            r["lid"] = {"bam": L[4 * k]["bam"], "fr": L[4 * k + 1]["fr"], "en": L[4 * k + 2]["en"],
                        "rt_bam": L[4 * k + 3]["bam"]}
    st = Counter(read=len(bt))
    cand = []
    for r in bt:
        if r["lid"]["bam"] < fc.get("bt_bam_lid_min", fc["bam_lid_min"]):
            st["drop_bam_source_lid"] += 1
            continue
        keep = {}
        for lg in ("fr", "en"):
            why = None
            if fc["drop_truncated"] and r["truncated"][lg]:
                why = "truncated"
            elif not r[lg]:
                why = "empty"
            elif r["lid"][lg] < fc["other_lid_min"]:
                why = "lid"
            elif not lo <= r["len_ratio"][lg] <= hi:
                why = "len_ratio"
            keep[lg] = why is None
            if why:
                st[f"drop_{lg}_{why}"] += 1
        if fc["drop_truncated"] and r["truncated"]["rt"]:
            st["drop_rt_truncated"] += 1
            continue
        if any(keep.values()):
            cand.append({**r, "keep": keep})
        else:
            st["drop_row_both_directions"] += 1
    st["pass_lid_ratio"] = len(cand)
    kept, thr = _drop_bottom(cand, lambda r: r["rt_chrf"], fc["rt_drop_fraction"], lambda r: r["source"])
    st["kept"] = len(kept)
    st["kept_fr"] = sum(r["keep"]["fr"] for r in kept)
    st["kept_en"] = sum(r["keep"]["en"] for r in kept)
    stats["bt"] = {"counts": dict(st), "rt_chrf_threshold_by_source": thr,
                   "kept_by_source": dict(Counter(r["source"] for r in kept))}
    _write_jsonl(out_dir / "bt_out_filtered.jsonl", sorted(kept, key=lambda r: r["id"]))

    # --- chat
    ch = _uniq([r for f in output_files(out_dir, "chat_bam") for r in read_jsonl(f)])
    need = [p for r in ch for p in r["parts"] if p.get("lid_bam") is None]
    if need:
        for p, sc in zip(need, lid([p["bam"] for p in need])):
            p["lid_bam"] = lid_scores(sc)["bam"]
    st = Counter(read=len(ch))
    sw, rej = cc.get("langid", {}).get("short_words", 9), cc.get("langid", {}).get("short_reject_conf", 0.9)
    short = [p for r in ch for p in r["parts"] if len(p["bam"].split()) < sw and p["bam"]]
    for p, pr in zip(short, lid([p["bam"] for p in short])):
        lab = max(pr, key=pr.get) if pr else ""
        p["_short_ok"] = lab in manding_labels() or pr.get(lab, 0) < rej
    cand = []
    for r in ch:
        why = None
        for p in r["parts"]:
            if fc["drop_truncated"] and p["truncated"]:
                why = "truncated"
            elif not p["bam"]:
                why = "empty"
            elif len(p["bam"].split()) >= sw and p["lid_bam"] < fc["bam_lid_min"]:
                why = "lid"
            elif len(p["bam"].split()) < sw and not p.get("_short_ok", True):
                why = "lid_short"
            elif len(p["src"]) >= 20 and not lo <= p["len_ratio"] <= hi:
                why = "len_ratio"
            if why:
                break
        if why:
            st[f"drop_{r['source']}_{why}"] += 1
        else:
            cand.append(r)
    for r in ch:
        for p in r["parts"]:
            p.pop("_short_ok", None)
    st["pass_lid_ratio"] = len(cand)

    def conv_chrf(r):
        w = [len(p["src"]) for p in r["parts"]]
        return sum(p["rt_chrf"] * x for p, x in zip(r["parts"], w)) / max(1, sum(w))

    if fc.get("chat_rt_level", "conversation") == "message":
        thr = {}
        by = defaultdict(list)
        for r in cand:
            by[r["source"]] += [p["rt_chrf"] for p in r["parts"]]
        for g, v in by.items():
            v = sorted(v)
            thr[g] = v[int(len(v) * fc["rt_drop_fraction"])] if v else 0
        kept = [r for r in cand if all(p["rt_chrf"] >= thr[r["source"]] for p in r["parts"])]
    else:
        kept, thr = _drop_bottom(cand, conv_chrf, fc["rt_drop_fraction"], lambda r: r["source"])
    st["kept"] = len(kept)
    stats["chat"] = {"counts": dict(st), "rt_chrf_threshold_by_source": thr, "rt_level": fc.get("chat_rt_level"),
                     "kept_by_source": dict(Counter(r["source"] for r in kept))}
    _write_jsonl(out_dir / "chat_bam_filtered.jsonl", sorted(kept, key=lambda r: r["id"]))
    stats["thresholds"] = fc
    (out_dir / "v3_filter.stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "thresholds"}, indent=1, ensure_ascii=False))

    # --- review sheets
    rev_dir.mkdir(parents=True, exist_ok=True)
    n = fc["review_per_category"]
    cats = {"bt": [], "chat_oasst2": [], "chat_dolly": []}
    for r in kept_bt_rows(out_dir):
        cats["bt"].append({"id": r["id"], "category": f"bt:{r['source']}",
                           "source_text": (f"FR: {r['fr']}" if r["keep"]["fr"] else "FR: (dropped)") + "\n" +
                                          (f"EN: {r['en']}" if r["keep"]["en"] else "EN: (dropped)"),
                           "bambara_text": r["bam"],
                           "scores": json.dumps({"rt_chrf": r["rt_chrf"], "lid": r["lid"], "len_ratio": r["len_ratio"]})})
    for r in kept:
        cats[f"chat_{r['source']}"].append({
            "id": r["id"], "category": f"chat:{r['source']}",
            "source_text": "\n\n".join(f"[{p['name']}/{p['role']}] {p['src']}" for p in r["parts"]),
            "bambara_text": "\n\n".join(f"[{p['name']}/{p['role']}] {p['bam']}" for p in r["parts"]),
            "scores": json.dumps({"rt_chrf": [p["rt_chrf"] for p in r["parts"]],
                                  "lid_bam": [p["lid_bam"] for p in r["parts"]],
                                  "len_ratio": [p["len_ratio"] for p in r["parts"]]})})
    cols = ["id", "category", "source_text", "bambara_text", "scores", "decision (ok / fix / no)", "corrected", "comment"]
    for cat, rows in cats.items():
        sample = rng.sample(rows, min(n, len(rows)))
        with (rev_dir / f"v3_{cat}_sample.csv").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            for r in sample:
                wr.writerow({**r, "decision (ok / fix / no)": "", "corrected": "", "comment": ""})
        print(f"review sheet: {rev_dir / f'v3_{cat}_sample.csv'} ({len(sample)} rows)")
    return 0


def kept_bt_rows(out_dir: Path) -> list[dict]:
    return read_jsonl(out_dir / "bt_out_filtered.jsonl")


# ----------------------------------------------------------------------------------------------- selftest / plan
def run_selftest(a, cfg) -> int:
    b = make_backend(a.backend, a.model or cfg["model"], cfg["generation"], a.seed, a.dtype)
    T = Translator(b, cfg)
    t0 = time.time()
    bam = ["N bɛ taa sugu la sini.", "Denmisɛnw bɛ kalan kɛ lakɔli la."]
    fr = T.translate(bam, "bam", "fr", False)
    en = T.translate(bam, "bam", "en", True)
    back = T.translate(["La maison est grande.", "The children are playing outside."][:1], "fr", "bam", False)
    back += T.translate(["The children are playing outside."], "en", "bam", False)
    for x in fr + en + back:
        print("selftest:", x)
    ok = all(x["text"] and not x["truncated"] for x in fr + back)
    print(f"selftest {'OK' if ok else 'FAILED'} ({b.name}, {b.n_gen} tokens, {time.time() - t0:.1f}s)")
    return 0 if ok else 1


def run_plan(a, cfg) -> int:
    """Counts and a rough estimate. Output tokens per item are approximated from characters (fr/en ≈ chars/4,
    Bambara ≈ chars/2.6, measured on SFT v2 outputs), throughput from --tok-s (generated tokens/s per GPU)."""
    in_dir, out_dir = Path(a.in_dir), Path(a.out_dir)
    bt = read_jsonl(in_dir / "bt_sources.jsonl")
    bdone = done_ids(out_dir, "bt_out")
    bc = sum(len(r["text"]) for r in bt if r["id"] not in bdone)
    ch = chat_items(in_dir)
    cdone = done_ids(out_dir, "chat_bam")
    cc_ = sum(len(p["src"]) for r in ch if r["id"] not in cdone for p in r["parts"])
    bt_tok = bc / 4 * 2 + bc / 2.6          # fr + en + round-trip Bambara
    ch_tok = (cc_ / 2.6 + cc_ / 4) * 1.5    # Bambara + round trip, x1.5 for loop retries (MLX smoke run)
    h = (bt_tok + ch_tok) / a.tok_s / 3600 / a.gpus
    print(f"plan: bt {len(bt) - len(bdone & {r['id'] for r in bt}):,} rows left ({bt_tok / 1e6:.1f}M output tokens);"
          f" chat {len(ch) - len(cdone):,} items left ({ch_tok / 1e6:.1f}M output tokens)")
    print(f"plan: ≈ {h:.1f} h on {a.gpus} GPU(s) at {a.tok_s:,.0f} generated tok/s per GPU (rough; the jobs print"
          f" a measured ESTIMATE after their first chunk)")
    return 0


# ----------------------------------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bslm.v3data.generate", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, gen=True):
        p.add_argument("--config", default="configs/v3_generate.yaml")
        p.add_argument("--in-dir", default=str(IN_DIR), help="inputs (bt_sources, oasst2_threads, dolly_grounded)")
        p.add_argument("--out-dir", default=str(IN_DIR), help="outputs (append-only, resumable)")
        p.add_argument("--seed", type=int, default=13)
        if gen:
            p.add_argument("--backend", choices=["vllm", "hf", "mlx"], default="hf")
            p.add_argument("--model", default=None, help="merged SFT v2 folder (default: config 'model')")
            p.add_argument("--dtype", default=None, help="hf: float16 (default) / float32; vllm: half")
            p.add_argument("--shard", default=None, help="i/n: process ids with crc32(id) %% n == i")
            p.add_argument("--limit", type=int, default=None)
            p.add_argument("--max-hours", type=float, default=None)
            p.add_argument("--stop-at", type=float, default=None, help="unix time; stop cleanly before it")
        return p

    p = common(sub.add_parser("prep-bt", help="build bt_sources.jsonl (CPU)"), gen=False)
    p.add_argument("--interim", default="data/interim")
    p = common(sub.add_parser("bt", help="back-translation bam -> fr, en (+ fr -> bam round trip)"))
    p.add_argument("--decode", choices=["sample", "greedy"], default="sample")
    p.add_argument("--chunk", type=int, default=512, help="segments per write")
    p = common(sub.add_parser("chat", help="oasst2 + dolly -> Bambara (+ round trip)"))
    p.add_argument("--decode", choices=["sample", "greedy"], default="greedy")
    p.add_argument("--chunk-chars", type=int, default=60000, help="source characters per write")
    p.add_argument("--sources", default=None, help="comma list: oasst2,dolly (default both)")
    p = common(sub.add_parser("filter", help="thresholds -> *_filtered.jsonl, stats, review sheets"), gen=False)
    p.add_argument("--review-dir", default="reports/review")
    common(sub.add_parser("selftest", help="translate a few sentences"))
    p = common(sub.add_parser("plan", help="counts and a rough time estimate"), gen=False)
    p.add_argument("--tok-s", type=float, default=700.0)
    p.add_argument("--gpus", type=int, default=2)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    fn = {"prep-bt": run_prep_bt, "bt": run_bt, "chat": run_chat, "filter": run_filter, "selftest": run_selftest,
          "plan": run_plan}[a.cmd]
    return fn(a, cfg)


if __name__ == "__main__":
    sys.exit(main())
