"""Baseline evaluation with MLX on Apple silicon (brief §10).

- MT: few-shot prompting of a base model, greedy decoding, BLEU + chrF++ (sacrebleu).
- Perplexity: per-token on Bambara validation text, and on Belebele passages in bam / fra / eng.
Outputs reports/eval/<run_name>/{mt_scores.json, ppl.json, generations/*.jsonl, summary.md}.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import sacrebleu
import yaml
from mlx_lm import batch_generate, load
from rich.console import Console

console = Console()


def _jsonl(p: str) -> list[dict]:
    return [json.loads(line) for line in open(p)]


def _pairs(eval_set: str, other: str) -> list[tuple[str, str]]:
    """(bambara, other) pairs from data/eval/<set>.jsonl for the requested other language."""
    out = []
    for r in _jsonl(f"data/eval/{eval_set}.jsonl"):
        if r["tgt_lang"] == other:
            out.append((r["src_text"], r["tgt_text"]))
    return out


def _few_shot(cfg: dict, other: str, rng: random.Random) -> list[tuple[str, str]]:
    src = cfg["mt"]["few_shot_source"][other]
    cands = []
    for r in _jsonl(src):
        if r.get("tgt_lang") != other:
            continue
        if (r.get("align_score") if r.get("align_score") is not None else 1.0) < 0.9 or r.get("source", "") == "nllb_mined":
            continue
        if 6 <= len(r["src_text"].split()) <= 25:
            cands.append((r["src_text"], r["tgt_text"]))
    return rng.sample(cands, cfg["mt"]["few_shot"])


def _prompt(src_lang: str, tgt_lang: str, shots: list[tuple[str, str]], text: str, names: dict) -> str:
    a, b = names[src_lang], names[tgt_lang]
    head = f"Translate from {a} to {b}.\n\n"
    body = "".join(f"{a}: {s}\n{b}: {t}\n\n" for s, t in shots)
    return head + body + f"{a}: {text}\n{b}:"


def run_mt(model, tok, cfg: dict, out_dir: Path) -> dict:
    names, rng = cfg["lang_names"], random.Random(cfg["seed"])
    shots = {lg: _few_shot(cfg, lg, random.Random(cfg["seed"])) for lg in ("fr", "en")}
    (out_dir / "few_shot.json").write_text(json.dumps(shots, ensure_ascii=False, indent=1))
    scores = {}
    gdir = out_dir / "generations"
    gdir.mkdir(parents=True, exist_ok=True)
    for es, spec in cfg["mt"]["sets"].items():
        for d in spec["directions"]:
            s_lang, t_lang = d.split("-")
            other = t_lang if s_lang == "bam" else s_lang
            pairs = _pairs(es, other)
            pairs = rng.sample(pairs, min(spec["n"], len(pairs))) if spec["n"] < len(pairs) else pairs
            src = [p[0] if s_lang == "bam" else p[1] for p in pairs]
            ref = [p[1] if s_lang == "bam" else p[0] for p in pairs]
            sh = [(a, b) if s_lang == "bam" else (b, a) for a, b in shots[other]]
            if cfg.get("_chat"):
                # chat models (after SFT): zero-shot instruction through the chat template, thinking disabled
                instr = cfg["mt"]["chat_instruction"][t_lang]
                prompts = [tok.apply_chat_template([{"role": "user", "content": instr.format(x=x)}],
                                                   add_generation_prompt=True, enable_thinking=False) for x in src]
            else:
                prompts = [tok.encode(_prompt(s_lang, t_lang, sh, x, names)) for x in src]
            hyps, t0 = [], time.time()
            B = cfg["mt"]["batch_size"]
            done_file = gdir / f"{es}__{d}.jsonl"
            if done_file.exists():  # resume: reuse a finished set (same fixed sample and prompts)
                prev = [json.loads(line) for line in done_file.open()]
                if [r["src"] for r in prev] == src:
                    hyps = [r["hyp"] for r in prev]
                    console.print(f"{es}:{d}: reusing {len(hyps)} saved generations")
            if hyps:
                prompts = []
            for i in range(0, len(prompts) if not hyps else 0, B):
                batch = prompts[i:i + B]
                mx_tokens = [int(cfg["mt"]["max_new_tokens_factor"] * len(tok.encode(x)) + 16) for x in src[i:i + B]]
                res = batch_generate(model, tok, batch, max_tokens=mx_tokens)
                hyps += [t.replace("<think>", "").replace("</think>", "").strip().split("\n")[0].strip()
                         for t in res.texts]
            bleu = sacrebleu.corpus_bleu(hyps, [ref]).score
            chrf = sacrebleu.corpus_chrf(hyps, [ref], word_order=2).score
            key = f"{es}:{d}"
            scores[key] = {"n": len(src), "bleu": round(bleu, 2), "chrf++": round(chrf, 2),
                           "seconds": round(time.time() - t0, 1)}
            console.print(f"{key}: n={len(src)} BLEU={bleu:.2f} chrF++={chrf:.2f}")
            with (gdir / f"{es}__{d}.jsonl").open("w") as f:
                for s, h, r in zip(src, hyps, ref):
                    f.write(json.dumps({"src": s, "hyp": h, "ref": r}, ensure_ascii=False) + "\n")
    return scores


def _nll(model, tok, text: str, max_tokens: int) -> tuple[float, int]:
    ids = tok.encode(text)[:max_tokens]
    if len(ids) < 2:
        return 0.0, 0
    x = mx.array(ids)[None]
    logits = model(x[:, :-1]).astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = x[:, 1:]
    nll = -mx.take_along_axis(logp, tgt[..., None], axis=-1).sum()
    return float(nll.item()), len(ids) - 1


def run_ppl(model, tok, cfg: dict) -> dict:
    out, rng = {}, random.Random(cfg["seed"])
    p = cfg["ppl"]
    by_reg: dict[str, list[str]] = {}
    for r in _jsonl(p["bam_val"]["file"]):
        by_reg.setdefault(r["register"], []).append(r["text"])
    caps = p["bam_val"]["n_per_register"]
    by_reg = {k: rng.sample(v, min(caps.get(k, 500), len(v))) for k, v in sorted(by_reg.items())}
    for reg, texts in by_reg.items():
        tot, n = 0.0, 0
        for t in texts:
            a, b = _nll(model, tok, t, p["max_tokens"])
            tot, n = tot + a, n + b
        out[f"bam_val_{reg}"] = {"texts": len(texts), "tokens": n, "ppl": round(math.exp(tot / max(n, 1)), 2)}
    bele = _jsonl(p["belebele"]["file"])
    texts = {"bam": [], "fr": [], "en": []}
    seen = set()
    for r in bele:
        if r["src_text"] not in seen:
            seen.add(r["src_text"])
            texts["bam"].append(r["src_text"])
        texts[r["tgt_lang"]].append(r["tgt_text"])
    for lg, ts in texts.items():
        if cfg.get("_max_n"):
            ts = ts[: cfg["_max_n"]]
        tot, n = 0.0, 0
        for t in ts:
            a, b = _nll(model, tok, t, p["max_tokens"])
            tot, n = tot + a, n + b
        out[f"belebele_{lg}"] = {"texts": len(ts), "tokens": n, "ppl": round(math.exp(tot / max(n, 1)), 2)}
    for k, v in out.items():
        console.print(f"ppl {k}: {v['ppl']} ({v['tokens']:,} tokens)")
    return out


def run(config: str = "configs/eval.yaml", run_name: str = "baseline_qwen3_1.7b_base",
        model_path: str | None = None, adapter_path: str | None = None, skip_mt: bool = False,
        chat: bool = False, max_n: int | None = None) -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    cfg["_chat"] = chat
    cfg["_max_n"] = max_n
    if max_n:  # quick checks only (smoke tests); real evaluations use the fixed config
        for spec in cfg["mt"]["sets"].values():
            spec["n"] = min(spec["n"], max_n)
        cfg["ppl"]["bam_val"]["n_per_register"] = {k: min(v, max_n) for k, v in cfg["ppl"]["bam_val"]["n_per_register"].items()}
    out_dir = Path("reports/eval") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = load(model_path or cfg["model"], adapter_path=adapter_path)
    res = {"model": model_path or cfg["model"], "adapter": adapter_path, "config": cfg}
    if (out_dir / "ppl.json").exists():  # resume: perplexity already computed for this run
        res["ppl"] = json.loads((out_dir / "ppl.json").read_text())
        console.print("reusing saved perplexity results")
    else:
        res["ppl"] = run_ppl(model, tok, cfg)
        (out_dir / "ppl.json").write_text(json.dumps(res["ppl"], indent=2))
    if not skip_mt:
        res["mt"] = run_mt(model, tok, cfg, out_dir)
        (out_dir / "mt_scores.json").write_text(json.dumps(res["mt"], indent=2))
    L = [f"# Evaluation: {run_name}", "", f"Model: `{res['model']}`  adapter: `{adapter_path}`", "",
         "## Perplexity (lower is better)", "", "| set | texts | tokens | ppl |", "|---|---|---|---|"]
    L += [f"| {k} | {v['texts']} | {v['tokens']:,} | {v['ppl']} |" for k, v in res["ppl"].items()]
    if "mt" in res:
        L += ["", "## Translation (few-shot, greedy)", "", "| set:direction | n | BLEU | chrF++ |", "|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['bleu']} | {v['chrf++']} |" for k, v in res["mt"].items()]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")
    console.print(f"wrote {out_dir/'summary.md'}")
