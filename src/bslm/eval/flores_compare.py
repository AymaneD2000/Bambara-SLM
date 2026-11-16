"""FLORES-200 devtest comparison: our models vs NLLB baselines (1012 sentences, 4 directions).

    PYTHONPATH=src .venv/bin/python -m bslm.eval.flores_compare --system v3|cpt|nllb|nllb_ft
    PYTHONPATH=src .venv/bin/python -m bslm.eval.flores_compare --contamination
    PYTHONPATH=src .venv/bin/python -m bslm.eval.flores_compare --summary

- v3 / cpt: MLX, exactly the prompts and decoding of `bslm.eval.baseline` (configs/eval.yaml):
  v3 = chat instruction (eval-baseline --chat); cpt = 5-shot completion prompt with the fixed shots.
- nllb / nllb_ft: HF transformers (MPS if available), beam 4, max_length 256, batch 16, forced target BOS.
- Metrics: sacrebleu corpus BLEU (default 13a) and chrF++ (word_order=2), as in baseline.run_mt.
Outputs reports/eval/flores_devtest/{<system>__<dir>.jsonl, scores.json, contamination.json, summary.md}.
Finished directions are reused (resume) when their sources match.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import sacrebleu
import yaml

FLORES = Path("data/raw/flores200/flores200_dataset/devtest")
OUT = Path("reports/eval/flores_devtest")
CODES = {"bam": "bam_Latn", "fr": "fra_Latn", "en": "eng_Latn"}
DIRS = ["bam-fr", "fr-bam", "bam-en", "en-bam"]
SYSTEMS = {
    "v3": {"path": "runs/sft_v3/merged", "kind": "mlx", "chat": True, "dirs": DIRS},
    "cpt": {"path": "runs/cpt_with_nllb/merged", "kind": "mlx", "chat": False, "dirs": DIRS},
    "nllb": {"path": "facebook/nllb-200-distilled-600M", "kind": "hf", "dirs": DIRS,
             "files": ["config.json", "generation_config.json", "pytorch_model.bin", "tokenizer*",
                       "sentencepiece.bpe.model", "special_tokens_map.json"]},
    # bam<->fr only; root of the repo = checkpoint 14500; skip optimizer/trainer state
    "nllb_ft": {"path": "Youssouf0123/bambara-french-nllb", "kind": "hf", "dirs": ["bam-fr", "fr-bam"],
                "files": ["config.json", "generation_config.json", "model.safetensors", "tokenizer*"]},
}


def flores() -> dict[str, list[str]]:
    out = {lg: (FLORES / f"{c}.devtest").read_text(encoding="utf-8").splitlines() for lg, c in CODES.items()}
    assert len({len(v) for v in out.values()}) == 1, "FLORES files are not aligned"
    return out


def score(hyps: list[str], ref: list[str]) -> dict:
    return {"n": len(hyps), "bleu": round(sacrebleu.corpus_bleu(hyps, [ref]).score, 2),
            "chrf++": round(sacrebleu.corpus_chrf(hyps, [ref], word_order=2).score, 2)}


def _saved(system: str, d: str, src: list[str]) -> list[str] | None:
    f = OUT / f"{system}__{d}.jsonl"
    if f.exists():
        rows = [json.loads(line) for line in f.open()]
        if [r["src"] for r in rows] == src:
            return [r["hyp"] for r in rows]
    return None


def _mlx_translator(spec: dict):
    from mlx_lm import load

    from bslm.eval.baseline import _few_shot, mt_generate, mt_prompts

    cfg = yaml.safe_load(Path("configs/eval.yaml").read_text())
    cfg["_chat"] = spec["chat"]
    model, tok = load(spec["path"])
    shots = {lg: _few_shot(cfg, lg, random.Random(cfg["seed"])) for lg in ("fr", "en")}

    def tr(s_lang: str, t_lang: str, src: list[str]) -> list[str]:
        other = t_lang if s_lang == "bam" else s_lang
        sh = [(a, b) if s_lang == "bam" else (b, a) for a, b in shots[other]]
        return mt_generate(model, tok, cfg, mt_prompts(tok, cfg, s_lang, t_lang, src, sh), src)

    return tr, (model, tok)


def _hf_translator(spec: dict, beams: int = 4, max_length: int = 256, batch: int = 16):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    local = snapshot_download(spec["path"], allow_patterns=spec["files"])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(local)
    model = AutoModelForSeq2SeqLM.from_pretrained(local).to(dev).eval()

    def tr(s_lang: str, t_lang: str, src: list[str]) -> list[str]:
        tok.src_lang = CODES[s_lang]
        bos = tok.convert_tokens_to_ids(CODES[t_lang])
        order = sorted(range(len(src)), key=lambda i: -len(src[i]))  # length-sorted batches, order restored
        out: list[str] = [""] * len(src)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            enc = tok([src[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(dev)
            assert enc["input_ids"][0, 0].item() == tok.convert_tokens_to_ids(CODES[s_lang])
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=beams, max_length=max_length)
            for j, h in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[j] = h.strip()
        return out

    return tr, (model, tok)


def run_system(system: str) -> None:
    spec, data = SYSTEMS[system], flores()
    OUT.mkdir(parents=True, exist_ok=True)
    sf = OUT / "scores.json"
    scores = json.loads(sf.read_text()) if sf.exists() else {}
    tr, t_load = None, time.time()
    for d in spec["dirs"]:
        s_lang, t_lang = d.split("-")
        src, ref = data[s_lang], data[t_lang]
        t0, hyps = time.time(), _saved(system, d, src)
        if hyps is None:
            if tr is None:
                tr, _held = (_mlx_translator if spec["kind"] == "mlx" else _hf_translator)(spec)
                print(f"{system}: loaded {spec['path']} in {time.time() - t_load:.0f}s", flush=True)
                t0 = time.time()  # generation time only (excludes download / load)
            hyps = tr(s_lang, t_lang, src)
            with (OUT / f"{system}__{d}.jsonl").open("w") as f:
                for s, h, r in zip(src, hyps, ref):
                    f.write(json.dumps({"src": s, "hyp": h, "ref": r}, ensure_ascii=False) + "\n")
            secs = round(time.time() - t0, 1)
        else:
            secs = scores.get(system, {}).get(d, {}).get("seconds")
            print(f"{system}:{d}: reusing saved generations")
        scores.setdefault(system, {})[d] = {**score(hyps, ref), "seconds": secs}
        print(f"{system}:{d}: {scores[system][d]}", flush=True)
        sf.write_text(json.dumps(scores, indent=2))
    tr = _held = None
    gc.collect()
    write_summary()


# ---------------------------------------------------------------- contamination
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " " + " ".join(s.split()) + " "


def contamination() -> dict:
    """FLORES devtest sentences found in training data: exact (normalised) match of a whole message/line,
    and containment (normalised, word-bounded) for sentences of >= 6 words."""
    data = flores()
    corpora = {"sft_v3": Path("data/sft_v3/train.jsonl"), "cpt_with_nllb": Path("data/cpt/with_nllb/train.jsonl")}
    res = {}
    for name, p in corpora.items():
        units = []
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if "messages" in r:
                units += [m["content"] for m in r["messages"]]
            else:
                units += r["text"].split("\n")
        exact = {_norm(u) for u in units}
        blob = "\n".join(_norm(u) for u in units)
        res[name] = {}
        for lg, sents in data.items():
            ex = [i for i, s in enumerate(sents) if _norm(s) in exact]
            long_ = [i for i, s in enumerate(sents) if len(s.split()) >= 6]
            cont = [i for i in long_ if _norm(s := sents[i]) in blob]
            res[name][lg] = {"exact": len(ex), "contained_ge6w": len(cont), "checked_ge6w": len(long_),
                             "examples": [sents[i] for i in (cont or ex)[:3]]}
            print(name, lg, {k: v for k, v in res[name][lg].items() if k != "examples"}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contamination.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    write_summary()
    return res


# ---------------------------------------------------------------- summary
NAMES = {"v3": "v3 (ours, Qwen3-1.7B SFT, MLX)", "cpt": "CPT base (ours, 5-shot)",
         "nllb": "NLLB-200-distilled-600M", "nllb_ft": "Youssouf0123/bambara-french-nllb"}


def write_summary() -> None:
    sf, cf = OUT / "scores.json", OUT / "contamination.json"
    scores = json.loads(sf.read_text()) if sf.exists() else {}
    L = ["# FLORES-200 devtest: Bambara model vs NLLB baselines", "",
         "Full FLORES-200 devtest (1012 aligned sentences), directions bam→fr, fr→bam, bam→en, en→bam.",
         "Metrics: sacrebleu corpus BLEU (13a tokenisation) and chrF++ (`word_order=2`), same calls as "
         "`bslm.eval.baseline`. Hypotheses: `<system>__<dir>.jsonl`; re-run with "
         "`PYTHONPATH=src .venv/bin/python -m bslm.eval.flores_compare --system <name>`.", "",
         "## Scores (BLEU / chrF++)", "", "| system | " + " | ".join(DIRS) + " | time (s) |",
         "|---|" + "---|" * (len(DIRS) + 1)]
    for s in [k for k in NAMES if k in scores]:
        cells = [f"{scores[s][d]['bleu']:.2f} / {scores[s][d]['chrf++']:.2f}" if d in scores[s] else "n/a"
                 for d in DIRS]
        secs = sum(v["seconds"] or 0 for v in scores[s].values())
        L.append(f"| {NAMES[s]} | " + " | ".join(cells) + f" | {secs:.0f} |")
    L += ["", "## Setup", "",
          "- **v3** (`runs/sft_v3/merged`): zero-shot chat instruction from `configs/eval.yaml` "
          "(`Traduis en français : {x}` / `Traduis en bambara : {x}` / `Translate into English: {x}`), chat template "
          "with thinking disabled, greedy, max tokens = 2 × source tokens + 16, batch 16, first output line kept "
          "(`mt_prompts` / `mt_generate` in `bslm.eval.baseline`, the `eval-baseline --chat` path).",
          "- **CPT base** (`runs/cpt_with_nllb/merged`, if present): the repo's 5-shot completion prompt with the "
          "fixed shots (fr shots: Bayelemabaga train; en shots: FLORES *dev* via SIB-200 train, not devtest).",
          "- **NLLB** (`facebook/nllb-200-distilled-600M`) and **NLLB-FT** (`Youssouf0123/bambara-french-nllb`, root "
          "= checkpoint 14500, bam↔fr only): HF transformers on MPS, fp32, codes bam_Latn / fra_Latn / eng_Latn, "
          "`forced_bos_token_id` = target code, beam 4, max_length 256, batch 16.", "",
          "## Caveats", "",
          "- Decoding differs by design: our model uses the repo's fixed greedy decoding, NLLB uses beam 4. "
          "Prompts/decoding were not tuned on devtest for any system.",
          "- NLLB-200 was trained with FLORES-200 as its official evaluation set (dev used for development); "
          "no FLORES training contamination is documented, but the NLLB data pipeline (incl. mined data) is not "
          "auditable here.",
          "- NLLB-FT: its model card lists training data as jeli-asr transcripts + Bayelemabaga + human-reviewed "
          "Bambara corpora and does not mention FLORES; overlap with FLORES cannot be ruled out (not verified).",
          "- The **v2** model is not included: it was trained on FLORES dev+devtest sentences (via the SIB-200 "
          "splits), so its FLORES scores would be contaminated.",
          "- The repo's `sib200_test` scores (reports/eval/sft_v3) are on a 191–204 sentence subset of the same "
          "devtest; these full-devtest numbers supersede them for FLORES comparisons.",
          "- Single reference, corpus-level scores; Bambara orthography varies, which depresses BLEU into Bambara.",
          "- v3 reproduces the repo's eval up to batch numerics: on the 79 devtest sentences shared with "
          "`sib200_test`, 39/79 bam→fr hypotheses are identical to reports/eval/sft_v3 and chrF++ on that overlap "
          "is 26.3 vs 26.7 (MLX batched greedy decoding depends on batch composition/padding).",
          "- Times are generation only (MPS / MLX on a 16 GB M-series Mac, under memory pressure); NLLB beam search "
          "is slowed by runaway repetitions up to max_length (e.g. 34/1012 fr→bam outputs > 2× source length + 5).",
          "- CPT base (`--system cpt`) is implemented but was not run for this report (time budget).", "",
          "## Licences", "",
          "- facebook/nllb-200-distilled-600M: CC-BY-NC-4.0 (used for evaluation only).",
          "- Youssouf0123/bambara-french-nllb: CC-BY-NC-4.0 (fine-tune of NLLB; evaluation only).",
          "- FLORES-200: CC-BY-SA-4.0."]
    if cf.exists():
        c = json.loads(cf.read_text())
        L += ["", "## Contamination check (FLORES devtest vs our training data)", "",
              "Normalised (NFC, lower-case, punctuation stripped) match. *exact* = a whole message / line equals "
              "the sentence; *contained* = the sentence (≥ 6 words) appears word-bounded inside any message / line. "
              "Positive control: the same check finds ~305 devtest sentences per language in data/sft_v2/train.jsonl.",
              "", "| corpus | lang | exact | contained (≥6 words) / checked |", "|---|---|---|---|"]
        for corp, per in c.items():
            for lg, v in per.items():
                L.append(f"| {corp} | {lg} | {v['exact']} | {v['contained_ge6w']} / {v['checked_ge6w']} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(SYSTEMS))
    ap.add_argument("--contamination", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.contamination:
        contamination()
    if a.system:
        run_system(a.system)
    if a.summary or not (a.system or a.contamination):
        write_summary()
