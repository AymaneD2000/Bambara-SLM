"""Stage 2 supervised fine-tuning (brief §11): LoRA on the merged CPT model, assistant-only loss.

- Chat template: Qwen3 ChatML with thinking DISABLED, identical at training and inference. The loss is computed on the
  assistant answer + <|im_end|> only (system, user and the empty think block are masked).
- Monitoring: SFT validation loss (held-out chat examples) + raw Bambara / French / English perplexity
  (same validation texts as CPT) to catch forgetting.
- Selection: lowest SFT validation loss among checkpoints whose Bambara, French and English perplexity stay within
  `regression_limit` of the starting (CPT) model. Merged into a standard HF model like Stage 1.
- Shares resume / time budget / final-step eval / pruning / NaN guard with Stage 1 (bslm.train.cpt).

Usage: python -m bslm.train.sft --config configs/train/sft_with_nllb.yaml [--dry-run]
       BSLM_BASE_MODEL=<path to merged CPT model> overrides `model` (e.g. a Kaggle input folder).
"""

from __future__ import annotations

import os

# less fragmentation on small GPUs; must be set before torch initialises CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from bslm.train.cpt import Collator, FinalEval, History, NanGuard, Pruner, TimeBudget, _env_report

LANGS = ("bam", "fr", "en")


CHAT_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")


def _chat_token_ids(tok) -> list[int]:
    ids = [tok.convert_tokens_to_ids(t) for t in CHAT_TOKENS]
    assert len(set(ids)) == len(ids) and all(isinstance(i, int) and i > tok.eos_token_id for i in ids), ids
    return ids


def _init_chat_token_rows(model, tok, init: bool = True) -> list[int]:
    """Qwen3-*-Base ships ALL special tokens after <|endoftext|> with one identical, untrained embedding row
    (norm 0.375 vs median ~1.6; tied to lm_head). LoRA on attn/MLP cannot separate them, so the SFT model never
    emits <|im_end|>. Give the chat tokens distinct starting rows (<|im_end|> <- trained <|endoftext|> row, the
    others <- mean row + small noise) and train exactly those rows via PEFT trainable_token_indices (which saves
    absolute row values, so merging onto the unmodified base model reproduces them).
    init=False (train config `init_chat_tokens: false`): the starting model is already chat-trained (e.g. a v3
    stage on top of an SFT'd model), so its rows are kept as they are; the same ids are still trained."""
    ids = _chat_token_ids(tok)
    if not init:
        print("chat token rows kept (init_chat_tokens: false):", dict(zip(CHAT_TOKENS, ids)))
        return ids
    E = model.get_input_embeddings().weight
    with torch.no_grad():
        ref = E[: tok.convert_tokens_to_ids("<|endoftext|>")].float()
        mean, std = ref.mean(0), ref.std(0)
        g = torch.Generator().manual_seed(0)
        for t, i in zip(CHAT_TOKENS, ids):
            row = E[tok.eos_token_id].float() if t == "<|im_end|>" else \
                mean + 0.5 * std * torch.randn(mean.shape, generator=g).to(mean.device)
            E[i] = row.to(E.dtype)
    print("chat token rows initialised:", dict(zip(CHAT_TOKENS, ids)))
    return ids


def _earlier_assistant_spans(tok, ids: list[int], end: int) -> list[tuple[int, int]]:
    """[start, stop) token spans of every assistant turn that starts before position `end` (the final answer's
    prompt length): content + closing <|im_end|>. Found in the token ids: <|im_start|> assistant <newline> ...
    <|im_end|>. An empty think block right after the header (only rendered before the final answer by the Qwen3
    template with enable_thinking=False, but checked anyway) stays masked."""
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    role = tok("assistant", add_special_tokens=False)["input_ids"]
    think = tok("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    spans, i = [], 0
    while i < end:
        if ids[i] == im_start and ids[i + 1: i + 1 + len(role)] == role:
            a = i + 1 + len(role) + 1                     # skip the newline after the role name
            if ids[a: a + len(think)] == think:
                a += len(think)
            b = a
            while b < end and ids[b] != im_end:
                b += 1
            if b < end:                                   # closed turn: content + <|im_end|>
                spans.append((a, b + 1))
            i = b + 1
        else:
            i += 1
    return spans


def _chat_features(tok, max_len: int, all_turns: bool = False):
    """Tokenise chat rows with the Qwen3 template (thinking disabled) and mask everything but the answer.

    all_turns=False (v2): loss on the LAST assistant turn only (its content, <|im_end|> and the trailing newline).
    all_turns=True (train config `loss_all_assistant_turns`): also on every earlier assistant turn (content +
    <|im_end|>). The final turn is labelled exactly as with False, so single-turn rows are identical either way.
    Rows longer than max_len are DROPPED, never truncated: no answer is ever cut and a conversation never loses
    its final answer (multi-turn rows are rare and short; the drop count is printed by load_chat)."""
    def f(batch):
        out = {"input_ids": [], "labels": []}
        texts = batch["text"] if "text" in batch else [None] * len(batch["messages"])
        for msgs, text in zip(batch["messages"], texts):
            if not msgs:   # raw-text replay row ({"text": ...}): plain LM loss on every token, ends with <|endoftext|>
                ids = tok(text, add_special_tokens=False)["input_ids"][: max_len - 1] + [tok.eos_token_id]
                out["input_ids"].append(ids if len(ids) > 8 else []); out["labels"].append(ids if len(ids) > 8 else [])
                continue
            full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(full, add_special_tokens=False)["input_ids"]
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            n_prompt = len(p_ids)
            if len(ids) > max_len or n_prompt >= len(ids) or (all_turns and ids[:n_prompt] != p_ids):
                out["input_ids"].append([]); out["labels"].append([])   # dropped below (too long / inconsistent)
                continue
            labels = [-100] * n_prompt + ids[n_prompt:]
            if all_turns:
                for a, b in _earlier_assistant_spans(tok, ids, n_prompt):
                    labels[a:b] = ids[a:b]
            out["input_ids"].append(ids)
            out["labels"].append(labels)
        return out
    return f


def _select_keys(cfg: dict) -> list[str]:
    """`select_key`: one eval metric or a '+'-joined sum, e.g. eval_sft_loss+eval_chat_loss (default eval_sft_loss)."""
    return [k.strip() for k in cfg.get("select_key", "eval_sft_loss").split("+")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.dry_run:
        cfg.update(cfg.get("dry_run_overrides", {}))
    cfg["model"] = os.environ.get("BSLM_BASE_MODEL") or cfg["model"]
    out = Path(os.environ.get("BSLM_RUN_DIR") or cfg["output_dir"])
    set_seed(cfg["seed"])

    cuda = torch.cuda.is_available()
    native_bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    if cuda and cfg.get("bf16", True) and native_bf16:
        dtype, amp = torch.bfloat16, "bf16"
    elif cuda:
        dtype, amp = torch.float32, "fp16"
    else:
        dtype, amp = torch.float32, None
    if cfg["batch_size"] == "auto":
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0
        cfg["batch_size"] = 16 if mem >= 70 else 8 if mem >= 35 else 4
        cfg["grad_accum"] = max(1, cfg["examples_per_step"] // cfg["batch_size"])
    # evaluation batch: the raw-text forgetting check uses long sequences (raw_eval_seq_len), whose logits
    # (batch x seq x 152k vocab, fp32) are large: 4 x 2048 tokens is ~5 GB. Keep it small on small GPUs.
    if cfg.get("eval_batch_size", "auto") == "auto":
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0
        cfg["eval_batch_size"] = 4 if mem >= 70 else 2 if mem >= 35 else 1
    cfg["_resolved"] = {"base_model": cfg["model"], "weights_dtype": str(dtype), "mixed_precision": amp,
                        "batch_size": cfg["batch_size"], "grad_accum": cfg["grad_accum"],
                        "eval_batch_size": cfg["eval_batch_size"], "output_dir": str(out),
                        "gpu": torch.cuda.get_device_name(0) if cuda else None}
    print("resolved settings:", json.dumps(cfg["_resolved"]))
    _env_report(out, cfg)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype,
                                                 attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lc = cfg["lora"]
    chat_ids = _init_chat_token_rows(model, tok, init=cfg.get("init_chat_tokens", True))
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM",
                                             trainable_token_indices={"embed_tokens": chat_ids}))
    model.print_trainable_parameters()

    sft_dir, raw_dir = Path(cfg["data_dir"]), Path(cfg["raw_eval_dir"])

    all_turns = bool(cfg.get("loss_all_assistant_turns", False))

    def load_chat(fn, limit=None):
        path = fn if isinstance(fn, Path) else sft_dir / fn     # chat_eval_file: a path (relative to the cwd)
        ds = load_dataset("json", data_files=str(path), split="train")
        if limit:
            ds = ds.shuffle(seed=cfg["seed"]).select(range(min(limit, len(ds))))
        ds = ds.map(_chat_features(tok, cfg["max_len"], all_turns), batched=True, remove_columns=ds.column_names)
        n = len(ds)
        ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
        if n > len(ds):
            print(f"{path}: {n - len(ds)} of {n} rows dropped (longer than max_len={cfg['max_len']})")
        return ds

    def load_raw(fn, limit=None):
        ds = load_dataset("json", data_files=str(raw_dir / fn), split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg["raw_eval_seq_len"], add_special_tokens=False),
                      batched=True, remove_columns=ds.column_names)

    train_ds = load_chat("train.jsonl", cfg.get("limit_examples")).shuffle(seed=cfg["seed"])
    eval_ds = {"sft": load_chat("valid.jsonl", cfg["sft_eval_examples"])}
    if cfg.get("chat_eval_file"):    # multi-turn chat eval set, logged as eval_chat_loss (usable in select_key)
        eval_ds["chat"] = load_chat(Path(cfg["chat_eval_file"]), cfg.get("chat_eval_examples"))
    eval_ds.update({lg: load_raw(f"valid_{lg}.jsonl", cfg["raw_eval_examples"]) for lg in LANGS})
    sel = _select_keys(cfg)
    missing = [k for k in sel if k.removeprefix("eval_").removesuffix("_loss") not in eval_ds]
    if missing:
        sys.exit(f"select_key {cfg.get('select_key')!r}: no eval set for {missing} (chat_eval_file missing?)")
    print(f"train examples: {len(train_ds):,}; eval: " + ", ".join(f"{k}={len(v)}" for k, v in eval_ds.items()))

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=cfg["epochs"], max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine", warmup_steps=float(cfg["warmup_ratio"]), weight_decay=cfg.get("weight_decay", 0.0),
        bf16=amp == "bf16", fp16=amp == "fp16", logging_steps=cfg.get("logging_steps", 25),
        eval_strategy="steps", eval_steps=cfg["eval_steps"], save_strategy="steps", save_steps=cfg["eval_steps"],
        save_total_limit=None, report_to="none", seed=cfg["seed"], data_seed=cfg["seed"],
        train_sampling_strategy="group_by_length",  # transformers 5: batches of similar length, less padding
        dataloader_num_workers=cfg.get("num_workers", 2), remove_unused_columns=False,
    )
    hist = History(out / "eval_history.json")
    pruner = Pruner(out / "checkpoints", hist, cfg["regression_limit"], select_key="+".join(sel), limit_langs=LANGS,
                    keep_best_unqualified=cfg.get("fallback_selection") == "best_sft")
    budget = TimeBudget(float(os.environ["BSLM_MAX_HOURS"]) if os.environ.get("BSLM_MAX_HOURS") else None)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=Collator(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
                      callbacks=[hist, pruner, NanGuard(), budget, FinalEval()])

    ckpt_dir = out / "checkpoints"
    resume = ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint-*"))
    start_file = out / "start_eval.json"
    if start_file.exists():
        start = json.loads(start_file.read_text())
    else:
        start = trainer.evaluate()
        start_file.write_text(json.dumps(start, indent=2))
    pruner.start = start
    if resume:
        print(f"resuming from the last checkpoint in {ckpt_dir}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    hours = (time.time() - t0) / 3600
    if budget.hit:
        print("STOPPED ON TIME BUDGET: not finished. Run the same command again (any machine) to resume.")
        sys.exit(3)

    lang_keys = [f"eval_{lg}_loss" for lg in LANGS]
    need = list(dict.fromkeys(["eval_sft_loss"] + sel + lang_keys))
    rows = [r for r in hist.rows if r["step"] > 0 and all(k in r for k in need) and (ckpt_dir / f"checkpoint-{r['step']}").exists()]
    ok = [r for r in rows if pruner._qualified(r)]
    chosen = min(ok, key=pruner.score) if ok else None
    fallback = None
    if not chosen and cfg.get("fallback_selection") == "best_sft" and rows:
        # user decision 2026-11-08: still ship the best-SFT-loss (select_key) checkpoint, flagged in selection.json
        chosen = min(rows, key=pruner.score)
        fallback = f"no checkpoint within the +{cfg['regression_limit']:.0%} limit; fallback = best {'SFT loss' if sel == ['eval_sft_loss'] else pruner.select_key} (step {chosen['step']})"
    extra = [k for k in need if k not in ("eval_sft_loss", *lang_keys)]   # e.g. eval_chat_loss
    summary = {"start": {k.replace("eval_", "").replace("_loss", ""): (math.exp(start[k]) if k in lang_keys else start[k])
                         for k in need if k in start},
               **({"select_key": pruner.select_key} if pruner.select_key != "eval_sft_loss" else {}),
               "regression_limit": cfg["regression_limit"], "train_hours": round(hours, 2),
               "history": [{"step": r["step"], "sft_loss": r["eval_sft_loss"],
                            **{k.removeprefix("eval_"): r[k] for k in extra},
                            **{lg: math.exp(r[f"eval_{lg}_loss"]) for lg in LANGS}} for r in hist.rows if all(k in r for k in need)],
               "chosen_step": chosen["step"] if chosen else None,
               "note": fallback or (None if chosen else "NO checkpoint kept Bambara/French/English perplexity within the limit.")}
    (out / "selection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not chosen:
        sys.exit(2)
    if (out / "NAN_STOP").exists():
        sys.exit("training stopped on a non-finite loss: no model merged")

    del trainer, model
    if cuda:
        torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16 if cuda else torch.float32)
    merged = PeftModel.from_pretrained(base, str(ckpt_dir / f"checkpoint-{chosen['step']}")).merge_and_unload()
    # chat model: stop on <|im_end|> as well as <|endoftext|> (HF generate, MLX, Ollama import read this)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    merged.config.eos_token_id = eos
    merged.generation_config.eos_token_id = eos
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")

    dev = "cuda" if cuda else "cpu"
    merged.eval().to(dev)
    samples = []
    for p in cfg.get("sample_prompts", []):
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt", return_dict=True).to(dev)
        with torch.no_grad():
            y = merged.generate(**ids, max_new_tokens=96, do_sample=False, repetition_penalty=1.1)
        samples.append({"prompt": p, "answer": tok.decode(y[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)})
    (out / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    print(f"done: merged model in {out/'merged'}; chosen step {chosen['step']}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
