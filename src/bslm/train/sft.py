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


def _init_chat_token_rows(model, tok) -> list[int]:
    """Qwen3-*-Base ships ALL special tokens after <|endoftext|> with one identical, untrained embedding row
    (norm 0.375 vs median ~1.6; tied to lm_head). LoRA on attn/MLP cannot separate them, so the SFT model never
    emits <|im_end|>. Give the chat tokens distinct starting rows (<|im_end|> <- trained <|endoftext|> row, the
    others <- mean row + small noise) and train exactly those rows via PEFT trainable_token_indices (which saves
    absolute row values, so merging onto the unmodified base model reproduces them)."""
    ids = [tok.convert_tokens_to_ids(t) for t in CHAT_TOKENS]
    assert len(set(ids)) == len(ids) and all(isinstance(i, int) and i > tok.eos_token_id for i in ids), ids
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


def _chat_features(tok, max_len: int):
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
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
            if len(ids) > max_len or n_prompt >= len(ids):
                out["input_ids"].append([]); out["labels"].append([])   # dropped below (too long)
                continue
            out["input_ids"].append(ids)
            out["labels"].append([-100] * n_prompt + ids[n_prompt:])
        return out
    return f


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
    chat_ids = _init_chat_token_rows(model, tok)
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM",
                                             trainable_token_indices={"embed_tokens": chat_ids}))
    model.print_trainable_parameters()

    sft_dir, raw_dir = Path(cfg["data_dir"]), Path(cfg["raw_eval_dir"])

    def load_chat(fn, limit=None):
        ds = load_dataset("json", data_files=str(sft_dir / fn), split="train")
        if limit:
            ds = ds.shuffle(seed=cfg["seed"]).select(range(min(limit, len(ds))))
        ds = ds.map(_chat_features(tok, cfg["max_len"]), batched=True, remove_columns=ds.column_names)
        return ds.filter(lambda x: len(x["input_ids"]) > 0)

    def load_raw(fn, limit=None):
        ds = load_dataset("json", data_files=str(raw_dir / fn), split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg["raw_eval_seq_len"], add_special_tokens=False),
                      batched=True, remove_columns=ds.column_names)

    train_ds = load_chat("train.jsonl", cfg.get("limit_examples")).shuffle(seed=cfg["seed"])
    eval_ds = {"sft": load_chat("valid.jsonl", cfg["sft_eval_examples"])}
    eval_ds.update({lg: load_raw(f"valid_{lg}.jsonl", cfg["raw_eval_examples"]) for lg in LANGS})
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
    pruner = Pruner(out / "checkpoints", hist, cfg["regression_limit"], select_key="eval_sft_loss", limit_langs=LANGS)
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

    need = ["eval_sft_loss"] + [f"eval_{lg}_loss" for lg in LANGS]
    rows = [r for r in hist.rows if r["step"] > 0 and all(k in r for k in need) and (ckpt_dir / f"checkpoint-{r['step']}").exists()]
    ok = [r for r in rows if pruner._qualified(r)]
    chosen = min(ok, key=lambda r: r["eval_sft_loss"]) if ok else None
    fallback = None
    if not chosen and cfg.get("fallback_selection") == "best_sft" and rows:
        # user decision 2026-11-08: still ship the best-SFT-loss checkpoint, flagged in selection.json
        chosen = min(rows, key=lambda r: r["eval_sft_loss"])
        fallback = f"no checkpoint within the +{cfg['regression_limit']:.0%} limit; fallback = best SFT loss (step {chosen['step']})"
    summary = {"start": {k.replace("eval_", "").replace("_loss", ""): (math.exp(start[k]) if k != "eval_sft_loss" else start[k])
                         for k in need},
               "regression_limit": cfg["regression_limit"], "train_hours": round(hours, 2),
               "history": [{"step": r["step"], "sft_loss": r["eval_sft_loss"],
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
