"""Stage 1 continued pretraining with transformers + PEFT LoRA (CUDA; also runs on MPS/CPU for dry runs).

Brief §11: LoRA on all linear layers, high rank, cosine schedule, bf16, gradient checkpointing.
Monitoring: validation loss on Bambara, French and English, measured BEFORE training and every `eval_steps`.
Checkpoint selection: lowest Bambara validation loss among checkpoints whose French AND English perplexity stay within
`regression_limit` (e.g. +10%) of the starting model. The selected adapter is merged into the base weights and saved
as a standard Hugging Face model (loadable by mlx-lm on the Mac for evaluation, and the starting point for SFT).

Usage:  python -m bslm.train.cpt --config configs/train/cpt_with_nllb.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)


class Collator:
    """Right-pad a batch; labels = input ids, padding masked with -100 (EOS separators inside examples are learned)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), n), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(feats), n), dtype=torch.long)
        for i, f in enumerate(feats):
            x = torch.tensor(f["input_ids"], dtype=torch.long)
            ids[i, : len(x)] = x
            att[i, : len(x)] = 1
        labels = ids.clone()
        labels[att == 0] = -100
        for i, f in enumerate(feats):           # SFT: per-token labels given (prompt masked with -100)
            if "labels" in f:
                y = torch.tensor(f["labels"], dtype=torch.long)
                labels[i, : len(y)] = y
        return {"input_ids": ids, "attention_mask": att, "labels": labels}


class History(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.rows = json.loads(path.read_text()) if path.exists() else []  # kept across resumes

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        # With several eval datasets the Trainer fires one event per dataset: merge them by step.
        losses = {k: v for k, v in (metrics or {}).items() if k.endswith("_loss")}
        if self.rows and self.rows[-1]["step"] == state.global_step:
            self.rows[-1].update(losses)
        else:
            self.rows.append({"step": state.global_step, **losses})
        self.path.write_text(json.dumps(self.rows, indent=2))


class NanGuard(TrainerCallback):
    """Stop at once if the training loss becomes NaN/inf (e.g. fp16 overflow) instead of wasting GPU hours."""

    def on_log(self, args, state, control, logs=None, **kw):
        loss = (logs or {}).get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            print(f"!!! non-finite training loss at step {state.global_step}: stopping. "
                  "Use a GPU with bf16 (A100/L4) or lower the learning rate.")
            control.should_training_stop = True
            (Path(args.output_dir).parent / "NAN_STOP").write_text(str(state.global_step))


class FinalEval(TrainerCallback):
    """Evaluate and save at the very last step too, so the end of training (lowest learning rate) is a
    selectable checkpoint even when max_steps is not a multiple of eval_steps."""

    def on_step_end(self, args, state, control, **kw):
        if state.max_steps and state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True


class TimeBudget(TrainerCallback):
    """Stop cleanly (with a saved checkpoint) after BSLM_MAX_HOURS of training, so a session limit (Kaggle 12 h,
    Colab) never kills the run between two saves. The run is then resumed in the next session."""

    def __init__(self, hours: float | None):
        self.hours, self.t0, self.hit = hours, time.time(), False

    def on_step_end(self, args, state, control, **kw):
        if self.hours and (time.time() - self.t0) / 3600 >= self.hours:
            print(f"time budget of {self.hours} h reached at step {state.global_step}: saving and stopping")
            control.should_save = True
            control.should_training_stop = True
            self.hit = True


class Pruner(TrainerCallback):
    """Keep only the latest checkpoint (to resume) and the best qualified one so far (lowest Bambara loss with
    French/English perplexity within the regression limit). Caps disk use at ~2 checkpoints (Colab Drive quota).
    select_key may be a '+'-joined sum of eval metrics (e.g. "eval_sft_loss+eval_chat_loss").
    keep_best_unqualified=True (SFT `fallback_selection: best_sft`): ALSO keep the best checkpoint by select_key
    ignoring the regression limit, so the fallback can still pick it at the end (up to 3 checkpoints on disk)."""

    def __init__(self, ckpt_dir: Path, hist: "History", limit: float,
                 select_key: str = "eval_bam_loss", limit_langs: tuple = ("fr", "en"),
                 keep_best_unqualified: bool = False):
        self.ckpt_dir, self.hist, self.limit, self.start = ckpt_dir, hist, limit, None
        self.select_key, self.limit_langs = select_key, limit_langs
        self.keys = [k.strip() for k in select_key.split("+")]
        self.keep_best_unqualified = keep_best_unqualified

    def score(self, r: dict) -> float:
        return sum(r[k] for k in self.keys)

    def _qualified(self, r: dict) -> bool:
        keys = self.keys + [f"eval_{lg}_loss" for lg in self.limit_langs]
        return all(k in r for k in keys) and all(
            math.exp(r[f"eval_{lg}_loss"]) <= (1 + self.limit) * math.exp(self.start[f"eval_{lg}_loss"])
            for lg in self.limit_langs)

    def on_save(self, args, state, control, **kw):
        if self.start is None:
            return
        ok = [r for r in self.hist.rows if r["step"] > 0 and self._qualified(r)
              and (self.ckpt_dir / f"checkpoint-{r['step']}").exists()]
        keep = {f"checkpoint-{state.global_step}"}
        if ok:
            keep.add(f"checkpoint-{min(ok, key=self.score)['step']}")
        if self.keep_best_unqualified:
            rows = [r for r in self.hist.rows if r["step"] > 0 and all(k in r for k in self.keys)
                    and (self.ckpt_dir / f"checkpoint-{r['step']}").exists()]
            if rows:
                keep.add(f"checkpoint-{min(rows, key=self.score)['step']}")
        for d in self.ckpt_dir.glob("checkpoint-*"):
            if d.name not in keep:
                shutil.rmtree(d, ignore_errors=True)


def _env_report(out: Path, cfg: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out / "pip_freeze.txt").write_text(freeze)
    info = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu"), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "env.json").write_text(json.dumps(info, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="tiny run to validate the code path (any device)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.dry_run:
        cfg.update(cfg.get("dry_run_overrides", {}))
    out = Path(os.environ.get("BSLM_RUN_DIR") or cfg["output_dir"])  # e.g. a Google Drive folder on Colab
    set_seed(cfg["seed"])

    cuda = torch.cuda.is_available()
    # precision: native bf16 needs compute capability >= 8.0 (A100, L4, H100). torch.cuda.is_bf16_supported()
    # also returns True for *emulated* bf16 (e.g. T4), which is slow, so we check the capability instead.
    # Older GPUs (T4, cc 7.5): fp32 weights + fp16 mixed precision (fp16 weights risk overflow in Qwen3).
    native_bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    if cuda and cfg.get("bf16", True) and native_bf16:
        dtype, amp = torch.bfloat16, "bf16"
    elif cuda:
        dtype, amp = torch.float32, "fp16"
    else:
        dtype, amp = torch.float32, None
    # batch: "auto" picks the per-device batch from GPU memory and keeps the effective batch fixed
    if cfg["batch_size"] == "auto":
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0
        cfg["batch_size"] = 8 if mem >= 70 else 4 if mem >= 35 else 2 if mem >= 20 else 1
        cfg["grad_accum"] = max(1, cfg["sequences_per_step"] // cfg["batch_size"])
    cfg["_resolved"] = {"weights_dtype": str(dtype), "mixed_precision": amp, "batch_size": cfg["batch_size"],
                        "grad_accum": cfg["grad_accum"], "output_dir": str(out),
                        "gpu": torch.cuda.get_device_name(0) if cuda else None,
                        "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None}
    print("resolved settings:", json.dumps(cfg["_resolved"]))
    _env_report(out, cfg)
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("model_revision"))
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg.get("model_revision"), dtype=dtype,
                                                 attn_implementation=cfg.get("attn_implementation", "sdpa"))
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    data_dir = Path(cfg["data_dir"])
    seq_len = cfg["seq_len"]

    def load(fn, limit=None):
        ds = load_dataset("json", data_files=str(data_dir / fn), split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=seq_len, add_special_tokens=False),
                      batched=True, remove_columns=ds.column_names)

    lim = cfg.get("limit_examples")
    train_ds = load("train.jsonl", lim).shuffle(seed=cfg["seed"])
    eval_ds = {lg: load(f"valid_{lg}.jsonl", cfg.get("limit_eval_examples")) for lg in ("bam", "fr", "en")}

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=cfg["epochs"], max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=float(cfg["warmup_ratio"]),  # transformers 5: a float in [0, 1) is a ratio of total steps
        weight_decay=cfg.get("weight_decay", 0.0),
        bf16=amp == "bf16", fp16=amp == "fp16", logging_steps=cfg.get("logging_steps", 10),
        eval_strategy="steps", eval_steps=cfg["eval_steps"], save_strategy="steps", save_steps=cfg["eval_steps"],
        save_total_limit=None, report_to=cfg.get("report_to", "none"), seed=cfg["seed"], data_seed=cfg["seed"],
        dataloader_num_workers=cfg.get("num_workers", 2), remove_unused_columns=False,
    )
    hist = History(out / "eval_history.json")
    pruner = Pruner(out / "checkpoints", hist, cfg["regression_limit"])
    budget = TimeBudget(float(os.environ["BSLM_MAX_HOURS"]) if os.environ.get("BSLM_MAX_HOURS") else None)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=Collator(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
                      callbacks=[hist, pruner, NanGuard(), budget, FinalEval()])

    ckpt_dir = out / "checkpoints"
    resume = ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint-*"))
    start_file = out / "start_eval.json"
    if start_file.exists():
        start = json.loads(start_file.read_text())       # resumed run: keep the original step-0 reference
    else:
        start = trainer.evaluate()                       # step-0 reference for the regression limit
        start_file.write_text(json.dumps(start, indent=2))
    pruner.start = start
    if resume:
        print(f"resuming from the last checkpoint in {ckpt_dir}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    train_hours = (time.time() - t0) / 3600
    if budget.hit:
        print("STOPPED ON TIME BUDGET: not finished. Run the same command again (any machine) to resume.")
        sys.exit(3)

    # ---- checkpoint selection under the French/English regression limit
    base = {lg: start[f"eval_{lg}_loss"] for lg in ("bam", "fr", "en")}
    limit = cfg["regression_limit"]
    need = [f"eval_{lg}_loss" for lg in ("bam", "fr", "en")]
    rows = [r for r in hist.rows if r["step"] > 0 and all(k in r for k in need)
            and (out / "checkpoints" / f"checkpoint-{r['step']}").exists()]
    ok = [r for r in rows if all(math.exp(r[f"eval_{lg}_loss"]) <= (1 + limit) * math.exp(base[lg]) for lg in ("fr", "en"))]
    chosen = min(ok, key=lambda r: r["eval_bam_loss"]) if ok else None
    summary = {"start_ppl": {lg: math.exp(v) for lg, v in base.items()},
               "regression_limit": limit, "train_hours": round(train_hours, 2),
               "history_ppl": [{"step": r["step"], **{lg: math.exp(r[f"eval_{lg}_loss"]) for lg in ("bam", "fr", "en")}}
                               for r in hist.rows if all(k in r for k in need)],   # full curve incl. step 0
               "chosen_step": chosen["step"] if chosen else None,
               "note": None if chosen else "NO checkpoint met the French/English regression limit: re-run with a higher "
                                           "replay share or a lower learning rate (brief §11)."}
    (out / "selection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not chosen:
        sys.exit(2)

    # ---- merge the chosen adapter into the base weights and save a standard HF model
    del trainer, model
    if cuda:
        torch.cuda.empty_cache()
    if (out / "NAN_STOP").exists():
        sys.exit("training stopped on a non-finite loss: no model merged (see NAN_STOP)")
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg.get("model_revision"),
                                                      dtype=torch.bfloat16 if cuda else torch.float32)
    merged = PeftModel.from_pretrained(base_model, str(out / "checkpoints" / f"checkpoint-{chosen['step']}")).merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")

    # ---- a few fixed sample continuations for a quick human look
    merged.eval()
    dev = "cuda" if cuda else "cpu"
    merged.to(dev)
    samples = []
    for p in cfg.get("sample_prompts", []):
        x = tok(p, return_tensors="pt").to(dev)
        with torch.no_grad():
            y = merged.generate(**x, max_new_tokens=80, do_sample=False, repetition_penalty=1.1)
        samples.append({"prompt": p, "continuation": tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True)})
    (out / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    print(f"done: merged model in {out/'merged'}; chosen step {chosen['step']}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
