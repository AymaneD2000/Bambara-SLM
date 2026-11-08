"""Merge a saved LoRA checkpoint into its base model on any machine (CPU is enough), bf16 output.

Used to turn an SFT checkpoint that was not auto-selected (e.g. the final step) into a standard HF model
that mlx-lm can load for evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base: str, adapter: str, out: str) -> Path:
    out_p = Path(out)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged.save_pretrained(out_p, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_p)
    (out_p / "merge_info.json").write_text(json.dumps({"base": base, "adapter": adapter}, indent=2))
    return out_p
