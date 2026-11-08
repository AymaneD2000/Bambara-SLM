"""Generate answers to the frozen native-speaker prompt set and write a scoring sheet (brief §10).

The prompt file must match its frozen hash. Decoding is greedy and fixed. For a base model (no chat template) the
prompt is wrapped in a fixed completion template; for chat models (after SFT) the repo's chat template is used.
Sheet: reports/eval/<run>/human_review.csv, scores 1-5 for adequacy, fluency, orthography, hallucination.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mlx_lm import batch_generate, load
from rich.console import Console

from bslm.prompts import OUT as PROMPTS, check_frozen

console = Console()
BASE_TEMPLATE = "Instruction: {prompt}\nResponse:"
MAX_TOKENS = {"generate": 256, "summarize": 200, "qa": 160}


def run(run_name: str = "baseline_qwen3_1.7b_base", model_path: str = "Qwen/Qwen3-1.7B-Base",
        adapter_path: str | None = None, chat: bool = False, batch_size: int = 8) -> Path:
    if not check_frozen():
        raise SystemExit("data/eval/human_prompts.jsonl does not match its frozen hash: refusing to evaluate")
    prompts = [json.loads(line) for line in open(PROMPTS)]
    model, tok = load(model_path, adapter_path=adapter_path)

    def encode(p: str) -> list[int]:
        if chat:
            msgs = [{"role": "user", "content": p}]
            return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        return tok.encode(BASE_TEMPLATE.format(prompt=p))

    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        res = batch_generate(model, tok, [encode(p["text"]) for p in batch],
                             max_tokens=[MAX_TOKENS.get(p["category"], 128) for p in batch])
        for p, t in zip(batch, res.texts):
            # base models keep writing new "Instruction:" blocks: cut there and at a blank line
            t = t.split("\nInstruction:")[0].split("\n\n")[0].strip() if not chat else t.strip()
            outs.append({**p, "output": t})
        console.print(f"{min(i + batch_size, len(prompts))}/{len(prompts)}")

    out_dir = Path("reports/eval") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "human_prompts_outputs.jsonl").write_text(
        "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in outs))
    sheet = out_dir / "human_review.csv"
    with sheet.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "prompt", "reference", "model_output",
                    "adequacy_1_5", "fluency_1_5", "orthography_1_5", "hallucination_1_5", "comment"])
        for o in outs:
            w.writerow([o["id"], o["category"], o["text"], o["tgt_text"], o["output"], "", "", "", "", ""])
    console.print(f"wrote {sheet}")
    return sheet
