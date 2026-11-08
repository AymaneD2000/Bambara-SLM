"""bslm command-line interface. Commands are added stage by stage, gated by supervisor approval."""

from __future__ import annotations

import typer

app = typer.Typer(help="Bambara SLM pipeline", no_args_is_help=True)


@app.command()
def inventory(
    config: str = "configs/sources.yaml",
    out: str = "reports/gate0_inventory.json",
) -> None:
    """Gate 0: verify existence, size, schema, license and access of every source."""
    from bslm.inventory import run_inventory

    run_inventory(config, out)


@app.command()
def fetch(only: list[str] = typer.Option(None, help="Source names to fetch (default: all)")) -> None:
    """Download approved sources (text columns only) into data/raw/."""
    import json

    from bslm.sources.fetch import fetch as _fetch

    status = _fetch(only)
    typer.echo(json.dumps(status, indent=2))


@app.command()
def verify() -> None:
    """Verify data/raw: checksums vs MANIFEST.json and row counts vs published numbers."""
    from bslm.sources.verify import run

    ok = run()
    typer.echo(f"{'PASS' if ok else 'FAIL'} - see reports/raw_verification.md")
    raise typer.Exit(0 if ok else 1)


@app.command()
def clean(config: str = "configs/clean.yaml") -> None:
    """Normalize, filter, deduplicate and report (Gate 1)."""
    from bslm.clean import run

    run(config)


@app.command()
def align() -> None:
    """Score every parallel pair (needs LASER embeddings from scripts/laser_embed.py)."""
    from bslm.align import run

    run()


@app.command()
def decontam() -> None:
    """Remove training text matching evaluation text (exact + near-duplicate)."""
    from bslm.decontam import run

    run()


@app.command()
def splits(config: str = "configs/align.yaml") -> None:
    """Build train/val splits by document group, applying the alignment threshold."""
    from bslm.splits import run

    run(config)


@app.command("eval-baseline")
def eval_baseline(config: str = "configs/eval.yaml", run_name: str = "baseline_qwen3_1.7b_base",
                  model: str = typer.Option(None), adapter: str = typer.Option(None),
                  skip_mt: bool = False, chat: bool = False,
                  max_n: int = typer.Option(None, help="Cap sentences per set (smoke tests only)")) -> None:
    """Perplexity + translation evaluation with MLX (fixed prompts from configs/eval.yaml)."""
    from bslm.eval.baseline import run

    run(config, run_name, model, adapter, skip_mt, chat, max_n)


@app.command("freeze-prompts")
def freeze_prompts(csv_path: str = "prompts/human_prompts_draft.csv", min_rows: int = 50,
                   force: bool = typer.Option(False, help="Override checks / re-freeze (supervisor approval only)")) -> None:
    """Validate and freeze the native-speaker prompt set into data/eval/human_prompts.jsonl."""
    from bslm.prompts import freeze

    raise typer.Exit(0 if freeze(csv_path, min_rows, force) else 1)


@app.command("inspect-results")
def inspect_results(target: float = 0.8) -> None:
    """Summarize the manual alignment inspection and propose per-source thresholds."""
    from bslm.inspect_results import run

    for g, v in run(target).items():
        thr = "DROP" if v["threshold"] is None else f">= {v['threshold']:.1f}"
        typer.echo(f"{g:22s} {thr:8s} kept {v['pairs_kept']:>7,}  est. precision {v['est_precision']:.0%}")


@app.command("human-review-sheet")
def human_review_sheet(run_name: str = "baseline_qwen3_1.7b_base", model: str = "Qwen/Qwen3-1.7B-Base",
                       adapter: str = typer.Option(None), chat: bool = False) -> None:
    """Generate answers to the frozen prompt set and write a native-speaker scoring sheet."""
    from bslm.eval.human_review import run

    run(run_name, model, adapter, chat)


@app.command("build-cpt")
def build_cpt(config: str = "configs/cpt_mix.yaml") -> None:
    """Build the continued-pretraining mix (both variants) into data/cpt/<variant>/."""
    from bslm.build_cpt import run

    run(config)


@app.command("build-sft")
def build_sft(config: str = "configs/sft.yaml") -> None:
    """Build chat-format SFT data (translation) into data/sft/."""
    import json

    from bslm.build_sft import run

    typer.echo(json.dumps(run(config), indent=2))


@app.command("merge-adapter")
def merge_adapter(base: str, adapter: str, out: str) -> None:
    """Merge a LoRA checkpoint (e.g. runs/.../checkpoints/checkpoint-3706) into its base model (bf16)."""
    from bslm.train.merge import merge

    typer.echo(f"merged model written to {merge(base, adapter, out)}")


@app.command("tokenizer-audit")
def tokenizer_audit() -> None:
    """Tokens-per-word of the Qwen3 tokenizer on Bambara vs French/English."""
    from bslm.tokenizer_audit import audit

    r = audit()
    typer.echo(f"bam/fra ratio = {r['ratio_bam_fra']:.2f} -> {r['decision']}")


def _not_yet(name: str) -> None:
    typer.echo(f"`{name}` is not implemented yet (blocked on supervisor gate).")
    raise typer.Exit(1)


for _cmd in [

    "train-cpt", "train-sft", "eval", "export",
]:
    app.command(name=_cmd)(lambda _n=_cmd: _not_yet(_n))


if __name__ == "__main__":
    app()
