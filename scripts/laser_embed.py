"""Embed sentences with LASER (LASER3 for bam_Latn, LASER2 for fra/eng). Runs in .venv-laser (Python 3.10).

LASER is BSD-licensed (commercial use OK). laser_encoders defaults to batch size 1 and CUDA only;
here we batch by tokens and run on Apple MPS when available.

Usage: .venv-laser/bin/python scripts/laser_embed.py IN.jsonl FIELD LANG OUT.npy
"""

import json
import sys
import time

import numpy as np
import torch
from laser_encoders import LaserEncoderPipeline

MODEL_DIR = "models/laser"


def load(lang: str, device: str) -> LaserEncoderPipeline:
    p = LaserEncoderPipeline(lang=lang, model_dir=MODEL_DIR)
    enc = p.encoder
    enc.max_sentences, enc.max_tokens = None, 12000
    enc.encoder.to(device)
    for layer in getattr(enc.encoder, "layers", []):  # fairseq "BetterTransformer" path needs ops missing on MPS
        layer.can_use_fastpath = False

    def _process_batch(batch, _enc=enc):
        with torch.no_grad():
            out = _enc.encoder(batch.tokens.to(device), batch.lengths.to(device))["sentemb"]
        return out.detach().float().cpu().numpy()

    enc._process_batch = _process_batch
    return p


def main(inp: str, field: str, lang: str, out: str) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.backends.mha.set_fastpath_enabled(False)  # nested-tensor fast path is not implemented on MPS
    texts = [json.loads(line)[field].replace("\n", " ") for line in open(inp)]
    p = load(lang, device)
    t, embs, B = time.time(), [], 20000
    for i in range(0, len(texts), B):
        embs.append(p.encode_sentences(texts[i:i + B], normalize_embeddings=True).astype(np.float16))
        print(f"{lang}: {min(i + B, len(texts))}/{len(texts)} ({time.time() - t:.0f}s)", flush=True)
    np.save(out, np.concatenate(embs) if embs else np.zeros((0, 1024), np.float16))


if __name__ == "__main__":
    main(*sys.argv[1:5])
