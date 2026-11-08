"""google/smol en_bm (CC-BY-4.0, revision fdaff3a1a0): professional English->Bambara translations.
SmolSent (sentences), SmolDoc (multi-sentence documents, kept whole) and GATITOS (words/short phrases)
-> data/interim/smol_bam_en.jsonl in the parallel format, with a deterministic 5% validation split.
Usage: python -m bslm.smol   (files downloaded to data/raw/smol/ with hf_hub_download)
"""
import hashlib, json
from pathlib import Path

RAW, OUT = Path("data/raw/smol"), Path("data/interim/smol_bam_en.jsonl")


def _split(key: str) -> str:
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 20 == 0 else "train"


def main() -> None:
    n = 0
    with OUT.open("w") as f:
        for sub in ("smolsent", "smoldoc", "gatitos"):
            for i, line in enumerate(open(RAW / sub / "en_bm.jsonl")):
                r = json.loads(line)
                en = " ".join(r["srcs"]) if "srcs" in r else r["src"]
                if sub == "gatitos":          # trgs = alternative translations of one term: keep the first
                    bm = r["trgs"][0] if r.get("trgs") else ""
                else:                         # smoldoc: list of aligned sentences of one document
                    bm = " ".join(r["trgs"]) if "trgs" in r else r["trg"]
                if not en.strip() or not bm.strip():
                    continue
                rid = f"smol-{sub}-{r.get('id', i)}"
                f.write(json.dumps({"id": rid, "split": _split(rid), "src_lang": "bam", "tgt_lang": "en",
                                    "src_text": bm.strip(), "tgt_text": en.strip(), "source": f"smol_{sub}",
                                    "license": "CC-BY-4.0", "synthetic": False}, ensure_ascii=False) + "\n")
                n += 1
    print(f"smol: {n} pairs -> {OUT}")


if __name__ == "__main__":
    main()
