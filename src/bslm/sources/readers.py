"""Read data/raw/<source> into interim records (brief §6 schema). No filtering here, only shaping.

Mono record:     id, text, lang, source, license, register, synthetic, url_or_ref, split_group
Parallel record: id, src_lang, tgt_lang, src_text, tgt_text, source, license, register, synthetic,
                 url_or_ref, split_group, align_method, align_score
Eval record:     same as parallel/mono plus `eval_set`.
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path("data/raw")

LICENSES = {
    "bayelemabaga": "CC-BY-4.0",  # supervisor decision 2026-11-04 (README says CC-BY-SA-4.0)
    "afvoices": "CC-BY-4.0", "afvoices_notag": "CC-BY-4.0", "kunkado": "CC-BY-SA-4.0",
    "jeli_asr": "CC-BY-4.0", "bam_asr_all_extra_mali_pense": "CC-BY-4.0", "bam_asr_all_extra_rt": "CC-BY-4.0",
    "asr_bam_only": "CC-BY-4.0",  # bam side of jeli-asr / bam-asr-all segments with no French translation
    "wikipedia_bm": "CC-BY-SA-4.0", "madlad_bm": "ODC-BY", "nllb_mined": "ODC-BY",
    "mafand_fr_bam": "CC-BY-NC-4.0", "belebele": "CC-BY-SA-4.0", "sib200": "CC-BY-SA-4.0",
}


def _rows(pattern: str) -> Iterator[tuple[str, dict]]:
    for f in sorted(glob.glob(str(RAW / pattern))):
        for r in pq.read_table(f).to_pylist():
            yield Path(f).name, r


def _mono(src, i, text, register, ref, group):
    return {"id": f"{src}-{i:07d}", "text": text or "", "lang": "bam", "source": src,
            "license": LICENSES[src], "register": register, "synthetic": False,
            "url_or_ref": ref, "split_group": group}


def _par(src, i, bam, other, other_lang, register, ref, group, method, score=None):
    return {"id": f"{src}-{other_lang}-{i:07d}", "src_lang": "bam", "tgt_lang": other_lang,
            "src_text": bam or "", "tgt_text": other or "", "source": src, "license": LICENSES[src],
            "register": register, "synthetic": False, "url_or_ref": ref, "split_group": group,
            "align_method": method, "align_score": score}


# ---------------- monolingual ----------------

def wikipedia_bm() -> Iterator[dict]:
    for i, (_, r) in enumerate(_rows("wikipedia_bm/*.parquet")):
        text = re.sub(r"__[A-Z_]+__", "", r["text"])  # MediaWiki magic words (e.g. __LEAD_SECTION__)
        yield _mono("wikipedia_bm", i, text, "written", r["url"], f"wiki:{r['id']}")


def madlad_bm() -> Iterator[dict]:
    for part in ("clean", "noisy"):
        with gzip.open(RAW / f"madlad_bm/bm_{part}_0000.jsonl.gz", "rt") as f:
            for i, line in enumerate(f):
                # MADLAD stores newlines as literal "\n" sequences
                text = json.loads(line)["text"].replace("\\n", "\n").replace("\\t", " ")
                rec = _mono("madlad_bm", i, text, "web", f"madlad:{part}:{i}", f"madlad:{part}:{i}")
                rec["id"] = f"madlad_bm-{part}-{i:07d}"
                rec["subset"] = part
                yield rec


def afvoices() -> Iterator[dict]:
    for i, (fn, r) in enumerate(_rows("afvoices/*.parquet")):
        yield {**_mono("afvoices", i, r["text"], "spoken", fn, f"afvoices:{fn}"),
               "orig_split": "test" if "__test-" in fn else "train"}


def afvoices_notag() -> Iterator[dict]:
    for i, (fn, r) in enumerate(_rows("afvoices_notag/*.parquet")):
        yield {**_mono("afvoices_notag", i, r["text"], "spoken", fn, f"afvoices:spk:{r['participantId']}"),
               "orig_split": r.get("split") or ("test" if "test" in fn else "train")}


def kunkado() -> Iterator[dict]:
    for i, (fn, r) in enumerate(_rows("kunkado/*.parquet")):
        yield {**_mono("kunkado", i, r["corrected-label"], "spoken", fn, f"kunkado:{fn}"),
               "orig_split": "test" if "__test-" in fn else "train"}


# ---------------- parallel (spoken transcripts with French translation) ----------------

def _asr_pairs(src: str, pattern: str) -> Iterator[dict]:
    for i, (fn, r) in enumerate(_rows(pattern)):
        rec = _par(src, i, r.get("bam"), r.get("french"), "fr", "spoken", fn, f"{src}:{fn}", "given")
        rec["orig_split"] = "test" if "__test" in fn else "train"
        yield rec


def jeli_asr():
    yield from _asr_pairs("jeli_asr", "jeli_asr/*.parquet")


def bam_asr_all_extra_mali_pense():
    yield from _asr_pairs("bam_asr_all_extra_mali_pense", "bam_asr_all_extra_mali_pense/*.parquet")


def bam_asr_all_extra_rt():
    yield from _asr_pairs("bam_asr_all_extra_rt", "bam_asr_all_extra_rt/*.parquet")


def asr_bam_only() -> Iterator[dict]:
    """Bambara side of ASR segments that have no French translation (they cannot be pairs)."""
    i = 0
    for src, pat in (("jeli_asr", "jeli_asr/*.parquet"),
                     ("bam_asr_all_extra_mali_pense", "bam_asr_all_extra_mali_pense/*.parquet"),
                     ("bam_asr_all_extra_rt", "bam_asr_all_extra_rt/*.parquet")):
        for fn, r in _rows(pat):
            if (r.get("bam") or "").strip() and not (r.get("french") or "").strip():
                rec = _mono("asr_bam_only", i, r["bam"], "spoken", f"{src}:{fn}", f"{src}:{fn}")
                rec["orig_split"] = "test" if "__test" in fn else "train"
                yield rec
                i += 1


# ---------------- parallel (written) ----------------

def bayelemabaga() -> Iterator[dict]:
    for split in ("train", "validation", "test"):
        for i, (fn, r) in enumerate(_rows(f"bayelemabaga/bam-fr__{split}__*.parquet")):
            t = r["translation"]
            rec = _par("bayelemabaga", i, t["bam"], t["fr"], "fr", "written", fn, f"bayelemabaga:{split}", "given")
            rec["id"] = f"bayelemabaga-{split}-{i:07d}"
            rec["orig_split"] = split
            yield rec


def nllb_mined() -> Iterator[dict]:
    for other, code in (("fra_Latn", "fr"), ("eng_Latn", "en")):
        with gzip.open(RAW / f"nllb_mined/bam_Latn-{other}.gz", "rt", errors="replace") as f:
            for i, line in enumerate(f):
                c = line.rstrip("\n").split("\t")
                if len(c) < 3:
                    continue
                url = c[6] if len(c) > 6 else ""
                yield _par("nllb_mined", i, c[0], c[1], code, "web", url, f"nllb:{url.split('/')[2] if url.count('/') >= 2 else 'na'}",
                           "laser_mined", float(c[2]))


# ---------------- evaluation only ----------------

def eval_sets() -> Iterator[dict]:
    for split in ("dev", "test", "train"):
        for i, line in enumerate((RAW / f"mafand_fr_bam/{split}.json").read_text().splitlines()):
            t = json.loads(line)["translation"]
            rec = _par("mafand_fr_bam", i, t["bam"], t["fr"], "fr", "news", f"mafand:{split}", f"mafand:{split}", "given")
            rec["id"] = f"mafand-{split}-{i:07d}"
            rec["eval_set"] = f"mafand_{split}"
            yield rec
    langs = {"bam_Latn": "bam", "fra_Latn": "fr", "eng_Latn": "en"}
    bele = {}
    for k in langs:  # one passage per link; pair by link, never by row order
        rows = [json.loads(x) for x in (RAW / f"belebele/data__{k}.jsonl").read_text().splitlines()]
        bele[k] = {r["link"]: r["flores_passage"] for r in rows}
    for j, (link, bam) in enumerate(bele["bam_Latn"].items()):
        for k, code in (("fra_Latn", "fr"), ("eng_Latn", "en")):
            if link not in bele[k]:
                continue
            rec = _par("belebele", j, bam, bele[k][link], code, "written", link, f"belebele:{link}", "given")
            rec["eval_set"] = "belebele_passages"
            yield rec
    for split in ("train", "dev", "test"):
        tabs = {}
        for k in langs:
            with open(RAW / f"sib200/data__{k}__{split}.tsv", newline="") as f:
                tabs[k] = {r["index_id"]: r["text"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
        for idx, bam in tabs["bam_Latn"].items():
            for k, code in (("fra_Latn", "fr"), ("eng_Latn", "en")):
                rec = _par("sib200", int(idx), bam, tabs[k].get(idx, ""), code, "written", f"sib200:{split}:{idx}",
                           f"sib200:{split}", "given")
                rec["eval_set"] = f"sib200_{split}"
                yield rec


MONO = {"wikipedia_bm": wikipedia_bm, "madlad_bm": madlad_bm, "afvoices": afvoices,
        "afvoices_notag": afvoices_notag, "kunkado": kunkado, "asr_bam_only": asr_bam_only}
PARALLEL = {"bayelemabaga": bayelemabaga, "jeli_asr": jeli_asr,
            "bam_asr_all_extra_mali_pense": bam_asr_all_extra_mali_pense,
            "bam_asr_all_extra_rt": bam_asr_all_extra_rt, "nllb_mined": nllb_mined}
