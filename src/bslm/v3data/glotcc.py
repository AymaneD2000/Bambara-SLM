"""cis-lmu/GlotCC-V1, bam-Latn (CC0 packaging; text under Common Crawl terms). Reader only: the cleaning,
segmentation and dedup pipeline is in bslm.v3data.hplt, which writes both sources into mono_bam.jsonl."""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow.parquet as pq

from bslm.v3data import download

REPO, FILE, SUBDIR = "cis-lmu/GlotCC-V1", "v1.0/bam-Latn/bam-Latn_0.parquet", "glotcc"


def read() -> Iterator[dict]:
    download(REPO, "README.md", SUBDIR)
    p = download(REPO, FILE, SUBDIR)
    f = pq.ParquetFile(p)
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=["content", "warc-target-uri", "warc-record-id"]).to_pylist()
        for r in t:
            yield {"doc_id": (r["warc-record-id"] or "").removeprefix("<urn:uuid:").rstrip(">"), "text": r["content"] or "", "url": r["warc-target-uri"],
                   "source": "glotcc_v1"}
