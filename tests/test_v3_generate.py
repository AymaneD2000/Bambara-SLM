"""v3 phase 3 generation: sentence splitting / rejoining and resumable outputs (no model needed)."""

import json

import pytest
import yaml

from bslm.v3data import generate as G

CFG = yaml.safe_load(open("configs/v3_generate.yaml"))


def _t(parts):
    return [p for k, p in parts if k == "t"]


def test_split_keeps_paragraphs_lists_and_whitespace():
    text = ("Here are tips. Mr. Smith said so in 2001. It works!\n\n"
            "1. First item. Second sentence.\n2) Second item\n- bullet\n   * nested: yes\n\n## Header\nEnd... ok  ")
    parts = G.split_text(text)
    assert "".join(p for _, p in parts) == text
    assert G.join_text(parts, lambda s: s) == text
    assert _t(parts) == ["Here are tips.", "Mr. Smith said so in 2001.", "It works!", "First item.",
                         "Second sentence.", "Second item", "bullet", "nested: yes", "Header", "End... ok"]
    for marker in ("1. ", "2) ", "- ", "   * ", "## "):
        assert ("k", marker) in parts
    out = G.join_text(parts, lambda s: s.upper())
    assert out.startswith("HERE ARE TIPS. MR. SMITH") and "\n\n1. FIRST ITEM. SECOND SENTENCE.\n2) SECOND ITEM\n- " in out
    assert out.endswith("\n\n## HEADER\nEND... OK  ")


def test_split_bambara_lowercase_and_long_sentences():
    assert _t(G.split_text("a ka di. ne bɛ taa.", lower_ok=True)) == ["a ka di.", "ne bɛ taa."]
    assert _t(G.split_text("a ka di. ne bɛ taa.", lower_ok=False)) == ["a ka di. ne bɛ taa."]
    long = " ".join(["w"] * 50) + "; " + " ".join(["x"] * 50) + "."
    parts = G.split_text(long, max_words=60)
    assert len(_t(parts)) == 2 and G.join_text(parts, lambda s: s) == long
    assert all(len(s.split()) <= 60 for s in _t(G.split_text(" ".join(["y"] * 200), max_words=60)))
    assert G.split_text("12.5 % -- 3") == [("k", "12.5 % -- 3")]       # no letters: kept verbatim


class FakeBackend(G.Backend):
    name = "fake"

    def __init__(self):
        super().__init__("none", CFG["generation"])
        self.calls = 0

    def n_tokens(self, text):
        return len(text.split())

    def generate(self, prompts, max_new, sample, rep=1.0):
        self.calls += len(prompts)
        return [("<" + p.split(": ", 1)[-1].split(" : ", 1)[-1] + ">", False) for p in prompts]


def _args(tmp, **kw):
    a = dict(in_dir=str(tmp), out_dir=str(tmp), backend="hf", model="x", dtype=None, seed=0, shard=None, limit=None,
             max_hours=None, stop_at=None, decode="sample", chunk=2, chunk_chars=50, sources=None)
    a.update(kw)
    return type("A", (), a)()


@pytest.fixture
def fake(monkeypatch):
    fb = FakeBackend()
    monkeypatch.setattr(G, "make_backend", lambda *a, **k: fb)
    monkeypatch.setattr(G, "try_lid", lambda texts: None)
    return fb


def test_bt_resumes_and_shards(tmp_path, fake):
    rows = [{"id": f"s{i}", "text": f"N bɛ taa {i}. A ka di.", "source": "madlad_bm", "license": "ODC-BY"}
            for i in range(7)]
    (tmp_path / "bt_sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert G.run_bt(_args(tmp_path, limit=3), CFG) == 0
    out = tmp_path / "bt_out.jsonl"
    assert len(G.read_jsonl(out)) == 3
    with out.open("a") as f:                                      # simulate a crash in the middle of a write
        f.write('{"id": "s9", "bam": "torn')
    G.run_bt(_args(tmp_path), CFG)
    got = G.read_jsonl(out)
    assert sorted(r["id"] for r in got) == sorted(r["id"] for r in rows)   # torn line skipped, no duplicates
    assert got[0]["fr"] == "<N bɛ taa 0.> <A ka di.>" and got[0]["rt_bam"].startswith("<<")
    calls = fake.calls
    G.run_bt(_args(tmp_path), CFG)                                 # everything done: no generation at all
    assert fake.calls == calls
    # shards split the remaining work and see each other's outputs
    (tmp_path / "bt_sources.jsonl").write_text("".join(json.dumps({**r, "id": "n" + r["id"]}) + "\n" for r in rows))
    for s in ("0/2", "1/2"):
        G.run_bt(_args(tmp_path, shard=s), CFG)
    ids = [r["id"] for f in G.output_files(tmp_path, "bt_out") for r in G.read_jsonl(f)]
    assert len(ids) == len(set(ids)) == 14 and len(G.output_files(tmp_path, "bt_out")) == 3


def test_chat_resumes(tmp_path, fake):
    th = [{"id": f"t{i}", "lang": "en", "messages": [{"role": "user", "content": "Hi there. How are you?"},
                                                      {"role": "assistant", "content": "Fine.\n\n- one\n- two"}]}
          for i in range(4)]
    (tmp_path / "oasst2_threads.jsonl").write_text("".join(json.dumps(r) + "\n" for r in th))
    G.run_chat(_args(tmp_path, decode="greedy", limit=2), CFG)
    G.run_chat(_args(tmp_path, decode="greedy"), CFG)
    got = G.read_jsonl(tmp_path / "chat_bam.jsonl")
    assert [r["id"] for r in got] == ["t0", "t1", "t2", "t3"]
    assert got[0]["parts"][1]["bam"] == "<Fine.>\n\n- <one>\n- <two>"
    assert got[0]["parts"][0]["role"] == "user" and "rt_chrf" in got[0]["parts"][0]


def test_deadline_stops_cleanly(tmp_path, fake):
    rows = [{"id": f"s{i}", "text": "A ka di.", "source": "kunkado", "license": "CC-BY-SA-4.0"} for i in range(4)]
    (tmp_path / "bt_sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert G.run_bt(_args(tmp_path, stop_at=1.0), CFG) == 3       # deadline already passed: nothing written
    assert G.read_jsonl(tmp_path / "bt_out.jsonl") == []
