"""v3 SFT builder: stitched multi-turn conversations and the pivot/BT-only synthetic cap."""

import random

from bslm.build_sft import _augment_informal, _chat, _syn_cap, stitch

CFG = {"system_prompt": "SYS", "system_prompt_share": 0.5,
       "stitch": {"min_turns": 2, "max_turns": 4, "max_turn_chars": 1000, "max_chars": 2200,
                  "pools": {"translation": ["human"], "aya": ["aya_dataset"], "qa": ["qa_synthetic"]},
                  "weights": {"translation": 0.6, "aya": 0.3, "qa": 0.1}},
       "informal_spelling": {"share": 1.0}}


def _rows(rng):
    rows = []
    for i in range(40):
        for d, (u, a) in {"bam-fr": (f"Traduis en français : mɔgɔ jɔn {i}", f"qui {i}"),
                          "fr-bam": (f"Traduis en bambara : qui {i}", f"mɔgɔ jɔn {i}")}.items():
            e = _chat(CFG, rng, u, a, {"id": f"p{i}", "direction": d, "source": "human", "synthetic": False})
            if d == "bam-fr":
                e["_bam_input"] = ("Traduis en français : {x}", f"mɔgɔ jɔn {i}")
            rows.append(e)
        rows.append(_chat(CFG, rng, f"Question {i} ?", f"Answer {i}.",
                          {"id": f"a{i}", "direction": "instr-en", "source": "aya_dataset", "synthetic": False}))
        rows.append(_chat(CFG, rng, f"Ɲininkali {i}: jɔn bɛ yen?", f"Cɛ dɔ bɛ yen {i}.",
                          {"id": f"q{i}", "direction": "task-qa", "source": "qa_synthetic", "synthetic": True}))
    rows.append({"text": "raw", "meta": {"id": "r", "direction": "raw", "source": "cpt_replay", "synthetic": False}})
    return rows


def test_stitch_structure_and_targets():
    rng = random.Random(0)
    rows = _rows(rng)
    pairs = {(m[-2]["content"], m[-1]["content"]) for m in (e["messages"] for e in rows if "messages" in e)}
    convs = stitch(CFG, rows, 200, rng, "train")
    assert len(convs) >= 190
    for e in convs:
        msgs = [m for m in e["messages"] if m["role"] != "system"]
        assert 4 <= len(msgs) <= 8 and len(msgs) % 2 == 0
        assert [m["role"] for m in msgs] == ["user", "assistant"] * (len(msgs) // 2)
        turns = list(zip(msgs[::2], msgs[1::2]))
        assert all((u["content"], a["content"]) in pairs for u, a in turns)   # every target is an original target
        assert len(set(e["meta"]["parts"])) == len(turns) == e["meta"]["n_turns"]
        assert e["meta"]["synthetic"] == any(p.startswith("q") for p in e["meta"]["parts"])
        # reply language follows each user turn and is mixed within the conversation
        langs = {"fr" if u["content"].startswith("Traduis en français") else "en" if u["content"].startswith("Question")
                 else "bam" for u, _ in turns}
        assert len(langs) >= 2
        assert all(e["messages"][i]["role"] == "user" for i, *_ in e["_turns"])


def test_stitch_deterministic_and_informal_on_bambara_inputs_only():
    def build():
        rng = random.Random(1)
        rows = _rows(rng)
        convs = stitch(CFG, rows, 50, rng, "train")
        _augment_informal(convs, CFG, rng)
        return convs
    a, b = build(), build()
    assert [e["messages"] for e in a] == [e["messages"] for e in b]
    for e in a:
        for m in e["messages"]:
            if m["role"] == "assistant":
                assert "ou" not in m["content"]                      # targets are never changed
            if m["role"] == "user" and m["content"].startswith("Tra"):
                assert m["content"].startswith(("Traduis en français : ", "Traduis en bambara : "))


def test_syn_cap_drops_only_pivot_bt():
    rng = random.Random(0)
    mk = lambda i, syn, drop: {"messages": [], "meta": {"id": i, "source": "s", "synthetic": syn,
                                                        **({"cap_droppable": True} if drop else {})}}
    ex = [mk(f"r{i}", False, False) for i in range(60)] + [mk(f"c{i}", True, False) for i in range(20)] + \
         [mk(f"p{i}", True, True) for i in range(40)]
    info = _syn_cap(ex, 0.35, rng)
    assert sum(e["meta"]["id"].startswith("c") for e in ex) == 20 and sum(e["meta"]["id"].startswith("r") for e in ex) == 60
    syn = sum(e["meta"]["synthetic"] for e in ex) / len(ex)
    assert syn <= 0.35 and info["dropped"] == 40 - (int(0.35 * 60 / 0.65) - 20)
