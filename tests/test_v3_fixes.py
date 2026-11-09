"""v3 SFT fixes: all-assistant-turn loss, informal spelling on Bambara only, Pruner fallback, synth_qa split key."""

import json
import random
from pathlib import Path

import pytest

from bslm.build_sft import _augment_informal, informal_spelling
from bslm.synth_qa import split_key

TOK_DIR = Path(__file__).resolve().parents[1] / "runs" / "sft_v2" / "merged"


@pytest.fixture(scope="module")
def tok():
    if not (TOK_DIR / "tokenizer_config.json").exists():
        pytest.skip(f"no tokenizer at {TOK_DIR}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOK_DIR))


def _feats(tok, rows, all_turns, max_len=512):
    from bslm.train.sft import _chat_features
    return _chat_features(tok, max_len, all_turns)({"messages": rows})


def _unmasked(tok, ids, labels):
    """Decoded runs of consecutive unmasked label positions."""
    runs, cur = [], []
    for i, y in zip(ids, labels):
        if y != -100:
            assert y == i
            cur.append(i)
        elif cur:
            runs.append(tok.decode(cur)); cur = []
    if cur:
        runs.append(tok.decode(cur))
    return runs


def test_single_turn_labels_identical(tok):
    rows = [[{"role": "user", "content": "Traduis en français : N bɛ taa sugu la."},
             {"role": "assistant", "content": "Je vais au marché."}],
            [{"role": "system", "content": "I ye dɛmɛbaga ye."}, {"role": "user", "content": "I ni ce!"},
             {"role": "assistant", "content": "N ba, i ni ce!"}]]
    a, b = _feats(tok, rows, False), _feats(tok, rows, True)
    assert a == b and all(a["input_ids"])


def test_three_turn_all_assistant_turns(tok):
    msgs = [{"role": "system", "content": "S"},
            {"role": "user", "content": "I ni ce!"}, {"role": "assistant", "content": "N ba, i ni ce!"},
            {"role": "user", "content": "I ka kɛnɛ wa?"}, {"role": "assistant", "content": "Tɔɔrɔ si tɛ."},
            {"role": "user", "content": "I tɔgɔ ye di?"}, {"role": "assistant", "content": "N tɔgɔ ye Bslm."}]
    f = _feats(tok, [msgs], True)
    ids, labels = f["input_ids"][0], f["labels"][0]
    # rendering unchanged (Qwen3 template, empty think block before the final answer only)
    assert ids == tok(tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False),
                      add_special_tokens=False)["input_ids"]
    # final turn labelled as in v2 (content + <|im_end|> + trailing newline); earlier ones content + <|im_end|>
    assert _unmasked(tok, ids, labels) == ["N ba, i ni ce!<|im_end|>", "Tɔɔrɔ si tɛ.<|im_end|>",
                                           "N tɔgɔ ye Bslm.<|im_end|>\n"]
    think = set(tok.convert_tokens_to_ids(["<think>", "</think>"]))
    assert not any(y in think for y in labels)
    # flag off: only the last turn
    g = _feats(tok, [msgs], False)
    assert _unmasked(tok, g["input_ids"][0], g["labels"][0]) == ["N tɔgɔ ye Bslm.<|im_end|>\n"]


def test_too_long_rows_dropped_not_cut(tok):
    long = " ".join(["kɛlɛ"] * 400)
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}, {"role": "assistant", "content": long}]
    f = _feats(tok, [msgs], True, max_len=128)
    assert f["input_ids"] == [[]] and f["labels"] == [[]]
    f = _feats(tok, [msgs[:2] + [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]], True,
               max_len=128)
    assert len(f["input_ids"][0]) <= 128 and len(_unmasked(tok, f["input_ids"][0], f["labels"][0])) == 2


def _cfg(share=1.0):
    return {"informal_spelling": {"share": share}}


def test_informal_spelling_only_on_bambara():
    instr = "Traduis en français : {x}"
    src = "Jɔn bɛ taa sugu la ?"
    tr = {"messages": [{"role": "user", "content": instr.format(x=src)}, {"role": "assistant", "content": "Qui va ?"}],
          "meta": {"direction": "bam-fr"}, "_bam_input": (instr, src)}
    fr_bam = {"messages": [{"role": "user", "content": "Traduis en bambara : Qui va au marché ?"},
                           {"role": "assistant", "content": "Jɔn bɛ taa sugu la ?"}], "meta": {"direction": "fr-bam"}}
    task = {"messages": [{"role": "user", "content": "I ni ce! I ka kɛnɛ wa?"}, {"role": "assistant", "content": "Tɔɔrɔ si tɛ."},
                         {"role": "user", "content": "Lis ce texte en bambara et réponds en bambara : Musow bɛ ɲɔ "
                                                     "susu.\n\nQuestion : Que font les femmes ?"},
                         {"role": "assistant", "content": "U bɛ ɲɔ susu."}], "meta": {"direction": "task-qa"}}
    fr_chat = {"messages": [{"role": "user", "content": "Que peux-tu faire ?"}, {"role": "assistant", "content": "Je traduis."}],
               "meta": {"direction": "task-chat"}}
    ex = [tr, fr_bam, task, fr_chat]
    before = json.loads(json.dumps(ex))
    _augment_informal(ex, _cfg(), random.Random(0))
    u = tr["messages"][0]["content"]
    assert u.startswith("Traduis en français : ") and "Tradouis" not in u and "ɛ" not in u and "ɔ" not in u
    assert fr_bam == before[1]                                   # X->bam: not eligible
    t_hist, t_last = task["messages"][0]["content"], task["messages"][2]["content"]
    assert "ɛ" not in t_hist and "ɲ" not in t_last              # Bambara user text changed, history included
    assert t_last.startswith("Lis ce texte en bambara et réponds en bambara : ")
    assert t_last.endswith("\n\nQuestion : Que font les femmes ?")
    assert all(m["content"] == b["content"] for e, eb in zip(ex, before) for m, b in zip(e["messages"], eb["messages"])
               if m["role"] == "assistant")                      # targets untouched
    assert fr_chat["messages"][0]["content"] == "Que peux-tu faire ?"


def test_informal_spelling_share_and_seed():
    def make():
        return [{"messages": [{"role": "user", "content": f"Translate into English: {i} ɲɔ bɛ yan"},
                              {"role": "assistant", "content": "x"}], "meta": {"direction": "bam-en"},
                 "_bam_input": ("Translate into English: {x}", f"{i} ɲɔ bɛ yan")} for i in range(2000)]
    a, b = make(), make()
    na, nb = _augment_informal(a, _cfg(0.15), random.Random(13)), _augment_informal(b, _cfg(0.15), random.Random(13))
    assert na == nb and a == b and 0.12 < na / 2000 < 0.18
    assert informal_spelling("jɔn", random.Random(1)) == informal_spelling("jɔn", random.Random(1))


def test_pruner_keeps_best_unqualified(tmp_path):
    from bslm.train.cpt import Pruner

    class H:
        rows = []
    h = H()
    start = {"eval_bam_loss": 1.0, "eval_fr_loss": 1.0, "eval_en_loss": 1.0}
    for step, sft, bam in ((2, 0.5, 1.5), (4, 0.9, 1.0), (6, 0.8, 1.5)):   # step 2: best SFT loss but over the limit
        (tmp_path / f"checkpoint-{step}").mkdir()
        h.rows.append({"step": step, "eval_sft_loss": sft, "eval_chat_loss": 1.0, "eval_bam_loss": bam,
                       "eval_fr_loss": 1.0, "eval_en_loss": 1.0})
    class S:
        global_step = 6
    for fb, kept in ((False, {"checkpoint-4", "checkpoint-6"}), (True, {"checkpoint-2", "checkpoint-4", "checkpoint-6"})):
        for s in (2, 4, 6):
            (tmp_path / f"checkpoint-{s}").mkdir(exist_ok=True)
        p = Pruner(tmp_path, h, 0.10, select_key="eval_sft_loss+eval_chat_loss", limit_langs=("bam", "fr", "en"),
                   keep_best_unqualified=fb)
        p.start = start
        p.on_save(None, S(), None)
        assert {d.name for d in tmp_path.iterdir()} == kept
        assert p.score(h.rows[0]) == 1.5


def test_synth_qa_split_key():
    assert split_key({"id": "b2-bayelemabaga-train-0014779-3"}) == "bayelemabaga-train-0014779"
    assert split_key({"id": "b1-jeli_asr-train-12-1"}) == "jeli_asr-train-12"
    assert split_key({"id": "x", "passage_id": "p-1"}) == "p-1"
    assert split_key({"id": "c2-1"}) == "c2-1"
