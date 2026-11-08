from bslm.clean import _quality_reason, _transcript
from bslm.dedup import exact_dedup, near_dedup
from bslm.pii import scrub
from bslm.text_norm import normalize, script_of

import yaml

TCFG = yaml.safe_load(open("configs/clean.yaml"))["transcripts"]
CONF = yaml.safe_load(open("configs/clean.yaml"))["normalize"]["confusables"]
Q = {"min_chars": 10, "max_chars": 20000, "min_letter_ratio": 0.6, "max_digit_ratio": 0.3,
     "max_dup_line_ratio": 0.3, "max_top_trigram_ratio": 0.2}


def test_normalize_nfc_apostrophes_confusables():
    s = "A bε n’a fo​  «ko» è"
    out = normalize(s, {"ε": "ɛ"})
    assert out == "A bɛ n'a fo \"ko\" è"


def test_normalize_keeps_bambara_letters_and_shha():
    s = "ɲɛmɔgɔ ŋaniya ʃ"
    assert normalize(s, {"ε": "ɛ"}) == s  # ʃ is not mapped silently


def test_script_detection():
    assert script_of("ߒ ߓߊ߯ߙߊ߲") == "nko"
    assert script_of("I ni ce") == "latin"


def test_transcript_markers():
    assert _transcript("[noise] mɔgɔ caman de b'a bila", TCFG) == ("mɔgɔ caman de b'a bila", "")
    assert _transcript("Ɲɛ dimi bi o la[?]", TCFG)[0] is None
    assert _transcript("[cs] a kan ka kalan", TCFG)[0] is None
    assert _transcript("[cs ] a kan ka kalan", TCFG)[0] is None
    assert _transcript("Bɔda bɔ a la ka to ka ?]", TCFG)[0] is None           # malformed tag
    assert _transcript("<INCOMPRÉHENSIBLE> kelen __ professeur__ Madu", TCFG)[0] is None
    assert _transcript("<RIRES> __Professeur__ n'a y'a sɔrɔ", TCFG) == ("Professeur n'a y'a sɔrɔ", "")
    assert _transcript("<MUSIQUE EN SOURDINE> a bɛ yen", TCFG) == ("a bɛ yen", "")


def test_confusables():
    assert normalize("damankɛπɛni dวnnibagaw kכnכ Лini kὲlɛ", CONF) == "damankɛɲɛni dɔnnibagaw kɔnɔ ɲini kɛ̀lɛ"


def test_pii():
    t, c = scrub("Wele 76 12 34 56 walima +223 76123456, email a.b@gmail.com")
    assert "<PHONE>" in t and "<EMAIL>" in t and "76" not in t
    t2, c2 = scrub("San 2021 la, mɔgɔ 250 nana.")
    assert t2 == "San 2021 la, mɔgɔ 250 nana." and not c2


def test_quality():
    assert _quality_reason("123 456 789 000 111", Q) in ("mostly_digits", "low_letter_ratio")
    assert _quality_reason("I ni ce, i ka kɛnɛ wa?", Q) is None


def test_dedup():
    recs = [{"id": "a", "text": "Ne bɛ Ala deli hali sa."}, {"id": "b", "text": "ne bɛ ala deli hali sa"},
            {"id": "c", "text": "Denba k'a janto a sinbara la, o ka fisa a n'a den bɛɛ ka kɛnɛya ma kosɛbɛ."},
            {"id": "d", "text": "Denba k'a janto a sinbara la, o ka fisa a n'a den bɛɛ ka kɛnɛya ma kosɛbɛ!!"}]
    keep, drop = exact_dedup(recs)
    assert [r["id"] for r in keep] == ["a", "c"]
    keep2, drop2 = near_dedup(recs, 0.8, 64, 5)
    assert [r["id"] for r in keep2] == ["a", "c"]


def test_decontam_index(tmp_path):
    import json
    from bslm.decontam import EvalIndex
    ev = tmp_path / "e.jsonl"
    ev.write_text(json.dumps({"eval_set": "t", "src_text": "Sanjani y'a terike deli kosɛbɛ ka taa so.",
                              "tgt_text": "Sandiani se mit à implorer son ami."}) + "\n")
    idx = EvalIndex(str(tmp_path))
    assert idx.match("sanjani y'a terike deli kosɛbɛ ka taa so")[0] == "exact"
    assert idx.match("Sanjani y'a terike deli kosɛbɛ ka taa so!!")[0] in ("exact", "near")
    assert idx.match("Ne bɛ taa sugu la sini.") is None


def test_freeze_prompts_rejects_placeholders(tmp_path, monkeypatch):
    import bslm.prompts as P
    csvf = tmp_path / "p.csv"
    csvf.write_text("id,category,instruction_lang,prompt,reference,domain,notes,approved\n"
                    "a,qa,fr,\"Réponds en bambara : [À ÉCRIRE EN BAMBARA]\",,x,,y\n"
                    "b,qa,fr,Réponds en bambara : Quel fleuve traverse Bamako ?,Jɔliba,x,,y\n", encoding="utf-8")
    monkeypatch.setattr(P, "OUT", tmp_path / "out.jsonl")
    monkeypatch.setattr(P, "SHA", tmp_path / "out.sha256")
    assert P.freeze(str(csvf), min_rows=1) is False          # placeholder in row a
    csvf.write_text("id,category,instruction_lang,prompt,reference,domain,notes,approved\n"
                    "b,qa,fr,Réponds en bambara : Quel fleuve traverse Bamako ?,Jɔliba,x,,y\n", encoding="utf-8")
    monkeypatch.setattr(P, "_training_overlap", lambda recs: [])
    assert P.freeze(str(csvf), min_rows=1) is True
    assert P.check_frozen() is True
    assert P.freeze(str(csvf), min_rows=1) is False          # refuses to overwrite a frozen set
