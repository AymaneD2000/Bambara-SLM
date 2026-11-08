"""Shared text normalization (brief §7.1-7.4). Conservative: no orthographic normalization."""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_APOSTROPHES = dict.fromkeys(map(ord, "’‘ʼʻ`´′"), "'")
_QUOTES = dict.fromkeys(map(ord, "“”„«»″"), '"')
_SPACES = dict.fromkeys(map(ord, "     　\t"), " ")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_MULTISPACE = re.compile(r"[ ]{2,}")
_NKO = re.compile(r"[߀-߿]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏɐ-ʯ]")


def normalize(text: str, confusables: dict[str, str] | None = None) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    text = text.translate(_ZERO_WIDTH).translate(_APOSTROPHES).translate(_QUOTES).translate(_SPACES)
    if confusables:
        for a, b in confusables.items():
            if a != b:
                text = text.replace(a, b)
    lines = [_MULTISPACE.sub(" ", ln).strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return unicodedata.normalize("NFC", text)


def script_of(text: str) -> str:
    n_nko, n_lat = len(_NKO.findall(text)), len(_LATIN.findall(text))
    if n_nko > n_lat:
        return "nko"
    return "latin" if n_lat else "other"


def strip_tags(text: str, tags: list[str]) -> str:
    for t in tags:
        text = text.replace(t, " ")
    return _MULTISPACE.sub(" ", text).strip()
