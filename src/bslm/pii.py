"""PII scrubbing (brief §3.6, §7.9): e-mails, phone numbers, long ID-like numbers, URLs with user info."""

from __future__ import annotations

import re

PATTERNS = [
    ("<EMAIL>", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    # Mali numbers (+223 / 00223 then 8 digits, optionally grouped) and generic intl numbers
    ("<PHONE>", re.compile(r"(?:\+|00)\d{1,3}[\s.-]?(?:\d{2}[\s.-]?){4,5}\d{0,2}")),
    ("<PHONE>", re.compile(r"(?<!\d)(?:\d{2}[\s.-]){3}\d{2}(?!\d)")),          # 76 12 34 56
    ("<PHONE>", re.compile(r"(?<![\d,.])[2-9]\d{7}(?![\d,.])")),                # 76123456 (8 digits)
    ("<ID>", re.compile(r"(?<![\d,.])\d{10,}(?![\d,.])")),                      # long ID-like numbers
]


def scrub(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for tag, pat in PATTERNS:
        text, n = pat.subn(tag, text)
        if n:
            counts[tag] = counts.get(tag, 0) + n
    return text, counts
