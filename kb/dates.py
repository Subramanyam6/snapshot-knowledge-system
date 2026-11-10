"""Dates written in a question.

Only dates that carry their year count: "May 12, 2026", "12 May 2026", "5/12/2026", "2026-05-12",
and "May 2026" for a whole month. A day or a month without a year ("since May", "on June 9") is
not read as a date here: which year, and for a month which day, would be a guess.

Two words need care in this subject. "APR" is a rate, not April, and "may" is usually a verb, so
"APR 2024" and "may 2024 models" are a model year and not a month.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
from typing import List, Tuple

_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")
_NUMBER = {name: i for i, name in enumerate(_MONTHS, start=1)}
_NUMBER.update({name[:3]: i for i, name in enumerate(_MONTHS, start=1)})
_NUMBER["sept"] = 9
_NAME = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
# For a month with no day, "apr" alone is the rate and is left out.
_NAME_OF_WHOLE_MONTH = _NAME.replace("apr(?:il)?", "april")
_YEAR = r"((?:19|20)\d{2})"
_A_MODEL_YEAR_FOLLOWS = r"(?!\s*(?:-\s*)?(?:models?|model-year|vehicles?|cars?|trucks?|suvs?|autos?)\b)"
_FORMS = (                                             # pattern, then the groups holding year, month, day
    (re.compile(r"\b(%s)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+%s\b" % (_NAME, _YEAR), re.I), (3, 1, 2)),              # May 12th, 2026
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(%s)\.?,?\s+%s\b" % (_NAME, _YEAR), re.I), (3, 2, 1)),    # 12 May 2026
    (re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/%s(?![\d/])" % _YEAR), (3, 1, 2)),                                     # 5/12/2026
    (re.compile(r"(?<![\d-])%s-(\d{2})-(\d{2})(?![\d-])" % _YEAR), (1, 2, 3)),                                         # 2026-05-12
    (re.compile(r"\b(%s)\.?,?\s+%s\b%s" % (_NAME_OF_WHOLE_MONTH, _YEAR, _A_MODEL_YEAR_FOLLOWS), re.I), (2, 1, None)),  # May 2026
)


def _month(text: str) -> int:
    return int(text) if text.isdigit() else _NUMBER.get(text.lower()[:4].rstrip("."), _NUMBER.get(text.lower()[:3], 0))


def spans(text: str) -> List[Tuple[int, int, dt.date, bool]]:
    """Every dated mention as (start, end, date, whole_month), in reading order. A whole month is
    given by its last day, the way "as of May 2026" is read elsewhere."""
    found: List[Tuple[int, int, dt.date, bool]] = []
    for pattern, (y, mo, d) in _FORMS:
        for m in pattern.finditer(text):
            if any(m.start() < end and m.end() > start for start, end, _, _ in found):
                continue                             # part of a fuller date already read
            year, month = int(m.group(y)), _month(m.group(mo))
            try:
                day = int(m.group(d)) if d else calendar.monthrange(year, month)[1]
                found.append((m.start(), m.end(), dt.date(year, month, day), d is None))
            except (ValueError, IndexError, calendar.IllegalMonthError):
                continue                             # "February 30", "13/45/2026": not a date
    return sorted(found)


def dates_in(text: str) -> List[dt.date]:
    """The dates a question names, in reading order."""
    return [day for _, _, day, _ in spans(text)]


def without_dates(text: str) -> str:
    """The text with its dates blanked out, for readers that look for other numbers (a term, a
    model year): the 2026 of "May 12, 2026" is not the 2026 of a "2026-23 Model Year" band."""
    out, last = [], 0
    for start, end, _, _ in spans(text):
        out.append(text[last:start])
        out.append(" ")
        last = end
    out.append(text[last:])
    return "".join(out)
