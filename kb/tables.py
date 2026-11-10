"""Read rate tables out of a page.

A rate table is a markdown table with percentages in it. For each one we keep what a careful
reader would note: which product it is about (the caption above it), every row with its tier or
term, each rate with its label (rate, APR, APY...), and the "effective as of" date printed below.
That date is the strongest evidence of which copy of a table is current.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")
_EFFECTIVE = re.compile(r"effective(?:\s+as\s+of)?:?\s+(\d{1,2})/(\d{1,2})/(\d{2,4})", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_RANGE = re.compile(r"^\D*(\d+(?:\.\d+)?)\s*%\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*%")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_ONLY_LINKS = re.compile(r"^(?:\s*\[[^\]]*\]\([^)]*\)\s*)+$")
_ATTRIBUTE_HEADER = re.compile(r"\b(min|minimum|max|maximum|payment|points|fee|fees)\b", re.I)
_FINE_PRINT = re.compile(r"%|\bAPR\b|\bAPY\b|\brates?\b|dividend|annual percentage|subject to (?:change|approval)|"
                         r"payment (?:example|obligation)|minimum (?:amount|balance)", re.I)
_BLANK_CELLS = {"", "-", "—", "–", "n/a", "none"}


@dataclass
class RateRow:
    label: str                      # e.g. "2021-25 Model Year | 60 Months"
    values: Dict[str, str]          # metric -> value as printed, e.g. {"APR": "6.05%"}
    numbers: Dict[str, Tuple[float, ...]]   # metric -> numeric form used to compare copies
    extras: Dict[str, str]          # remaining cells, e.g. {"Min. to Earn": "$25,000.00"}
    key: object = None              # identity of the fact (facts.RowKey); set by facts.assign_keys
    qualifiers: object = None       # term / tier / model-year band (facts.Qualifiers)


@dataclass
class RateTable:
    section: str                    # broader heading, e.g. "Credit Cards"
    caption: str                    # e.g. "Platinum Visa"
    header: List[str]
    label_header: str               # headings of the label columns, e.g. "Loan Program | Term"
    rows: List[RateRow]
    effective: Optional[dt.date]
    first_line: int                 # index of the caption line in page.lines
    last_line: int                  # last row, or the "effective as of" line when present
    notes_end: int = 0              # last line of the notes that belong to this table
    page: object = None             # the Page this table sits on
    stale: bool = False             # set by facts.py when a newer copy disagrees

    @property
    def name(self) -> str:
        return self.caption or self.section


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _plain(line: str) -> str:
    """Strip markdown decoration from a would-be caption line."""
    line = _LINK.sub(r"\1", line)
    return re.sub(r"^[#>\-\s*_]+|[*_\s]+$", "", line).strip()


def _is_caption(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("|") or _ONLY_LINKS.match(stripped):
        return False
    is_link_bullet = bool(re.match(r"^[-*]\s*\[[^\]]+\]\([^)]*\)\s*$", stripped))
    if re.match(r"^[-*]\s", stripped) and not is_link_bullet:      # a feature bullet, not a title
        return False
    text = _plain(stripped)
    if not text or len(text) > 70 or text[-1] in ".:!?," or ":" in text or _EFFECTIVE.search(text):
        return False
    return bool(re.search(r"[A-Za-z]", text))


def metric_name(header: str, cell: str) -> str:
    """Name the kind of rate a cell holds. The cell wins when it says APR/APY itself, because
    some copied tables carry the wrong column heading."""
    in_cell = re.search(r"\b(APR|APY)\b", cell)
    low = re.sub(r"[*\\]", "", header.lower())
    extra = ""
    m = re.search(r"\b(for (?:the )?1st .*|for (?:the )?first .*|after .*|during .*)$", low)
    if m:
        extra = " " + m.group(1).strip()
        low = low[: m.start()]
    if in_cell:
        base = in_cell.group(1)
    elif "apy" in low or "percentage yield" in low:
        base = "APY"
    elif "apr" in low or "percentage rate" in low:
        base = "APR"
    elif "dividend" in low:
        base = "dividend rate"
    else:
        base = "rate"
    return base + extra


def _value(cell: str) -> Tuple[str, Tuple[float, ...]]:
    """Printed value and its numeric form. Only the leading rate counts for comparison, so
    '2.90% APR for 6 mos. (after that ...)' and '2.90%' are the same value."""
    rng = _RANGE.match(cell)
    if rng:
        return "%s%% - %s%%" % rng.groups(), (float(rng.group(1)), float(rng.group(2)))
    first = _PERCENT.search(cell)
    shown = re.sub(r"\s*\b(APR|APY)\b\s*$", "", cell).strip()
    return shown, (float(first.group(1)),)


def _effective_date(m: "re.Match[str]") -> Optional[dt.date]:
    month, day, year = (int(x) for x in m.groups())
    if year < 100:
        year += 2000
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def parse_tables(lines: List[str]) -> List[RateTable]:
    """Find every rate table in a page's cleaned lines."""
    tables: List[RateTable] = []
    captions: List[Tuple[int, str]] = []      # caption-like lines seen since the previous table
    section = ""
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        is_header = line.lstrip().startswith("|") and i + 1 < n and _SEPARATOR.match(lines[i + 1])
        if not is_header:
            if _is_caption(line):
                captions.append((i, _plain(line)))
            i += 1
            continue

        header = _cells(line)
        j = i + 2
        raw_rows: List[List[str]] = []
        while j < n and lines[j].lstrip().startswith("|"):
            raw_rows.append(_cells(lines[j]))
            j += 1

        caption, first_line = "", i
        if captions:
            first_line, caption = captions[-1]
            if len(captions) >= 2:
                section = captions[-2][1]
        captions = []

        rows, label_header = _rate_rows(header, raw_rows)
        if rows:
            effective, last_line = None, j - 1
            seen = 0
            for k in range(j, n):
                if not lines[k].strip():
                    continue
                if lines[k].lstrip().startswith("|") or seen >= 12:
                    break
                seen += 1
                m = _EFFECTIVE.search(lines[k])
                if m:
                    effective, last_line = _effective_date(m), k
                    break
            tables.append(RateTable(section=section, caption=caption, header=header, label_header=label_header,
                                    rows=rows, effective=effective, first_line=first_line, last_line=last_line))
        i = j

    # The fine print under a table (footnotes, payment examples, "see details" links) belongs to it
    # and goes stale with it. It ends at the first line that is ordinary page content.
    for idx, table in enumerate(tables):
        limit = tables[idx + 1].first_line - 1 if idx + 1 < len(tables) else n - 1
        end = table.last_line
        for k in range(table.last_line + 1, limit + 1):
            if not lines[k].strip():
                continue
            if lines[k].startswith("#") or not (_FINE_PRINT.search(lines[k]) or _ONLY_LINKS.match(lines[k].strip())):
                break
            end = k
        table.notes_end = end
    return tables


def _rate_rows(header: List[str], raw_rows: List[List[str]]) -> Tuple[List[RateRow], str]:
    """Split each row into its label, its rates and its other cells. Returns ([], "") when the
    table holds no rates."""
    width = len(header)
    raw_rows = [r + [""] * (width - len(r)) for r in raw_rows if any(c.strip() for c in r)]
    if not raw_rows:
        return [], ""
    value_cols, label_cols, extra_cols = [], [], []
    for c in range(width):
        column = [r[c] for r in raw_rows]
        with_percent = sum(1 for cell in column if _PERCENT.search(cell))
        if _ATTRIBUTE_HEADER.search(header[c]):
            extra_cols.append(c)
        elif with_percent * 2 >= len(column) and with_percent:
            value_cols.append(c)
        else:
            label_cols.append(c)
    if not value_cols:
        return [], ""
    rows: List[RateRow] = []
    for r in raw_rows:
        values, numbers = {}, {}
        for c in value_cols:
            if not _PERCENT.search(r[c]):
                continue
            metric = metric_name(header[c], r[c])
            if metric in values:                       # two columns with the same label
                metric = "%s (%s)" % (metric, header[c].strip())
            values[metric], numbers[metric] = _value(r[c])
        if not values:
            continue
        label = " | ".join(r[c] for c in label_cols if r[c].strip().lower() not in _BLANK_CELLS)
        extras = {header[c].strip(): r[c] for c in extra_cols if r[c].strip().lower() not in _BLANK_CELLS}
        rows.append(RateRow(label=label, values=values, numbers=numbers, extras=extras))
    return rows, " | ".join(header[c] for c in label_cols)
