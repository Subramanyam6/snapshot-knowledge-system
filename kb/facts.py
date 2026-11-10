"""Decide which copy of each rate is current.

The same rate table is printed on many pages (the rates page, the product page, a calculator, a
campaign landing page...). Copies drift apart: one gets updated, another does not. This module

  1. gives every table row an identity - product, variant flags, term, balance tier - so that
     "60 months" is never compared with "84 months", nor "business" with "personal";
  2. gathers the copies of each row across the site and picks the current one:
         newest "effective as of" date  >  more authoritative page kind  >  most recent crawl;
  3. marks the copies that disagree with the current one as stale, with the reason.

Rows whose identity cannot be established with confidence are left alone: they stay searchable,
but they are never called a conflict.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from .pages import Page
from .tables import RateRow, RateTable
from .text import normalise, stem

# Words arrive here as stems (text.stem cuts "introductory" to "introdu"), so both word lists
# below are looked up by stem. Looking a stem up among whole words would miss every word the
# stemmer shortens.
# Words that describe a row without telling products apart.
GENERIC = frozenset(stem(word) for word in (
    "rate rates account accounts loan loans online only annual percentage yield apr apy dividend "
    "type tier balance program product and the of or less more up to as low a an for with new "
    "model year years month months mo mos term standard conventional credit union amount").split()
)
# Words that split one product into variants which must never be mixed up.
FLAGS = frozenset(
    "business jumbo roth traditional purchase refinance special regular variable introductory "
    "intro equity cash heloc promotional youth student".split()
)
FLAG_OF_STEM = {stem(word): word for word in FLAGS}        # "introdu" -> "introductory"

_MONEY = r"\$\s?([\d,]+(?:\.\d+)?)"
_AMOUNT_RANGE = re.compile(_MONEY + r"\s*(?:-|–|to)\s*\$?\s?([\d,]+(?:\.\d+)?)")
_AMOUNT_MIN = re.compile(_MONEY + r"\s*(?:\+|and (?:up|over|above)|or more)")
_AMOUNT_MAX = re.compile(_MONEY + r"\s*(?:or less|or under|and (?:under|below))")
_TERM_MONTHS = re.compile(r"(\d{1,3})\s*-?\s*(?:months?|mos?\b)", re.I)
_TERM_YEARS = re.compile(r"(?<![\d-])(\d{1,2})\s*-?\s*years?\b", re.I)
_YEAR_BAND = re.compile(r"((?:19|20)\d{2})\s*[-–]\s*((?:19|20)?\d{2})\b")
_YEAR_ONE = re.compile(r"\b((?:19|20)\d{2})\b")
_NAME_TOKEN = re.compile(r"[a-z][a-z0-9']*|\d+(?:st|nd|rd|th)")


def _num(text: str) -> float:
    return float(text.replace(",", ""))


@dataclass(frozen=True)
class Qualifiers:
    """The measurable part of a row label (or of a question)."""
    term: Optional[int] = None                        # months
    amount: Optional[Tuple[float, float]] = None      # dollar range, inclusive
    years: Optional[Tuple[int, int]] = None           # model-year band


def parse_qualifiers(text: str) -> Tuple[Qualifiers, str]:
    """Pull term, dollar range and model-year band out of a label. Returns them plus the rest."""
    rest = normalise(text)
    amount = None
    m = _AMOUNT_RANGE.search(rest)
    if m:
        amount = (round(_num(m.group(1)), 2), round(_num(m.group(2)), 2))
    else:
        m = _AMOUNT_MIN.search(rest)
        if m:
            amount = (round(_num(m.group(1)), 2), math.inf)
        else:
            m = _AMOUNT_MAX.search(rest)
            if m:
                amount = (0.0, round(_num(m.group(1)), 2))
    if m:
        rest = rest[: m.start()] + " " + rest[m.end():]

    years = None
    if "model year" in rest:
        m = _YEAR_BAND.search(rest)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < 100:
                b += (a // 100) * 100
            years = (min(a, b), max(a, b))
        else:
            m = _YEAR_ONE.search(rest)
            if m:
                year = int(m.group(1))
                years = (0, year) if "older" in rest else (year, year)
        if m:
            rest = rest[: m.start()] + " " + rest[m.end():]

    term = None
    m = _TERM_MONTHS.search(rest)
    if m:
        term = int(m.group(1))
    else:
        m = _TERM_YEARS.search(rest)
        if m:
            term = int(m.group(1)) * 12
    if m:
        rest = rest[: m.start()] + " " + rest[m.end():]
    return Qualifiers(term=term, amount=amount, years=years), rest


def name_words(text: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Split the wording of a label into (product words, variant flags)."""
    found = {stem(t.replace("'", "")) for t in _NAME_TOKEN.findall(text)}
    flags = frozenset(FLAG_OF_STEM[t] for t in found if t in FLAG_OF_STEM)
    names = frozenset(t for t in found if t not in FLAG_OF_STEM and t not in GENERIC and len(t) > 1)
    return names, flags


@dataclass(frozen=True)
class RowKey:
    """Identity of one table row. Two rows with compatible names and equal everything else state
    the same fact."""
    names: FrozenSet[str]
    flags: FrozenSet[str]
    term: Optional[int]
    amount: Optional[Tuple[float, float]]
    band: Optional[tuple]          # model-year band by place: ("newest",), ("place", 1, 4), ("older",)

    @property
    def shape(self) -> tuple:
        return (self.flags, self.term, self.amount, self.band)


def names_match(a: FrozenSet[str], b: FrozenSet[str]) -> bool:
    """Same product? Yes when one name merely abbreviates the other ("Auto Loans" for "Auto &
    Motorcycle Loans"). When each side has a word the other lacks ("Platinum Visa" against
    "Rewards Visa"), they are two different products, however many other words they share."""
    return bool(a) and bool(b) and (a <= b or b <= a)


def assign_keys(table: RateTable) -> None:
    """Give every row of a table its identity."""
    page: Page = table.page
    caption_names, caption_flags = name_words(parse_qualifiers(table.caption)[1])
    page_flags = frozenset({"business"}) if re.search(r"/business\b|business-", page.url.lower()) else frozenset()
    parsed = []
    for row in table.rows:
        quals, rest = parse_qualifiers(row.label)
        names, flags = name_words(rest)
        parsed.append((row, quals, names, flags))
    bands = sorted({q.years for _, q, _, _ in parsed if q.years}, key=lambda b: (b[1], b[0]), reverse=True)
    # A section heading is a short title ("Certificates"); a longer line is marketing copy.
    section = table.section if len(table.section.split()) <= 5 else ""
    section_names = name_words(parse_qualifiers(section)[1])[0]
    for row, quals, names, flags in parsed:
        product = names | caption_names
        if not product:
            # A caption such as "Regular" names a variant, not a product; the section heading
            # above it ("Certificates") says which product the table belongs to.
            product = section_names
        band = None
        if quals.years:
            # Model-year bands roll forward ("2018-20" becomes "2018-21"), so a band is known by
            # its place in the table's ladder, not by its years: the newest, the "and older" one,
            # or the n-th of a ladder of that length.
            place = bands.index(quals.years)
            if place == 0:
                band = ("newest",)
            elif quals.years[0] == 0:
                band = ("older",)
            else:
                band = ("place", place, len(bands))
        row.key = RowKey(names=frozenset(product), flags=flags | caption_flags | page_flags,
                         term=quals.term, amount=quals.amount, band=band)
        row.qualifiers = quals


def anchor_names(pages: Iterable[Page]) -> None:
    """Name the product of a table that sits on a campaign page, in a calculator or in an article.

    Such a table is often headed by a slogan or a feature bullet ("Online and mobile banking
    services") instead of the product name, so its rows would be compared with nothing and an
    out-of-date copy would pass for a product of its own. The official rates and product pages
    say which products exist. When a row's own name matches none of them but the section the
    table sits in does, the section names the product."""
    pages = list(pages)
    official = {row.key.names for page in pages if page.kind in ("rates", "product")
                for table in page.tables for row in table.rows if row.key.names}
    for page in pages:
        if page.kind in ("rates", "product", "expired"):
            continue
        for table in page.tables:
            section = table.section if len(table.section.split()) <= 5 else ""
            section_names = name_words(parse_qualifiers(section)[1])[0]
            if not section_names or not any(names_match(section_names, known) for known in official):
                continue
            for row in table.rows:
                if row.key.names and any(names_match(row.key.names, known) for known in official):
                    continue
                row.key = RowKey(names=section_names, flags=row.key.flags, term=row.key.term,
                                 amount=row.key.amount, band=row.key.band)


# ------------------------------------------------------------------------------ resolution


@dataclass
class Occurrence:
    """One printed copy of a rate."""
    table: RateTable
    row: RateRow
    metric: str

    @property
    def page(self) -> Page:
        return self.table.page

    @property
    def value(self) -> str:
        return self.row.values[self.metric]

    @property
    def number(self) -> Tuple[float, ...]:
        return self.row.numbers[self.metric]

    def rank(self) -> tuple:
        """Sort key for 'most likely current': effective date, then page kind, then crawl date."""
        return (self.table.effective or dt.date.min, self.page.authority, self.page.fetched_date)


@dataclass
class Fact:
    """One rate with every copy found on the site in one snapshot."""
    key: RowKey
    metric: str
    current: Occurrence
    copies: List[Occurrence] = field(default_factory=list)     # agree with `current`
    stale: List[Occurrence] = field(default_factory=list)      # disagree with `current`

    def reason(self, loser: Occurrence) -> str:
        win, lose = self.current, loser
        if win.table.effective and lose.table.effective and win.table.effective > lose.table.effective:
            return "newer effective date (%s vs %s)" % (win.table.effective.isoformat(), lose.table.effective.isoformat())
        if win.table.effective and not lose.table.effective:
            return "dated %s; the other copy carries no effective date" % win.table.effective.isoformat()
        if win.page.authority > lose.page.authority:
            return "%s page outranks %s page" % (win.page.kind, lose.page.kind)
        if win.page.fetched_date > lose.page.fetched_date:
            return "crawled more recently (%s vs %s)" % (win.page.fetched_date, lose.page.fetched_date)
        return "same date and page kind; kept the first copy found"


def _cluster(keys: List[RowKey]) -> Dict[RowKey, int]:
    """Group row identities that name the same product. Only rows with identical flags, term,
    tier and band are ever compared, so 'similar name' cannot merge different tiers."""
    by_shape: Dict[tuple, List[RowKey]] = {}
    for k in keys:
        by_shape.setdefault(k.shape, []).append(k)
    group: Dict[RowKey, int] = {}
    next_id = 0
    for same_shape in by_shape.values():
        anchors: List[Tuple[RowKey, int]] = []
        # Longest names first: a shorter name may join a longer one it abbreviates, but only when
        # exactly one candidate exists. "Savings" alone could be any savings product, so it joins none.
        for k in sorted(same_shape, key=lambda x: (-len(x.names), sorted(x.names))):
            if k in group:
                continue
            homes = {gid for other, gid in anchors if names_match(k.names, other.names)}
            if len(homes) == 1:
                group[k] = homes.pop()
                continue
            group[k] = next_id
            anchors.append((k, next_id))
            next_id += 1
    return group


def resolve(pages: Iterable[Page]) -> List[Fact]:
    """Compare every copy of every rate in one snapshot and mark the stale ones."""
    occurrences: List[Occurrence] = []
    for page in pages:
        if page.kind == "expired":
            continue
        for table in page.tables:
            table.stale = False
            for row in table.rows:
                for metric in row.values:
                    occurrences.append(Occurrence(table, row, metric))
    group = _cluster(sorted({o.row.key for o in occurrences}, key=lambda k: (sorted(k.names), sorted(k.flags), str(k.shape))))
    buckets: Dict[tuple, List[Occurrence]] = {}
    for o in occurrences:
        # A row without a product name can only be itself: key it to its own table.
        ident = (group[o.row.key], o.metric) if o.row.key.names else (id(o.table), o.row.label, o.metric)
        buckets.setdefault(ident, []).append(o)

    facts: List[Fact] = []
    for members in buckets.values():
        members.sort(key=lambda o: (o.rank(), o.page.url), reverse=True)
        current = members[0]
        fact = Fact(key=current.row.key, metric=current.metric, current=current)
        for other in members[1:]:
            if other.number == current.number:
                fact.copies.append(other)
            elif other.table is not current.table:
                fact.stale.append(other)
                other.table.stale = True
        facts.append(fact)
    return facts
