"""Find the rows of the current rate tables that a question is asking about.

This is ordinary matching, not language understanding: product words are scored the way a search
engine would score them, and the measurable parts of the question (a term, a balance, a model
year, "jumbo", "business") are checked against each row exactly. A 60-month question can never
return the 84-month row, whatever the wording.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .dates import without_dates
from .facts import FLAG_OF_STEM, Qualifiers
from .search import QUERY_SYNONYMS
from .tables import RateRow, RateTable
from .text import digits_for_number_words, normalise, stem, strip_request, tokens

RATE_WORDS = frozenset("rate apr apy interest dividend yield percent percentage earn earning pay paying".split())
_HIGH = re.compile(r"\b(highest|top|best|max|maximum|most)\b", re.I)
_LOW = re.compile(r"\b(lowest|minimum|min|cheapest|least)\b", re.I)
_Q_MONTHS = re.compile(r"(?<![\d.$])(\d{1,3})\s*-?\s*(?:months?|mos?\b)", re.I)
_Q_YEARS = re.compile(r"(?<![\d.$-])(\d{1,2})\s*-?\s*(?:years?|yrs?)\b", re.I)
_Q_MODEL_YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
_Q_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|million)?\b|\b(\d[\d,]{3,}(?:\.\d+)?)\s*(?:dollars|bucks)\b|\b(\d{1,3})\s*(k|thousand|million)\b", re.I)
_Q_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)")
_A_YEAR_AS_TIME = re.compile(r"\b(?:in|for|during|since|through|until|as of|(?:beginning|start|end) of)\s+(?:the\s+)?(?:year\s+)?"
                             r"(?:19|20)\d{2}\b(?!\s*(?:-\s*)?(?:models?|model-year|vehicles?|cars?|trucks?|suvs?|autos?)\b)", re.I)
_VEHICLE =re.compile(r"\b(model|vehicle|car|auto|truck|motorcycle|suv|atv|boat|rv)\b", re.I)
_FLAG_SYNONYMS = {stem(word): flag for word, flag in {
    "refi": "refinance", "refinancing": "refinance", "buy": "purchase", "buying": "purchase",
    "commercial": "business", "company": "business"}.items()}
_SCALE = {"k": 1e3, "thousand": 1e3, "million": 1e6}
_NEGATED = re.compile(r"\b(?:non|not|no|without)[\s-]+(?:an? |the )?([a-z]+)")
_STANDARD = re.compile(r"\bstandard\b")
_FOR_EACH = re.compile(r"\b(each|every|all (?:the )?(?:terms|tiers|rates)|list)\b")
_NOT_A_VALUE = re.compile(
    r"^\s*(?:when|who|whom|whose|why|how (?:do|does|did|can|could|should|would|will|long|often|soon|early|late|far)"
    r"|(?:in |on |at |by |during |for )?(?:which|what) (?:month|day|date|year|time|week|quarter|season|days|months|years))\b", re.I)
_SHARE_CERTIFICATE = re.compile(r"\bshare[\s-]+certificate(?=s?\b)")
PHRASE_BONUS = 1.0
OWN_NAME_BONUS = 1.0           # the product word the member used is part of the row's own name
# Kinds that together make up one product. An IRA is a Roth or a Traditional one: a question about
# IRAs that names neither asks about both, so neither counts as a variant nobody asked for.
_KINDS_OF = {"ira": frozenset({"roth", "traditional"})}
_NOT_A_NAME = frozenset(stem(w) for w in RATE_WORDS)


def _pairs(words: Sequence[str]) -> set:
    """Neighbouring word pairs, order ignored; rate words and numbers never name a product."""
    return {frozenset((a, b)) for a, b in zip(words, words[1:])
            if a != b and not (a in _NOT_A_NAME or b in _NOT_A_NAME or a[0].isdigit() or b[0].isdigit())}


def _tightest_tiers(rows: List[RateRow]) -> List[RateRow]:
    """For a stated amount, the tightest tier that holds it. Tiers written "$100,000 or less",
    "$200,000 or less", ... all contain a $90,000 loan, but the first is the one that applies."""
    narrowest: Dict[tuple, float] = {}
    group = lambda row: (row.qualifiers.term, row.qualifiers.years, row.key.flags)
    width = lambda row: row.qualifiers.amount[1] - row.qualifiers.amount[0]
    for row in rows:
        if row.qualifiers.amount is not None:
            narrowest[group(row)] = min(width(row), narrowest.get(group(row), width(row)))
    return [row for row in rows if row.qualifiers.amount is None or width(row) <= narrowest[group(row)]]


def _in_reading_order(rows: List[RateRow]) -> List[RateRow]:
    """Rows of one answer in the order a person would read them out: terms as the table lists
    them; within a term the standard row before the jumbo one, and tiers from the smallest balance
    or loan amount up."""
    first_seen: Dict[tuple, int] = {}
    for row in rows:
        first_seen.setdefault((row.qualifiers.term, row.qualifiers.years), len(first_seen))
    return sorted(rows, key=lambda r: (first_seen[(r.qualifiers.term, r.qualifiers.years)], "jumbo" in r.key.flags,
                                       r.qualifiers.amount or (0.0, 0.0)))


@dataclass
class Asked:
    """The measurable parts of a question."""
    words: List[str]                         # content words, plus everyday synonyms
    own_words: List[str] = field(default_factory=list)       # content words exactly as asked
    synonyms: Dict[str, List[str]] = field(default_factory=dict)
    term: Optional[int] = None
    amounts: List[float] = field(default_factory=list)
    model_year: Optional[int] = None
    flags: FrozenSet[str] = frozenset()
    excluded: FrozenSet[str] = frozenset()   # variants the question rules out ("non-jumbo")
    open_kinds: FrozenSet[str] = frozenset() # variants that are all asked for at once ("IRA": Roth and Traditional)
    percents: List[float] = field(default_factory=list)
    other_numbers: List[str] = field(default_factory=list)   # numbers that are none of the above
    wants_rate: bool = False                 # mentions a rate, or earning / paying a percentage
    names_rate: bool = False                 # uses the word rate / APR / APY / interest / dividend
    extreme: Optional[str] = None            # "high" or "low"


_RATE_NOUNS = frozenset("rate apr apy interest dividend yield".split())


def read_question(question: str) -> Asked:
    # The polite lead-in goes first: in "Could you share the rates ..." the word "share" is a
    # verb, not the Share Savings product.
    text = normalise(digits_for_number_words(strip_request(question)))
    # The site's footnotes call its certificates "share certificates"; its tables say "certificates".
    # Left as two words, "share" would point at Share Savings.
    text = _SHARE_CERTIFICATE.sub("certificate", text)
    # A date is read by the caller, never matched against row labels: the "2026" of "May 12, 2026"
    # is not the "2026" of a "2026-23 Model Year" band. ("APR 2024" is a rate and a model year,
    # not April 2024: see dates.py.)
    undated = without_dates(text)
    words = tokens(undated.replace("-", " "))
    synonyms = {w: [stem(x) for x in QUERY_SYNONYMS[w]] for w in words if w in QUERY_SYNONYMS}
    asked = Asked(words=words + [s for extra in synonyms.values() for s in extra], own_words=words, synonyms=synonyms)
    rest = undated
    m = _Q_MONTHS.search(rest)
    if m:
        asked.term = int(m.group(1))
    else:
        m = _Q_YEARS.search(rest)
        if m:
            asked.term = int(m.group(1)) * 12
    if m:
        rest = rest[: m.start()] + " " + rest[m.end():]
    for m in _Q_MONEY.finditer(rest):
        if m.group(1):
            asked.amounts.append(float(m.group(1).replace(",", "")) * _SCALE.get((m.group(2) or "").lower(), 1))
        elif m.group(3):
            asked.amounts.append(float(m.group(3).replace(",", "")))
        else:
            asked.amounts.append(float(m.group(4)) * _SCALE[m.group(5).lower()])
    rest = _Q_MONEY.sub(" ", rest)
    asked.percents = [float(x) for x in _Q_PERCENT.findall(rest)]
    rest = _Q_PERCENT.sub(" ", rest)
    # A year named as a time ("the APY in 2026") says when, not which product: it is neither a
    # model year nor a number a table has to show.
    rest = _A_YEAR_AS_TIME.sub(" ", rest)
    if _VEHICLE.search(text):
        m = _Q_MODEL_YEAR.search(rest)
        if m:
            asked.model_year = int(m.group(1))
            rest = rest[: m.start()] + " " + rest[m.end():]
    # Any number still left (the 7 and 1 of a "7/1 ARM") names a product rather than a quantity, so
    # the matching table must show it too. An ordinal such as "1st" stays a word and is not taken.
    asked.other_numbers = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)", rest)
    raw_words = {stem(w) for w in re.findall(r"[a-z]+", text)}
    flags = ({FLAG_OF_STEM[w] for w in raw_words if w in FLAG_OF_STEM}
             | {_FLAG_SYNONYMS[w] for w in raw_words if w in _FLAG_SYNONYMS})
    # "non-jumbo", "not the business one": the variant is named in order to rule it out.
    excluded = {FLAG_OF_STEM[stem(w)] for w in _NEGATED.findall(text) if stem(w) in FLAG_OF_STEM}
    if _STANDARD.search(text):
        excluded |= {"jumbo", "introductory", "intro", "promotional"} - flags
    asked.excluded = frozenset(excluded)
    asked.flags = frozenset(flags - excluded)
    for word, kinds in _KINDS_OF.items():
        if word in raw_words and not asked.flags & kinds:
            asked.open_kinds |= kinds
    # "When are dividends paid?" and "How do I earn the bonus?" mention rate words without asking
    # for a rate. A question that opens like these is about a date, a person or a procedure.
    value_question = not _NOT_A_VALUE.match(strip_request(question))
    asked.wants_rate = value_question and (bool(raw_words & RATE_WORDS) or "%" in text)
    asked.names_rate = value_question and bool(raw_words & _RATE_NOUNS)
    asked.extreme = "high" if _HIGH.search(text) else ("low" if _LOW.search(text) else None)
    if asked.extreme and _FOR_EACH.search(text):
        asked.extreme = None        # "the lowest APR for each term" asks for every row, not for one
    return asked


@dataclass
class Match:
    table: RateTable
    rows: List[RateRow]
    score: float
    exact: bool                # every measurable part of the question was matched by these rows


class RateLookup:
    """Scores the rows of one snapshot's current tables against a question."""

    def __init__(self, tables: Sequence[RateTable], ignore: FrozenSet[str] = frozenset()):
        self.tables = list(tables)
        self.row_words: Dict[int, Counter] = {}
        self.product_words: set = set()          # every word that names a product with a rate table
        df: Counter = Counter()
        n = 0
        for table in self.tables:
            context = " ".join([table.page.title.split("|")[0], table.section, table.caption, table.label_header])
            self.product_words.update(tokens((table.section + " " + table.caption).replace("-", " ")))
            for row in table.rows:
                words = Counter(tokens((context + " " + row.label + " " + " ".join(row.values)).replace("-", " ")))
                self.row_words[id(row)] = words
                self.product_words.update(row.key.names)
                df.update(words.keys())
                n += 1
        self.idf = {w: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.ignore = ignore                      # the organisation's own name: it names no product
        # Word pairs that stand side by side in a table's own name: "home loans", "share savings".
        self.name_pairs: Dict[int, set] = {id(t): _pairs(tokens(t.section.replace("-", " ")))
                                           | _pairs(tokens(t.caption.replace("-", " "))) for t in self.tables}

    def _fits(self, row: RateRow, asked: Asked) -> Tuple[bool, float, bool]:
        """(row is compatible, bonus, row matched something measurable)."""
        q: Qualifiers = row.qualifiers
        bonus, matched = 0.0, False
        if asked.term is not None and q.term is not None:
            if q.term != asked.term:
                return False, 0.0, False
            bonus, matched = bonus + 3.0, True
        if asked.model_year is not None and q.years is not None:
            if not q.years[0] <= asked.model_year <= q.years[1]:
                return False, 0.0, False
            bonus, matched = bonus + 3.0, True
        if asked.amounts and q.amount is not None:
            if not all(q.amount[0] <= a <= q.amount[1] for a in asked.amounts):
                return False, 0.0, False
            bonus, matched = bonus + 2.0, True
        return True, bonus, matched

    def find(self, asked: Asked) -> List[Match]:
        """Best-matching tables, each with the rows that fit the question."""
        matches: List[Match] = []
        query = set(asked.words)
        # A product named by two words ("home loan", or "mortgage", which members use for it) means
        # the table whose name has those words together, not every table that has both somewhere:
        # "Home Loans", not "Home Improvement Loans".
        phrases = _pairs(asked.own_words)
        for alternatives in asked.synonyms.values():
            phrases |= _pairs(alternatives)
        # A row whose own name (its label and its table's caption: facts.RowKey.names) holds the
        # product word the member used outranks a row that only shares a word with the title of the
        # page it sits on: a "Starter Savings" account that sits on a page about retirement savings
        # and IRAs is not an IRA.
        own_words = {w for w in query if w not in _NOT_A_NAME and w not in FLAG_OF_STEM and w not in self.ignore and not w[0].isdigit()}
        for table in self.tables:
            named = PHRASE_BONUS if phrases & self.name_pairs[id(table)] else 0.0
            scored: List[Tuple[float, RateRow, bool]] = []
            for row in table.rows:
                words = self.row_words[id(row)]
                base = sum(self.idf.get(w, 0.0) for w in query if w in words)
                if base <= 0:
                    continue
                ok, bonus, matched = self._fits(row, asked)
                if not ok:
                    continue
                row_flags = row.key.flags
                if asked.excluded & row_flags:
                    continue                              # the question ruled this variant out
                if asked.flags - row_flags and any(asked.flags <= r.key.flags for r in table.rows):
                    continue                              # the table has the asked variant; this row is not it
                unasked = row_flags - asked.flags - asked.open_kinds
                # "Jumbo" is not in this list: it is the same product at a higher minimum balance, so
                # a question that leaves it open gets the standard row and the jumbo row.
                penalty = 2.0 * len(unasked & {"business", "roth", "traditional", "refinance", "equity", "cash", "introductory", "intro"})
                own = OWN_NAME_BONUS if own_words & row.key.names else 0.0
                scored.append((base + bonus + named + own - penalty, row, matched))
            if not scored:
                continue
            best = max(s for s, _, _ in scored)
            rows = _in_reading_order([r for s, r, _ in scored if s >= best - 0.01])
            if asked.amounts:
                rows = _tightest_tiers(rows)
            wanted = sum(x is not None for x in (asked.term, asked.model_year)) + bool(asked.amounts)
            exact = wanted == 0 or all(self._covers(r, asked) for r in rows)
            if asked.extreme and len(rows) > 1:
                rows = [self._extreme(rows, asked.extreme)]
            matches.append(Match(table=table, rows=rows, score=best, exact=exact))
        matches.sort(key=lambda m: (-m.score, -m.table.page.authority, m.table.page.url, m.table.first_line))
        return matches

    @staticmethod
    def _covers(row: RateRow, asked: Asked) -> bool:
        q: Qualifiers = row.qualifiers
        if asked.term is not None and q.term != asked.term:
            return False
        if asked.model_year is not None and q.years is None:
            return False
        return True

    @staticmethod
    def _extreme(rows: List[RateRow], which: str) -> RateRow:
        def headline(row: RateRow) -> float:
            for metric in ("APY", "APR", "rate", "dividend rate"):
                if metric in row.numbers:
                    return row.numbers[metric][-1 if which == "high" else 0]
            return next(iter(row.numbers.values()))[0]
        return (max if which == "high" else min)(rows, key=headline)
