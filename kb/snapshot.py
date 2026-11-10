"""Everything we know about the website on one date, already cleaned of what is out of date.

Building a SnapshotView is where "what is current" gets decided, once, before any question:
  1. read every page and its rate tables;
  2. compare the copies of each rate and mark the stale tables (facts.py);
  3. find repeated dated statements and mark the superseded copies (statements.py);
  4. cut the stale tables, the sentences that quote their numbers and the superseded statements
     out of the text, and index what remains for search.
Whatever answers a question later only ever sees step 4's output.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from . import facts as facts_mod
from .hours import HoursBlock, Location, find_hours, find_locations, locator_url
from .pages import Page, make_page
from .search import Index, Passage
from .statements import Statement, find_statements
from .tables import RateTable, parse_tables
from .text import sentences, tokens

PASSAGE_CHARS = 1000           # a passage is closed at the first blank line after this much text
HARD_LIMIT = 1600              # ... and in any case at this much, blank line or not


@dataclass
class SnapshotView:
    snapshot_id: str
    taken_on: dt.date
    pages: List[Page]
    facts: List[facts_mod.Fact]
    statements: List[Statement]
    index: Index
    removed: Dict[str, List[str]] = field(default_factory=dict)    # url -> removed text, for audit
    hours: List[HoursBlock] = field(default_factory=list)          # every block of branch hours on the site
    locator: Optional[str] = None                                  # the branch-locator page, if the site has one
    locations: List[Location] = field(default_factory=list)        # the entries that page lists
    site_words: frozenset = frozenset()                            # the organisation's own name, as stems

    @property
    def current_tables(self) -> List[RateTable]:
        """One table per group of identical copies: the copy chosen as current."""
        seen, out = set(), []
        for fact in self.facts:
            table = fact.current.table
            if id(table) not in seen and not table.stale:
                seen.add(id(table))
                out.append(table)
        out.sort(key=lambda t: (-t.page.authority, t.page.url, t.first_line))
        return out


def build_view(snapshot_id: str, taken_on: str, raw_pages, embedder=None) -> SnapshotView:
    """raw_pages: objects with url, title, markdown, fetched_at (the harness Page).
    embedder: optional meaning.Embedder; with it the index can also be searched by meaning."""
    pages: List[Page] = []
    for raw in raw_pages:
        page = make_page(raw.url, raw.title, raw.markdown, raw.fetched_at or taken_on, snapshot_id)
        page.tables = parse_tables(page.lines)
        for table in page.tables:
            table.page = page
            facts_mod.assign_keys(table)
        pages.append(page)

    facts_mod.anchor_names(pages)
    facts = facts_mod.resolve(pages)
    statements = find_statements(pages)
    today = dt.date.fromisoformat(taken_on[:10])

    stale_values: Dict[str, Set[str]] = {}
    for fact in facts:
        for loser in fact.stale:
            stale_values.setdefault(loser.page.url, set()).add(loser.value)
    superseded: Dict[str, Set[str]] = {}
    for statement in statements:
        for mention in statement.superseded:
            superseded.setdefault(mention.page.url, set()).add(mention.sentence)

    # The same outdated figure, worded differently, on pages published before the current statement.
    rewordings = [(s.current.page.published_on, pattern) for s in statements if s.replaced_by is None
                  for pattern in s.reworded_patterns()]

    locator = locator_url(pages)
    removed: Dict[str, List[str]] = {}
    passages: List[Passage] = []
    for page in pages:
        if page.kind == "expired":
            continue
        older = [pattern for published, pattern in rewordings if page.published_on and page.published_on < published]
        passages.extend(_passages(page, stale_values.get(page.url, set()), superseded.get(page.url, set()), older, removed))
    # The current copy of each repeated statement also stands alone, so a question about it finds
    # the sentence itself rather than a long press release that happens to end with it.
    for statement in statements:
        if statement.replaced_by is None:
            passages.append(Passage(page=statement.current.page, heading="About the organization (most recent statement)",
                                    text=statement.current.sentence))
    return SnapshotView(snapshot_id=snapshot_id, taken_on=today, pages=pages, facts=facts,
                        statements=statements, index=Index(_merge_duplicates(passages), today, embedder=embedder),
                        removed=removed, hours=find_hours(pages, _site_words(pages)), locator=locator,
                        locations=find_locations(next((p for p in pages if locator and p.url.rstrip("/") == locator), None)),
                        site_words=_site_words(pages))


def _site_words(pages: List[Page]) -> frozenset:
    """The organisation's own name: the words that at least half of the page titles carry
    ("... | Acme Credit Union"). In a question they say whose rate is meant, never which product.
    A site of a few pages has no such pattern, so it gets none."""
    if len(pages) < 20:
        return frozenset()
    in_titles: Dict[str, int] = {}
    for page in pages:
        for word in set(tokens(page.title)):
            in_titles[word] = in_titles.get(word, 0) + 1
    return frozenset(w for w, n in in_titles.items() if n * 2 >= len(pages) and not w.isdigit())


def _value_pattern(values: Set[str]) -> Optional["re.Pattern[str]"]:
    numbers = set()
    for v in values:
        numbers.update(re.findall(r"\d+(?:\.\d+)?(?=\s*%)", v))
    if not numbers:
        return None
    # "5.50" must also catch "5.5%" and "5.500%", but never "15.50%" or "5.505%".
    alternatives = []
    for n in sorted(numbers):
        whole, _, fraction = n.partition(".")
        fraction = fraction.rstrip("0")
        alternatives.append(re.escape(whole) + (r"\." + fraction + "0*" if fraction else r"(?:\.0+)?"))
    return re.compile(r"(?<![\d.])(?:%s)\s*%%" % "|".join(alternatives))


def _passages(page: Page, stale: Set[str], superseded: Set[str], outdated: List["re.Pattern[str]"],
              removed: Dict[str, List[str]]) -> List[Passage]:
    lines = list(page.lines)
    dropped: List[str] = []
    table_at: Dict[int, RateTable] = {}
    skip: Set[int] = set()
    for table in page.tables:
        if table.stale:
            span = range(table.first_line, table.notes_end + 1)
            skip.update(span)
            dropped.append("[stale table] " + (table.caption or table.section))
        else:
            table_at[table.first_line] = table

    stale_number = _value_pattern(stale)
    out: List[Passage] = []
    heading = ""
    buffer: List[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        text = "\n".join(buffer).strip()
        if len(text) >= 25:
            out.append(Passage(page=page, heading=heading, text=text))
        buffer, size = [], 0

    def add(text: str) -> None:
        """Add a line to the passage being built. A passage is closed at a blank line once it is
        long enough; text that never has a blank line (a PDF, a long list) is closed at the hard
        limit, so that no passage is too long to be shown whole."""
        nonlocal size
        for piece in _pieces(text):
            buffer.append(piece)
            size += len(piece)
            if size >= HARD_LIMIT:
                flush()

    i, n = 0, len(lines)
    while i < n:
        if i in skip:
            i += 1
            continue
        table = table_at.get(i)
        if table is not None:
            flush()
            block = "\n".join(l for l in lines[table.first_line: table.last_line + 1] if l.strip())
            out.append(Passage(page=page, heading=table.section or heading, text=block, table=table))
            i = table.last_line + 1
            continue
        line = lines[i]
        i += 1
        if not line.strip():
            if size >= PASSAGE_CHARS:
                flush()
            elif buffer:
                buffer.append("")
            continue
        if line.startswith("#"):
            # A section heading (#, ##, ###) starts a new passage. A deeper heading is a label for
            # the lines right under it: a person's name above their title, a question above its
            # answer. Cutting there would separate the two, so it stays inside the passage.
            if len(line) - len(line.lstrip("#")) <= 3:
                flush()
                heading = line.lstrip("# ").strip()[:90]
        elif not line.lstrip().startswith("|"):
            kept = []
            for sentence in sentences(line):
                if (sentence in superseded or (stale_number is not None and stale_number.search(sentence))
                        or any(pattern.search(sentence) for pattern in outdated)):
                    dropped.append(sentence)
                else:
                    kept.append(sentence)
            line = " ".join(kept)
            if not line:
                continue
        add(line)
    flush()
    for position, passage in enumerate(out):
        passage.position = position
    if dropped:
        removed[page.url] = dropped
    return out


def _pieces(line: str) -> List[str]:
    """A line as it is, or, when it is a wall of text, cut at sentence ends into passage-sized pieces."""
    if len(line) <= HARD_LIMIT:
        return [line]
    pieces, current = [], ""
    for sentence in sentences(line):
        while len(sentence) > HARD_LIMIT:                      # no sentence ends at all: cut at a space
            cut = sentence.rfind(" ", 0, PASSAGE_CHARS) + 1 or PASSAGE_CHARS
            pieces.append((current + " " + sentence[:cut]).strip())
            current, sentence = "", sentence[cut:]
        if current and len(current) + len(sentence) > PASSAGE_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        pieces.append(current)
    return pieces


def _merge_duplicates(passages: List[Passage]) -> List[Passage]:
    """The same block (a shared rate widget, an alias URL) appears on several pages. Keep one copy,
    on the most authoritative page, and remember the others as additional sources."""
    by_text: Dict[str, Passage] = {}
    for p in sorted(passages, key=lambda p: (-p.page.authority, p.page.url)):
        digest = hashlib.sha1(re.sub(r"\s+", " ", p.text).encode("utf-8")).hexdigest()
        first = by_text.get(digest)
        if first is None:
            by_text[digest] = p
        elif all(p.page.url != q.url for q in first.pages):
            first.also_on.append(p.page)
    return list(by_text.values())
