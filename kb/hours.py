"""Branch hours, and the page where a member looks a branch up.

A site like this prints hours inside articles about individual branches ("Branch hours are:
Lobby ... Drive-up ..."), never in one place. A general question ("what is the usual lobby schedule?")
can only be answered well from all of those blocks together, and keyword search brings up a
different two or three of them for every wording. So the blocks are gathered once, at load time,
into one list that is shown with every question about branches or hours.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .pages import Page
from .text import stem

_LABEL = re.compile(r"^[*_\s]*(lobby|drive[- ]?up|drive[- ]?thr(?:u|ough))(?: hours)?[*_\s]*:?[*_\s]*(.*)$", re.I)
_DAY = r"(?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?s?\.?"
_SCHEDULE = re.compile(r"^[*_\s]*((?:%s)(?:\s*(?:-|–|—|to|through|&|and|,)\s*(?:%s))*|weekdays?|weekends?|daily)[*_\s]*:\s*(\S.*)$"
                       % (_DAY, _DAY), re.I)
_BRANCH_NAME = re.compile(r"\b[Tt]he ([A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,3}) branch\b")
_DATELINE = re.compile(r"^([A-Z][A-Z .'-]{2,30}),\s")
# A link such as ".../contact#locator": the page the site itself sends members to for branches.
_LOCATOR_LINK = re.compile(r"\((https?://[^\s)#]+?)/?#[\w-]*locat[\w-]*\)", re.I)


@dataclass
class HoursBlock:
    """One printed schedule: the page it is on, the sentence that introduces it, and its rows."""
    page: Page
    intro: str
    sections: Dict[str, List[str]] = field(default_factory=dict)     # "Lobby" -> ["Monday-Friday: 9 a.m.-6 p.m.", ...]
    lede: str = ""                 # the article's opening sentence: it says which branch this is
    site_words: frozenset = frozenset()

    @property
    def place(self) -> str:
        """A short name for the branch: "the Main Street branch" in the introducing sentence,
        else the town in the article's dateline ("RIVERTON, Ohio - ...")."""
        m = _BRANCH_NAME.search(self.intro)
        if m:
            # "the Brookfield Acme branch": the organisation's own name is not part of the place
            words = [w for w in m.group(1).split() if stem(w.lower()) not in self.site_words]
            if words:
                return " ".join(words)
        m = _DATELINE.match(self.lede)
        return m.group(1).title() if m else ""

    def text(self) -> str:
        return " ".join("%s - %s." % (name, "; ".join(rows)) for name, rows in self.sections.items() if rows)


def _tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("*", "")).strip()


def _schedule_follows(lines: List[str], start: int) -> bool:
    for line in lines[start: start + 4]:
        if line.strip():
            return bool(_SCHEDULE.match(line.strip()))
    return False


def _intro(lines: List[str], at: int) -> str:
    """The sentence before the block; it usually names the branch ("The Brookfield branch is ...")."""
    for line in reversed(lines[max(0, at - 6): at]):
        if line.strip():
            return _tidy(line)[:260]
    return ""


def _lede(lines: List[str]) -> str:
    """The first sentence of the page's first paragraph."""
    for line in lines:
        text = line.strip()
        if len(text) >= 60 and text[0] not in "[!#>|*-":
            first = re.split(r"(?<=[a-z0-9)]\.)\s+(?=[A-Z])", _tidy(text), maxsplit=1)[0]
            return first[:240]
    return ""


def find_hours(pages: Iterable[Page], site_words: frozenset = frozenset()) -> List[HoursBlock]:
    """Every block of branch hours on the site, newest page first."""
    blocks: List[HoursBlock] = []
    for page in pages:
        if page.kind == "expired":
            continue
        lines, current = page.lines, None
        i, n = 0, len(lines)
        while i < n:
            text = lines[i].strip()
            label = _LABEL.match(text) if text else None
            rest = label.group(2).strip() if label else ""
            if label and (_SCHEDULE.match(rest) or (not rest and _schedule_follows(lines, i + 1))):
                if current is None:
                    current = HoursBlock(page=page, intro=_intro(lines, i), lede=_lede(lines), site_words=site_words)
                name = "Lobby" if label.group(1).lower().startswith("lobby") else "Drive-up"
                rows = current.sections.setdefault(name, [])
                if rest:
                    rows.append(_tidy(rest))
                i += 1
                while i < n and (not lines[i].strip() or _SCHEDULE.match(lines[i].strip())):
                    if lines[i].strip():
                        rows.append(_tidy(lines[i]))
                    i += 1
                continue
            if current is not None and text:          # ordinary text again: the block is complete
                blocks.append(current)
                current = None
            i += 1
        if current is not None:
            blocks.append(current)
    blocks = [b for b in blocks if any(b.sections.values())]
    blocks.sort(key=lambda b: (b.page.published_on.isoformat() if b.page.published_on else b.page.fetched_date, b.page.url), reverse=True)
    return blocks


_FIELD = re.compile(r"^\*\*([A-Z][\w /&.-]{2,28}?):?\*\*:?\s*(.*)$")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_CITY_LINE = re.compile(r"^[A-Z][\w .'-]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?$")


def contact_details(page: Optional[Page]) -> List[str]:
    """The labelled lines at the top of a contact page: "Address: P.O. Box 100, Springfield,
    ST 00000", "Phone: 800-555-0100", "Routing Number: ...". These are the organisation's own
    statement of how to reach it, and questions about them should not depend on which article a
    search happens to bring up."""
    if page is None:
        return []
    found: List[str] = []
    lines = [line.strip() for line in page.lines[:60]]
    for i, line in enumerate(lines):
        m = _FIELD.match(line)
        if not m:
            continue
        value = _MD_LINK.sub(r"\1", m.group(2)).strip()
        following = next((l for l in lines[i + 1: i + 3] if l), "")
        if _CITY_LINE.match(following):                  # the second line of a postal address
            value = "%s, %s" % (value, following) if value else following
        if value and len(value) <= 120:
            found.append("%s: %s" % (m.group(1).strip(), value))
    return found[:8]


_STREET_ADDRESS = re.compile(r"^\d+[\w .'#/-]*,\s*([A-Za-z .'-]+),\s*[A-Z]{2},?\s*\d{5}(?:-\d{4})?$")


@dataclass
class Location:
    """One entry of the locator: what it is ("Acme Branch"), where, and what it offers."""
    kind: str
    address: str
    city: str
    features: str = ""


def find_locations(page: Optional[Page]) -> List[Location]:
    """The entries a locator page lists: a line with a street address, the line above it saying
    what the place is, and the line of services below it. A locator shows the places nearest
    its default search, so the list is what the page prints, not every branch there is."""
    if page is None:
        return []
    lines = [line.strip() for line in page.lines]
    found: List[Location] = []
    for i, line in enumerate(lines):
        m = _STREET_ADDRESS.match(line)
        if not m:
            continue
        above = [l for l in lines[max(0, i - 3): i] if l and not l.startswith("#")]
        below = [l for l in lines[i + 1: i + 5] if l]
        features = next((l for l in below[:3] if "·" in l or "ATM" in l or "Teller" in l), "")
        if above and len(above[-1]) <= 40:
            found.append(Location(kind=above[-1], address=line, city=m.group(1).strip(), features=features if not features.startswith("#") else ""))
    return found


def locator_url(pages: Iterable[Page]) -> Optional[str]:
    """The page the site links to as its branch locator, if it has one (three or more such links)."""
    counts: Counter = Counter()
    for page in pages:
        for line in page.lines:
            if "#" in line and "locat" in line.lower():
                for m in _LOCATOR_LINK.finditer(line):
                    counts[m.group(1).rstrip("/")] += 1
    if not counts:
        return None
    url, seen = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return url if seen >= 3 else None
