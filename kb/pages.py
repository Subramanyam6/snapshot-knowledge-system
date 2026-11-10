"""One crawled page as we remember it: what kind of page it is and which dates it carries.

Three dates matter when pages disagree, and a page can carry all of them:
  * fetched_at   - when the crawler downloaded this copy (from the manifest)
  * published_on - the date an article or press release prints under its headline
  * effective    - the "rates are effective as of ..." date printed under a rate table (see tables.py)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

from .text import clean_markdown

# How much a page's *kind* counts when two pages state the same fact with the same date evidence.
# Official rate pages and product pages are maintained on purpose; calculators and campaign landing
# pages are built once and forgotten; articles are true for the day they were published.
AUTHORITY = {
    "rates": 6, "product": 5, "document": 4, "news": 3, "blog": 2, "listing": 1,
    "calculator": 1, "promo": 1, "expired": 0,
}

_MONTHS = ("january february march april may june july august september october november december").split()
_MONTH_INDEX = {m: i + 1 for i, m in enumerate(_MONTHS)}
_MONTH_INDEX.update({m[:3]: i + 1 for i, m in enumerate(_MONTHS)})
_MONTH_INDEX["sept"] = 9
_DATE_LINE = re.compile(r"^(?:posted(?: on)?:?\s*|published(?: on)?:?\s*)?([A-Za-z]{3,9})\.? (\d{1,2}), ((?:19|20)\d{2})$", re.I)
_DOCUMENT_EXT = (".pdf", ".xls", ".xlsx", ".csv", ".doc", ".docx", ".ppt", ".pptx")


def parse_month_date(month: str, day: str, year: str) -> Optional[dt.date]:
    m = _MONTH_INDEX.get(month.lower().rstrip("."))
    if not m:
        return None
    try:
        return dt.date(int(year), m, int(day))
    except ValueError:
        return None


def published_date(markdown: str) -> Optional[dt.date]:
    """The date printed alone on a line near the top of an article, if any."""
    seen = 0
    for line in markdown.splitlines():
        line = line.strip().strip("*_ ")
        if not line:
            continue
        seen += 1
        if seen > 14:
            break
        m = _DATE_LINE.match(line)
        if m:
            return parse_month_date(*m.groups())
    return None


def page_kind(url: str, title: str, text: str) -> str:
    """Classify a page from its address, title and shape. Generic web conventions, no site list."""
    path = urlparse(url).path.lower().rstrip("/")
    last = path.rsplit("/", 1)[-1]
    low_title = (title or "").lower()
    if "offer has expired" in low_title or "page not found" in low_title:
        return "expired"
    if path.endswith(_DOCUMENT_EXT) or "/hubfs/" in path or "/downloadables/" in path:
        return "document"
    if "calculator" in path or "calculator" in low_title:
        return "calculator"
    if re.search(r"/(tag|author|category)/", path) or text.count("[Keep reading]") >= 3:
        return "listing"
    if re.search(r"(^|/)(newsroom|news|press|press-releases|media)(/|$)", path):
        return "news" if path.count("/") >= 2 else "listing"
    if re.search(r"(^|/)(blog|learn|resources|articles|stories)(/|$)", path) or re.search(r"/(19|20)\d\d/\d\d/", path):
        return "blog" if path.count("/") >= 2 else "listing"
    if last.endswith("-lp") or re.search(r"(offer|promo|landing|campaign)", last):
        return "promo"
    if last == "rates" or last.endswith("-rates"):
        return "rates"
    return "product"


@dataclass
class Page:
    """A page as it stood in one snapshot."""
    url: str
    title: str
    snapshot_id: str
    fetched_at: str
    kind: str
    published_on: Optional[dt.date]
    lines: List[str]                       # cleaned content, one entry per line
    content_hash: str
    tables: list = field(default_factory=list)        # filled in by tables.parse_tables

    @property
    def authority(self) -> int:
        return AUTHORITY.get(self.kind, 2)

    @property
    def fetched_date(self) -> str:
        return (self.fetched_at or "")[:10]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def make_page(url: str, title: str, markdown: str, fetched_at: str, snapshot_id: str) -> Page:
    text = clean_markdown(markdown)
    published = published_date(markdown)
    return Page(
        url=url,
        title=(title or "").strip(),
        snapshot_id=snapshot_id,
        fetched_at=fetched_at or "",
        kind=page_kind(url, title or "", text),
        published_on=published,
        lines=text.split("\n"),
        content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest(),
    )
