"""The fact sheet: what the site currently states, already resolved, in a form a reader can scan.

It holds every current rate table (one copy each, stale copies left out) and the newest copy of
each repeated statement. The model reads this sheet, with passages from the cleaned pages rather than
the raw ones, so it never sees a superseded rate and never has to decide which copy to trust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .hours import contact_details
from .pages import Page
from .snapshot import SnapshotView
from .tables import RateRow, RateTable


@dataclass
class Sheet:
    text: str
    sources: Dict[str, List[Page]] = field(default_factory=dict)     # "T3" / "O1" / "H2" -> pages that back it
    hours_text: str = ""                                             # the branch schedules, shown on demand
    items: Dict[str, str] = field(default_factory=dict)              # "T3" / "O1" / "H2" -> what that item says


def _label(table: RateTable, row: RateRow) -> str:
    """Row label without the product name repeated in front of it."""
    parts = [p.strip() for p in row.label.split("|")]
    name = table.caption.strip().lower()
    parts = [p for p in parts if p and p.lower() != name] or parts
    return ", ".join(p for p in parts if p)


def _row_line(table: RateTable, row: RateRow) -> str:
    values = "; ".join("%s %s" % (metric, value) for metric, value in row.values.items())
    extras = "; ".join("%s %s" % (k.replace("*", "").strip(), v) for k, v in row.extras.items()
                       if "payment" not in k.lower() and "points" not in k.lower())
    label = _label(table, row)
    line = "- %s: %s" % (label, values) if label else "- " + values
    return line + (" (%s)" % extras if extras else "")


def table_text(table: RateTable) -> str:
    """One rate table as plain lines: its name, the date printed with it, its rows."""
    heading = " > ".join(x for x in (table.section if table.section != table.caption else "", table.caption) if x) or "Rates"
    dated = "effective %s" % table.effective.strftime("%m/%d/%Y") if table.effective else "no effective date printed"
    return "%s (%s)\n%s" % (heading, dated, "\n".join(_row_line(table, row) for row in table.rows))


def build_sheet(view: SnapshotView) -> Sheet:
    copies: Dict[int, List[Page]] = {}
    for fact in view.facts:
        pages = copies.setdefault(id(fact.current.table), [fact.current.page])
        for other in fact.copies:
            if all(other.page.url != p.url for p in pages):
                pages.append(other.page)

    lines = [
        "RATE SHEET - every rate published on the website as of %s." % view.taken_on.strftime("%B %d, %Y").replace(" 0", " "),
        "Each table is the current copy; out-of-date copies found elsewhere on the site were removed.",
        "A product, term, tier or variant that is not in this sheet has no published rate.",
    ]
    sheet = Sheet(text="")
    for number, table in enumerate(view.current_tables, start=1):
        ref = "T%d" % number
        heading = " > ".join(x for x in (table.section if table.section != table.caption else "", table.caption) if x) or "Rates"
        kind = table.page.kind
        note = "" if kind in ("rates", "product") else " [from a %s page]" % kind
        dated = "effective %s" % table.effective.strftime("%m/%d/%Y") if table.effective else "no effective date printed"
        pages = copies.get(id(table), [table.page])
        # Where a member can see this table: up to three rate or product pages that carry it, the
        # product's own page first, since that is the page the site itself links to for these rates.
        # A table found only on another kind of page (a promo, an article) points to that page.
        official = sorted((p for p in pages if p.kind in ("rates", "product")),
                          key=lambda p: (p.kind != "product", p.url.rstrip("/").count("/"), len(p.url), p.url))
        shown_on = [p.url for p in official][:3] or [table.page.url]
        lines.append("")
        lines.append("[%s] %s (%s)%s - shown on %s" % (ref, heading, dated, note, " and ".join(shown_on)))
        rows = [_row_line(table, row) for row in table.rows]
        lines.extend(rows)
        sheet.sources[ref] = pages
        sheet.items[ref] = "%s %s %s" % (heading, table.page.title, " ".join(rows))

    current = [s for s in view.statements if s.replaced_by is None]
    if current:
        lines.append("")
        lines.append("ORGANIZATION FACTS - the most recently published statement of each figure. Older pages that")
        lines.append("gave other figures were superseded and removed.")
        for number, statement in enumerate(current, start=1):
            ref = "O%d" % number
            mention = statement.current
            lines.append("[%s] (published %s, %s) %s" % (ref, mention.page.published_on.isoformat(), mention.page.url, mention.sentence))
            sheet.sources[ref] = [mention.page]
            sheet.items[ref] = mention.sentence

    contact_page = next((p for p in view.pages if view.locator and p.url.rstrip("/") == view.locator), None)
    details = contact_details(contact_page)
    if details:
        lines.append("")
        lines.append("CONTACT DETAILS - as printed on the contact page (%s). Use these for the organisation's own address," % contact_page.url)
        lines.append("phone number and routing number; a number printed for one department or branch is not the general one.")
        lines.append("[C1] " + ". ".join(details) + ".")
        sheet.sources["C1"] = [contact_page]
        sheet.items["C1"] = "contact address phone mail mailing routing number " + " ".join(details)
    sheet.text = "\n".join(lines)

    # Branch hours are kept apart from the sheet: they are shown only with questions about
    # branches or hours (reader.py). Placed in front of every question they drew unrelated
    # questions towards them.
    if view.hours:
        lines = ["BRANCH HOURS - every branch schedule the website prints. Hours differ from branch to branch; each",
                 "entry comes from an article about one branch. A branch that has no entry has no published hours."]
        for number, block in enumerate(view.hours, start=1):
            ref = "H%d" % number
            page = block.page
            dated = "published %s" % page.published_on.isoformat() if page.published_on else "crawled %s" % page.fetched_date
            about = " ".join(x for x in (block.lede, block.intro if block.intro != block.lede else "") if x)
            lines.append("[%s] %s (%s, %s) %s %s" % (ref, page.title.split("|")[0].strip(), dated, page.url, about, block.text()))
            sheet.sources[ref] = [page]
            sheet.items[ref] = lines[-1] + " branch hours lobby drive-up"
        sheet.hours_text = "\n".join(lines)

    branches = [place for place in view.locations if "branch" in place.kind.lower()]
    if branches:
        lines = ["BRANCH LOCATIONS - the branches the locator on the contact page (%s) lists. It shows the ones nearest its" % view.locator,
                 "default search, not every branch; a branch that is not here may still exist."]
        for number, place in enumerate(branches, start=1):
            ref = "L%d" % number
            lines.append("[%s] %s in %s: %s%s" % (ref, place.kind, place.city, place.address, " (%s)" % place.features if place.features else ""))
            sheet.sources[ref] = [p for p in view.pages if p.url.rstrip("/") == view.locator]
            sheet.items[ref] = "branch address location located where " + lines[-1]
        sheet.hours_text = (sheet.hours_text + "\n\n" if sheet.hours_text else "") + "\n".join(lines)
    return sheet
