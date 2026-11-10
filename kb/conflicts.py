"""List the places where the website contradicts itself, with the value we believe and why.

Four kinds of contradiction are reported, each tied to one fact:
  * live:      another page in the newest snapshot still shows a different value for the same row
               of a rate table (same product, variant, term and tier);
  * changed:   the official value of that row was different in an earlier snapshot;
  * statement: dated pages repeat one sentence with different numbers (branch count, members...);
  * restated:  a sentence on an official page keeps its wording from one snapshot to the next
               while a figure in it changes ("grants of up to $1,000" became "up to $1,500").

Different tiers, terms or products are never reported against each other: that is the job of
the row identity in facts.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .facts import Fact
from .history import followed_back
from .snapshot import SnapshotView
from .statements import _template
from .text import sentences

# Everyday names for the site's product words, added to topics so they read naturally.
_ALIASES = (("home loan", "mortgage"), ("certificate", "CD"), ("auto", "car / vehicle loan"))


@dataclass
class Record:
    topic: str
    values: List[str]
    resolution: str
    reason: str
    sources: List[Tuple[str, str, str]] = field(default_factory=list)   # (url, snapshot_id, quote)
    kind: str = ""


def _add(values: List[str], seen: set, value: str, number: Tuple[float, ...]) -> None:
    if number not in seen:
        seen.add(number)
        values.append(value)


def _topic(fact: Fact, metrics: Sequence[str]) -> str:
    table, row = fact.current.table, fact.current.row
    term = fact.key.term
    lead = ""
    if term:
        in_years = term % 12 == 0 and re.search(r"\d\s*-?\s*year", row.label, re.I)
        lead = "%d-year " % (term // 12) if in_years else "%d-month " % term
    product = re.sub(r"\s*&\s*\w+", "", table.name).strip()
    singular = re.sub(r"s\b$", "", product).strip()
    topic = "%s%s %s" % (lead, singular.lower(), " / ".join(metrics))
    detail = ", ".join(x for x in (table.name, row.label.replace(" | ", ", ")) if x)
    aliases = [alias for word, alias in _ALIASES if word in (table.name + " " + table.section).lower()]
    if aliases:
        detail += "; " + ", ".join(aliases)
    return "%s (%s)" % (topic.strip(), detail)


def build_conflicts(views: Sequence[SnapshotView]) -> List[Record]:
    if not views:
        return []
    latest = views[-1]
    by_row: Dict[int, List[Fact]] = {}
    for fact in latest.facts:
        if not fact.current.table.stale:
            by_row.setdefault(id(fact.current.row), []).append(fact)

    live: List[Record] = []
    changed: List[Record] = []
    for facts in by_row.values():
        facts.sort(key=lambda f: list(f.current.row.values).index(f.metric))
        first = facts[0]
        values: List[str] = []
        seen: set = set()
        sources: List[Tuple[str, str, str]] = []
        reasons: List[str] = []
        has_live = has_change = False
        for fact in facts:
            for loser in fact.stale:
                has_live = True
                _add(values, seen, loser.value, loser.number)
                sources.append((loser.page.url, latest.snapshot_id, "%s: %s %s" % (loser.row.label, loser.value, fact.metric)))
                reasons.append(fact.reason(loser))
            for view, old in followed_back(views, fact):
                if old.current.number != fact.current.number:
                    has_change = True
                    _add(values, seen, old.current.value, old.current.number)
                    sources.append((old.current.page.url, view.snapshot_id,
                                    "%s: %s %s" % (old.current.row.label, old.current.value, old.metric)))
        if not (has_live or has_change):
            continue
        for fact in facts:
            _add(values, seen, fact.current.value, fact.current.number)
        current = first.current
        sources.append((current.page.url, latest.snapshot_id,
                        "%s: %s" % (current.row.label, ", ".join("%s %s" % (v, m) for m, v in current.row.values.items()))))
        resolution = ", ".join("%s %s" % (f.current.value, f.metric) for f in facts)
        why = []
        if has_live:
            why.append("the current table wins over the other live copy: " + sorted(set(reasons))[0])
        if has_change:
            why.append("the official value changed between snapshots; the newest snapshot (%s) is current" % latest.taken_on.isoformat())
        if current.table.effective:
            why.append("current table is dated effective %s" % current.table.effective.isoformat())
        record = Record(topic=_topic(first, [f.metric for f in facts]), values=values, resolution=resolution,
                        reason="; ".join(why), sources=_unique(sources), kind="live" if has_live else "changed")
        (live if has_live else changed).append(record)

    statements: List[Record] = []
    for statement in latest.statements:
        if statement.replaced_by is not None:
            continue
        current = statement.current
        for slot in range(statement.slots()):
            seen_values = statement.slot_values(slot)
            for other in latest.statements:
                if other.replaced_by is statement:
                    for s2 in range(other.slots()):
                        if other.slot_context(s2) == statement.slot_context(slot):
                            seen_values += [v for v in other.slot_values(s2) if v not in seen_values]
            if len(seen_values) < 2:
                continue
            label = statement.slot_label(slot).replace("#", "(number)")
            if "membership" in label:
                label += " - members"
            # Each value is given with the words that say what it counts: "12 branch locations",
            # not a bare "12" that could be anything.
            unit = _unit(statement.slot_context(slot, width=3)[1])
            hedge = {}
            for mention in statement.mentions:          # "over 600", "nearly 700": as the site words it
                found = re.search(r"\b(more than|at least|close to|over|nearly|almost|about|approximately|around|roughly)\s+\$?%s"
                                  % re.escape(mention.values[slot].lstrip("$")), mention.sentence, re.I)
                if found:
                    hedge.setdefault(mention.values[slot], found.group(1).lower())
            worded = lambda v: " ".join(x for x in (hedge.get(v, ""), v, unit) if x)
            ordered = [worded(v) for v in seen_values if v != current.values[slot]][:6] + [worded(current.values[slot])]
            sources = [(current.page.url, latest.snapshot_id, current.sentence)]
            for mention in sorted(statement.superseded, key=lambda m: m.page.published_on, reverse=True)[:3]:
                sources.append((mention.page.url, latest.snapshot_id, mention.sentence))
            statements.append(Record(
                topic="%s (figure repeated across press releases and articles)" % label.strip(),
                values=ordered, resolution=worded(current.values[slot]),
                reason="most recently published page that states it (%s); older releases repeat earlier figures"
                       % current.page.published_on.isoformat(),
                sources=_unique(sources), kind="statement"))

    return _merge_same_values(live) + statements + _merge_same_values(changed + _restated(views))


_LITTLE_WORDS = frozenset("a an the of and or in on at to by for with from that which has have had is are was were as its their "
                          "throughout across over under into than then".split())
_DATE_ONLY = re.compile(r"\beffective\b|\bas of\b|\bupdated\b|\brevised\b|copyright|©", re.I)
_OFFICIAL = ("rates", "product", "document")


def _unit(following: str) -> str:
    """The words right after a number that name what it counts, up to the first little word."""
    words = []
    for word in following.split():
        if word in _LITTLE_WORDS or word == "#":
            break
        words.append(word)
    return " ".join(words[:3])


def _said_once(page) -> Dict[str, Tuple[Tuple[str, ...], str]]:
    """Sentences with a figure whose wording occurs once on the page: template -> (values, sentence)."""
    seen: Dict[str, List[Tuple[Tuple[str, ...], str]]] = {}
    for sentence in sentences(page.text):
        if sentence.startswith("|") or not 30 <= len(sentence) <= 300 or _DATE_ONLY.search(sentence):
            continue
        template, values = _template(sentence)
        if values and len(template.split()) >= 5:
            seen.setdefault(template, []).append((values, sentence))
    return {template: found[0] for template, found in seen.items() if len(found) == 1}


def _a_real_figure(before: Tuple[str, ...], after: Tuple[str, ...]) -> bool:
    """At least one of the figures that changed is an amount, a rate or a number of three digits
    or more. A footnote mark going from 10 to 1 is not a change of substance, and neither is a
    year rolling forward ("since 2025" becoming "since 2026") on its own."""
    if len(before) != len(after):
        return False
    changed = [value for pair in zip(before, after) if pair[0] != pair[1] for value in pair]
    return any("$" in v or "%" in v or (len(re.sub(r"\D", "", v)) >= 3 and not re.fullmatch(r"(?:19|20)\d{2}", v.strip()))
               for v in changed)


def _restated(views: Sequence[SnapshotView]) -> List[Record]:
    """A sentence on an official page that keeps its wording from one snapshot to the next while a
    figure in it changes. The wording must still stand, with the newest figure, in the newest
    snapshot: only then do we know which figure is current."""
    latest = views[-1]
    now = {page.url: _said_once(page) for page in latest.pages if page.kind in _OFFICIAL}
    history: Dict[Tuple[str, str], List[Tuple[str, Tuple[str, ...], str]]] = {}
    for view in views[:-1]:
        for page in view.pages:
            current = now.get(page.url)
            if not current:
                continue
            for template, (values, sentence) in _said_once(page).items():
                if template in current and values != current[template][0] and _a_real_figure(values, current[template][0]):
                    history.setdefault((page.url, template), []).append((view.snapshot_id, values, sentence))
    records: List[Record] = []
    titles = {page.url: page.title.split("|")[0].strip() for page in latest.pages}
    for (url, template), earlier in history.items():
        new_values, new_sentence = now[url][template]
        differs = [i for i in range(len(new_values)) if any(old[i] != new_values[i] for _, old, _ in earlier)]
        values: List[str] = []
        for _, old, _ in earlier:
            for i in differs:
                if old[i] != new_values[i] and old[i] not in values:
                    values.append(old[i])
        values += [new_values[i] for i in differs if new_values[i] not in values]
        sources = [(url, sid, sentence) for sid, _, sentence in earlier[-2:]] + [(url, latest.snapshot_id, new_sentence)]
        records.append(Record(
            topic="%s: %s" % (titles.get(url, url), template.replace("#", "(number)")[:150]),
            values=values, resolution=", ".join(new_values[i] for i in differs),
            reason="the official page changed this figure between snapshots; the newest snapshot (%s) is current" % latest.taken_on.isoformat(),
            sources=_unique(sources), kind="restated"))
    return records


def _unique(sources: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    seen, out = set(), []
    for s in sources:
        if (s[0], s[1]) not in seen:
            seen.add((s[0], s[1]))
            out.append(s)
    return out[:8]


def _merge_same_values(records: List[Record]) -> List[Record]:
    """Variants that carry identical numbers (a Roth and a Traditional IRA special, say) are one
    contradiction, not two."""
    merged: Dict[Tuple[str, ...], Record] = {}
    for r in records:
        key = tuple(sorted(r.values)) + (r.resolution,)
        if key in merged:
            first = merged[key]
            if r.topic not in first.topic and len(first.topic) < 300:
                first.topic += " | " + r.topic
            first.sources = _unique(first.sources + r.sources)
        else:
            merged[key] = r
    return list(merged.values())
