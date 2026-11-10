"""Find sentences that many dated pages repeat with different numbers, and pick the newest.

Press releases and articles end with the same "about us" paragraph, year after year:
"... It has over 400 employees and 12 branch locations." Each release was right on the day it
was published, so the site ends up stating 9, 10, 11 and 12 branches at once. We find
such sentences by hiding their numbers and comparing what is left. The copy on the most recently
published page is the current statement; older copies with other numbers are superseded.

Only widely repeated sentences qualify. A sentence that differs on every page it appears on
("The phone number for the ... branch is ...") states a different fact each time and is left alone.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .pages import Page
from .text import digits_for_number_words, sentences, tokens

_ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth"
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?|\b(?:" + _ORDINALS.replace(" ", "|") + r")\b", re.I)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# "over 600", "nearly 700" and "more than 650" are the same sentence with a different number.
_HEDGE = re.compile(r"\b(?:more than|at least|close to|over|nearly|almost|about|approximately|around|roughly)\s+(?=#)")
_NOT_A_NOUN = frozenset("with and that which from into over under about than then this these those their there "
                        "million billion thousand percent".split())
MIN_PAGES = 5                  # the sentence must be repeated on at least this many dated pages...
MIN_PAGES_PER_VARIANT = 2.0    # ...and each wording must, on average, be shared by several of them
# A running count: "the fourth time", "a third consecutive year".
_RANK = {word: rank for rank, word in enumerate(_ORDINALS.split(), start=1)}
_RUNNING = re.compile(r"\b(?:the|a|an)\s+(%s)\s+(?:consecutive\s+|straight\s+)?(?:time|year)\b" % _ORDINALS.replace(" ", "|"), re.I)


@dataclass
class Mention:
    page: Page
    sentence: str
    values: Tuple[str, ...]


@dataclass
class Statement:
    """One repeated sentence with all its numeric variants."""
    template: str                      # the sentence with numbers replaced by '#'
    mentions: List[Mention]
    current: Mention = None            # the copy on the most recently published page
    superseded: List[Mention] = field(default_factory=list)
    replaced_by: "Statement" = None    # a newer sentence that states the same figure in other words
    running: bool = False              # a running count ("the fifth time ..."), see _running_counts

    def slots(self) -> int:
        return self.template.count("#")

    def slot_values(self, slot: int) -> List[str]:
        """Distinct values seen in one number position, newest first."""
        seen: List[str] = []
        for m in sorted(self.mentions, key=lambda m: m.page.published_on, reverse=True):
            if m.values[slot] not in seen:
                seen.append(m.values[slot])
        return seen

    def slot_context(self, slot: int, width: int = 2) -> Tuple[str, str]:
        parts = self.template.split("#")
        return " ".join(parts[slot].split()[-width:]), " ".join(parts[slot + 1].split()[:width])

    def reworded_patterns(self) -> List["re.Pattern[str]"]:
        """Patterns for an outdated figure stated in other words. If the repeated sentence says
        "12 branch locations" today and older releases said 11, then "Of our 11 local
        branches ..." on an older page is the same outdated figure. A pattern is the old value
        followed, within two words, by the noun that follows the number in the repeated sentence."""
        patterns = []
        if self.running:
            return patterns          # "fourth ... time" is far too common a phrase to hunt for elsewhere
        parts = self.template.split("#")
        for slot in range(self.slots()):
            following = parts[slot + 1].split()[:1]
            if not following or following[0] in _NOT_A_NOUN or len(following[0]) < 4:
                continue
            noun = re.escape(following[0].rstrip("s"))
            for value in self.slot_values(slot):
                if value != self.current.values[slot]:
                    patterns.append(re.compile(r"(?<![\d.,$])%s(?![\d.,%%])(?:\W+[\w-]+){0,2}?\W+%s" % (re.escape(value), noun), re.I))
        return patterns

    def slot_label(self, slot: int) -> str:
        """A few words around a number position, to name it: '# branch locations'."""
        before, after = self.slot_context(slot, width=4)
        return (before + " # " + after).strip()


def _template(sentence: str) -> Tuple[str, Tuple[str, ...]]:
    plain = digits_for_number_words(_LINK.sub(r"\1", sentence)).replace("#", " ")   # '#' marks a number below
    values = tuple(v.lower() for v in _NUMBER.findall(plain))
    masked = _NUMBER.sub("#", plain.lower())
    masked = re.sub(r"[^a-z# ]+", " ", masked)
    masked = _HEDGE.sub("", re.sub(r"\s+", " ", masked))
    return masked.strip(), values


def find_statements(pages: Iterable[Page]) -> List[Statement]:
    """Repeated, dated sentences whose numbers disagree, each with its current copy chosen."""
    by_template: Dict[str, List[Mention]] = defaultdict(list)
    for page in pages:
        if not page.published_on or page.kind not in ("news", "blog"):
            continue
        seen_here = set()
        for sentence in sentences(page.text):
            if sentence.startswith("|") or not 40 <= len(sentence) <= 400:
                continue
            template, values = _template(sentence)
            if not values or template in seen_here or len(template.split()) < 6:
                continue
            seen_here.add(template)
            by_template[template].append(Mention(page, sentence, values))

    found: List[Statement] = []
    for template, mentions in by_template.items():
        variants = {m.values for m in mentions}
        if len(variants) < 2 or len(mentions) < MIN_PAGES:
            continue
        if len(mentions) / len(variants) < MIN_PAGES_PER_VARIANT:
            continue
        newest = max(m.page.published_on for m in mentions)
        latest = [m for m in mentions if m.page.published_on == newest]
        # Several pages on the newest date: take the wording most of them use.
        counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        for m in latest:
            counts[m.values] += 1
        winner_values = max(sorted(counts), key=lambda v: counts[v])
        statement = Statement(template=template, mentions=mentions)
        statement.current = next(m for m in sorted(latest, key=lambda m: m.page.url) if m.values == winner_values)
        statement.superseded = [m for m in mentions if m.values != winner_values]
        found.append(statement)

    _link_rewordings(found)
    found.extend(_running_counts(pages))
    found.sort(key=lambda s: (-len(s.mentions), s.template))
    return found


def _running_counts(pages: Iterable[Page]) -> List[Statement]:
    """"This is the fourth time Acme has been named a best employer", and a year later "... the
    fifth time ...". A running count is out of date as soon as a later article counts one
    more, even though each sentence appears on a single page. Two sentences are the same count when
    their content words are identical once the ordinal is taken out; the count must never go down."""
    groups: Dict[str, List[Mention]] = defaultdict(list)
    for page in pages:
        if not page.published_on or page.kind not in ("news", "blog"):
            continue
        seen_here = set()
        for sentence in sentences(page.text):
            if sentence.startswith("|") or not 25 <= len(sentence) <= 400:
                continue
            plain = _LINK.sub(r"\1", sentence)
            m = _RUNNING.search(plain)
            if not m or len(_template(sentence)[1]) != 1:           # the ordinal must be the only number
                continue
            key = " ".join(tokens(plain[: m.start(1)] + " " + plain[m.end(1):]))
            if len(key.split()) < 3 or key in seen_here:
                continue
            seen_here.add(key)
            groups[key].append(Mention(page, sentence, (m.group(1).lower(),)))

    found: List[Statement] = []
    for mentions in groups.values():
        if len({m.values for m in mentions}) < 2:
            continue
        newest = max(mentions, key=lambda m: (m.page.published_on, _RANK[m.values[0]], m.page.url))
        if any(_RANK[m.values[0]] > _RANK[newest.values[0]] for m in mentions):
            continue
        statement = Statement(template=_template(newest.sentence)[0], mentions=mentions, running=True)
        statement.current = newest
        statement.superseded = [m for m in mentions if m.values != newest.values]
        found.append(statement)
    return found


def _link_rewordings(statements: List[Statement]) -> None:
    """The boilerplate gets reworded over the years ("serves # members with assets of # billion"
    became "has grown to # members with assets of # billion"). When two repeated sentences share
    the words around a number, the one last published earlier is retired as a whole: all its copies
    are superseded by the newer sentence."""
    by_context: Dict[Tuple[str, str], List[Statement]] = defaultdict(list)
    for s in statements:
        for slot in range(s.slots()):
            before, after = s.slot_context(slot)
            if len((before + " " + after).split()) >= 3:
                by_context[(before, after)].append(s)
    for group in by_context.values():
        if len(group) < 2:
            continue
        leader = max(group, key=lambda s: (s.current.page.published_on, len(s.mentions)))
        for s in group:
            if s is not leader and s.replaced_by is None and s.current.page.published_on < leader.current.page.published_on:
                s.replaced_by = leader
                s.superseded = list(s.mentions)
