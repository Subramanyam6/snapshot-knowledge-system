"""Answer "how did it change?" by comparing the copies of the website taken on different dates.

Every other answer is read from one snapshot. A question about a change needs two:

    "How has the 60-month auto loan APR changed since May 12, 2026?"
    "What did the 13-month special pay on May 19, 2026, and what does it pay now?"
    "What was the APY on June 9, 2026 and on August 27, 2026?"

Such an answer states an earlier value on purpose. Everywhere else an earlier value is exactly
what must not be said, so this path is entered only when the wording leaves no doubt (comparison):
three fixed shapes, dates that carry their year, nothing about the member's own account, nothing
about the future. Anything else is answered from one snapshot, as if this module did not exist.

  * For a rate row the answer is written by rules: the row is followed back through the earlier
    snapshots (followed_back) and its value on each date asked about is stated (rows_answer).
  * For anything else the model is shown the earlier copy next to today's (Earlier).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .dates import spans
from .facts import Fact, names_match
from .lookup import RATE_WORDS, Asked
from .rules_answer import MAX_ROWS, Draft, RulesAnswerer, _effective, _tier
from .snapshot import SnapshotView
from .text import normalise, stem, strip_request

# ------------------------------------------------------------------ is it a comparison over time?

_IN_SENTENCE = r"(?:[^?.!;]|(?<=\d)\.(?=\d))"          # any character of one sentence; "$24,999.99" does not end it
_SENTENCE_END = re.compile(r"[?!;]|(?<!\d)\.|\.(?!\d)")  # the same rule, as a place to cut


def _in_order(text: str, *patterns: str) -> bool:
    """Each pattern occurs in `text`, each one after the end of the one before it."""
    position = 0
    for pattern in patterns:
        found = re.compile(pattern, re.I).search(text, position)
        if found is None:
            return False
        position = found.end()
    return True
# Something the website publishes as a figure. "Has ... changed" must be about one: "has the rate
# changed", not "has anything changed" or "did the branch move".
_MEASURE = re.compile(r"[%$]|\b(?:rates?|apr|apy|dividends?|yields?|interest|fees?|charges?|prices?|costs?|limits?|minimums?|maximums?|"
                      r"amounts?|payments?|bonus(?:es)?|rewards?|grants?|hours|percent(?:age)?)\b", re.I)
# "Has the APR changed", "How much have auto loan rates gone up", "Did the fee go down", "What did the
# APY change from". The question must open with it, so that "have" is the helping verb of the
# change and not possession: "Does a 15-year loan have a lower rate?" is about today.
_CHANGED = (r"changed|moved|gone up|gone down|increased|decreased|risen|fallen|dropped|stayed the same|remained the same|"
            r"been (?:raised|lowered|cut|reduced|increased|decreased|changed|adjusted|updated)")
_CHANGE = r"change|move|go up|go down|increase|decrease|rise|fall|drop|stay the same|remain the same"
_OPENING = (r"^(?:(?:since|between|from|as\s+of|on|in)\s+[^?]{4,40}?,\s+"                  # "Since May 12, 2026, how has ..."
            r"(?=(?:(?:and|so|but)\s+)?(?:how|by|when|why|in\s+what|what|has|have|had|hasn't|haven't|did|didn't)\b))?"
            r"(?:(?:and|so|but|if|whether)\s+)?(?:(?:how(?:\s+much|\s+far)?|by\s+how\s+much|when|why|in\s+what\s+way|what)\s+)?")
_HAS_CHANGED = re.compile(_OPENING + r"(?:(?:has|have|had|hasn't|haven't)\s+(%(s)s{2,120}?)\s(?:%(done)s)|(?:did|didn't)\s+(%(s)s{2,120}?)\s(?:%(do)s))\b"
                          % {"s": _IN_SENTENCE, "done": _CHANGED, "do": _CHANGE}, re.I)
_WHAT_CHANGED = re.compile(r"^(?:(?:and|so|but)\s+)?(?:what(?:'s|\s+has|\s+have|\s+had)?\s+changed\b"
                           r"|(?:has|have)\s+there\s+been\s+(?:a|an|any)\s+(?:rate\s+)?(?:change|increase|decrease|rise|drop|cut)\b)", re.I)
# The present, and the words that set it beside a date: "... on May 12, 2026, and what is it now?",
# "what is it now, and what was it back on June 9, 2026?", "May 12, 2026 compared with today".
NOW = re.compile(r"\b(?:now|today|today's|currently|at present)\b", re.I)
_BESIDE = r"\b(?:and|but|vs\.?|versus|compared\s+(?:with|to)|against|than)\b"
_PAST_VERB = r"\b(?:was|were|did|difference)\b"      # "the difference between ... on May 12, 2026 and today" asks the same
_CURRENT = r"\b(?:now|today|today's|currently|current|present|at present)\b"
# "from May 25, 2026 to September 7, 2026" is a stretch of time, not two values side by side.
_OPENS_A_RANGE = re.compile(r"\b(?:between|from)\s+(?:the\s+)?$", re.I)
# No copy of the website records the member's own account, and none records the future.
_MEMBER = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|my|me|mine|myself|we|we're|we've|our|ours|us)\b", re.I)
_FUTURE = re.compile(r"\b(?:will|won't|going\s+to|gonna|plan(?:ning)?\s+to|next\s+(?:week|month|quarter|year)|tomorrow|upcoming|expect(?:ed)?\s+to)\b", re.I)
# A time that cannot be pinned to a copy of the website: a month or a day without its year, or a
# stretch counted back from today. With one of these in the question no comparison is attempted.
_MONTH_NAME = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|april|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_VAGUE_TIME = re.compile(
    r"\b(?:in|since|from|between|before|after|until|till|during|through|of|and|to|by|on|early|mid|late)[\s-]+%(m)s\b(?!\.?,?\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?(?:19|20)\d{2}\b)"
    r"|\b%(m)s\.?\s+\d{1,2}(?:st|nd|rd|th)?\b(?!,?\s+(?:19|20)\d{2}\b)"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?%(m)s\b(?!\.?,?\s+(?:19|20)\d{2}\b)"
    r"|\b(?:last|past|this|previous|prior|coming)\s+(?:\w+\s+)?(?:days?|weeks?|months?|quarters?|years?|spring|summer|fall|autumn|winter)\b"
    r"|\b(?:days?|weeks?|months?|quarters?|years?)\s+(?:ago|back|earlier)\b"
    r"|\b(?:recently|lately|yesterday|so far|a while|earlier this|start of the|beginning of the|end of the|spring|summer|autumn|winter)\b"
    % {"m": _MONTH_NAME}, re.I)


# A month named with its year stands for a day of it, chosen by the word in front:
#   "since / from / between May 2026"  -> its start ("start");
#   "did it change in / during May 2026" -> the whole month, start to end ("within");
#   otherwise ("as of May 2026", "what was it in May 2026") -> its last day ("end"), as elsewhere.
_MONTH_START = re.compile(r"\b(?:since|from|between)\s+(?:the\s+(?:very\s+)?(?:beginning|start)\s+of\s+)?$", re.I)
_MONTH_WITHIN = re.compile(r"\b(?:in|during|over|throughout)\s+(?:the\s+month\s+of\s+)?$", re.I)


def _marked(text: str, named) -> List[Tuple[dt.date, str]]:
    marks = []
    for start, _, day, whole_month in named:
        before = text[:start]
        kind = "day" if not whole_month else "within" if _MONTH_WITHIN.search(before) else "start" if _MONTH_START.search(before) else "end"
        marks.append((day, kind))
    return marks


@dataclass
class Period:
    """The period a question about a change names."""
    days: List[Tuple[dt.date, str]]      # each date named, with how to read it: "day", or for a month
                                         # "start", "end" or "within"; empty for a bare "has it changed?"
    to_today: bool                       # True when the period runs to today


def comparison(question: str) -> Optional[Period]:
    """The period asked about, when the question is without doubt about how a published figure
    differs between two times; None for every other question.

    Three shapes count:
      1. it opens with "has/have ... changed/gone up/...", or "did ... change/go up/...", about a
         figure: "How has the 60-month auto loan APR changed since May 12, 2026?";
      2. it names two dates, with their years, side by side: "on May 12, 2026 and on June 9, 2026",
         "between ... and ...", "... vs ...";
      3. it names one dated past value and today's: "what was it on May 12, 2026, and what is it now?"
    """
    text = strip_request(question)
    if _MEMBER.search(text) or _FUTURE.search(text) or _VAGUE_TIME.search(text):
        return None
    named = spans(text)
    marks = _marked(text, named)
    days = sorted({(day, "end" if kind == "within" else kind) for day, kind in marks})
    opened = _HAS_CHANGED.match(text)
    if (opened and _MEASURE.search(opened.group(1) or opened.group(2))) or (_WHAT_CHANGED.match(text) and _MEASURE.search(text)):
        if not days and NOW.search(text):
            return None                              # "did it change today?": no copy of the website can say
        if len(marks) == 1 and marks[0][1] == "within":
            return Period(days=[(marks[0][0], "start"), (marks[0][0], "end")], to_today=False)   # "did it change in May 2026?"
        return Period(days=days, to_today=len(days) < 2 or bool(NOW.search(text)))
    if len(days) >= 2:
        between = text[named[0][1]:named[1][0]]
        one_sentence = not re.search(r"[?.!;]", re.sub(r"(?<=\d)\.(?=\d)", "", between))
        if len(between) <= 80 and one_sentence and re.search(_BESIDE + r"|\bthen\b|,", between, re.I) \
                and not _OPENS_A_RANGE.search(text[:named[0][0]]):
            return Period(days=days, to_today=bool(NOW.search(text)))
        return None
    if len(days) == 1:
        # Only the sentence that holds the date can say "then" and "now" about it. Within one
        # sentence, "these words, in this order" is a plain left-to-right search, which stays fast
        # however long the question is. (A single pattern doing the same took over a minute on a
        # long rambling question.)
        before = _SENTENCE_END.split(text[:named[0][0]])[-1]
        after = _SENTENCE_END.split(text[named[0][1]:])[0]
        compare = r"\bcompar(?:e|es|ed|ing)\b"
        then_now = _in_order(before, _PAST_VERB) and _in_order(after, _BESIDE, NOW.pattern)
        now_then = _in_order(before, NOW.pattern, _BESIDE, _PAST_VERB)
        compared = ((_in_order(before, compare) and _in_order(after, r"\b(?:with|to|against|and)\b", _CURRENT))
                    or _in_order(before, _CURRENT, r"\bcompar(?:e|es|ed|ing)\s+(?:with|to|against)\b")
                    or _in_order(before, compare, _CURRENT, r"\b(?:with|to|against)\b"))
        if then_now or now_then or compared:
            return Period(days=days, to_today=True)
    return None


# ------------------------------------------------------------------ one row, followed through time

def _same_row(a: Fact, b: Fact, model_year: Optional[int]) -> bool:
    if model_year is None or b.key.band is None:
        return a.key.shape == b.key.shape
    years = a.current.row.qualifiers.years
    return ((a.key.flags, a.key.term, a.key.amount) == (b.key.flags, b.key.term, b.key.amount)
            and years is not None and years[0] <= model_year <= years[1])


def followed_back(views: Sequence[SnapshotView], fact: Fact, model_year: Optional[int] = None) -> List[Tuple[SnapshotView, Fact]]:
    """The same row in each earlier snapshot of `views`; `fact` belongs to the last one. A row is
    followed only when exactly one earlier row has the same variant, term and tier and a
    compatible product name.

    Model-year bands are followed by their place in the table ("the newest band"), because their
    labels roll forward every year. When the question names a model year, the band that held that
    year on the earlier date is followed instead: a 2022 car sat in "2022-25" in May and in
    "2020-22" by September."""
    found: List[Tuple[SnapshotView, Fact]] = []
    for view in views[:-1]:
        same = [f for f in view.facts
                if f.metric == fact.metric and _same_row(f, fact, model_year)
                and (names_match(f.key.names, fact.key.names) or not f.key.names or not fact.key.names)
                and not f.current.table.stale]
        if len(same) > 1:
            same = [f for f in same if f.current.page.url == fact.current.page.url] or same
        if len(same) > 1:
            same = [f for f in same if f.key.names == fact.key.names]
        if len(same) == 1:
            found.append((view, same[0]))
    return found


@dataclass
class Run:
    """A stretch of snapshots on which a row showed the same values under the same name."""
    first: dt.date
    last: dt.date
    text: str                       # "6.05% APR"
    numbers: tuple
    label: str                      # the row as it was named then: "2022-25 Model Year, 60 Months"
    named: str                      # the same, written one way, to tell whether the name changed


def _label(table, row) -> str:
    tier = _tier(row, table.name)
    return "%s, %s" % (table.name, tier) if tier else table.name


def _named(row, table) -> str:
    """What tells this row from its neighbours, written one way: '$5,000 - $24,999.99' and
    '$5,000 – $24,999.99' are the same tier; '2021-25 Model Year' and '2022-25 Model Year' are not."""
    return re.sub(r"[^a-z0-9$%+]+", " ", re.sub(r"\.00\b", "", normalise(_tier(row, table.name)))).strip()


def timeline(views: Sequence[SnapshotView], answerer: RulesAnswerer, table, row, model_year: Optional[int]) -> List[Run]:
    """What `row` (of the last snapshot) showed on each snapshot of `views`, as runs. A snapshot
    in which the row cannot be traced is left out: nothing is claimed about that date."""
    metrics = list(row.values)
    earlier: dict = {}
    for metric in metrics:
        fact = answerer.fact_for.get((id(row), metric))
        for view, old in (followed_back(views, fact, model_year) if fact else []):
            earlier.setdefault(view.snapshot_id, {})[metric] = old.current
    runs: List[Run] = []
    for view in views:
        if view is views[-1]:
            text = ", ".join("%s %s" % (value, metric) for metric, value in row.values.items())
            numbers = tuple(row.numbers[metric] for metric in metrics)
            label, named = _tier(row, table.name), _named(row, table)
        else:
            seen = earlier.get(view.snapshot_id, {})
            if len(seen) != len(metrics) or len({id(seen[metric].row) for metric in metrics}) != 1:
                continue
            text = ", ".join("%s %s" % (seen[metric].value, metric) for metric in metrics)
            numbers = tuple(seen[metric].number for metric in metrics)
            then = seen[metrics[0]]
            label, named = _tier(then.row, then.table.name), _named(then.row, then.table)
        if runs and runs[-1].numbers == numbers and runs[-1].named == named:
            runs[-1].last = view.taken_on
        else:
            runs.append(Run(first=view.taken_on, last=view.taken_on, text=text, numbers=numbers, label=label, named=named))
    return runs


def _written(day: dt.date) -> str:
    return day.strftime("%B %d, %Y").replace(" 0", " ")


def _story(name: str, runs: List[Run], days: Sequence[dt.date]) -> Optional[str]:
    """What a row showed on each of `days`: the two ends of the period asked about, and any date
    the question names in between. Values in between that were not asked about are left out:
    they answer nothing and could be taken for the current one. None when the row cannot be
    traced on the first or the last of those days.

      "Regular Certificates, 12 Month: 2.75% APY on May 12, 2026; 3.15% APY on September 28, 2026"

    Where the row itself was named differently on the earlier date (model-year bands roll
    forward), each value is given under the name the row had that day:

      "Auto Loans: 2020-24 Model Year, 60 Months: 6.05% APR on May 12, 2026;
       2021-25 Model Year, 60 Months: 6.25% APR on September 28, 2026"
    """
    shown: List[Tuple[Run, List[dt.date]]] = []
    for day in days:
        run = next((r for r in runs if r.first <= day <= r.last), None)
        if run is None:
            if day in (days[0], days[-1]):
                return None
            continue
        if shown and shown[-1][0].numbers == run.numbers and shown[-1][0].named == run.named:
            shown[-1][1].append(day)
        else:
            shown.append((run, [day]))
    on = lambda dates: " and on ".join(_written(day) for day in dates)  # noqa: E731
    if len({run.named for run, _ in shown}) == 1:
        parts = ["%s on %s" % (run.text, on(dates)) for run, dates in shown]
        if len(shown) == 1 and len(shown[0][1]) > 1:
            parts[0] += ", the same on both dates" if len(shown[0][1]) == 2 else ", the same on each date"
        return "%s: %s" % (name, "; ".join(parts))
    table_name = name.split(",")[0]
    pieces = "; ".join("%s: %s on %s" % (run.label or table_name, run.text, on(dates)) for run, dates in shown)
    return pieces if any((run.label or table_name).startswith(table_name) for run, _ in shown) else "%s: %s" % (table_name, pieces)


# Words a question about a change uses besides the name of the thing that changed.
CHANGE_WORDS = frozenset(stem(word) for word in (
    "change changed changes changing increase increased increases decrease decreased decreases rise rises risen rose rising "
    "fall falls fallen fell falling drop drops dropped dropping raise raised lower lowered reduce reduced cut update updated "
    "adjust adjusted compare compared comparison versus vs same unchanged stay stayed remain remained since between "
    "snapshot copy copies version went gone been far").split())


def beyond_the_rows(answerer: RulesAnswerer, asked: Asked, drafted: Draft) -> List[str]:
    """Words of a question about a change that neither the matched rows nor the vocabulary of
    change account for. Any such word means more is asked than how the rows moved."""
    return [w for w in answerer._beyond_the_rows(asked, drafted.matched) if w not in CHANGE_WORDS]


def mentions_rate(question: str) -> bool:
    return "%" in question or bool({stem(w) for w in re.findall(r"[a-z]+", normalise(question))} & RATE_WORDS)


def rows_answer(views: Sequence[SnapshotView], answerer: RulesAnswerer, drafted: Draft, asked: Asked,
                days: Sequence[dt.date]) -> Optional[Draft]:
    """The rate rows the question names, with what each showed on `days`: the two ends of the
    period and any date named in between. `views` are the snapshots of the period, oldest first;
    `answerer` and `drafted` (its plain rate answer) belong to the last of them.

    None when a row cannot be traced back to the start of the period. Whether it was not offered
    then, or was listed under another name, cannot be told from the tables alone, so nothing is
    claimed and the question goes to the model with that copy's tables."""
    rows = [(match.table, row) for match in drafted.matched for row in match.rows[:MAX_ROWS]][:MAX_ROWS]
    told = []
    for table, row in rows:
        story = _story(_label(table, row), timeline(views, answerer, table, row, asked.model_year), days)
        if story is None:
            return None
        told.append((table, story))
    if not told:
        return None
    # The date printed with the table is given when every row comes from one table.
    dated = _effective(told[0][0]) if len({id(table) for table, _ in told}) == 1 else ""
    text = " ".join("%s." % story for _, story in told[:-1]) + (" " if len(told) > 1 else "") + "%s%s." % (told[-1][1], dated)
    return Draft(text=text, sources=drafted.sources, confident=True, conflict_note=drafted.conflict_note, route="history")


def row_lines(views: Sequence[SnapshotView], answerer: RulesAnswerer, drafted: Draft, asked: Asked,
              days: Sequence[dt.date]) -> List[str]:
    """The same record as plain lines, for the model to read when the question asks for more
    than the rows hold."""
    lines = []
    for match in drafted.matched:
        for row in match.rows[:MAX_ROWS]:
            story = _story(_label(match.table, row), timeline(views, answerer, match.table, row, asked.model_year), days)
            if story:
                lines.append(story)
    return lines[:MAX_ROWS]


@dataclass
class Earlier:
    """What the model is shown, besides today's material, when a question is about a change."""
    view: SnapshotView                               # the copy of the website at the start of the period
    lines: List[str] = field(default_factory=list)   # rate rows followed through the period (row_lines)
    tables: List[str] = field(default_factory=list)  # that copy's rate tables that share words with the question

    def heading(self) -> str:
        return ("EARLIER COPY OF THE WEBSITE, taken on %s (shown because the question asks how something changed)"
                % _written(self.view.taken_on))

    def instruction(self) -> str:
        return ("To say how something changed, give its value in the earlier copy and its value today, each with its date. "
                "If the earlier copy does not show it, say only what is current.")
