"""The knowledge system the harness talks to: ingest snapshots, answer questions, list conflicts."""
from __future__ import annotations

import calendar
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

from .conflicts import Record, build_conflicts
from .dates import dates_in
from .history import Earlier, beyond_the_rows, comparison, mentions_rate, row_lines, rows_answer
from .lookup import read_question
from .meaning import make_embedder
from .pages import _MONTH_INDEX
from .rules_answer import NOT_LISTED, Draft, RulesAnswerer
from .sheet import table_text
from .snapshot import SnapshotView, build_view
from .text import expand_shorthand, is_yes_no, strip_request

_MONTH_NAMES = "|".join(sorted(_MONTH_INDEX, key=len, reverse=True))
_DATED_QUESTION = re.compile(r"\b(?:as of|back in|in|during|at)\s+(?:(?:the )?(end|start|beginning) of\s+)?"
                             r"(%s)\.?\s+((?:19|20)\d{2})\b" % _MONTH_NAMES, re.I)
_DAY_IN_QUESTION = re.compile(
    r"\b(?:as of|on|by|back on)\s+(?:the\s+)?(?:"
    r"(%(m)s)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})"            # May 15, 2026
    r"|(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(%(m)s)\.?,?\s+((?:19|20)\d{2})"  # 15 May 2026
    r"|(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})"                                   # 5/15/2026
    r"|((?:19|20)\d{2})-(\d{2})-(\d{2})"                                       # 2026-05-15
    r")\b" % {"m": _MONTH_NAMES}, re.I)
_PAST = re.compile(r"\b(was|were|did|had|as of|back in|back on|used to)\b", re.I)
_SAYS_NOT_PUBLISHED = re.compile(r"\b(does not|doesn't|do not|don't|isn't|is not|are not|aren't|no)\s+(?:\w+\s+){0,3}?"
                                 r"(publish|published|list|listed|provide|provided|show|shown|available|specific)\b", re.I)
_RELATIVE_YEAR = re.compile(r"\b(last|previous|prior|past|this) (?:year|yr)\b", re.I)
_OPENS_WITH_NOT_LISTED = re.compile(
    # "The website does not list ...", "The credit union does not publish ...", "... is not listed",
    # "There is no ...", "The Main Street branch does not have published lobby hours ..."
    r"^(?:[^.!?;]{0,80}\b(?:website|site|material|pages?|credit union|it|they|we)\b[^.!?;]{0,60}\b(?:does not|doesn't|do not|don't)\s+(?:\w+\s+){0,2}?"
    r"(?:list|publish|report|provide|give|mention|show|have|include|state|specify)\w*"
    r"|[^.!?;]{0,120}\b(?:is|are|was|were)\s+not\s+(?:listed|published|reported|provided|given|mentioned|shown|available|stated|specified)\b"
    r"|[^.!?;]{0,120}\b(?:does not|doesn't|do not|don't)\s+have\s+(?:any\s+)?(?:published|listed|posted)\b"
    r"|[^.!?;]{0,120}\bhas no\s+(?:published|listed|posted)\b"
    r"|there (?:is|are) no\b)", re.I)
_ABOUT_BRANCHES = re.compile(r"\b(branch(?:es)?|lobby|drive[- ]?(?:up|thru|through)|hours|locations?|located|address|atms?)\b", re.I)
TABLES_THEN = 3                 # rate tables of the earlier copy shown to the model for a question about a change
MAX_QUESTION_CHARS = 4000       # a member's question is a few sentences; nothing beyond this is read
# The time all model calls for one answer may take together. A healthy call takes about a second;
# this only matters when the API is slow or failing, and keeps every answer far inside the
# scorer's 60-second limit.
ANSWER_SECONDS = float(os.environ.get("KB_ANSWER_SECONDS", "15"))
_NEEDS_JUDGMENT = re.compile(r"^\s*(why|how come)\b|"
                             r"\b(compare|compared|versus|vs|difference|better|worse|cheaper|changed?|increase[ds]?|decrease[ds]?)\b", re.I)


def date_in_question(question: str) -> Optional[str]:
    """The date a question looks back to, when it names one and the harness passed none:
    'What was the rate as of May 2026?' -> '2026-05-31' (a month means its last day, unless the
    question says "the start of" or "the beginning of" it);
    'What did it pay on June 9, 2026?'  -> '2026-06-09'.
    Only for questions in the past tense or with "as of": a date can also be part of the subject
    ("the event on March 31, 2026"), and then it says nothing about which snapshot to read."""
    if not _PAST.search(question):
        return None
    m = _DAY_IN_QUESTION.search(question)
    if m:
        try:
            if m.group(1):
                return dt.date(int(m.group(3)), _MONTH_INDEX[m.group(1).lower().rstrip(".")], int(m.group(2))).isoformat()
            if m.group(4):
                return dt.date(int(m.group(6)), _MONTH_INDEX[m.group(5).lower().rstrip(".")], int(m.group(4))).isoformat()
            if m.group(7):
                return dt.date(int(m.group(9)), int(m.group(7)), int(m.group(8))).isoformat()
            return dt.date(int(m.group(10)), int(m.group(11)), int(m.group(12))).isoformat()
        except (ValueError, KeyError):
            return None                              # "February 30", or a month name in odd letters: no claim
    m = _DATED_QUESTION.search(question)
    month = _MONTH_INDEX.get(m.group(2).lower().rstrip(".")) if m else None
    if month is None:
        return None
    year = int(m.group(3))
    first = (m.group(1) or "").lower() in ("start", "beginning")
    return dt.date(year, month, 1 if first else calendar.monthrange(year, month)[1]).isoformat()


def iso_day(value) -> Optional[str]:
    """`as_of` as "YYYY-MM-DD", whatever form it arrives in: a date or datetime, an ISO string with
    or without a time, "5/19/2026", "May 19, 2026". None when it names no date. (Comparing raw
    strings would put "05/19/2026" before every snapshot.)"""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    text = str(value).strip()
    m = re.match(r"((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})(?!\d)", text)
    try:
        if m:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    except ValueError:
        return None
    found = dates_in(text)
    return found[0].isoformat() if found else None


def make_reader():
    """The model-backed reader, or None when no API key is configured (or KB_MODE=rules).
    Without it the system still answers, from rules alone."""
    if os.environ.get("KB_MODE", "").lower() == "rules":
        return None
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    try:
        from .reader import Reader
        return Reader()
    except Exception as error:                       # SDK missing or misconfigured: stay usable
        print("knowledge system: model reader unavailable (%s); answering from rules only" % error, file=sys.stderr)
        return None


class KnowledgeSystem:
    """Keeps one cleaned view of the website per snapshot date."""

    def __init__(self, answer_types=None):
        self.views: List[SnapshotView] = []
        self._answerers: Dict[str, RulesAnswerer] = {}
        self._answers: Dict[Tuple[str, Optional[str]], object] = {}
        self.routes: Dict[Tuple[str, Optional[str]], str] = {}
        self._conflicts: Optional[List[Record]] = None
        self._types = answer_types
        self.reader = make_reader()
        # Search by meaning feeds the model's reading list; it is of no use to the rules-only path,
        # which quotes sentences by word overlap. So it is loaded only alongside the reader.
        self.embedder = make_embedder() if self.reader is not None or os.environ.get("KB_EMBED", "").lower() == "on" else None
        # "hybrid" (default): plain look-ups that the rules settle exactly (a rate row, a rate link,
        # a printed schedule or address) are answered without a model call; everything else goes to
        # the model. "model": every question goes to the model.
        self.mode = os.environ.get("KB_MODE", "hybrid").lower()
        self.fallbacks = 0
        self._pool = ThreadPoolExecutor(max_workers=4)      # for the second look, started early (see draft)

    # ------------------------------------------------------------------------- ingest

    def ingest(self, snapshot) -> None:
        view = build_view(snapshot.snapshot_id, snapshot.taken_on, snapshot.pages, embedder=self.embedder)
        self.views = sorted([v for v in self.views if v.snapshot_id != view.snapshot_id] + [view],
                            key=lambda v: (v.taken_on, v.snapshot_id))
        self._answerers.clear()
        self._answers.clear()
        self.routes.clear()
        self._conflicts = None
        if self.reader is not None:
            self.reader.warm(view)

    # ------------------------------------------------------------------------- answer

    def view_for(self, as_of: Optional[str]) -> Optional[SnapshotView]:
        """The snapshot that shows the website as it stood on `as_of` (the newest one by default).
        A date before the first snapshot falls back to the first: it is the earliest record."""
        if not self.views:
            return None
        day = iso_day(as_of)
        if not day:
            return self.views[-1]
        earlier = [v for v in self.views if v.taken_on.isoformat() <= day]
        return earlier[-1] if earlier else self.views[0]

    def _answerer(self, view: SnapshotView) -> RulesAnswerer:
        answerer = self._answerers.get(view.snapshot_id)
        if answerer is None:
            answerer = self._answerers[view.snapshot_id] = RulesAnswerer(view, newest=view is self.views[-1])
        return answerer

    def _how_it_changed(self, question: str, as_of: Optional[str], deadline: Optional[float] = None) -> Optional[Tuple[Draft, SnapshotView]]:
        """A question that is without doubt about a change (history.comparison) is answered from
        the snapshots at both ends of the period it names: by rules when it is about rate rows
        and nothing else, otherwise by the model with the earlier copy beside today's. Returns
        None in every other case; the usual answer, read from one snapshot, then applies."""
        period = comparison(question)
        if period is None:
            return None
        first, newest = self.views[0], self.views[-1]

        def concrete(day: dt.date, kind: str) -> dt.date:
            if kind == "start":                      # the month's first day; the first copy if the month began before it
                return first.taken_on if day.replace(day=1) < first.taken_on <= day else day.replace(day=1)
            if kind == "end" and day.replace(day=1) <= newest.taken_on:
                return min(day, newest.taken_on)     # a month still running is read up to today
            return day
        dated = [concrete(day, kind) for day, kind in period.days]
        if any(day < first.taken_on or day > newest.taken_on for day in dated):
            return None                              # a date that no copy of the website covers
        asked_view = self.view_for(as_of) if as_of else newest
        named = [self.view_for(day.isoformat()) for day in dated]
        if period.to_today:
            start, end = (named[0] if named else first), asked_view
        elif asked_view is newest:
            start, end = named[0], named[-1]
        else:
            return None                              # two dates and a third from the caller: left to the usual path
        if start.taken_on >= end.taken_on or any(view.taken_on > end.taken_on for view in named):
            return None                              # no earlier copy inside the period to compare with
        views = [v for v in self.views if start.taken_on <= v.taken_on <= end.taken_on]
        # The dates to report: both ends, and any date the question names in between.
        days = sorted({start.taken_on, end.taken_on} | {view.taken_on for view in named})
        answerer = self._answerer(end)
        question = answerer.spelled(question)
        asked = read_question(question)
        if "jumbo" not in asked.flags | asked.excluded:
            # As with any look at a past date, only the product that was named: an old jumbo
            # figure is easily taken for the standard one (see RulesAnswerer.rate_answer).
            named_only = replace(asked, excluded=asked.excluded | {"jumbo"})
            if answerer.rate_answer(question, named_only) is not None:
                asked = named_only
        drafted = answerer.rate_answer(question, asked)
        lines: List[str] = []
        tables: List[str] = []
        if mentions_rate(question):
            # For the model, should the rules below not settle it: the earlier copy's rate tables
            # that share words with the question. Today's are in its rate sheet already.
            tables = [table_text(match.table) for match in self._answerer(start).lookup.find(asked)[:TABLES_THEN]]
            if drafted is not None and drafted.matched:
                if drafted.route == "rate" and not beyond_the_rows(answerer, asked, drafted):
                    settled = rows_answer(views, answerer, drafted, asked, days)
                    if settled is not None:
                        return settled, end
                lines = row_lines(views, answerer, drafted, asked, days)
        if self.reader is None:
            return None
        try:
            day = dt.date.fromisoformat(str(as_of)[:10]) if as_of else None
        except ValueError:
            day = None
        try:
            by_model = self.reader.answer(end, question, day=day, earlier=Earlier(view=start, lines=lines, tables=tables),
                                          deadline=deadline)
        except Exception as error:                   # the usual path below gets its own try
            print("knowledge system: model call failed (%s: %s); comparing copies was skipped"
                  % (type(error).__name__, str(error)[:120]), file=sys.stderr)
            return None
        if not by_model.confident or _OPENS_WITH_NOT_LISTED.search(by_model.text):
            return None
        by_model.route = "history, model"
        return by_model, end

    def draft(self, question: str, as_of: Optional[str] = None, deadline: Optional[float] = None) -> Tuple[Draft, Optional[SnapshotView]]:
        """The answer to one question, and the snapshot it was read from. All model calls for it
        share one time budget (ANSWER_SECONDS): past it, the rules answer stands."""
        as_of = iso_day(as_of)
        if deadline is None:
            deadline = time.monotonic() + ANSWER_SECONDS
        if self.views:
            try:
                compared = self._how_it_changed(question, as_of, deadline)
            except Exception as error:                   # this path may add an answer, never cost one
                print("knowledge system: comparing copies failed (%s: %s); answered from one copy"
                      % (type(error).__name__, str(error)[:120]), file=sys.stderr)
                compared = None
            if compared is not None:
                return compared
        asked_for = as_of or date_in_question(question)
        view = self.view_for(asked_for)
        try:
            day = dt.date.fromisoformat(str(asked_for)[:10]) if asked_for else None
        except ValueError:
            day = None
        if view is None:
            return Draft(text="I don't know yet: no website snapshot has been loaded.", confident=False), None
        answerer = self._answerer(view)
        by_rules = answerer.answer(question)
        if self.reader is None or (self.mode == "hybrid" and self._plain_lookup(question, by_rules)):
            return by_rules, view
        try:
            # The second look (passages found by meaning) is only used when the first finds no
            # answer, but it is started at the same time: the member then waits for the slower of
            # two calls, never for one after the other. When the first look answers, the second is
            # simply not used; it costs a fraction of a cent.
            early = None
            if not by_rules.route.startswith("rate") and self.reader.can_look_again(view):
                early = self._pool.submit(self.reader.answer, view, question, by_meaning=True, day=day, deadline=deadline)
            by_model = self.reader.answer(view, question, day=day, deadline=deadline)
            if self._worth_a_second_look(by_rules, by_model) and early is not None:
                try:
                    again = early.result(timeout=max(0.0, deadline - time.monotonic()))
                except Exception:                    # out of time, or the second look failed: the first look stands
                    again = None
                if again is not None and again.confident:
                    again.route = "model, second look"
                    by_model = again
        except Exception as error:                   # timeout, rate limit, bad reply: never fail the member
            self.fallbacks += 1
            print("knowledge system: model call failed (%s: %s); used the rules answer"
                  % (type(error).__name__, str(error)[:120]), file=sys.stderr)
            return by_rules, view
        return self._cross_check(by_rules, by_model, question), view

    @staticmethod
    def _worth_a_second_look(by_rules: Draft, by_model: Draft) -> bool:
        """The model found no answer in the passages that keyword search brought up. Unless the
        question is about a rate (those are settled by the rate tables), the cause may be wording:
        the member's words differ from the website's. Then the second look, with passages found by
        meaning, is used. It was started alongside the first, so it adds no waiting of its own."""
        return by_model.route == "model-not-listed" and not by_rules.route.startswith("rate")

    @staticmethod
    def _plain_lookup(question: str, by_rules: Draft) -> bool:
        """True when the rules matched an exact rate row and the question simply asks for it, however
        politely ("Can you tell me the rate for ..."). Real yes/no, 'why' and comparison questions
        need wording judgment, so they go to the model; a yes/no about one stated percentage is
        the exception, because the rows settle it."""
        if by_rules.route in ("rate-link", "rate-yes-no"):
            return True
        if by_rules.route in ("hours", "hours-branch", "branch-address"):
            # The printed schedules and addresses are the whole answer; of the yes/no questions only
            # a question asking whether every branch keeps the same schedule is settled by them.
            return not is_yes_no(question) or by_rules.text.startswith(("No. ", "Yes. "))
        if by_rules.route != "rate" or not by_rules.complete:
            return False
        asked = strip_request(question)
        return not (is_yes_no(question) or _NEEDS_JUDGMENT.search(asked) or re.search(r"\d\s*(%|percent)", question))

    @staticmethod
    def _cross_check(by_rules: Draft, by_model: Draft, question: str = "") -> Draft:
        """Where the rules found the exact rate row, the model's wording is kept only if it states
        that row's value; otherwise the rule-based answer stands. A model can misread a table or
        decline needlessly; an exact row match cannot."""
        if by_model.confident and not by_rules.confident and _OPENS_WITH_NOT_LISTED.search(by_model.text) \
                and not _RELATIVE_YEAR.search(question):
            # The model says the website does not list it, then goes on about other branches or
            # products. The rules found nothing either. That is a plain "not listed": anything
            # added would be an answer to a question the member did not ask.
            return Draft(text=NOT_LISTED, confident=False, route="model-not-listed")
        if by_rules.route == "rate" and not by_rules.complete:
            # The question asks for more than the rate row holds (a fee, a floor, a condition).
            # The row cannot overrule an answer about something it does not contain.
            return by_model
        if by_rules.route == "rate":
            exact = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", by_rules.text)}
            said = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", by_model.text)}
            if not by_model.confident or not exact & said:
                by_rules.route = "rate (model answer set aside)"
                return by_rules
            by_model.conflict_note = by_rules.conflict_note
        elif by_rules.route == "rate-link":
            if not by_model.confident or not any(url in by_model.text for url, _ in by_rules.sources):
                return by_rules
        elif by_model.route == "model-off-topic" and by_rules.confident:
            return by_rules                          # the model wandered off twice; the quoted passage is on the subject
        elif by_rules.route in ("hours", "hours-branch", "branch-address") and not by_model.confident:
            return by_rules                          # the schedules are printed; a "not listed" would be wrong
        elif by_rules.route in ("rate-not-found", "rate-not-listed") and _SAYS_NOT_PUBLISHED.search(by_model.text):
            # The member asked for a rate, no rate table has it, and the model says so too. A plain
            # "not listed" is the whole answer; nearby numbers about the product would only mislead.
            return by_rules
        return by_model

    def answer(self, question: str, as_of: Optional[str] = None):
        # Spacing carries no meaning. Collapsing it first means every part of the system, and the
        # stored answers, see the same question whatever spaces or line breaks it came with.
        question = " ".join(str(question or "").split())[:MAX_QUESTION_CHARS]
        as_of = iso_day(as_of)
        key = (question.lower(), as_of)
        if key in self._answers:
            return self._answers[key]
        draft, view = self.draft(expand_shorthand(question), as_of)
        text = draft.text
        if view is not None and draft.confident and view is not self.views[-1] and not draft.route.startswith("history"):
            # Stamped with the date the member asked about; the copy of the website it was read
            # from may be a few days older, and naming that date instead only confuses. (An answer
            # about a change names its own dates.)
            asked_for = str(as_of or date_in_question(question) or "")[:10]
            try:
                day = max(dt.date.fromisoformat(asked_for), view.taken_on)
            except ValueError:
                day = view.taken_on
            text = "As of %s: %s" % (day.strftime("%B %d, %Y").replace(" 0", " "), text)
        Answer, Source, _ = self._types
        cited = list(draft.sources)
        if view is not None and draft.confident and view.locator and _ABOUT_BRANCHES.search(question) \
                and all(url.rstrip("/") != view.locator for url, _ in cited):
            # Hours and addresses differ by branch, so the page where a member looks their own
            # branch up belongs with any answer about branches.
            cited.append((view.locator, ""))
        sources = [Source(url=url, snapshot_id=view.snapshot_id if view else None, quote=quote[:240] or None)
                   for url, quote in cited]
        answer = Answer(text=text, sources=sources, confident=draft.confident, conflict_note=draft.conflict_note)
        self._answers[key] = answer
        self.routes[key] = draft.route             # which path answered; read by the development reports
        return answer

    # ---------------------------------------------------------------------- conflicts

    def conflicts(self):
        if self._conflicts is None:
            self._conflicts = build_conflicts(self.views)
        _, Source, Conflict = self._types
        return [Conflict(topic=r.topic, values=list(r.values),
                         sources=[Source(url=u, snapshot_id=sid, quote=q[:240]) for u, sid, q in r.sources],
                         resolution=r.resolution, reason=r.reason)
                for r in self._conflicts]
