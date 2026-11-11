"""Answer one question with one model call, then check the answer against what the model saw.

The model is the only part of the system that understands wording. It is deliberately given
little room to go wrong:
  * it reads the resolved fact sheet and passages from the cleaned snapshot, never raw pages, so
    out-of-date numbers are not in front of it;
  * it must return a fixed JSON shape and name the items it relied on;
  * every number in its answer must appear in what it was shown, or the answer is discarded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .rules_answer import NOT_LISTED, Draft
from .search import Passage, query_tokens
from .sheet import Sheet, build_sheet
from .snapshot import SnapshotView
from .text import stem, tokens

MODEL = os.environ.get("KB_MODEL", "claude-haiku-5-5")
EFFORT = os.environ.get("KB_EFFORT", "low")
THINKING = os.environ.get("KB_THINKING", "disabled")       # "disabled" or "adaptive"
FORMAT = os.environ.get("KB_FORMAT", "json")               # "json": shape asked for in words; "schema": enforced by the API
PASSAGES = int(os.environ.get("KB_PASSAGES", "8"))         # blocks of website text shown per question
FULL_PAGES = int(os.environ.get("KB_FULL_PAGES", "2"))     # of which this many are whole pages
# A healthy call takes about a second. Past these limits the member is better served by the rules
# answer at once than by waiting: no automatic retries (a rate-limited key would otherwise sleep for
# however long the API asks), and after a few failures in a row the model is left alone for a while.
TIMEOUT_SECONDS = float(os.environ.get("KB_TIMEOUT", "8"))
MIN_CALL_SECONDS = 1.5         # never start a call with less time than this left in the answer's budget
FAILURES_BEFORE_PAUSE = 3
PAUSE_SECONDS = 60.0
MAX_TOKENS = 1500
SEARCH_DEPTH = 24
PASSAGE_CHARS = 1700           # a whole passage (snapshot.HARD_LIMIT is 1600)
EARLIER_PASSAGES = 5           # blocks from the earlier copy of the website, for questions about a change,
SAME_PAGES_THEN = 3            # of which up to this many are the earlier version of a page shown from today's copy
PAGE_CHARS = 4500

INSTRUCTIONS = """You answer questions from members of a credit union, using only what its website said on one date.

You are given these kinds of material:
- RATE SHEET: every rate the website publishes on that date. Out-of-date copies were already removed.
- ORGANIZATION FACTS: the most recently published statement of figures such as branch or member counts.
- CONTACT DETAILS: the organisation's own address, phone number and routing number, from its contact page.
- BRANCH HOURS and BRANCH LOCATIONS (only with questions about branches, hours or addresses): every branch schedule the website prints, each taken from an article about one branch; and the branches its locator lists, with their addresses.
- PASSAGES: excerpts of the website found for this question. Each shows its page type and its dates.

Rules
1. Use only this material. Never fill a gap from general knowledge, and never invent a number, date, name, address, phone number or link. If the material covers the question, answer from it, even when the answer sits on a policy page or in a news story; set answerable to false only when the material truly does not contain the answer.
2. Rates come only from the RATE SHEET. Members word products loosely: a CD is a certificate, a car loan is an auto loan, a mortgage is a home loan. Match such wording to the listed product. If the product, term, tier or variant asked about is truly not in the sheet (an adjustable-rate mortgage when only fixed ones are listed, say), that rate is not published: set answerable to false. Do not give the rate of a different product instead.
3. Use the exact row: the term, the balance or loan-amount tier, the model year and the variant (jumbo, business, Roth and so on) that the member asked about. When the question names the tier, give that row only and leave the other tiers out. When the question leaves a tier open, give the rate for each tier of that one product, or the range across them; do not ask the member to choose first. Personal products are meant unless the question says business; when the member asks about a whole category (all credit cards, say) without saying which, give the personal products first and then the business ones.
4. Passages can be old. When two passages disagree, the more recently published one is current, and the older figure must not be repeated. The ORGANIZATION FACTS are newer than any passage that gives a different figure for the same thing: use the ORGANIZATION FACTS figure and leave the other one out. Use them as well to make an answer specific: where a passage speaks only of the charter area or the communities served, name the states the ORGANIZATION FACTS give. Mention an earlier value only when the member asks how something changed or asks about a past date. A figure that a passage ties to a named year ("in 2023, 294 employees ...") is that year's figure, not an out-of-date one.
5. If the member names one specific branch, product or program and the material says nothing about that one, it is unknown. Do not answer with general information in its place.
6. A general question is answerable from particular cases. Asked for lobby or drive-up hours in general, say that hours vary by branch and give each distinct schedule in BRANCH HOURS, with a branch that keeps it. Asked about one named branch, use that branch's entry; if it has none, rule 5 applies.
7. Give one answer. Never add a second, different figure as an alternative, and do not remark that sources differ.
8. When the material announces a dated change (a branch closing, a new fee, an offer ending), answer questions about the present from it and state the date.
9. When the member says "last year" or "this year" (a NOTE under the question then names the year meant) and the material has no figure for that year, do not decline: say that year is not reported, then give the figure for every year the material does report, each with its year, newest first. When the member names the year outright ("in 2024") and the material has no figure for that year, that figure is not published: set answerable to false, and do not offer another year's figure in its place.

Style
- Answer directly in the first sentence, without lead-ins such as "according to" and without explaining how you worked it out. Use one to three sentences; when the member asks for a list or for all rates, give the whole list. Start the answer to a yes/no question with Yes or No, then give the fact that settles it.
- Copy values exactly as the material writes them, with their units: percentages with APR or APY, dollar amounts, times such as 8:30 a.m., dates such as March 31, 2026, phone numbers and links.
- Be specific. Include the details that define the answer: how many times, which date, what amount, which places, which phone number. Where the material gives a count, state it; never write "several" or "many times" instead.
- When the member asks for a link, where to find something, or how to do something online, include the full link; for rates, give the links the sheet shows for that table.
- Speak as the credit union's assistant, stating facts plainly. The member cannot see this material, so never refer to it: no "the passage", "the sheet", "the website shows", "according to its most recently published figures". Do not ask for a ZIP code, member number or other personal details.

Reply with:
- answerable: true whenever the material lets you tell the member something that answers the question, fully or in part; false only when it gives you nothing to answer with.
- answer: your reply to the member. When answerable is false, one sentence saying the website does not list this and suggesting they contact the credit union, with no numbers.
- cited: the ids of the items you relied on, such as T4, O1, C1, H2, L3 or P2."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "cited": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answerable", "answer", "cited"],
    "additionalProperties": False,
}
# Without an enforced schema, the reply's shape is asked for at the very end of the prompt, the last
# thing the model reads.
REPLY_FORMAT = ("\n\nReply with one JSON object only, with the keys answerable (true or false), answer (string) "
                "and cited (list of item ids), and nothing else.")


def parse_reply(text: str) -> dict:
    """The model's reply as {answerable, answer, cited}. The model writes the JSON object itself and
    now and then wraps it in a code fence or adds a remark after it, so the first complete {...} is
    read. Raises ValueError when there is no object to read; the caller then falls back to the rules
    answer."""
    try:
        reply = json.loads(text)
    except ValueError:
        start = text.find("{")
        if start < 0:
            raise ValueError("the reply holds no JSON object")
        reply, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(reply, dict):
        raise ValueError("the reply is not a JSON object")
    answerable = reply.get("answerable")
    if isinstance(answerable, str):
        answerable = answerable.strip().lower() in ("true", "yes")
    answer = reply.get("answer") or ""
    if isinstance(answer, list):                     # a list of lines instead of one string
        answer = " ".join(str(line) for line in answer)
    cited = reply.get("cited") or []
    if isinstance(cited, str):
        cited = re.findall(r"\b[A-Z]\d+\b", cited)
    return {"answerable": bool(answerable), "answer": str(answer),
            "cited": [str(ref) for ref in cited] if isinstance(cited, list) else []}

_NUMBER = re.compile(r"(\$)?\s?(\d[\d,]*(?:\.\d+)?)\s*(%)?")
_RELATIVE_YEAR = re.compile(r"\b(last|previous|prior|past|this) (?:year|yr)\b", re.I)
# How credit unions name the area a member must live or work in, and how a site says where that is.
_AREA = re.compile(r"\b(charter area|field of membership|service area|membership area)\b", re.I)
_SERVES = re.compile(r"\bserv(?:es|ing)\b.{0,80}\b(counties|communities|states)\b", re.I)
_ABOUT_HOURS = re.compile(r"\b(hours?|lobby|lobbies|drive[- ]?(?:up|thru|through)|opens?|opening|close[sd]?|closing|"
                          r"branch(?:es)?|weekends?|saturdays?|sundays?|address(?:es)?|located|locations?|where)\b", re.I)


def numbers_in(text: str) -> set:
    return {round(float(m.group(2).replace(",", "")), 4) for m in _NUMBER.finditer(text) if m.group(2).strip(",")}


MIN_SIMILARITY = 0.55


def on_topic(question: str, answer: str, relied_on: List[str], site_words=frozenset(), embedder=None) -> bool:
    """Does the answer have anything to do with the question? Only an answer about something
    else altogether fails, and it takes two signs together:
      * no content word of the question (compared by its first five letters) appears in the
        answer or in the items the model says it relied on, and
      * the embedding model, when one is loaded, finds question and answer far apart in meaning.
    Measured on 136 model answers: the one answer that was about something else scored 0.46 and
    shared no word with its question; every real answer scored 0.61 or more. Without an embedding
    model there is no second opinion, so nothing is rejected."""
    asked = {t[:5] for t in tokens(question.replace("-", " ")) if not t[0].isdigit() and t not in site_words}
    if not asked or embedder is None:
        return True
    seen = {t[:5] for t in tokens((answer + " " + " ".join(relied_on)).replace("-", " "))}
    if asked & seen:
        return True
    try:
        return float(embedder.question(question) @ embedder.question(answer[:1200])) >= MIN_SIMILARITY
    except Exception:
        return True


_TYPED_RATE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:%|percent\b|per\s+cent\b|(?=(?:APR|APY)\b))", re.I)


def material_shown(user: str) -> str:
    """The prompt, as evidence for the numbers in an answer. A rate the member typed in the
    question is not evidence for itself: "Is it still 5.34%?" must not let the model repeat 5.34%.
    Other numbers in the question (a balance, a loan amount, a model year) still count, since an
    answer may name the amount it was asked about."""
    before, marker, asked = user.rpartition("QUESTION: ")
    return before + _TYPED_RATE.sub(" ", asked) if marker else user


def ungrounded_numbers(answer: str, shown: str) -> List[str]:
    """Numbers in the answer that appear nowhere in the material the model was shown. Small whole
    numbers without a unit are exempt: 'two branches' or '3 options' is counting, not quoting."""
    known = numbers_in(shown)
    missing = []
    for m in _NUMBER.finditer(answer):
        raw = m.group(2).replace(",", "")
        if not raw:
            continue
        value = round(float(raw), 4)
        counting = "." not in raw and not m.group(1) and not m.group(3) and value <= 12
        if not counting and value not in known:
            missing.append(m.group(0).strip())
    return missing


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    seconds: List[float] = field(default_factory=list)


class Reader:
    def __init__(self, client=None):
        if client is None:
            import anthropic                  # imported here so the rules-only path needs no SDK
            client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=0)
        self.client = client
        self._failures = 0                    # failed calls in a row
        self._paused_until = 0.0              # time.monotonic() before which no call is made
        self._lock = threading.Lock()         # the second look calls from another thread
        self.sheets: Dict[str, Sheet] = {}
        self.usage = Usage()
        cache_dir = os.environ.get("KB_LLM_CACHE")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def sheet_for(self, view: SnapshotView) -> Sheet:
        if view.snapshot_id not in self.sheets:
            self.sheets[view.snapshot_id] = build_sheet(view)
        return self.sheets[view.snapshot_id]

    def _system(self, sheet_text: str) -> list:
        return [
            {"type": "text", "text": INSTRUCTIONS},
            # The sheet is identical for every question about the same date, so it is cached. One
            # hour covers a whole evaluation run even when a past date is asked about only rarely.
            {"type": "text", "text": sheet_text, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ]

    def warm(self, view: SnapshotView) -> None:
        """Called once per snapshot at ingest: write the instructions and this snapshot's sheet to
        the prompt cache, so that the first real question does not pay for it in waiting time."""
        if self.cache_dir is not None:
            return                                   # development runs replay saved replies
        try:
            self.client.messages.create(**self._request(
                self.sheet_for(view).text, "QUESTION: (warm-up, reply with answerable false)", max_tokens=64))
        except Exception:
            pass                                     # warming is an optimisation, never a requirement

    def _request(self, sheet_text: str, user: str, max_tokens: int = MAX_TOKENS) -> dict:
        """The request body. warm() and _call() share it, so the prefix that warm() caches is exactly
        the one the real calls read."""
        request = dict(model=MODEL, max_tokens=max_tokens, system=self._system(sheet_text),
                       messages=[{"role": "user", "content": user if FORMAT == "schema" else user + REPLY_FORMAT}],
                       output_config={"effort": EFFORT})
        if FORMAT == "schema":
            request["output_config"]["format"] = {"type": "json_schema", "schema": SCHEMA}
        if THINKING == "disabled":
            request["thinking"] = {"type": "disabled"}
        return request

    # ------------------------------------------------------------------------- prompt

    @staticmethod
    def _block(number: int, passage: Passage, text: str, stood: str = "") -> str:
        page = passage.page
        dates = "published %s" % page.published_on.isoformat() if page.published_on else "crawled %s" % page.fetched_date
        title = page.title.split("|")[0].strip() or page.url
        return "[P%d] %s - %s (%s page; %s%s)\n%s" % (number, title, page.url, page.kind, dates, stood, text)

    @staticmethod
    def _page_excerpt(view: SnapshotView, hits: List[Tuple[Passage, float]]) -> str:
        """A whole page, or for a long page the matching passages with their neighbours, in page order."""
        body = [p for p in view.index.by_page.get(hits[0][0].page.url, []) if p.table is None]
        if sum(len(p.text) for p in body) > PAGE_CHARS:
            keep, used = set(), 0
            for hit, _ in hits:
                for position in (hit.position, hit.position + 1, hit.position - 1):
                    near = next((p for p in body if p.position == position), None)
                    if near is not None and position not in keep and used + len(near.text) <= PAGE_CHARS:
                        keep.add(position)
                        used += len(near.text)
            body = [p for p in body if p.position in keep]
        pieces, previous = [], None
        for p in body:
            if previous is not None and p.position != previous + 1:
                pieces.append("[...]")
            pieces.append(p.text)
            previous = p.position
        return "\n".join(pieces)

    def context(self, view: SnapshotView, question: str, by_meaning: bool = False) -> List[Tuple[Passage, str]]:
        """What the model reads besides the sheet: the most relevant pages at length (an answer is
        often two paragraphs away from the best-matching one), then the best passages from other
        pages. "Relevant" is by keywords, or by meaning for the second look (see answer)."""
        if by_meaning:
            hits = view.index.nearest(question, k=SEARCH_DEPTH)
        else:
            hits = [(p, s) for p, s in view.index.search(question, k=SEARCH_DEPTH) if p.table is None]
        by_page: Dict[str, List[Tuple[Passage, float]]] = {}
        for p, s in hits:
            by_page.setdefault(p.page.url, []).append((p, s))
        if by_meaning:
            # Similarities all lie close together, so adding them up would only count passages.
            # The page with the single closest passage comes first.
            ranked = sorted(by_page.values(), key=lambda page_hits: -page_hits[0][1])
        else:
            ranked = sorted(by_page.values(), key=lambda page_hits: -sum(s for _, s in page_hits[:3]))
        blocks: List[Tuple[Passage, str]] = []
        at_length = set()
        for page_hits in ranked[:FULL_PAGES]:
            first = page_hits[0][0]
            if first.position >= 0:
                blocks.append((first, self._page_excerpt(view, page_hits)))
                at_length.add(first.page.url)
        for p, _ in hits:
            if len(blocks) >= PASSAGES:
                break
            if p.page.url not in at_length:
                blocks.append((p, p.text[:PASSAGE_CHARS]))
        return blocks

    def _context_then(self, then: SnapshotView, question: str, shown_now: List[str]) -> List[Tuple[Passage, str]]:
        """What the model reads from the earlier copy of the website. First the earlier version of
        the pages it is shown from today's copy: the same page, then and now, is where a change
        shows. Then the earlier copy's own best matches."""
        hits = [(p, s) for p, s in then.index.search(question, k=SEARCH_DEPTH) if p.table is None]
        wanted = set(query_tokens(question))
        blocks: List[Tuple[Passage, str]] = []
        used = set()
        for url in dict.fromkeys(shown_now):
            if len(blocks) >= SAME_PAGES_THEN:
                break
            page_hits = [(p, s) for p, s in hits if p.page.url == url and p.position >= 0]
            if not page_hits:
                # Not among that copy's best matches: take the passages of the page that share
                # the most words with the question.
                body = [p for p in then.index.by_page.get(url, []) if p.table is None and p.position >= 0]
                shared = sorted(((len(wanted & set(tokens(p.text))), -p.position, p) for p in body), key=lambda x: x[:2], reverse=True)
                page_hits = [(p, float(n)) for n, _, p in shared[:2] if n > 0]
            if page_hits:
                blocks.append((page_hits[0][0], self._page_excerpt(then, page_hits)))
                used.add(url)
        for p, _ in hits:
            if len(blocks) >= EARLIER_PASSAGES:
                break
            if p.page.url not in used:
                blocks.append((p, p.text[:PASSAGE_CHARS]))
        return blocks

    def build_prompt(self, view: SnapshotView, question: str, by_meaning: bool = False, day=None,
                     earlier=None) -> Tuple[str, str, List[Passage]]:
        """`day` is the date the member asked about, when it is later than the day this copy of
        the website was taken. The model is then told that date as "today": a question about
        June 8 read against a "today" of May 19 looked like a question about the future, and was
        declined.

        `earlier` (history.Earlier) is given only for a question about a change. The passages
        found in the earlier copy of the website then follow today's, numbered on from them, with
        the rate rows already followed through time. Without it the prompt is what it always was."""
        sheet = self.sheet_for(view)
        blocks = self.context(view, question, by_meaning)
        shown = "\n\n".join(self._block(i, p, text) for i, (p, text) in enumerate(blocks, start=1)) or "(none found)"
        written = lambda d: d.strftime("%B %d, %Y").replace(" 0", " ")
        today = "Website date: %s. Treat this as today." % written(view.taken_on)
        if day is not None and day > view.taken_on:
            today = ("Website date: %s. Treat this as today. The material is the copy of the website taken on %s, "
                     "the latest one on or before that date; it is what the website said on %s."
                     % (written(day), written(view.taken_on), written(day)))
        hours = "%s\n\n" % sheet.hours_text if sheet.hours_text and _ABOUT_HOURS.search(question) else ""
        then = ""
        if earlier is not None:
            old = self._context_then(earlier.view, question, [p.page.url for p, _ in blocks])
            stood = "; as the website stood on %s" % written(earlier.view.taken_on)
            parts = [earlier.heading()]
            if earlier.lines:
                parts.append("RATE ROWS OVER TIME\n%s" % "\n".join("- " + line for line in earlier.lines))
            elif earlier.tables:
                parts.append("RATE TABLES IN THAT COPY (those that share words with the question)\n%s" % "\n\n".join(earlier.tables))
            if old:
                parts.append("PASSAGES FROM THAT COPY\n%s" % "\n\n".join(
                    self._block(len(blocks) + i, p, text, stood) for i, (p, text) in enumerate(old, start=1)))
            parts.append(earlier.instruction())
            then = "\n\n".join(parts) + "\n\n"
            blocks = blocks + old
        note = "" if earlier is not None else self._note(view, question, day)
        user = "%s\n\n%sPASSAGES\n%s\n\n%sQUESTION: %s%s" % (today, hours, shown, then, question.strip(), note)
        return sheet.text, user, [p for p, _ in blocks]

    @staticmethod
    def _note(view: SnapshotView, question: str, day=None) -> str:
        """A reminder placed right under the question when it speaks of "last year" or "this year".
        The website reports yearly figures a year or two late, so such a question usually has no
        exact match; said here, next to the question, rule 9 is followed far more reliably."""
        m = _RELATIVE_YEAR.search(question)
        if not m:
            return ""
        year = (day or view.taken_on).year - (0 if m.group(1).lower() == "this" else 1)
        # "Still answerable" is spelt out: without it the model often declined these questions outright.
        return ("\nNOTE: \"%s\" means %d here. If the material has no figure for %d, the question is still answerable: "
                "set answerable to true, say that %d is not reported, then give the figure for every year the material "
                "does report, each with its year, newest first." % (m.group(0), year, year, year))

    # --------------------------------------------------------------------------- call

    def _call(self, sheet_text: str, user: str, deadline: float = None) -> dict:
        """One request. `deadline` (a time.monotonic() value) is when the whole answer must be done;
        the request gets whatever is left of that, up to TIMEOUT_SECONDS. Raises when there is no
        time left, while the model is paused after repeated failures, and on any API error."""
        key = hashlib.sha256(json.dumps([MODEL, EFFORT, THINKING, FORMAT, INSTRUCTIONS, sheet_text, user]).encode("utf-8")).hexdigest()
        cached = self.cache_dir / (key + ".json") if self.cache_dir else None
        if cached is not None and cached.exists():
            self.usage.cached_calls += 1
            return json.loads(cached.read_text())
        now = time.monotonic()
        if now < self._paused_until:
            raise RuntimeError("model paused after %d failed calls in a row" % FAILURES_BEFORE_PAUSE)
        timeout = TIMEOUT_SECONDS if deadline is None else min(TIMEOUT_SECONDS, deadline - now)
        if timeout < MIN_CALL_SECONDS:
            raise TimeoutError("no time left in this answer's budget for another model call")
        started = time.time()
        try:
            response = self.client.messages.create(**self._request(sheet_text, user), timeout=timeout)
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= FAILURES_BEFORE_PAUSE:
                    self._paused_until = time.monotonic() + PAUSE_SECONDS
            raise
        with self._lock:
            self._failures = 0
        self.usage.seconds.append(time.time() - started)
        self.usage.calls += 1
        usage = response.usage
        self.usage.input_tokens += usage.input_tokens or 0
        self.usage.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.usage.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.usage.output_tokens += usage.output_tokens or 0
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ValueError("model stopped early: %s" % response.stop_reason)
        parsed = parse_reply("".join(block.text for block in response.content if block.type == "text"))
        if cached is not None:
            cached.write_text(json.dumps(parsed))
        return parsed

    # ------------------------------------------------------------------------- answer

    def can_look_again(self, view: SnapshotView) -> bool:
        return view.index.vectors is not None

    def answer(self, view: SnapshotView, question: str, by_meaning: bool = False, day=None, earlier=None,
               deadline: float = None) -> Draft:
        """One model call, or two when the first reply is unusable (see below). Raises on API
        failure or when the answer's time budget runs out; the caller falls back to the rules answer.

        by_meaning=True is the second look. The first look reads passages found by keywords. When
        the model finds no answer in them, the question may simply be worded differently from the
        website, so the answer from passages found by meaning is used instead. The caller starts
        both looks together and uses the second only when the first finds nothing. The two reading
        lists are kept apart on purpose: adding meaning matches to every question was measured and
        it made the model decline questions that keywords alone had answered correctly."""
        sheet_text, user, passages = self.build_prompt(view, question, by_meaning, day, earlier)
        sheet = self.sheet_for(view)
        # A small model now and then trails off mid-list, or answers some other question from the
        # material in front of it. Both are easy to see and cheap to put right: ask once more,
        # saying what was wrong. A second miss is not passed on to the member.
        for attempt in (1, 2):
            reply = self._call(sheet_text, user, deadline)
            text = " ".join(str(reply.get("answer", "")).split())
            if not reply.get("answerable") or not text:
                return Draft(text=NOT_LISTED, confident=False, route="model-not-listed")
            refs = [str(ref).strip().strip("[]") for ref in reply.get("cited") or []]
            relied_on = [sheet.items[r] for r in refs if r in sheet.items]
            relied_on += [passages[int(r[1:]) - 1].text for r in refs if r[:1] == "P" and r[1:].isdigit() and 1 <= int(r[1:]) <= len(passages)]
            if text.endswith(("...", "…")) or len(text) < 12:
                problem = "Your previous reply broke off before the end. Reply again with every sentence finished."
            elif not on_topic(question, text, relied_on, view.site_words, view.index.embedder if view.index.vectors is not None else None):
                problem = "Your previous reply answered a different question. Answer only the QUESTION above."
            else:
                break
            if attempt == 2:
                if problem.startswith("Your previous reply broke off"):
                    raise ValueError("model returned an unfinished answer")
                return Draft(text=NOT_LISTED, confident=False, route="model-off-topic")
            user = "%s\n\n%s" % (user, problem)
        missing = ungrounded_numbers(text, sheet_text + "\n" + material_shown(user))
        if missing:
            return Draft(text=NOT_LISTED, confident=False, route="model-ungrounded",
                         conflict_note="discarded an answer containing %s, which the website material does not show" % ", ".join(missing))
        sources: List[Tuple[str, str]] = []
        for ref in refs:
            pages = []
            if ref in sheet.sources:
                pages = sheet.sources[ref]
            elif ref[:1] == "P" and ref[1:].isdigit() and 1 <= int(ref[1:]) <= len(passages):
                pages = passages[int(ref[1:]) - 1].pages
            for page in pages:
                if all(page.url != url for url, _ in sources):
                    sources.append((page.url, ""))
        if not sources:
            sources = [(p.page.url, "") for p in passages[:2]]
        for url in self._also_stated_on(view, text):
            if len(sources) < 6 and all(url != u for u, _ in sources):
                sources.append((url, ""))
        for stated in (self._state_the_count(view, question, text), self._name_the_area(view, text)):
            if stated is not None:
                text = "%s %s" % (text, stated.sentence)
                if all(stated.page.url != u for u, _ in sources):
                    sources.insert(0, (stated.page.url, stated.sentence))
        return Draft(text=text, sources=sources[:6], confident=True, route="model")

    @staticmethod
    def _name_the_area(view: SnapshotView, answer: str):
        """A credit union's rules say "live or work in our charter area" and leave the area to a
        map. When the answer repeats that phrase without saying where the area is, the site's
        current statement of where it serves is added. Returns the statement to add, or None."""
        if not _AREA.search(answer):
            return None
        for statement in view.statements:
            sentence = statement.current.sentence
            if statement.replaced_by is None and _SERVES.search(sentence):
                places = {w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", sentence) if stem(w.lower()) not in view.site_words}
                if places and sum(1 for place in places if place in answer) < min(2, len(places)):
                    return statement.current
        return None

    @staticmethod
    def _state_the_count(view: SnapshotView, question: str, answer: str):
        """The site keeps a running count ("This was the fifth time a magazine named us ...") and the
        answer says "several times", or "five times" in words of its own. Then the count is added
        in the site's own sentence. Returns the statement to add, or None. Only when the question
        names the subject of that count (by one of the two least common words of the sentence)
        and the answer does not already give the count as the site words it."""
        asked = set(tokens(question))
        for statement in view.statements:
            if not statement.running or statement.replaced_by is not None:
                continue
            current = statement.current
            words = [t for t in dict.fromkeys(tokens(current.sentence))
                     if t not in view.site_words and t not in current.values and view.index.df.get(t, 0) > 0]
            subject = sorted(words, key=lambda t: view.index.df[t])[:2]
            if any(t in asked for t in subject) and current.values[0] not in answer.lower():
                return current
        return None

    @staticmethod
    def _also_stated_on(view: SnapshotView, answer: str) -> List[str]:
        """Other pages that print the answer's most specific value (a routing number, a rate).
        They corroborate the answer, so they are cited too."""
        values = [m.group(0) for m in re.finditer(r"\d[\d,]*(?:\.\d+)?%|\b\d{5,}\b", answer)]
        if not values:
            return []
        pages = {}
        for passage in view.index.passages:
            if values[0] in passage.text:
                for page in passage.pages:
                    pages[page.url] = page
        if len(pages) > 12:              # printed everywhere: not specific enough to count as support
            return []
        return [p.url for p in sorted(pages.values(), key=lambda p: (-p.authority, p.url))]
