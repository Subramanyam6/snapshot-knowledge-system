"""Answer without a language model.

Used when no API key is configured or a model call fails, as the fast path for plain rate
look-ups, branch hours and branch addresses, and as the check on the model's rate answers:
  * rate questions are answered from the current rate tables (lookup.py), in a fixed sentence
    shape, so the same question always produces the same text;
  * branch hours and addresses are read from the printed schedules and the branch locator;
  * everything else is answered by quoting the best-matching sentences of the cleaned site.
Each route declines, or leaves the question to the next route, rather than guess when the match
is weak.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

from .facts import FLAG_OF_STEM
from .lookup import Asked, Match, RateLookup, _pairs, read_question
from .search import MAX_RESPELLED, QUERY_SYNONYMS, SYNONYM_PHRASES, query_tokens
from .snapshot import SnapshotView
from .tables import RateRow, RateTable
from .text import is_yes_no, normalise, sentences, stem, strip_opening, tokens

NOT_LISTED = ("I'm sorry, the website doesn't list that, so I can't confirm it. "
              "Please contact the credit union directly for details.")
# Variants that make a different product. A question that names one ("student loan refinance",
# "business certificate") is not answered from a table that lacks it.
_STRICT_VARIANTS = frozenset("business jumbo roth traditional student youth equity heloc".split())
# Words that pad a question without naming a product.
_FILLER = frozenset(stem(word) for word in (
    "current currently today now right offer offered available list provide looking like would new get typical "
    "general standard regular normal usual much many long often year month day percent percentage number amount "
    "rate interest apr apy dividend charge charged pay paid earn cost fee balance account loan member credit union "
    "between under over above below least most highest lowest best top maximum minimum type kind tier "
    "online website link page where look find see view check show").split())
# Words that ask for nothing: dates, list words, ways of saying "what the site states".
_NEUTRAL = frozenset(stem(word) for word in (
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec annual percentage yield "
    "quote quoted quoting advertise advertised advertising carry carries carried going go goes apply applied applies "
    "moment present presently buy buying finance financing financed keep kept put back inside start starting started "
    "label labeled labelled option options one along dollar dollars each every all list listed lists term terms "
    "non up down when then end beginning middle date dated day days today week weeks ago earlier later previously time point "
    "said says say state stated states show showed shows shown give gave given gives tell told posted post published printed "
    "website site official officially exactly precisely specifically actual actually really just thanks thank "
    "stand stood stands open opened opening hold held holds charged charging offering offers money got getting").split())
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)\)")
_WANTS_LINK = re.compile(r"\b(link|url|web ?site|web ?page|web ?address|internet address|which page|"
                         r"where (?:can|do|could|should|online)|online)\b", re.I)
# The address a site uses when it points at a page's rate table: ".../savings-account#rates".
_RATES_ANCHOR = re.compile(r"(https?://[^\s)\"'<>\]#]+?)/?#(?:\w+-)*rates?(?:-\w+)*\b", re.I)
_HEADLINE_METRICS = ("APY", "APR", "rate", "dividend rate")
# The kinds of rate a member can attach to a figure, matched against the start of a column's metric
# name. A bare "dividends" is left out: "pays 3.40% in dividends" usually means the yield.
_NAMED_METRIC = (("APY", r"apy|annual percentage yield"), ("APR", r"apr|annual percentage rate"),
                 ("dividend rate", r"dividend rate"))
_FIGURE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)


def _kind_of_figure(question: str, target: float) -> Optional[str]:
    """The kind of rate the member attached to the figure asked about: "3.96% APY", "an APY of
    3.96%", "is 3.96% the APY". None when the figure stands alone; in "6.75% with a 6.88% APR" the
    APR belongs to the second figure, not the first."""
    for m in _FIGURE.finditer(question):
        if abs(float(m.group(1)) - target) > 1e-9:
            continue
        after = re.split(r"\d", question[m.end():m.end() + 40])[0]          # up to the next figure
        before = re.split(r"\d", question[max(0, m.start() - 30):m.start()])[-1]
        for kind, pattern in _NAMED_METRIC:
            if re.match(r"\s*(?:the\s+|an?\s+)?(?:%s)\b" % pattern, after, re.I) \
                    or re.search(r"\b(?:%s)\s*(?:of|is|at|was|=)?\s*$" % pattern, before, re.I):
                return kind
        return None
    return None
_LOBBY = re.compile(r"\blobby\b", re.I)
_DRIVE_UP = re.compile(r"\bdrive[- ]?(?:up|thru|through)\b", re.I)
_ADDRESS_QUESTION = re.compile(r"\b(address(?:es)?|located|locations?|where|directions)\b", re.I)
_ADDRESS_WORDS = frozenset(stem(word) for word in (
    "address addresses located location locations where branch branches office offices directions street physical exact "
    "full find map visit go get give tell need credit union town city").split())
_SAME_EVERYWHERE = re.compile(r"\b(same|identical|alike|differ|different|vary|varies)\b", re.I)
_HOURS_QUESTION = re.compile(r"\b(lobby|drive[- ]?(?:up|thru|through)|(?:branch|office|business|banking|opening|your|our|the) hours)\b", re.I)
# Every word a question about hours in general can contain. One word outside this list (a town, a
# street, a store name) means the member has a particular branch in mind.
_HOURS_WORDS = frozenset(stem(word) for word in (
    "lobby lobbies drive up thru through drivethrough drivethru window hour hours open opens opening close closes closing closed "
    "time times when schedule schedules branch branches office offices location locations credit union "
    "general generally usual usually typical typically normal normally regular standard business banking bank operating "
    "operation operations today now current currently right weekday weekdays weekend weekends day days week "
    "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun "
    "all same identical alike similar different differ vary varies every each everywhere across keep keeps kept available "
    "late early start starts end ends until till "
    "stay run list teller tellers service services inside walk lane lanes here there").split())
MAX_ROWS = 12
MAX_TABLES = 4
MAX_SOURCES = 8


@dataclass
class Draft:
    text: str
    sources: List[Tuple[str, str]] = field(default_factory=list)      # (url, supporting quote)
    confident: bool = True
    conflict_note: Optional[str] = None
    route: str = ""
    complete: bool = True          # rate answers only: the rows cover everything the question asks
    matched: list = field(default_factory=list)     # rate answers only: the tables and rows it was read from


def _plain(text: str) -> str:
    """Markdown to speakable text; link targets are kept because members ask for them."""
    def link(m: "re.Match[str]") -> str:
        label, url = m.group(1).strip(), m.group(2)
        if not url.startswith("http") or label.startswith("http"):
            return label or url
        return "%s (%s)" % (label, url) if label else url
    text = _MD_LINK.sub(link, text)
    text = re.sub(r"[*_`#>]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _row_text(row: RateRow, minimum: bool = False, name: str = "") -> str:
    """One row as printed. `minimum` adds the balance the row needs, which is what tells a
    standard certificate from its jumbo twin. With the table's `name`, the label is only what
    the row adds to it ("$25,000+", not "Share Savings | $25,000+")."""
    values = ", ".join("%s %s" % (value, metric) for metric, value in row.values.items())
    if minimum:
        floors = [v for k, v in row.extras.items() if k.lower().startswith("min")]
        to_earn = [v for k, v in row.extras.items() if k.lower().startswith("min") and "earn" in k.lower()]
        if to_earn or floors:
            values += " (%s minimum)" % (to_earn or floors)[0]
    label = _tier(row, name) if name else row.label
    return "%s: %s" % (label, values) if label else values


def _tier(row: RateRow, table_name: str) -> str:
    """The part of a row label that names its tier: 'Share Savings | $25,000+' -> '$25,000+'."""
    parts = [p.strip() for p in row.label.split("|") if p.strip()]
    return ", ".join(p for p in parts if p.lower() != table_name.strip().lower())


def _page_order(page) -> tuple:
    """Which page to point a member to first: the product's own page, the higher up the better,
    then the general rates page, then anything else."""
    return ({"product": 0, "rates": 1}.get(page.kind, 2), page.url.rstrip("/").count("/"), len(page.url), page.url)


def _effective(table: RateTable) -> str:
    return " (rates effective %s)" % table.effective.strftime("%m/%d/%Y") if table.effective else ""


class RulesAnswerer:
    def __init__(self, view: SnapshotView, newest: bool = True):
        self.view = view
        self.newest = newest                     # False when answering about a past date
        self.lookup = RateLookup(view.current_tables, ignore=frozenset(view.site_words))
        self.fact_for = {}
        for fact in view.facts:
            self.fact_for[(id(fact.current.row), fact.metric)] = fact
        vocabulary = set()
        for words in self.lookup.row_words.values():
            vocabulary.update(words)
        self.rate_vocabulary = vocabulary
        self._anchors = None
        self.site_words = view.site_words          # the organisation's own name (snapshot.py)
        # The words, as written, that name the rate tables and their rows (see spelled).
        names = set(QUERY_SYNONYMS)
        for table in self.lookup.tables:
            names.update(re.findall(r"[a-z]{6,}", normalise(" ".join(
                [table.page.title.split("|")[0], table.section, table.caption] + [row.label for row in table.rows]))))
        self._names = sorted(n for n in names if len(n) >= 6 and n.isalpha())

    def spelled(self, question: str) -> str:
        """The question with a misspelt product word put right ("certficate" -> "certificate"),
        so that a typing slip does not hide the product from the rate tables. Only a word of six
        letters or more that the whole site never uses, and only when it is one slip away from a
        word in the rate tables' own names: the same first letter and at most one letter more or
        fewer. That keeps real words apart ("intern" is not "internet", "equality" is not "equity")."""
        tried = []                       # a long run of unknown words is not worth a search per word

        def fix(m: "re.Match[str]") -> str:
            word = m.group(0)
            lower = normalise(word)
            if stem(lower) in self.rate_vocabulary or stem(lower) in QUERY_SYNONYMS or self.view.index.df.get(stem(lower), 0):
                return word
            if len(tried) >= MAX_RESPELLED:
                return word
            tried.append(word)
            close = [name for name in difflib.get_close_matches(lower, self._names, n=3, cutoff=0.85)
                     if name[0] == lower[0] and abs(len(name) - len(lower)) <= 1]
            return close[0] if close else word
        return re.sub(r"[A-Za-z]{6,}", fix, question)

    def _rates_link(self, url: str) -> str:
        """The page address with the anchor the site itself uses for that page's rate table
        (seen at least twice), or the plain address."""
        if self._anchors is None:
            counts: dict = {}
            for page in self.view.pages:
                for line in page.lines:
                    if "#" in line:
                        for m in _RATES_ANCHOR.finditer(line):
                            key = (m.group(1).rstrip("/").lower(), m.group(0))
                            counts[key] = counts.get(key, 0) + 1
            best: dict = {}
            for (base, full), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                if n >= 2:
                    best.setdefault(base, full)
            self._anchors = best
        return self._anchors.get(url.rstrip("/").lower(), url)

    # ------------------------------------------------------------------ rate questions

    def rate_answer(self, question: str, asked: Optional[Asked] = None) -> Optional[Draft]:
        """Answer from the rate tables, or None when the question is not about a listed rate."""
        asked = asked or read_question(question)
        matches = self._matches(asked)
        if not self.newest and "jumbo" not in asked.flags | asked.excluded:
            # Looking back at a past date, only the product that was named is given. The jumbo row
            # of that day often carries the figure the standard row moved to later, and a member
            # reading both would take the old jumbo figure for the new standard one.
            matches = self._matches(replace(asked, excluded=asked.excluded | {"jumbo"})) or matches
        if not matches:
            return None
        # The same table printed on two pages under slightly different headings is read out once.
        seen, distinct = set(), []
        for m in matches:
            signature = tuple(sorted((_tier(r, m.table.name).lower(), tuple(sorted(r.numbers.items()))) for r in m.rows[:MAX_ROWS]))
            if signature not in seen:
                seen.add(signature)
                distinct.append(m)
        matches = distinct
        best = matches[0]
        keep = [m for m in matches[:MAX_TABLES] if m.score >= best.score - 0.5]
        if len(keep) >= 2 and "business" not in asked.flags:
            # A question about a whole category ("your credit cards") covers the business
            # products of that category too; they follow the personal ones.
            keep += [m for m in matches if m not in keep and m.score >= best.score - 2.5
                     and "business" in m.rows[0].key.flags][:MAX_TABLES]
        strict = asked.flags & _STRICT_VARIANTS
        if not best.exact or any(strict - r.key.flags for r in best.rows):
            return Draft(text=NOT_LISTED, confident=False, route="rate-not-listed")
        if asked.percents and is_yes_no(question):
            sources, note = self._backing(keep)
            return Draft(text=self._yes_or_no(keep, asked, question), sources=sources, confident=True, conflict_note=note,
                         route="rate-yes-no", matched=list(keep))
        if _WANTS_LINK.search(question):
            return self._where_listed(keep, asked)

        lines = []
        for match in keep:
            rows = match.rows[:MAX_ROWS]
            name = match.table.name
            # Where a standard row and its jumbo twin are both shown, their minimums tell them apart.
            kinds: dict = {}
            for row in rows:
                kinds.setdefault(row.qualifiers.term, set()).add("jumbo" in row.key.flags)
            twin = lambda row: len(kinds[row.qualifiers.term]) == 2
            if len(rows) == 1 and len(keep) == 1:
                row = _row_text(rows[0], name=name)
                lines.append("%s%s%s%s." % (name, ", " if ":" in row else ": ", row, _effective(match.table)))
            else:
                lines.append("%s%s: %s." % (name, _effective(match.table), "; ".join(_row_text(r, twin(r), name) for r in rows)))
        sources, note = self._backing(keep)
        return Draft(text=" ".join(lines), sources=sources, confident=True, conflict_note=note, route="rate",
                     complete=not self._beyond_the_rows(asked, keep), matched=list(keep))

    def _beyond_the_rows(self, asked: Asked, keep: Sequence[Match]) -> List[str]:
        """Words of the question that the matched tables do not account for ("late", "floor",
        "qualify"). When there are any, the member is asking for more than a rate row holds: a
        fee, a limit, a condition. The rows then inform the answer but do not settle it."""
        table_words = set()
        for match in keep:
            for row in match.table.rows:
                table_words.update(self.lookup.row_words[id(row)])
        return [w for w in asked.own_words
                if not (w in table_words or w in _FILLER or w in _NEUTRAL or w in self.site_words or w in FLAG_OF_STEM
                        or w[0].isdigit() or any(s in table_words for s in asked.synonyms.get(w, ())))]

    def _matches(self, asked: Asked) -> List[Match]:
        """The tables that really are the product asked about, best first. A table that merely
        shares a loose word with the question ("online", "look") is not one of them. The official
        rates and product pages are authoritative: a table that survives only on a campaign page
        or in a calculator is used when no official page has the product at all."""
        matches = [m for m in self.lookup.find(asked) if self._names_this_product(asked, m)]
        return [m for m in matches if m.table.page.kind in ("rates", "product")] or matches

    def _pages_of(self, match: Match) -> list:
        """Every page that prints the current copy of this table."""
        pages = []
        for row in match.rows[:MAX_ROWS]:
            for metric in row.values:
                fact = self.fact_for.get((id(row), metric))
                for occurrence in ([fact.current] + fact.copies) if fact else []:
                    if all(occurrence.page.url != p.url for p in pages):
                        pages.append(occurrence.page)
        return pages or [match.table.page]

    def _backing(self, keep: Sequence[Match]) -> Tuple[List[Tuple[str, str]], Optional[str]]:
        """Where the rows of an answer are printed (the product's own page first), and a note on
        any out-of-date copy that was set aside."""
        sources, notes = [], []
        for match in keep:
            quote = _row_text(match.rows[0])
            for page in sorted(self._pages_of(match), key=_page_order):
                if all(page.url != u for u, _ in sources):
                    sources.append((page.url, quote))
            for row in match.rows[:MAX_ROWS]:
                for metric in row.values:
                    fact = self.fact_for.get((id(row), metric))
                    for loser in fact.stale if fact else []:
                        notes.append("%s still shows %s for %s (%s)" % (loser.page.url, loser.value, row.label or match.table.name, fact.reason(loser)))
        # One page per table before a second page of any table, so a long answer still cites them all.
        first = list(dict.fromkeys(sorted(self._pages_of(m), key=_page_order)[0].url for m in keep))
        ordered = [(u, q) for u, q in sources if u in first] + [(u, q) for u, q in sources if u not in first]
        return ordered[:MAX_SOURCES], "; ".join(dict.fromkeys(notes)) or None

    def _yes_or_no(self, keep: Sequence[Match], asked: Asked, question: str = "") -> str:
        """'Is 0.42% paid on any savings tier?' is settled by the rows themselves. A "no" says
        what the rows do pay; it does not repeat the figure asked about, which may be an old one.
        When the question names the kind of rate ("3.96% APY"), only that column counts: a row
        paying a 3.96% dividend rate and a 4.00% APY does not pay 3.96% APY."""
        target = asked.percents[0]
        kind = _kind_of_figure(question, target)
        if kind and not any(metric.startswith(kind) for m in keep for row in m.rows[:MAX_ROWS] for metric in row.numbers):
            kind = None                     # a kind these rows do not have (an APR on a savings account): read them all

        def pays(row) -> bool:
            return any(abs(n - target) < 1e-9 for metric, numbers in row.numbers.items()
                       if kind is None or metric.startswith(kind) for n in numbers)
        hits = ["%s, %s" % (m.table.name, _row_text(row, name=m.table.name)) for m in keep for row in m.rows[:MAX_ROWS] if pays(row)]
        if hits:
            return "Yes. %s." % "; ".join(hits[:4])
        mine = [m for m in keep if "business" in asked.flags or "business" not in m.rows[0].key.flags] or list(keep)
        return "No. " + " ".join(self._span(m, kind) for m in mine[:MAX_TABLES])

    @staticmethod
    def _span(match: Match, kind: Optional[str] = None) -> str:
        """What one table pays, in a sentence: 'Share Savings: 0.00% to 0.20% APY (0.20% for $25,000+).'
        `kind` picks the column the member asked about (a "No" to "a 4.00% dividend rate?" gives the
        dividend rate). Without one, rows that print two kinds of rate are read out whole: "Is the
        30-year mortgage rate still 6.04%?" must get the interest rates, not only the APRs beside them."""
        rows, name = match.rows[:MAX_ROWS], match.table.name
        if kind is None and any(len(r.numbers) > 1 for r in rows):
            return "%s: %s." % (name, "; ".join(_row_text(r, name=name) for r in rows))
        metric = next((k for r in rows for k in r.numbers if k.startswith(kind)), None) if kind else None
        metric = metric or next((m for m in _HEADLINE_METRICS if any(m in r.numbers for r in rows)), None)
        having = [r for r in rows if metric in r.numbers and len(r.numbers[metric]) == 1]
        if metric is None or len(having) != len(rows):
            return "%s: %s." % (name, "; ".join(_row_text(r, name=name) for r in rows))       # ranges or mixed columns: read the rows out
        low = min(having, key=lambda r: r.numbers[metric][0])
        high = max(having, key=lambda r: r.numbers[metric][0])
        if low.numbers[metric] == high.numbers[metric]:
            return "%s: %s %s." % (name, high.values[metric], metric)
        tier = _tier(high, name)
        return "%s: %s to %s %s%s." % (name, low.values[metric], high.values[metric], metric,
                                       " (%s for %s)" % (high.values[metric], tier) if tier else "")

    def _where_listed(self, matches: Sequence[Match], asked: Asked) -> Draft:
        """For 'which link shows the ... rates': one link. The product's own page when it prints the
        current table, with the anchor the site uses for its rate section; else the rates page."""
        mine = [m for m in matches if "business" in asked.flags or "business" not in m.rows[0].key.flags] or list(matches)
        pages = []
        for match in mine:
            for page in self._pages_of(match):
                if page.kind in ("rates", "product") and all(page.url != p.url for p in pages):
                    pages.append(page)
        if not pages:
            return Draft(text=NOT_LISTED, confident=False, route="rate-not-listed")
        # A page the site itself links to as "...#rates" is the page it means members to use.
        pages.sort(key=lambda p: (p.kind != "product", self._rates_link(p.url) == p.url) + _page_order(p))
        sections = {m.table.section for m in mine if m.table.section}
        names = list(dict.fromkeys(m.table.name for m in mine))
        label = sections.pop() if len(sections) == 1 else " and ".join(names[:2])
        text = "Current %s rates are listed at %s." % (label, self._rates_link(pages[0].url))
        return Draft(text=text, sources=[(p.url, "") for p in pages[:MAX_SOURCES]], confident=True, route="rate-link")

    def _names_this_product(self, asked: Asked, match: Match) -> bool:
        """Is the matched table really the product the question names? No when
          * nothing but filler words matched ("rate", "loan");
          * the question carries a number the table knows nothing about (a "7/1" mortgage);
          * the question uses a word that names some *other* rate product."""
        table_words = set()
        for row in match.table.rows:
            table_words.update(self.lookup.row_words[id(row)])
        # Some word of the question must name the product itself. "Business" or "jumbo" alone names
        # a variant: a "business equipment loan" is not the business savings table.
        if not any(w in table_words and w not in _FILLER and w not in self.site_words and w not in FLAG_OF_STEM
                   and not w.isdigit() for w in asked.words):
            return False
        if any(n not in table_words for n in asked.other_numbers):
            return False
        for w in asked.own_words:
            if w in SYNONYM_PHRASES and w not in table_words:
                # "Mortgage" means a table whose name says "home loan", with the two words together;
                # "Home Equity Loan" has both words and is a different product.
                if not _pairs(asked.synonyms.get(w, ())) & self.lookup.name_pairs[id(match.table)]:
                    return False
            if w in table_words or w in _FILLER or w in self.site_words or w in FLAG_OF_STEM or w.isdigit():
                continue
            if any(s in table_words for s in asked.synonyms.get(w, ())):
                continue                # the member's word for something the table does name
            if w in self.lookup.product_words:
                return False
        return True

    # ----------------------------------------------------------------- other questions

    def passage_answer(self, question: str) -> Draft:
        hits = self.view.index.search(question, k=6)
        if not hits:
            return Draft(text=NOT_LISTED, confident=False, route="no-hit")
        wanted = set(query_tokens(question))
        rare = {w for w in wanted if 0 < self.view.index.df.get(w, 0) <= max(3, self.view.index.n // 150)}
        missing = {w for w in wanted if w not in self.view.index.df and len(w) > 3 and not w.isdigit()}
        best_passage, best_score = hits[0]
        passage_words = set(tokens(best_passage.text + " " + best_passage.heading + " " + best_passage.page.title))
        if missing or (rare and not rare <= passage_words):
            return Draft(text=NOT_LISTED, confident=False, route="weak-hit")

        scored = []
        for order, sentence in enumerate(sentences(best_passage.text)):
            words = set(tokens(sentence))
            overlap = sum(self.view.index.idf(w) for w in wanted if w in words)
            if overlap > 0 and not sentence.startswith("#"):
                scored.append((overlap, order, sentence))
        if not scored:
            return Draft(text=NOT_LISTED, confident=False, route="weak-hit")
        top = sorted(sorted(scored, key=lambda s: (-s[0], s[1]))[:2], key=lambda s: s[1])
        text = " ".join(_plain(s) for _, _, s in top)[:600]
        sources = [(p.url, _plain(top[0][2])[:200]) for p in best_passage.pages[:4]]
        return Draft(text=text, sources=sources, confident=True, route="passage")

    # ------------------------------------------------------------------- branch hours

    def hours_answer(self, question: str) -> Optional[Draft]:
        """Lobby and drive-up hours from the schedules the site prints (hours.py).

        A general question gets every distinct schedule with the branches that keep it. A question
        that names a branch gets that branch's schedule. Anything else returns None: a branch the
        site prints no hours for must not be answered with another branch's hours."""
        blocks = self.view.hours
        if not blocks or not _HOURS_QUESTION.search(question):
            return None
        wants = [name for name, pattern in (("Lobby", _LOBBY), ("Drive-up", _DRIVE_UP)) if pattern.search(question)] or ["Lobby", "Drive-up"]
        extra = [w for w in tokens(question.replace("-", " "))
                 if w not in _HOURS_WORDS and w not in self.site_words and not w.isdigit()]
        if extra:
            named = [b for b in blocks if set(extra) <= set(tokens(b.place.replace("-", " ")))]
            if len(named) != 1:
                return None
            block = named[0]
            parts = ["%s - %s" % (name, "; ".join(block.sections[name])) for name in wants if block.sections.get(name)]
            if not parts:
                return None
            text = "%s branch hours. %s." % (block.place, ". ".join(parts))
            return Draft(text=text, sources=[(block.page.url, block.text()[:200])], confident=True, route="hours-branch")

        sentences_out, pages = [], []
        for name in wants:
            groups: dict = {}
            for block in blocks:
                rows = block.sections.get(name)
                if rows:
                    key = re.sub(r"[\s–—-]+", "-", " ".join(rows).lower())
                    groups.setdefault(key, []).append(block)
            if not groups:
                continue
            ordered = sorted(groups.values(), key=lambda g: -len(g))
            pieces = []
            for group in ordered:
                places = list(dict.fromkeys(b.place for b in group if b.place))
                pieces.append("%s%s" % ("; ".join(group[0].sections[name]), " (%s)" % ", ".join(places) if places else ""))
                pages.extend(b.page for b in group)
            lead = "%s hours vary by branch" % name if len(ordered) > 1 else "%s hours" % name
            sentences_out.append("%s: %s." % (lead, ". ".join(pieces)))
        if not sentences_out:
            return None
        text = " ".join(sentences_out) + " Which branch do you need?"
        if is_yes_no(question) and _SAME_EVERYWHERE.search(question):
            # "Are the hours alike at every branch?" is settled by counting the schedules above.
            text = ("No. " if " vary by branch" in text else "Yes. ") + text
        sources = [(url, "") for url in dict.fromkeys(p.url for p in pages)]
        return Draft(text=text, sources=sources[:MAX_SOURCES], confident=True, route="hours")

    # --------------------------------------------------------------- branch addresses

    def address_answer(self, question: str) -> Optional[Draft]:
        """"Where is the Brookhaven branch located?" from the branches the locator lists. Only
        when every word that is not part of asking for an address is found in the address of a
        listed branch; a branch the locator does not list is left to the model, which will say
        so rather than give some other branch's address."""
        places = [place for place in self.view.locations if "branch" in place.kind.lower()]
        if not places or not _ADDRESS_QUESTION.search(question):
            return None
        named_by = [w for w in tokens(question.replace("-", " "))
                    if w not in _ADDRESS_WORDS and w not in self.site_words and not w.isdigit()]
        found = [place for place in places if named_by and set(named_by) <= set(tokens(place.address.replace("-", " ")))]
        if not found:
            return None
        if len(found) == 1:
            text = "The %s branch is at %s." % (found[0].city, found[0].address)
        else:
            text = "The locator lists %d branches in %s: %s." % (len(found), found[0].city, "; ".join(place.address for place in found))
        return Draft(text=text, sources=[(self.view.locator, found[0].address)], confident=True, route="branch-address")

    def answer(self, question: str) -> Draft:
        # A greeting or a spoken filler in front is not part of what is asked or searched for.
        question = self.spelled(strip_opening(question))
        asked = read_question(question)
        if asked.wants_rate:
            draft = self.rate_answer(question, asked)
            if draft is not None:
                return draft
            if asked.names_rate:
                # Every published rate sits in a rate table. A rate that is in none of them is not
                # published, whatever nearby text says about the product.
                return Draft(text=NOT_LISTED, confident=False, route="rate-not-found")
        return self.hours_answer(question) or self.address_answer(question) or self.passage_answer(question)
