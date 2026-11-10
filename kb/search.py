"""Search over the passages of one snapshot: by keywords, and by meaning when a model is loaded.

Keyword ranking is plain BM25 with two transparent adjustments, both of which favour the copy a
careful person would trust: the kind of page (a product page over a calculator) and, for dated
articles, how recently they were published.

Meaning ranking (see meaning.py) orders passages by how close their vector is to the question's.
It finds the page that answers a question in other words; keywords find exact names, numbers and
hours. The two are kept apart on purpose: `search` is keywords only and `nearest` is meaning only,
and the caller decides how to use each. (Merging them into one ranking was tried and measured: it
pushed out passages that keyword search had rightly put first.)
"""
from __future__ import annotations

import datetime as dt
import difflib
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .meaning import TEXT_CHARS
from .pages import Page
from .text import digits_for_number_words, stem, tokens

# Multipliers by page kind. Stale copies are already removed before indexing; this only orders
# the remaining pages when several say something relevant.
KIND_WEIGHT = {
    "rates": 1.30, "product": 1.25, "document": 1.10, "news": 1.00, "blog": 0.90,
    "promo": 0.70, "calculator": 0.60, "listing": 0.45, "expired": 0.0,
}
RECENCY_BONUS = 0.30          # an article published today scores up to 30% above a five-year-old one
RECENCY_WINDOW_DAYS = 5 * 365
MAX_RESPELLED = 8             # unknown question words looked up as possible typos, per question
# Meaning search has no such weighting, so it simply leaves out the kinds of page that keyword
# search trusts least: calculators (their figures are worked examples), listings and promotions.
MEANING_KINDS = frozenset(kind for kind, weight in KIND_WEIGHT.items() if weight >= 0.9)

# Everyday words members use for the words the site uses. Applied to the question only, and
# looked up by stem because that is how question words arrive.
QUERY_SYNONYMS = {stem(word): alternatives for word, alternatives in {
    "cd": ["certificate"], "car": ["auto"], "vehicle": ["auto"], "truck": ["auto"], "automobile": ["auto"],
    "mortgage": ["home", "loan"], "aba": ["routing"], "house": ["home"], "motorbike": ["motorcycle"],
    "location": ["branch"], "office": ["branch"], "drivethrough": ["drive", "up"],
    "staff": ["employee"], "join": ["member", "membership"], "signup": ["join"], "heloc": ["home", "equity", "line"],
    "interest": ["rate"], "earn": ["dividend", "apy"], "yield": ["apy"],
    "internet": ["online"], "web": ["online"], "enroll": ["join"], "register": ["join"], "sign": ["join"],
    "qualify": ["join"], "eligible": ["join"], "eligibility": ["join"],
}.items()}
# Words whose alternatives form one phrase ("mortgage" = "home loan"), not a choice of words.
SYNONYM_PHRASES = frozenset(stem(word) for word in ("mortgage", "heloc", "drivethrough"))


# What an answer to a question must look like, and how to recognise a passage that contains one.
# A question about hours is best answered by a passage that actually prints times of day.
ANSWER_SHAPES = (
    ("time", re.compile(r"\b(hours?|open|opens|close|closes|closing|opening)\b", re.I),
     re.compile(r"\b\d{1,2}(?::\d{2})?\s*[ap]\.?m\b|\bnoon\b", re.I)),
    ("count", re.compile(r"\bhow many\b", re.I), re.compile(r"\b\d[\d,]*\b")),
    ("address", re.compile(r"\b(address|located|where is|where's|directions)\b", re.I),
     re.compile(r"\b\d{2,5} [A-Z][\w.]*(?: [A-Z][\w.]*)* (?:St|Rd|Ave|Dr|Ln|Blvd|Hwy|Way|Ct|Street|Road|Avenue|Drive|Lane)\b")),
    ("date", re.compile(r"\b(when|what date|which day)\b", re.I),
     re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}(?:, \d{4})?\b")),
    ("phone", re.compile(r"\b(phone|call|telephone)\b", re.I), re.compile(r"\b\d{3}[-.)]\s?\d{3}[-.]\d{4}\b")),
    ("link", re.compile(r"\b(link|url|website|web ?page|online)\b", re.I), re.compile(r"https?://")),
)
SHAPE_BONUS = 1.35


@dataclass
class Passage:
    """A piece of one page, small enough to quote."""
    page: Page
    heading: str
    text: str
    table: object = None                              # the RateTable this passage shows, if any
    also_on: List[Page] = field(default_factory=list)   # other pages carrying the identical text
    position: int = -1                                # order within its page; -1 for stand-alone facts

    @property
    def pages(self) -> List[Page]:
        return [self.page] + self.also_on


def query_tokens(question: str) -> List[str]:
    base = tokens(digits_for_number_words(question).replace("-", " "))
    extra: List[str] = []
    for t in base:
        extra.extend(QUERY_SYNONYMS.get(t, ()))
    return base + [t for t in tokens(" ".join(extra)) if t not in base]


def meaning_text(p: Passage) -> str:
    """What the embedding model reads for a passage: where it is from, then what it says."""
    return "%s. %s\n%s" % (p.page.title.split("|")[0].strip(), p.heading, p.text[:TEXT_CHARS])


class Index:
    def __init__(self, passages: Sequence[Passage], today: dt.date, k1: float = 1.4, b: float = 0.6, embedder=None):
        self.passages = list(passages)
        self.k1, self.b = k1, b
        self.df: Counter = Counter()
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.length: List[int] = []
        for i, p in enumerate(self.passages):
            # Title and heading count twice: they say what the passage is about.
            counts = Counter(tokens(p.text) + 2 * tokens(p.page.title + " " + p.heading))
            self.length.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings[term].append((i, freq))
                self.df[term] += 1
        self.n = len(self.passages)
        self.avg_length = (sum(self.length) / self.n) if self.n else 1.0
        self.prior = [self._prior(p, today) for p in self.passages]
        self.shapes = [frozenset(name for name, _, shown in ANSWER_SHAPES if shown.search(p.text)) for p in self.passages]
        self.by_page: Dict[str, List[Passage]] = defaultdict(list)
        for p in sorted((p for p in self.passages if p.position >= 0), key=lambda p: p.position):
            self.by_page[p.page.url].append(p)
        self._vocabulary: Optional[List[str]] = None      # built on first use, see _respelled
        self._spelling: Dict[str, Optional[str]] = {}
        # Search by meaning: one vector per passage in `by_meaning`. Without an embedder, or if
        # embedding fails, `vectors` stays None and every search is by keywords.
        self.embedder = embedder
        self.vectors = None
        # Rate tables are left out as well: rates are read from the resolved rate sheet.
        self.by_meaning = [i for i, p in enumerate(self.passages)
                           if p.table is None and p.page.kind in MEANING_KINDS]
        if embedder is not None and self.by_meaning:
            try:
                self.vectors = embedder.passages([meaning_text(self.passages[i]) for i in self.by_meaning])
            except Exception as error:
                print("knowledge system: could not embed the passages (%s: %s); searching by keywords only"
                      % (type(error).__name__, str(error)[:120]), file=sys.stderr)

    @staticmethod
    def _prior(p: Passage, today: dt.date) -> float:
        weight = max(KIND_WEIGHT.get(page.kind, 1.0) for page in p.pages)
        published = p.page.published_on
        if published and p.page.kind in ("news", "blog"):
            age = max(0, (today - published).days)
            weight *= 1.0 + RECENCY_BONUS * max(0.0, 1.0 - age / RECENCY_WINDOW_DAYS)
        return weight

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, question: str, k: int = 8) -> List[Tuple[Passage, float]]:
        """Keyword search."""
        return [(self.passages[i], s) for s, i in self._keyword_ranking(question)[:k]]

    def nearest(self, question: str, k: int = 3) -> List[Tuple[Passage, float]]:
        """The passages closest in meaning to the question, best first, with their similarity
        (1.0 = same direction). Empty when no embedding model is loaded."""
        if self.vectors is None or k <= 0:
            return []
        try:
            np = self.embedder.np
            similarity = self.vectors @ self.embedder.question(question)       # one row per by_meaning entry
            k = min(k, len(similarity))
            top = np.argpartition(-similarity, k - 1)[:k]
            ranked = sorted((int(row) for row in top), key=lambda row: (-float(similarity[row]), row))
            return [(self.passages[self.by_meaning[row]], float(similarity[row])) for row in ranked]
        except Exception as error:
            print("knowledge system: search by meaning failed (%s); used keywords only" % type(error).__name__, file=sys.stderr)
            return []

    def _respelled(self, term: str) -> Optional[str]:
        """A question word the site never uses is probably a typo ("elligible", "morgage"). The
        closest word the site does use takes its place, when one is close enough."""
        if term not in self._spelling:
            if self._vocabulary is None:
                self._vocabulary = [t for t, seen in self.df.items() if seen >= 3 and len(t) >= 5 and t.isalpha()]
            close = difflib.get_close_matches(term, self._vocabulary, n=1, cutoff=0.85)
            self._spelling[term] = close[0] if close else None
        return self._spelling[term]

    def _keyword_ranking(self, question: str) -> List[Tuple[float, int]]:
        scores: Dict[int, float] = defaultdict(float)
        terms = set(query_tokens(question))
        # A real question has a typo or two; a long string of unknown words is not worth a
        # dictionary search per word, so only the first few are tried.
        unknown = [t for t in sorted(terms) if t not in self.postings and len(t) >= 6 and t.isalpha()][:MAX_RESPELLED]
        for term in unknown:
            if term not in self.postings and len(term) >= 6 and term.isalpha():
                fixed = self._respelled(term)
                if fixed:
                    terms.add(fixed)
                    terms.update(tokens(" ".join(QUERY_SYNONYMS.get(fixed, ()))))
        for term in terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for i, freq in postings:
                norm = freq + self.k1 * (1 - self.b + self.b * self.length[i] / self.avg_length)
                scores[i] += idf * freq * (self.k1 + 1) / norm
        wanted = {name for name, asks, _ in ANSWER_SHAPES if asks.search(question)}
        return sorted(((s * self.prior[i] * (SHAPE_BONUS if wanted & self.shapes[i] else 1.0), i)
                       for i, s in scores.items()), key=lambda x: (-x[0], x[1]))
