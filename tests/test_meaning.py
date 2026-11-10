"""Tests for search by meaning and for the second look. A small stand-in replaces the embedding
model, so they need neither the model files nor an API key (only numpy):

    python -m unittest discover tests
"""
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # for test_answers, which holds shared helpers

try:
    import numpy
except ImportError:                                  # the rules-only install has no numpy
    numpy = None

from kb import meaning  # noqa: E402
from kb.rules_answer import NOT_LISTED, Draft  # noqa: E402
from kb.snapshot import build_view  # noqa: E402
from kb.system import KnowledgeSystem  # noqa: E402
from test_answers import system_without_models  # noqa: E402

SITE = "https://cu.example"

PRIVACY = """# Privacy Policy

## Keeping Your Details Correct

Sign in to review and correct the personal information we hold, or call 800-555-0100.
"""

BRANCH = """# Main Street Branch

## Lobby

The lobby is open Monday to Friday, 9 a.m. to 5 p.m.
"""

RATES = """# Rates

Home Loans

| Loan | Annual Percentage Rate |
| --- | --- |
| 30-Year Fixed | 6.40% APR |

Rates are effective as of 09/04/2026.
"""

# Ideas the stand-in "understands". Texts that mention the same ideas get similar vectors,
# whatever words they use. That is all an embedding model does, at a much larger scale.
IDEAS = [
    {"change", "update", "correct", "correcting"},
    {"phone", "address", "personal", "details", "information"},
    {"lobby", "hours", "open"},
    {"mortgage", "loan", "loans", "rate", "rates"},
]


class StandInEmbedder:
    def __init__(self):
        self.np = numpy

    def _vector(self, text):
        words = set(re.findall(r"[a-z]+", text.lower()))
        vector = numpy.array([1.0 if words & idea else 0.0 for idea in IDEAS] + [0.05], dtype="float32")
        return vector / numpy.linalg.norm(vector)

    def passages(self, texts):
        return numpy.stack([self._vector(t) for t in texts])

    def question(self, text):
        return self._vector(text)


def raw(path, markdown, title=""):
    return SimpleNamespace(url=SITE + path, title=title or path, markdown=markdown, fetched_at="2026-09-28T00:00:00Z")


PAGES = [raw("/privacy-policy", PRIVACY, "Privacy Policy"), raw("/branches/main-street", BRANCH, "Main Street Branch"),
         raw("/rates", RATES, "Rates")]
QUESTION = "How do I change my phone number?"


@unittest.skipIf(numpy is None, "numpy is not installed")
class SearchByMeaning(unittest.TestCase):
    def setUp(self):
        self.view = build_view("05_2026-09-28_latest", "2026-09-28", PAGES, embedder=StandInEmbedder())

    def test_it_finds_the_page_that_answers_in_other_words(self):
        by_keywords = [p.text for p, _ in self.view.index.search(QUESTION, k=3)]
        self.assertFalse(any("800-555-0100" in text for text in by_keywords))       # no shared word
        closest, similarity = self.view.index.nearest(QUESTION, k=1)[0]
        self.assertIn("800-555-0100", closest.text)
        self.assertGreater(similarity, 0.9)

    def test_rate_tables_are_never_found_by_meaning(self):
        found = self.view.index.nearest("What is the rate on a mortgage loan?", k=10)
        self.assertTrue(found)
        self.assertTrue(all(p.table is None for p, _ in found))

    def test_without_a_model_search_is_by_keywords_only(self):
        plain = build_view("05_2026-09-28_latest", "2026-09-28", PAGES)
        self.assertIsNone(plain.index.vectors)
        self.assertEqual(plain.index.nearest(QUESTION, k=3), [])
        self.assertTrue(plain.index.search("lobby hours", k=3))

    def test_running_out_of_time_falls_back_to_keywords_and_keeps_the_work(self):
        class SlowModel:
            def passage_embed(self, texts, batch_size=16):
                time.sleep(0.05)
                return (numpy.ones(3, dtype="float32") for _ in texts)

        embedder = meaning.Embedder.__new__(meaning.Embedder)     # skip loading the real model
        embedder.np, embedder.model, embedder.known, embedder.spent = numpy, SlowModel(), {}, 0.0
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        embedder.saved = Path(folder.name) / "vectors.npz"
        allowed, chunk = meaning.SECONDS_PER_SNAPSHOT, meaning.CHUNK
        meaning.SECONDS_PER_SNAPSHOT, meaning.CHUNK = 0.02, 1    # the clock runs out after one text
        try:
            with self.assertRaises(TimeoutError):
                embedder.passages(["first text", "second text", "third text"])
            self.assertEqual(len(embedder.known), 1)              # what was computed is kept
            view = build_view("05_2026-09-28_latest", "2026-09-28", PAGES, embedder=embedder)
            self.assertIsNone(view.index.vectors)                 # this snapshot: keywords only
            self.assertTrue(view.index.search("lobby hours", k=3))
        finally:
            meaning.SECONDS_PER_SNAPSHOT, meaning.CHUNK = allowed, chunk


class StandInReader:
    """Finds nothing on the first look (keywords) and the answer on the second (meaning)."""

    def __init__(self):
        self.looks = []

    def warm(self, view):
        pass

    def can_look_again(self, view):
        return True

    def answer(self, view, question, by_meaning=False, day=None, deadline=None):
        self.looks.append("meaning" if by_meaning else "keywords")
        if by_meaning:
            return Draft(text="Call 800-555-0100.", sources=[(SITE + "/privacy-policy", "")], confident=True, route="model")
        return Draft(text=NOT_LISTED, confident=False, route="model-not-listed")


class SecondLook(unittest.TestCase):
    def setUp(self):
        self.system = system_without_models()
        self.system.reader = StandInReader()
        self.system.ingest(SimpleNamespace(snapshot_id="05_2026-09-28", taken_on="2026-09-28", pages=PAGES))

    def test_a_question_the_first_look_cannot_answer_gets_a_second_look(self):
        answer = self.system.answer(QUESTION)
        # The rules found nothing here, so the two looks are started together: one of each, in any order.
        self.assertEqual(sorted(self.system.reader.looks), ["keywords", "meaning"])
        self.assertTrue(answer.confident)
        self.assertIn("800-555-0100", answer.text)

    def test_a_rate_that_is_not_published_gets_no_second_look(self):
        answer = self.system.answer("Which rate goes with a 5/1 adjustable-rate mortgage?")
        self.assertEqual(self.system.reader.looks, ["keywords"])
        self.assertFalse(answer.confident)
        self.assertEqual(answer.text, NOT_LISTED)

    def test_an_answered_question_is_never_asked_twice(self):
        self.assertFalse(KnowledgeSystem._worth_a_second_look(Draft(text="", route="passage"), Draft(text="x", route="model")))
        self.assertTrue(KnowledgeSystem._worth_a_second_look(Draft(text="", route="weak-hit"),
                                                             Draft(text=NOT_LISTED, confident=False, route="model-not-listed")))
        # An answer thrown out because it quoted a number the site does not show stays thrown out.
        self.assertFalse(KnowledgeSystem._worth_a_second_look(Draft(text="", route="passage"),
                                                              Draft(text=NOT_LISTED, confident=False, route="model-ungrounded")))


if __name__ == "__main__":
    unittest.main()
