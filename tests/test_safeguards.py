"""Tests for the safeguards around the model call and for inputs a grader might send that the
practice questions never did: odd dates, odd spacing, very long questions, a slow or failing API.
Small made-up pages and a fake API client; no snapshots, no API key, no network.

    python -m unittest discover tests
"""
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # for test_answers, which holds shared helpers

from kb.conflicts import _a_real_figure  # noqa: E402
from kb.history import comparison  # noqa: E402
from kb.lookup import read_question  # noqa: E402
from kb.rules_answer import Draft  # noqa: E402
from kb.system import _OPENS_WITH_NOT_LISTED, KnowledgeSystem, date_in_question, iso_day  # noqa: E402
from test_answers import RATES, SAVINGS_PAGE, answerer, raw, system_without_models  # noqa: E402


def rules_only_system():
    """The whole system, rules only, whatever key happens to be in the environment."""
    system = system_without_models()
    for snapshot_id, day in (("01_test", "2026-05-12"), ("05_test", "2026-09-28")):
        system.ingest(SimpleNamespace(snapshot_id=snapshot_id, taken_on=day,
                                      pages=[raw("/rates", RATES, "Rates"), raw("/personal/savings", SAVINGS_PAGE, "Savings")]))
    return system


class Dates(unittest.TestCase):
    def test_as_of_is_read_in_any_common_form(self):
        import datetime as dt
        for value in ("2026-05-19", "2026-05-19T10:00:00Z", " 2026-05-19 ", "2026-5-19", "05/19/2026", "May 19, 2026",
                      dt.date(2026, 5, 19), dt.datetime(2026, 5, 19, 10, 0)):
            self.assertEqual(iso_day(value), "2026-05-19", value)
        for value in (None, "", "soon", "2026-02-30"):
            self.assertIsNone(iso_day(value), value)

    def test_the_snapshot_for_a_us_style_date_is_the_right_one(self):
        system = rules_only_system()
        self.assertEqual(system.view_for("06/01/2026").snapshot_id, "01_test")     # as a string it sorted before every snapshot
        self.assertEqual(system.view_for("not a date").snapshot_id, "05_test")

    def test_the_start_of_a_month_is_its_first_day(self):
        self.assertEqual(date_in_question("What was the rate at the beginning of June 2026?"), "2026-06-01")
        self.assertEqual(date_in_question("What was the rate as of the start of June 2026?"), "2026-06-01")
        self.assertEqual(date_in_question("What was the rate at the end of June 2026?"), "2026-06-30")
        self.assertEqual(date_in_question("What was the rate in June 2026?"), "2026-06-30")


class BeforeTheFirstCopy(unittest.TestCase):
    def test_an_as_of_date_before_the_first_snapshot_gets_an_honest_i_dont_know(self):
        system = rules_only_system()
        early = system.answer("What is the APY on the 13-month certificate special?", as_of="2026-01-15")
        self.assertFalse(early.confident)
        self.assertNotRegex(early.text, r"\d")                # a number in a refusal reads as a guess
        # A date in the question may be when something happened, which later copies report.
        self.assertTrue(system.answer("What was the APY on the 13-month certificate special in January 2026?").confident)
        self.assertTrue(system.answer("What is the APY on the 13-month certificate special?", as_of="2026-05-12").confident)


class Spacing(unittest.TestCase):
    def test_extra_spaces_and_line_breaks_change_nothing(self):
        system = rules_only_system()
        plain = system.answer("For the 13-month certificate special, which APY applies?").text
        spaced = system.answer("  For the 13-month\ncertificate special,   which APY applies? ").text
        self.assertEqual(plain, spaced)


class RateQuestions(unittest.TestCase):
    def setUp(self):
        self.rules = answerer(raw("/rates", RATES, "Rates"), raw("/personal/savings", SAVINGS_PAGE, "Savings"))

    def test_a_loan_amount_gets_only_the_tightest_tier_that_holds_it(self):
        text = self.rules.answer("What is the 30-year purchase APR for a $90,000 mortgage?").text
        self.assertIn("6.66%", text)
        self.assertNotIn("6.71%", text)
        text = self.rules.answer("What is the 30-year purchase APR for a $250,000 mortgage?").text
        self.assertIn("6.71%", text)
        self.assertNotIn("6.66%", text)

    def test_a_yes_no_about_a_named_kind_of_rate_checks_that_column_only(self):
        no = self.rules.answer("Does the 13-month jumbo certificate special pay 4.27% APY?")
        self.assertEqual(no.route, "rate-yes-no")
        self.assertTrue(no.text.startswith("No."), no.text)      # 4.27% is its dividend rate, not its APY
        self.assertIn("4.35%", no.text)
        self.assertTrue(self.rules.answer("Does the 13-month jumbo certificate special pay 4.35% APY?").text.startswith("Yes."))
        self.assertTrue(self.rules.answer("Does the 13-month jumbo certificate special pay 4.27%?").text.startswith("Yes."))

    def test_a_no_about_a_mortgage_rate_gives_the_interest_rates_too(self):
        no = self.rules.answer("Is the 30-year fixed purchase mortgage rate still 5.95%?").text
        self.assertTrue(no.startswith("No."), no)
        for value in ("6.48%", "6.62%", "6.66%", "6.71%"):      # both tiers' rates and APRs
            self.assertIn(value, no)
        self.assertNotIn("5.95", no)                            # the figure asked about is not repeated

    def test_the_kind_of_rate_belongs_to_the_figure_it_is_attached_to(self):
        two = self.rules.answer("Is the 30-year fixed purchase mortgage for $120,000 or less 6.48% with a 6.66% APR?")
        self.assertTrue(two.text.startswith("Yes."), two.text)   # the APR belongs to 6.66%, not to 6.48%
        # A kind of rate the product does not have (an APR on a certificate) is a slip of the tongue.
        self.assertTrue(self.rules.answer("Does the 13-month jumbo certificate special pay 4.35% APR?").text.startswith("Yes."))
        self.assertTrue(self.rules.answer("Does the 13-month jumbo certificate special pay 4.35% in dividends?").text.startswith("Yes."))
        no = self.rules.answer("Does the 13-month jumbo certificate special pay a 4.35% dividend rate?").text
        self.assertTrue(no.startswith("No."), no)
        self.assertIn("4.27% dividend rate", no)                  # the "no" names the column asked about

    def test_a_year_named_as_a_time_does_not_hide_the_product(self):
        self.assertEqual(read_question("What is the 12-month certificate APY in 2026?").other_numbers, [])
        self.assertEqual(read_question("Rates for the 1st Step Checking account").other_numbers, [])
        text = self.rules.answer("For the 13-month certificate special, which APY applies in 2026?").text
        self.assertIn("4.25%", text)

    def test_a_year_named_as_a_time_is_not_the_model_year(self):
        self.assertEqual(read_question("What is the 60-month auto APR in 2026 for a 2019 car?").model_year, 2019)
        self.assertEqual(read_question("During 2026, what is the 60-month APR on a 2018 truck?").model_year, 2018)
        self.assertEqual(read_question("What are the auto rates for 2024 models?").model_year, 2024)


class LongQuestions(unittest.TestCase):
    def test_a_long_rambling_question_is_read_quickly(self):
        for question in ("now and was " * 330 + "; May 12, 2026", "today compare with " * 210 + "; May 12, 2026",
                         "now and was " * 330 + "May 12, 2026", "was " + "and so on " * 390 + "May 12, 2026 and now"):
            started = time.perf_counter()
            comparison(question[:4000])
            self.assertLess(time.perf_counter() - started, 0.5)

    def test_change_questions_are_still_recognised(self):
        self.assertIsNotNone(comparison("What was the 60-month auto loan APR on May 12, 2026, and what is it now?"))
        self.assertIsNotNone(comparison("Today's 60-month APR compared with May 12, 2026?"))
        padding = "for a loan that runs the full term and is paid back on schedule " * 5     # one long sentence
        self.assertIsNotNone(comparison("What was the 60-month APR %son May 12, 2026, and what is it now?" % padding))


class NotListedWording(unittest.TestCase):
    def test_a_refusal_in_the_models_own_words_is_recognised(self):
        for text in ("The Main Street branch does not have published lobby hours on the website.",
                     "The branch has no posted Saturday hours.",
                     "The website does not list that rate."):
            self.assertTrue(_OPENS_WITH_NOT_LISTED.search(text), text)
        for text in ("The account does not have a monthly fee.",
                     "Yes. Members do not have to pay for the first box of checks."):
            self.assertFalse(_OPENS_WITH_NOT_LISTED.search(text), text)


class RateCrossCheck(unittest.TestCase):
    def test_a_model_answer_with_a_second_rate_beside_the_rows_is_set_aside(self):
        def rules():
            return Draft(text="Certificate Rate Specials, 13 Month Fixed Certificate Special: 4.18% dividend rate, 4.25% APY.",
                         route="rate", complete=True)
        good = Draft(text="The 13-month special pays 4.25% APY.", route="model")
        mixed = Draft(text="The 13-month special pays 4.25% APY, up from 4.10%.", route="model")
        self.assertIs(KnowledgeSystem._cross_check(rules(), good, "q"), good)
        kept = rules()
        self.assertIs(KnowledgeSystem._cross_check(kept, mixed, "q"), kept)

    def test_a_dropped_model_answer_gives_way_to_the_rate_row_not_a_refusal(self):
        partial = Draft(text="Certificate Rate Specials, 13 Month Fixed Certificate Special: 4.25% APY.",
                        route="rate", complete=False)
        dropped = Draft(text="I'm sorry, the website doesn't list that.", confident=False, route="model-ungrounded")
        self.assertIs(KnowledgeSystem._cross_check(partial, dropped, "I heard it pays 4.10%, what does it pay?"), partial)


class ContradictionFigures(unittest.TestCase):
    def test_a_year_rolling_forward_is_not_a_contradiction(self):
        self.assertFalse(_a_real_figure(("2025",), ("2026",)))
        self.assertTrue(_a_real_figure(("2025", "187"), ("2026", "213")))
        self.assertTrue(_a_real_figure(("$1,250",), ("$2,600",)))


# ----------------------------------------------------------------------------- the model call

class FakeClient:
    """Stands in for the API client: records each request and replies with `behaviour(request)`."""
    def __init__(self, behaviour):
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)
        self.behaviour = behaviour

    def _create(self, **request):
        self.requests.append(request)
        return self.behaviour(request)


def reply(text):
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(usage=usage, stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


class ModelCall(unittest.TestCase):
    def setUp(self):
        os.environ.pop("KB_LLM_CACHE", None)
        from kb import reader
        self.reader_module = reader

    def test_the_reply_is_read_even_when_wrapped(self):
        parse = self.reader_module.parse_reply
        self.assertEqual(parse('{"answerable": true, "answer": "Yes.", "cited": ["T4"]}')["cited"], ["T4"])
        self.assertFalse(parse('```json\n{"answerable": false, "answer": "No.", "cited": []}\n```')["answerable"])
        self.assertEqual(parse('{"answerable": "true", "answer": "x", "cited": "T1, P2"}'),
                         {"answerable": True, "answer": "x", "cited": ["T1", "P2"]})
        self.assertEqual(parse('{"answerable": true, "answer": "The fee is $5.", "cited": ["T1"]}\n\n(I relied on {T1}.)')["answer"],
                         "The fee is $5.")
        self.assertEqual(parse('{"answerable": true, "answer": ["Mon-Fri 9-5.", "Sat 9-12."], "cited": []}')["answer"],
                         "Mon-Fri 9-5. Sat 9-12.")
        for broken in ("no json here", "[1, 2]", "{broken"):
            with self.assertRaises(ValueError):
                parse(broken)

    def test_a_rate_typed_in_the_question_is_not_evidence_for_itself(self):
        r = self.reader_module
        user = "PASSAGES\n[P1] The APY is 5.39%.\n\nQUESTION: Is it still 5.34% on a $20,000 loan for a 2019 car?"
        shown = r.material_shown(user)
        self.assertEqual(r.ungrounded_numbers("Yes, it is still 5.34%.", shown), ["5.34%"])
        self.assertEqual(r.ungrounded_numbers("No, it is 5.39% APY for a $20,000 loan on a 2019 car.", shown), [])
        bare = r.material_shown("PASSAGES\n[P1] The APR is 5.39%.\n\nQUESTION: Is it still 5.34 APR?")
        self.assertEqual(r.ungrounded_numbers("Yes, it is still 5.34% APR.", bare), ["5.34%"])

    def test_last_year_gets_a_reminder_that_names_the_year(self):
        view = SimpleNamespace(taken_on=__import__("datetime").date(2026, 9, 28))
        for question in ("How many staff volunteered last year?", "how many staff volunteered last yr"):
            note = self.reader_module.Reader._note(view, question)
            self.assertIn("means 2025", note)
            self.assertIn("set answerable to true", note)
        self.assertEqual(self.reader_module.Reader._note(view, "How many staff volunteer?"), "")

    def test_no_call_is_started_without_time_left(self):
        client = FakeClient(lambda request: reply('{"answerable": true, "answer": "x", "cited": []}'))
        reader = self.reader_module.Reader(client=client)
        with self.assertRaises(TimeoutError):
            reader._call("sheet", "QUESTION: x", deadline=time.monotonic() + 0.5)
        self.assertEqual(client.requests, [])
        reader._call("sheet", "QUESTION: x", deadline=time.monotonic() + 5)
        self.assertLessEqual(client.requests[0]["timeout"], 5)

    def test_after_repeated_failures_the_model_is_left_alone_for_a_while(self):
        def fail(request):
            raise ConnectionError("API unreachable")
        client = FakeClient(fail)
        reader = self.reader_module.Reader(client=client)
        for _ in range(self.reader_module.FAILURES_BEFORE_PAUSE):
            with self.assertRaises(ConnectionError):
                reader._call("sheet", "QUESTION: x")
        with self.assertRaises(RuntimeError):
            reader._call("sheet", "QUESTION: x")             # paused: not sent at all
        self.assertEqual(len(client.requests), self.reader_module.FAILURES_BEFORE_PAUSE)

    def test_a_success_resets_the_count_of_failures(self):
        outcomes = iter([ConnectionError("x"), ConnectionError("x"), None, ConnectionError("x"), ConnectionError("x"), None])

        def sometimes(request):
            outcome = next(outcomes)
            if outcome:
                raise outcome
            return reply('{"answerable": true, "answer": "x", "cited": []}')
        reader = self.reader_module.Reader(client=FakeClient(sometimes))
        for _ in range(6):
            try:
                reader._call("sheet", "QUESTION: x")
            except ConnectionError:
                pass
        self.assertEqual(reader._paused_until, 0.0)


if __name__ == "__main__":
    unittest.main()
