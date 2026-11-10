"""Tests for questions about a change ("how has it changed since May?"). They use three small
made-up copies of a website, so they need neither the snapshots nor an API key:

    python -m unittest discover tests
"""
import datetime as dt
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # for test_answers, which holds shared helpers

from kb.dates import dates_in, without_dates  # noqa: E402
from kb.history import comparison  # noqa: E402
from kb.rules_answer import NOT_LISTED, Draft  # noqa: E402
from test_answers import system_without_models  # noqa: E402

SITE = "https://cu.example"
TODAY = dt.date(2026, 9, 28)

RATES = """# Rates

Auto Loans

Auto & RV Loans

| Model Year | Term | APR (As Low As) |
| --- | --- | --- |
%s

Rates are effective as of %s. Every loan is subject to credit approval.

Certificates

Certificate Rate Specials

| Certificate | Dividend Rate | Annual Percentage Yield (APY) | Minimum to Earn |
| --- | --- | --- | --- |
%s

Rates are effective as of %s and may change without notice.
"""
GRANT = """# Neighbor Relief Grant

The community fund gives a member who has had a sudden setback a single grant of up to %s.
"""


def rates(auto, auto_effective, specials, special_effective):
    return RATES % ("\n".join("| %s | %s | %s APR1 |" % row for row in auto), auto_effective,
                    "\n".join("| %s | %s | %s2 | $500.00 |" % row for row in specials), special_effective)


# In May a 2022 car is in the newest band. By September the bands have rolled on and it is in the older one.
OLD_AUTO = [("2022-26 Model Year", "60 Months", "5.96%"), ("2022-26 Model Year", "84 Months", "6.18%"),
            ("2019-21 Model Year", "60 Months", "6.23%")]
NEW_AUTO = [("2023-27 Model Year", "60 Months", "6.18%"), ("2023-27 Model Year", "84 Months", "6.41%"),
            ("2019-22 Model Year", "60 Months", "6.46%")]
ELEVEN = "11 Month Fixed Certificate Special"
THIRTEEN = "13 Month Fixed Certificate Special"
COPIES = [   # (snapshot id, day, rates page, grant limit)
    ("01_2026-05-12", "2026-05-12", rates(OLD_AUTO, "04/28/2026", [(ELEVEN, "3.42%", "3.45%")], "04/28/2026"), "$1,250"),
    ("03_2026-06-09", "2026-06-09", rates(OLD_AUTO, "04/28/2026", [(ELEVEN, "3.42%", "3.45%"), (THIRTEEN, "3.52%", "3.55%")], "06/03/2026"), "$1,250"),
    ("05_2026-09-28", "2026-09-28", rates(NEW_AUTO, "08/19/2026", [(ELEVEN, "3.62%", "3.65%"), (THIRTEEN, "4.18%", "4.25%")], "09/04/2026"), "$2,600"),
]


def page(path, markdown, title):
    return SimpleNamespace(url=SITE + path, title=title, markdown=markdown, fetched_at="2026-09-28T00:00:00Z")


def system_with(reader=None):
    system = system_without_models()
    system.reader = reader
    for snapshot_id, day, rates_page, limit in COPIES:
        system.ingest(SimpleNamespace(snapshot_id=snapshot_id, taken_on=day, pages=[
            page("/rates", rates_page, "Rates"), page("/giving/neighbor-relief-grant", GRANT % limit, "Neighbor Relief Grant")]))
    return system


class ReadingTheQuestion(unittest.TestCase):
    def test_questions_that_are_without_doubt_about_a_change(self):
        for question in ("How has the 60-month auto loan APR changed since May 12, 2026?",
                         "Has the APY on the 11-month certificate special gone up since June 9, 2026?",
                         "Um, so, have your certificate rates changed?",
                         "Did the 15-year fixed mortgage rate for loans of $640,000 or less go down between May 19, 2026 and June 9, 2026?",
                         "Has the dividend rate for balances of $5,000 to $19,999.99 changed since June 9, 2026?",
                         "How has the dividend on the traditional IRA 13-month special (regular rather than jumbo) changed since August 27, 2026?",
                         "What did the yield on the 13-month CD special move from and to between May 19, 2026 and September 28, 2026?",
                         "What changed in the auto loan rates between May 12, 2026 and September 28, 2026?",
                         "What was the APY on the 11-month certificate special on June 9, 2026 and on September 28, 2026?",
                         "quick question: 20 yr fixed refinance rate, June 9 2026 vs August 27 2026?",
                         "What was the 60-month auto loan APR on May 12, 2026, and what is it now?",
                         "what do you charge on a 48 month truck loan for a 2016 model today, and what did it cost back on august 27 2026?",
                         "Compare the 13-month certificate special's APY on August 27, 2026 with today's."):
            self.assertIsNotNone(comparison(question), question)

    def test_questions_that_only_look_like_it(self):
        for question in (
                # "have" as possession, two products compared on one date
                "Does a 15-year mortgage have a lower rate than a 30-year mortgage?",
                "Do you have a lower rate for a 60-month auto loan?",
                "Does Acorn Savings have a minimum balance, and can the rate change?",
                "Is the 12-month certificate APY higher than the current 6-month APY?",
                "Is the auto loan rate higher for vehicles older than 2020?",
                "Is the Platinum Visa APR lower than the Rewards Visa APR?",
                # "may" the verb, a date from the member's own story, a plan, the future
                "What is the 12-month certificate APY today? Its rate can vary and may change after the account is opened.",
                "I opened my certificate on May 12, 2026. What is the APY today?",
                "I was told to apply before June 30, 2026, what is the rate?",
                "The fee schedule was updated on June 9, 2026; what is the wire fee now?",
                "What's the 60-month auto loan APR if I buy between June and August?",
                "Will the rate go up after the introductory period?",
                "Has my auto loan rate changed?",
                # a time that cannot be pinned to a copy of the website
                "Has the 60-month auto loan APR changed since May?",
                "Has the 60-month auto loan APR changed since May 12?",
                "Has the 60-month auto loan APR changed in the last month?",
                "Have certificate rates gone up recently?",
                # what it was before: answered as always, from one snapshot
                "What was the previous APY on the 13-month special?",
                "What did the 12-month certificate use to pay?",
                "What was the APY on the 13-month special before September 2026?",
                "What was the APY on May 19, 2026?",
                # nothing measurable, or something the member does
                "How do I change my address?",
                "Did the credit union cut the ribbon on a new branch?",
                "Do I have to move the money into an IRA?"):
            self.assertIsNone(comparison(question), question)

    def test_dates_must_carry_their_year(self):
        iso = lambda text: [str(day) for day in dates_in(text)]  # noqa: E731
        self.assertEqual(iso("on May 12th, 2026, on 9 June 2026, on 8/27/2026 and on 2026-09-28"),
                         ["2026-05-12", "2026-06-09", "2026-08-27", "2026-09-28"])
        self.assertEqual(iso("between May 2026 and Sept. 2026"), ["2026-05-31", "2026-09-30"])   # a month: its last day
        self.assertEqual(iso("since May, since May 12, in June"), [])
        self.assertEqual(iso("Which APR 2025 trucks get on a 72-month loan?"), [])               # APR is a rate, not April
        self.assertEqual(iso("What APR may 2024 models get?"), [])                                # "may" is a verb here
        self.assertEqual(iso("on February 30, 2026 or 13/45/2026"), [])
        self.assertEqual(without_dates("the apr on may 12, 2026 for a 2021 vehicle").split(), "the apr on for a 2021 vehicle".split())


class RateRowsOverTime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = system_with(reader=None)                                              # rules only: no model at all

    def ask(self, question, as_of=None):
        answer = self.system.answer(question, as_of)
        key = (" ".join(question.split()).lower(), as_of)
        return answer.text, self.system.routes[key]

    def test_a_rate_that_changed_is_given_at_both_ends_with_their_dates(self):
        text, route = self.ask("How has the 60-month auto loan APR for a 2025 vehicle changed since May 12, 2026?")
        self.assertEqual(route, "history")
        # Each value under the name its row had on that date: the band was "2022-26" in May.
        self.assertIn("2022-26 Model Year, 60 Months: 5.96% APR on May 12, 2026", text)
        self.assertIn("2023-27 Model Year, 60 Months: 6.18% APR on September 28, 2026", text)

    def test_a_rate_that_did_not_change(self):
        text, route = self.ask("Did the APY on the 11-month certificate special change between May 12, 2026 and June 9, 2026?")
        self.assertEqual(route, "history")
        self.assertIn("3.45% APY on May 12, 2026 and on June 9, 2026, the same on both dates", text)
        self.assertNotIn("3.65%", text)                                                    # today's value was not asked for

    def test_a_named_model_year_follows_the_band_that_held_it(self):
        # A 2022 car: "2022-26" in May (5.96%), "2019-22" in September (6.46%). The older band of
        # May (6.23%) is the same place in the table but never applied to this car.
        text, _ = self.ask("For a 2022 car on a 60-month loan, what was the APR on May 12, 2026, and what is it today?")
        self.assertIn("2022-26 Model Year, 60 Months: 5.96% APR on May 12, 2026", text)
        self.assertIn("2019-22 Model Year, 60 Months: 6.46% APR on September 28, 2026", text)
        self.assertNotIn("6.23%", text)

    def test_now_and_a_past_date_in_either_order(self):
        for question in ("Tell me what the 11-month certificate special's APY was on June 9, 2026, and what it is now.",
                         "the 11 month certificate special - what's the apy now, and what was it back on june 9 2026?"):
            text, route = self.ask(question)
            self.assertEqual(route, "history", question)
            self.assertIn("3.45% APY on June 9, 2026", text)
            self.assertIn("3.65% APY on September 28, 2026", text)

    def test_two_dates_in_two_copies_are_a_comparison(self):
        text, route = self.ask("What was the APY on the 11-month certificate special on June 9, 2026 and on September 28, 2026?")
        self.assertEqual(route, "history")
        self.assertIn("3.45% APY on June 9, 2026", text)
        self.assertIn("3.65% APY on September 28, 2026", text)

    def test_two_dates_inside_one_copy_are_answered_from_that_copy(self):
        text, route = self.ask("What was the APY on the 11-month certificate special on May 13, 2026 and on May 20, 2026?")
        self.assertNotEqual(route, "history")
        self.assertIn("3.45% APY", text)
        self.assertNotIn("3.65%", text)

    def test_only_the_dates_asked_about_are_reported(self):
        # The 13-month special paid 3.55% in June and 4.25% in September; nothing in between is added.
        text, _ = self.ask("How has the APY on the 13-month certificate special changed since June 9, 2026?")
        self.assertIn("3.55% APY on June 9, 2026", text)
        self.assertIn("4.25% APY on September 28, 2026", text)
        text, _ = self.ask("What was the APY on the 11-month certificate special on May 12, 2026, on June 9, 2026 and on September 28, 2026?")
        self.assertIn("3.45% APY on May 12, 2026 and on June 9, 2026; ", text)
        self.assertIn("3.65% APY on September 28, 2026", text)

    def test_nothing_is_claimed_about_a_date_on_which_the_row_cannot_be_traced(self):
        # The 13-month special is not in the May tables. Not offered yet, or listed under another
        # name? The tables alone cannot tell, so the rules say nothing about May.
        text, route = self.ask("How has the APY on the 13-month certificate special changed since May 12, 2026?")
        self.assertNotEqual(route, "history")
        self.assertNotIn("not listed on", text)

    def test_a_named_month(self):
        # "in June 2026": within that month, from the copy in force on June 1 to the last one of June.
        text, route = self.ask("Did the APY on the 11-month certificate special change in June 2026?")
        self.assertEqual(route, "history")
        self.assertIn("3.45% APY on May 12, 2026 and on June 9, 2026, the same on both dates", text)
        self.assertNotIn("3.65%", text)
        # Only one copy in May here: nothing to compare, so the May value, as for any dated question.
        text, route = self.ask("Did the APY on the 11-month certificate special change in May 2026?")
        self.assertNotEqual(route, "history")
        self.assertIn("3.45% APY", text)
        self.assertNotIn("3.65%", text)
        # "since May 2026" and "since the beginning of May 2026": from the month's start.
        text, _ = self.ask("How has the APR on a 60-month auto loan for a 2025 vehicle changed since the beginning of May 2026?")
        self.assertIn("5.96% APR on May 12, 2026", text)
        self.assertIn("6.18% APR on September 28, 2026", text)
        text, _ = self.ask("Since May 12th, 2026, how has the APR on a 60-month auto loan for a 2025 vehicle changed?")
        self.assertIn("5.96% APR on May 12, 2026", text)

    def test_a_date_passed_by_the_caller_ends_the_period(self):
        text, _ = self.ask("How has the 60-month auto loan APR for a 2025 vehicle changed since May 12, 2026?", as_of="2026-06-30")
        self.assertIn("5.96% APR on May 12, 2026 and on June 9, 2026, the same on both dates", text)
        self.assertNotIn("6.18%", text)

    def test_what_it_was_before_is_not_a_comparison(self):
        # "Previous", "used to", "before": too many of these are about something else ("my previous
        # address", "used to finance"). They are answered as always, from today's copy.
        text, route = self.ask("What was the previous APY on the 11-month certificate special?")
        self.assertEqual(route, "rate")
        self.assertIn("3.65% APY", text)
        self.assertNotIn("3.45%", text)

    def test_a_look_alike_gets_todays_value_only(self):
        for question in ("Do you have a lower APR for a 60-month auto loan for a 2025 vehicle?",
                         "For a 2025 vehicle, what's the auto loan APR over 60 months?",
                         "Is the 60-month auto loan APR for a 2025 vehicle higher than the 84-month APR?"):
            text, route = self.ask(question)
            self.assertNotEqual(route, "history", question)
            self.assertNotIn("5.96%", text, question)

    def test_the_sites_own_word_for_a_certificate(self):
        # The footnotes say "share certificate"; the tables say "certificate". Not Share Savings.
        text, route = self.ask("On the 11-month share certificate special, what APY did it pay on June 9, 2026, and what does it pay at present?")
        self.assertEqual(route, "history")
        self.assertIn("3.45% APY on June 9, 2026", text)
        self.assertIn("3.65% APY on September 28, 2026", text)

    def test_a_typing_slip_in_the_product_name_does_not_hide_it(self):
        text, route = self.ask("13 month certficate special - what was the apy on june 9 2026 and on sept 28 2026?")
        self.assertEqual(route, "history")
        self.assertIn("3.55% APY on June 9, 2026", text)
        self.assertIn("4.25% APY on September 28, 2026", text)
        text, route = self.ask("For the 13-month certficate special, which APY applies?")
        self.assertEqual(route, "rate")
        self.assertIn("4.25% APY", text)

    def test_a_real_word_is_not_respelt(self):
        answerer = self.system._answerer(self.system.views[-1])
        for word in ("intern", "equality", "bimonthly", "unconventional"):
            self.assertEqual(answerer.spelled("Do you offer %s loans?" % word), "Do you offer %s loans?" % word)
        self.assertEqual(answerer.spelled("the certficate special"), "the certificate special")

    def test_a_rate_that_was_never_listed_is_still_declined(self):
        text, _ = self.ask("How has the rate on the 9-month certificate changed since May 12, 2026?")
        self.assertEqual(text, NOT_LISTED)


class StandInReader:
    """Records what it is shown. Answers only when it is given the earlier copy (or always)."""

    def __init__(self, only_with_earlier=True):
        self.calls = []
        self.only_with_earlier = only_with_earlier

    def warm(self, view):
        pass

    def can_look_again(self, view):
        return False

    def answer(self, view, question, by_meaning=False, day=None, earlier=None, deadline=None):
        self.calls.append((view.taken_on, earlier.view.taken_on if earlier else None))
        if earlier is None and self.only_with_earlier:
            return Draft(text=NOT_LISTED, confident=False, route="model-not-listed")
        if earlier is None:
            return Draft(text="Grants are up to $2,600.", sources=[(SITE + "/giving/neighbor-relief-grant", "")], confident=True, route="model")
        if self.only_with_earlier:
            return Draft(text="It was up to $1,250 on May 12, 2026 and is up to $2,600 now.",
                         sources=[(SITE + "/giving/neighbor-relief-grant", "")], confident=True, route="model")
        return Draft(text=NOT_LISTED, confident=False, route="model-not-listed")


class OtherFactsOverTime(unittest.TestCase):
    QUESTION = "Has the neighbor relief grant limit changed since May 12, 2026?"

    def test_the_model_is_shown_the_earlier_copy_next_to_todays(self):
        reader = StandInReader()
        system = system_with(reader)
        answer = system.answer(self.QUESTION)
        self.assertEqual(reader.calls, [(TODAY, dt.date(2026, 5, 12))])                     # one call: today's copy and May's
        self.assertIn("$1,250", answer.text)
        self.assertIn("$2,600", answer.text)

    def test_the_usual_answer_stands_when_the_comparison_finds_nothing(self):
        reader = StandInReader(only_with_earlier=False)
        system = system_with(reader)
        answer = system.answer(self.QUESTION)
        self.assertEqual([call[1] for call in reader.calls], [dt.date(2026, 5, 12), None])   # compared first, then as always
        self.assertEqual(answer.text, "Grants are up to $2,600.")


if __name__ == "__main__":
    unittest.main()
