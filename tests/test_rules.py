"""Tests for the rules that decide what is current. They use small made-up pages, so they run
without the website snapshots and without an API key:

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # for test_answers, which holds shared helpers

from kb import facts  # noqa: E402
from kb.lookup import read_question  # noqa: E402
from kb.pages import page_kind, published_date  # noqa: E402
from kb.reader import ungrounded_numbers  # noqa: E402
from kb.rules_answer import NOT_LISTED, RulesAnswerer  # noqa: E402
from kb.snapshot import build_view  # noqa: E402
from kb.system import date_in_question  # noqa: E402
from kb.tables import parse_tables  # noqa: E402
from kb.text import clean_markdown, digits_for_number_words  # noqa: E402
from test_answers import system_without_models  # noqa: E402

SITE = "https://cu.example"

AUTO_NEW = """# Rates

Auto Loans

Auto & RV Loans

| Model Year | Term | APR (As Low As) |
| --- | --- | --- |
| 2023-27 Model Year | 60 Months | 6.18% APR1 |
| 2023-27 Model Year | 84 Months | 6.41% APR1 |

Rates are effective as of 08/19/2026. Every loan is subject to credit approval.

Certificates

Certificate Rate Specials

| Certificate | Dividend Rate | Annual Percentage Yield (APY) | Minimum to Earn |
| --- | --- | --- | --- |
| 13 Month Fixed Jumbo Certificate Special | 4.27% | 4.35%1 | $80,000.00 |
| 13 Month Fixed Certificate Special | 4.18% | 4.25%2 | $750.00 |

Rates are effective as of 09/04/2026 and may change without notice.
"""

AUTO_STALE = """# Car Payment Calculator

Auto Loan Rates

| Vehicle | Length | Annual Percentage Yield |
| --- | --- | --- |
| 2022–26 Model Year | 60 Months | 6.12% APR |
| 2022–26 Model Year | 84 Months | 6.35% APR |

**Rates are effective as of 2/3/2026.**

Payment example: borrowing $24,000 over 60 months at 6.12% APR works out to about $465 a month.

Use the calculator to see what fits your budget.
"""


def raw(path, markdown, title="", fetched="2026-09-28T00:00:00Z"):
    return SimpleNamespace(url=SITE + path, title=title or path, markdown=markdown, fetched_at=fetched)


def press_release(number, published, branches):
    body = ("1 min read\n\n%s\n\n# Release %d\n\nSomething happened at the credit union this week.\n\n"
            "ABOUT US\n\nIt has more than 480 staff and %d branch offices. Learn more on our site.\n"
            % (published, number, branches))
    return raw("/newsroom/release-%d" % number, body, title="Release %d" % number)


def view_of(pages, snapshot_id="05_2026-09-28_latest", taken_on="2026-09-28"):
    return build_view(snapshot_id, taken_on, pages)


class CleaningAndTables(unittest.TestCase):
    def test_footnote_marks_are_not_part_of_the_value(self):
        tables = parse_tables(clean_markdown(AUTO_NEW).split("\n"))
        auto, special = tables
        self.assertEqual(auto.caption, "Auto & RV Loans")
        self.assertEqual(auto.section, "Auto Loans")
        self.assertEqual(auto.rows[0].values, {"APR": "6.18%"})
        self.assertEqual(str(auto.effective), "2026-08-19")
        self.assertEqual(special.rows[0].values, {"dividend rate": "4.27%", "APY": "4.35%"})

    def test_a_cell_that_says_apr_overrides_a_wrong_column_heading(self):
        table = parse_tables(clean_markdown(AUTO_STALE).split("\n"))[0]
        self.assertEqual(list(table.rows[0].values), ["APR"])

    def test_related_article_teasers_are_dropped(self):
        text = clean_markdown("# Story\n\nBody text.\n\n## Handpicked for You\n\nOld teaser with 9.99% in it.")
        self.assertNotIn("9.99", text)

    def test_page_kinds_and_publish_dates(self):
        self.assertEqual(page_kind(SITE + "/rates", "Rates", ""), "rates")
        self.assertEqual(page_kind(SITE + "/calculators/car-budget", "Calculator", ""), "calculator")
        self.assertEqual(page_kind(SITE + "/summer-offer-lp", "Offer", ""), "promo")
        self.assertEqual(page_kind(SITE + "/newsroom/a-release", "News", ""), "news")
        self.assertEqual(str(published_date("2 min read\n\n[Author](x)\n\nJune 20, 2025\n\n# Title")), "2025-06-20")


class RowIdentity(unittest.TestCase):
    def test_qualifiers(self):
        quals, _ = facts.parse_qualifiers("Purchase: 30-Year Fixed Conventional $640,000 or less")
        self.assertEqual((quals.term, quals.amount), (360, (0.0, 640000.0)))
        quals, _ = facts.parse_qualifiers("2023-27 Model Year | 60 Months")
        self.assertEqual((quals.term, quals.years), (60, (2023, 2027)))
        quals, _ = facts.parse_qualifiers("Share Savings | $5,000 – $19,999.99")
        self.assertEqual(quals.amount, (5000.0, 19999.99))

    def test_an_abbreviated_name_is_the_same_product_but_a_sibling_is_not(self):
        auto, auto_long = frozenset({"auto"}), frozenset({"auto", "motorcycle"})
        self.assertTrue(facts.names_match(auto, auto_long))
        self.assertFalse(facts.names_match(frozenset({"platinum", "visa"}), frozenset({"reward", "visa"})))
        self.assertFalse(facts.names_match(frozenset({"share", "saving"}), frozenset({"health", "saving"})))


class WhatIsCurrent(unittest.TestCase):
    def setUp(self):
        self.view = view_of([
            raw("/rates", AUTO_NEW, "Rates"),
            raw("/personal/loans/auto-loans", AUTO_NEW.replace("# Rates", "# Auto Loans"), "Auto Loans"),
            raw("/calculators/car-budget", AUTO_STALE, "Car Payment Calculator"),
        ])

    def test_newer_effective_date_wins_and_the_old_copy_is_stale(self):
        fact = next(f for f in self.view.facts if f.key.term == 60 and f.stale)
        self.assertEqual(fact.current.value, "6.18%")
        self.assertEqual([s.value for s in fact.stale], ["6.12%"])
        self.assertIn("newer effective date", fact.reason(fact.stale[0]))
        self.assertEqual([c.page.url for c in fact.copies], [SITE + "/personal/loans/auto-loans"])

    def test_different_terms_and_variants_are_never_conflicts(self):
        for fact in self.view.facts:
            for loser in fact.stale:
                self.assertEqual(loser.row.key.term, fact.key.term)
        jumbo = [f for f in self.view.facts if "jumbo" in f.key.flags]
        self.assertTrue(jumbo and all(not f.stale for f in jumbo))

    def test_stale_numbers_are_removed_from_what_can_be_searched(self):
        searchable = "\n".join(p.text for p in self.view.index.passages)
        self.assertNotIn("6.12", searchable)
        self.assertNotIn("6.35", searchable)
        self.assertIn("6.18", searchable)
        self.assertIn("Use the calculator", searchable)       # the rest of the page stays

    def test_rate_answers_use_the_exact_row(self):
        answerer = RulesAnswerer(self.view)
        draft = answerer.answer("What's the APR for a five year auto loan on a 2025 car?")
        self.assertIn("6.18%", draft.text)
        self.assertNotIn("6.41%", draft.text)
        self.assertIn(SITE + "/rates", [url for url, _ in draft.sources])
        self.assertIn("6.12%", draft.conflict_note)

    def test_a_rate_that_is_not_published_is_declined(self):
        answerer = RulesAnswerer(self.view)
        for question in ("What is the rate on a 60-month jumbo certificate?", "Which rate goes with a 5/1 adjustable-rate mortgage?"):
            draft = answerer.answer(question)
            self.assertFalse(draft.confident, question)
            self.assertEqual(draft.text, NOT_LISTED)


class RepeatedStatements(unittest.TestCase):
    def test_the_most_recently_published_figure_is_current(self):
        pages = [press_release(i, "March %d, 2024" % (i + 1), 15) for i in range(4)]
        pages += [press_release(10, "June 3, 2024", 16), press_release(11, "April 14, 2026", 14),
                  press_release(12, "May 6, 2026", 14)]
        view = view_of(pages)
        statement = view.statements[0]
        self.assertEqual(statement.current.values[-1], "14")
        self.assertEqual(statement.slot_values(1), ["14", "16", "15"])
        searchable = "\n".join(p.text for p in view.index.passages)
        self.assertNotIn("15 branch", searchable)
        self.assertNotIn("16 branch", searchable)
        self.assertIn("14 branch", searchable)


class QuestionsAndAnswers(unittest.TestCase):
    def test_numbers_in_words_and_measurable_parts(self):
        self.assertEqual(digits_for_number_words("fifteen thousand dollars over five years"), "15000 dollars over 5 years")
        asked = read_question("I'm getting a 2025 car and plan to pay it off over five years")
        self.assertEqual((asked.term, asked.model_year), (60, 2025))
        asked = read_question("What do I earn on $15,000 in a business savings account?")
        self.assertEqual((asked.amounts, sorted(asked.flags)), ([15000.0], ["business"]))

    def test_a_dated_question_names_its_own_date(self):
        self.assertEqual(date_in_question("What was the jumbo rate as of May 2026?"), "2026-05-31")
        self.assertIsNone(date_in_question("Will the downtown office still be open in October 2026?"))

    def test_the_decline_message_contains_no_number(self):
        self.assertFalse(any(ch.isdigit() for ch in NOT_LISTED))
        self.assertIn("please contact", NOT_LISTED.lower())

    def test_answers_must_quote_numbers_that_were_shown(self):
        shown = "Platinum Visa: APR 14.50% - 21.50%. Call 800-555-0100."
        self.assertEqual(ungrounded_numbers("The standard APR is 14.5% to 21.5%.", shown), [])
        self.assertEqual(ungrounded_numbers("The APR is 9.9%.", shown), ["9.9%"])
        self.assertEqual(ungrounded_numbers("There are two cards and 3 options.", shown), [])


IRA_RATES = """# Rates

Retirement Choices

IRA Specials

| IRA Kind | Dividend Rate | Annual Percentage Yield (APY) | Minimum to Earn |
| --- | --- | --- | --- |
| Roth: 13 Month Fixed Special | 4.18% | 4.25%2 | $750.00 |
| Traditional: 13 Month Fixed Special | 4.18% | 4.25%2 | $750.00 |

Rates are effective as of 09/04/2026.
"""

RETIREMENT_PAGE = """# Retirement Savings and IRA Options

Starter Savings Rates

| Account | Dividend Rate | Annual Percentage Yield (APY) |
| --- | --- | --- |
| Starter Savings | 2.86% | 2.91% |

Rates are effective as of 01/09/2026.
"""


class KindsOfOneProduct(unittest.TestCase):
    """Every IRA row is a Roth or a Traditional one. A question about IRAs asks about both, and a
    savings account that merely sits on a page called "... IRA Options" is not an IRA."""

    def setUp(self):
        self.view = view_of([raw("/rates", IRA_RATES, "Rates"),
                             raw("/personal/retirement", RETIREMENT_PAGE, "Retirement Savings and IRA Options")])

    def test_a_question_about_iras_gets_the_ira_rows(self):
        text = RulesAnswerer(self.view).answer("Which rates do your IRAs earn?").text
        self.assertIn("4.25%", text)
        self.assertNotIn("2.91%", text)

    def test_naming_one_kind_still_narrows_it(self):
        text = RulesAnswerer(self.view).answer("What does a Roth IRA pay?").text
        self.assertIn("Roth", text)
        self.assertNotIn("Traditional", text)

    def test_the_savings_account_is_still_found_by_its_own_name(self):
        text = RulesAnswerer(self.view).answer("What does Starter Savings pay?").text
        self.assertIn("2.91%", text)


class AsOf(unittest.TestCase):
    def test_a_past_date_is_answered_from_the_snapshot_of_that_time(self):
        system = system_without_models()
        old = AUTO_NEW.replace("6.18%", "5.96%").replace("08/19/2026", "04/28/2026")
        system.ingest(SimpleNamespace(snapshot_id="01_2026-05-12", taken_on="2026-05-12", pages=[raw("/rates", old, "Rates")]))
        system.ingest(SimpleNamespace(snapshot_id="05_2026-09-28", taken_on="2026-09-28", pages=[raw("/rates", AUTO_NEW, "Rates")]))
        question = "For a 2025 vehicle, what's the auto loan APR over 60 months?"
        self.assertIn("6.18%", system.answer(question).text)
        past = system.answer(question, as_of="2026-05-31")
        self.assertIn("5.96%", past.text)
        self.assertNotIn("6.18%", past.text)
        changed = [c for c in system.conflicts() if "60-month" in c.topic]
        self.assertEqual((changed[0].values, changed[0].resolution), (["5.96%", "6.18%"], "6.18% APR"))


if __name__ == "__main__":
    unittest.main()
