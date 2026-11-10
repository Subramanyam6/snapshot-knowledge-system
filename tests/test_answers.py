"""Tests for how answers are shaped once the rules have found the facts: direct yes/no answers,
one link, standard and jumbo together, branch hours, running counts, dates in questions.
Small made-up pages; no snapshots, no API key.

    python -m unittest discover tests
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kb.hours import find_hours, locator_url  # noqa: E402
from kb.rules_answer import RulesAnswerer  # noqa: E402
from kb.snapshot import build_view  # noqa: E402
from kb.system import KnowledgeSystem, date_in_question  # noqa: E402
from kb.text import is_yes_no, strip_request  # noqa: E402

SITE = "https://cu.example"

RATES = """# Rates

Savings Accounts

Share Savings

| Balance | Dividend Rate | Annual Percentage Yield (APY) | Minimum to Earn |
| --- | --- | --- | --- |
| $20,000+ | 0.27% | 0.27% | $20,000.00 |
| $50 – $19,999.99 | 0.05% | 0.05% | $50.00 |

Rates are effective as of 06/11/2026.

Certificates

Certificate Rate Specials

| Certificate | Dividend Rate | Annual Percentage Yield (APY) | Minimum to Earn |
| --- | --- | --- | --- |
| 13 Month Fixed Jumbo Certificate Special | 4.27% | 4.35%1 | $80,000.00 |
| 13 Month Fixed Certificate Special | 4.18% | 4.25%2 | $750.00 |

Rates are effective as of 09/04/2026.

Home Loans

Home Loans - Purchase

| Loan Program | Rate | APR |
| --- | --- | --- |
| Purchase: 30-Year Fixed Conventional $275,000 or less | 6.62% | 6.71% |
| Purchase: 30-Year Fixed Conventional $120,000 or less | 6.48% | 6.66% |

Rates are effective as of 09/22/2026.

Home Improvement Loans

Home Improvement Loans

| Amount | Term | APR |
| --- | --- | --- |
| $5,000.00 – $15,000.00 | 96 Months | 8.85% |

Rates are effective as of 06/05/2026.
"""

SAVINGS_PAGE = """# Savings

See our [certificate rates](https://cu.example/personal/savings#rates) and
compare [savings rates](https://cu.example/personal/savings#rates).

Certificates

Certificate Rate Specials

| Certificate | Dividend Rate | Annual Percentage Yield (APY) | Minimum to Earn |
| --- | --- | --- | --- |
| 13 Month Fixed Jumbo Certificate Special | 4.27% | 4.35%1 | $80,000.00 |
| 13 Month Fixed Certificate Special | 4.18% | 4.25%2 | $750.00 |

Rates are effective as of 09/04/2026.
"""


def raw(path, markdown, title=""):
    return SimpleNamespace(url=SITE + path, title=title or path, markdown=markdown, fetched_at="2026-09-28T00:00:00Z")


def article(path, published, body, title="News"):
    return raw(path, "2 min read\n\n%s\n\n# %s\n\n%s\n" % (published, title, body), title=title)


def branch_article(path, published, town, hours):
    lede = "%s, Ohio—Our newest branch, at 1 Main St. in %s, opened its doors to members this week." % (town.upper(), town)
    return article(path, published, lede + "\n\nThe phone number is (555) 010-0000. Branch hours are:\n\n" + hours, title="Branch news")


LAB = "**Lobby**\n\nMonday-Friday: 8:45 a.m.-5:15 p.m.\n\nSaturday: 8:45 a.m.-12:15 p.m.\n\n**Drive-up**\n\nMonday-Friday: 7 a.m.-7 p.m.\n\nSunday: ATM only\n"
CITY = "**Lobby** Monday-Friday: 10 a.m.-4 p.m.\n\nSaturday-Sunday: Closed\n\n**Drive-up** Monday-Friday: 10 a.m.-4 p.m.\n"


def answerer(*pages):
    return RulesAnswerer(build_view("05_test", "2026-09-28", list(pages)))


def system_without_models(answer_types=None):
    """The whole system with neither the model reader nor the embedding model built, whatever key
    or settings the environment holds (building them means a real API client and loading, the
    first time downloading, the embedding model). Its mode is the default, "hybrid"; a test that
    wants a reader installs its own stand-in afterwards."""
    saved = {name: os.environ.get(name) for name in ("KB_MODE", "KB_EMBED")}
    os.environ.update(KB_MODE="rules", KB_EMBED="off")
    try:
        any_type = lambda **kw: SimpleNamespace(**kw)  # noqa: E731
        system = KnowledgeSystem(answer_types=answer_types or (any_type, any_type, any_type))
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    system.mode = "hybrid"
    return system


class QuestionShape(unittest.TestCase):
    def test_a_polite_request_is_not_a_yes_no_question(self):
        self.assertEqual(strip_request("Hello, could you tell me the APR on a 15-year home equity loan?").strip(),
                         "the APR on a 15-year home equity loan?")
        self.assertFalse(is_yes_no("Can you tell me the APR on a 15-year home equity loan?"))
        self.assertFalse(is_yes_no("I'd like to know the 60-month auto rate"))
        self.assertTrue(is_yes_no("Is 0.42% paid on any savings tier?"))
        self.assertTrue(is_yes_no("Could you tell me if the lobby is open on Sunday?"))

    def test_a_date_in_a_question_that_looks_back(self):
        self.assertEqual(date_in_question("What was the rate as of May 2026?"), "2026-05-31")
        self.assertEqual(date_in_question("What did a 24-month certificate pay out on June 9, 2026?"), "2026-06-09")
        self.assertEqual(date_in_question("What was the APY on 15 May 2026?"), "2026-05-15")
        self.assertEqual(date_in_question("What was the APY as of 2026-08-27?"), "2026-08-27")
        self.assertEqual(date_in_question("What was the APY as of 8/27/2026?"), "2026-08-27")

    def test_a_date_that_is_part_of_the_subject_is_not_a_look_back(self):
        self.assertIsNone(date_in_question("When does the branch that closes on March 31, 2026 reopen?"))
        self.assertIsNone(date_in_question("Does the downtown office stay open today (October 2026)?"))


class RateAnswers(unittest.TestCase):
    def setUp(self):
        self.rules = answerer(raw("/rates", RATES, "Rates"), raw("/personal/savings", SAVINGS_PAGE, "Savings"))

    def test_an_open_question_gets_the_standard_row_then_its_jumbo_twin(self):
        text = self.rules.answer("For the 13-month certificate special, which APY applies?").text
        self.assertIn("4.25% APY ($750.00 minimum)", text)
        self.assertIn("4.35% APY ($80,000.00 minimum)", text)
        self.assertLess(text.index("4.25%"), text.index("4.35%"))

    def test_asking_for_the_jumbo_gives_only_the_jumbo(self):
        text = self.rules.answer("For the 13-month jumbo certificate special, which APY applies?").text
        self.assertIn("4.35%", text)
        self.assertNotIn("4.25%", text)

    def test_a_yes_no_about_a_percentage_is_settled_from_the_rows(self):
        no = self.rules.answer("Is there a Share Savings tier paying 0.42% APY?")
        self.assertEqual(no.route, "rate-yes-no")
        self.assertTrue(no.text.startswith("No."))
        self.assertIn("0.27%", no.text)
        self.assertNotIn("0.42", no.text)              # the figure asked about may be an old one
        yes = self.rules.answer("Does Share Savings pay 0.27% APY on any tier?")
        self.assertTrue(yes.text.startswith("Yes."))
        self.assertIn("$20,000+", yes.text)

    def test_mortgage_means_home_loans_and_tiers_run_from_the_smallest(self):
        text = self.rules.answer("What mortgage rates do you have today?").text
        self.assertIn("Home Loans - Purchase", text)
        self.assertNotIn("Home Improvement", text)
        self.assertLess(text.index("$120,000 or less"), text.index("$275,000 or less"))

    def test_a_link_question_gets_one_link_with_the_sites_own_anchor(self):
        draft = self.rules.answer("Which web page has the certificate rates on it?")
        self.assertEqual(draft.route, "rate-link")
        self.assertIn("https://cu.example/personal/savings#rates", draft.text)
        self.assertEqual(draft.text.count("https://"), 1)

    def test_a_table_sharing_only_a_loose_word_does_not_block_the_right_one(self):
        promo = raw("/checking-bonus-lp", "# Offer\n\nOnline and mobile access included\n\n| Tier | APY |\n| --- | --- |\n| $0+ | 2.00% |\n", "Offer")
        rules = answerer(raw("/rates", RATES, "Rates"), raw("/personal/savings", SAVINGS_PAGE, "Savings"), promo)
        self.assertEqual(rules.answer("Where online are the certificate rates posted?").route, "rate-link")


class ReadingTheQuestion(unittest.TestCase):
    def setUp(self):
        self.rules = answerer(raw("/rates", RATES, "Rates"), raw("/personal/savings", SAVINGS_PAGE, "Savings"))

    def test_non_jumbo_rules_the_jumbo_row_out(self):
        text = self.rules.answer("Non-jumbo 13-month certificate special: what APY does it earn?").text
        self.assertIn("4.25%", text)
        self.assertNotIn("4.35%", text)

    def test_a_question_about_a_date_or_a_procedure_is_not_a_rate_lookup(self):
        from kb.lookup import read_question
        self.assertFalse(read_question("When are Share Savings dividends paid?").wants_rate)
        self.assertFalse(read_question("How do I earn the higher certificate rate?").wants_rate)
        self.assertFalse(read_question("In which month does the Holiday Club pay out?").wants_rate)
        self.assertTrue(read_question("How much is a 13 month CD special paying these days?").wants_rate)

    def test_a_polite_verb_is_not_a_product_and_a_number_in_a_name_is_not_a_quantity(self):
        from kb.lookup import read_question
        self.assertNotIn("share", read_question("Could you share the certificate rates with me, please?").own_words)
        self.assertEqual(read_question("What does 1st Step Checking pay?").other_numbers, [])

    def test_chat_shorthand_is_written_out(self):
        from kb.text import expand_shorthand
        self.assertEqual(expand_shorthand("wat r ur hrs on sat"), "what are your hours on sat")
        self.assertEqual(expand_shorthand("update my acct info pls"), "update my account information please")
        self.assertEqual(expand_shorthand("Is the U.S. office open?"), "Is the U.S. office open?")

    def test_an_official_table_wins_over_a_campaign_page_copy(self):
        promo = raw("/savings-offer-lp", "# Offer\n\nShare Savings\n\nEarn more with us\n\n| Balance | Dividend Rate | Annual Percentage Yield (APY) |\n"
                                         "| --- | --- | --- |\n| $20,000+ | 0.42% | 0.42% |\n| $50 – $19,999.99 | 0.05% | 0.05% |\n", "Offer")
        view = build_view("05_test", "2026-09-28", [raw("/rates", RATES, "Rates"), promo])
        text = RulesAnswer_text(view, "Among the Share Savings tiers, which APY is highest?")
        self.assertIn("0.27%", text)
        self.assertNotIn("0.42%", text)
        stale = [loser.value for fact in view.facts for loser in fact.stale]
        self.assertIn("0.42%", stale)                        # and the old copy is known to be out of date


def RulesAnswer_text(view, question):
    return RulesAnswerer(view).answer(question).text


class ModelAnswerChecks(unittest.TestCase):
    def test_an_answer_about_something_else_needs_two_signs(self):
        from kb.reader import on_topic
        far = SimpleNamespace(question=lambda text: FakeVector(0.2))
        near = SimpleNamespace(question=lambda text: FakeVector(0.9))
        question, off = "Fairview branch address?", "1st Step Checking pays 1.75% APY on balances from $2,500."
        self.assertFalse(on_topic(question, off, [], embedder=far))
        self.assertTrue(on_topic(question, off, [], embedder=near))                  # close in meaning: kept
        self.assertTrue(on_topic(question, off, [], embedder=None))                  # no second opinion: kept
        self.assertTrue(on_topic(question, "The branch is at 1 Main St.", [], embedder=far))   # shares a word: kept


class LaterSafeguards(unittest.TestCase):
    def test_a_variant_the_table_lacks_means_another_product(self):
        rules = answerer(raw("/rates", RATES, "Rates"), raw("/personal/savings", SAVINGS_PAGE, "Savings"))
        self.assertFalse(rules.answer("What rate applies to refinancing a student home loan?").confident)
        self.assertFalse(rules.answer("What does business equipment financing cost in interest?").confident)

    def test_a_denial_followed_by_other_things_is_a_plain_not_listed(self):
        from kb.rules_answer import NOT_LISTED, Draft
        nothing = Draft(text=NOT_LISTED, confident=False, route="weak-hit")
        said = Draft(text="The website does not list lobby hours for the Elm Street branch. Branches with hours include Riverton.", route="model")
        checked = KnowledgeSystem._cross_check(nothing, said, "Lobby schedule for the Elm Street branch, please?")
        self.assertFalse(checked.confident)
        self.assertEqual(checked.text, NOT_LISTED)
        # "last year" is the one case where "not reported, but here are the other years" is the answer
        kept = KnowledgeSystem._cross_check(nothing, Draft(text="The website does not report 2025. In 2024 it was 187.", route="model"),
                                            "Last year, how many staff members took part in volunteering?")
        self.assertTrue(kept.confident)

    def test_a_misspelt_word_is_searched_as_the_word_the_site_uses(self):
        pages = [article("/blog/%d" % n, "May %d, 2024" % (n + 1), "A mortgage is a loan for buying a home. " * 3, title="Home buying %d" % n) for n in range(4)]
        index = build_view("05_test", "2026-09-28", pages).index
        self.assertEqual(index._respelled("morgage"), "mortgage")
        self.assertIsNone(index._respelled("zzzzzzq"))
        self.assertTrue(index.search("what is a morgage", k=2))


class FakeVector:
    """Stands in for an embedding: the dot product of two of them is their product."""
    def __init__(self, value):
        self.value = value

    def __matmul__(self, other):
        return self.value * other.value


class BranchHours(unittest.TestCase):
    def setUp(self):
        self.pages = [
            branch_article("/newsroom/lab-a", "March 21, 2024", "Riverton", LAB),
            branch_article("/newsroom/lab-b", "February 8, 2022", "Lakeside", LAB),
            branch_article("/newsroom/city", "October 11, 2022", "Fernhill", CITY),
            raw("/", "# Home\n\n[Find a location](https://cu.example/contact#locator)\n\n[Branches](https://cu.example/contact#locator)\n\n"
                     "[ATMs](https://cu.example/contact#locator-map)\n", "Home"),
            raw("/contact", "# Contact\n\nFind your closest branch.\n", "Contact"),
        ]
        self.view = build_view("05_test", "2026-09-28", self.pages)
        self.rules = RulesAnswerer(self.view)

    def test_blocks_and_the_locator_page_are_found(self):
        self.assertEqual(len(self.view.hours), 3)
        self.assertEqual(self.view.hours[0].place, "Riverton")
        self.assertEqual(self.view.hours[0].sections["Lobby"][0], "Monday-Friday: 8:45 a.m.-5:15 p.m.")
        self.assertEqual(self.view.locator, "https://cu.example/contact")

    def test_a_general_question_gets_every_distinct_schedule(self):
        draft = self.rules.answer("What is the usual lobby schedule?")
        self.assertEqual(draft.route, "hours")
        self.assertIn("8:45 a.m.-5:15 p.m.", draft.text)
        self.assertIn("10 a.m.-4 p.m.", draft.text)
        self.assertIn("(Riverton, Lakeside)", draft.text)
        self.assertNotIn("7 a.m.-7 p.m.", draft.text)        # that is a drive-up time
        drive = self.rules.answer("How late does the drive-up stay open?")
        self.assertIn("7 a.m.-7 p.m.", drive.text)

    def test_a_named_branch_gets_its_own_schedule_and_an_unknown_one_gets_none(self):
        named = self.rules.answer("What are the lobby hours in Fernhill?")
        self.assertEqual(named.route, "hours-branch")
        self.assertIn("10 a.m.-4 p.m.", named.text)
        self.assertNotIn("8:45", named.text)
        unknown = self.rules.answer("What are the lobby hours for the Elm Street branch?")
        self.assertFalse(unknown.confident)                 # never another branch's hours

    def test_the_locator_page_is_cited_with_branch_answers(self):
        from solution import Answer, Conflict, Source
        ks = system_without_models(answer_types=(Answer, Source, Conflict))
        ks.ingest(SimpleNamespace(snapshot_id="05_test", taken_on="2026-09-28", pages=self.pages))
        answer = ks.answer("What is the usual lobby schedule?")
        self.assertIn("https://cu.example/contact", [s.url for s in answer.sources])


class RunningCounts(unittest.TestCase):
    def test_a_later_article_that_counts_one_more_supersedes_the_earlier_one(self):
        pages = [
            article("/newsroom/award-2", "July 11, 2023", "A magazine again named us a top credit union. It was the fourth time the magazine recognized us."),
            article("/newsroom/award-3", "June 25, 2024", "A magazine again named us a top credit union. This was the fifth time the magazine recognized us."),
            article("/newsroom/other", "May 2, 2022", "For the first time, members can open accounts from their phones in minutes."),
        ]
        view = build_view("05_test", "2026-09-28", pages)
        running = [s for s in view.statements if s.running]
        self.assertEqual(len(running), 1)
        self.assertEqual(running[0].current.values, ("fifth",))
        text = " ".join(p.text for p in view.index.passages)
        self.assertIn("fifth time", text)
        self.assertNotIn("fourth time", text)
        self.assertIn("first time", text)                    # an unrelated "first time" is left alone


if __name__ == "__main__":
    unittest.main()
