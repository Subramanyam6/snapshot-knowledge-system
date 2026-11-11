# Notes

How the system works, how it decides what is current, and what it needs to run. The results and
how they were measured are in [README.md](README.md).

## How it works

**Clean first, then answer.** The website contradicts itself: a rate changes while an old calculator
page or press release keeps the old number. So nothing answers questions from the raw pages. When
each snapshot is ingested, the system decides which copy of every fact is current and removes the
copies that lost: the stale table, its fine print, and sentences that quote its numbers. Whatever
answers a question afterwards only ever sees the cleaned site, so it has no out-of-date number to
repeat.

**How "current" is decided.** For two copies of the same rate, the first rule that separates them
wins: the newer "effective as of" date printed under the rate table; then the more official kind of
page (rates page, then product page, document, news, blog, and last calculators, listings and
promotions; an expired promotion never counts); then the more recent download. Two rows are the same
rate only when product, variant (business, Roth, jumbo...), term, balance or loan-amount tier and
model-year band all match, so different tiers are never treated as a contradiction. For figures
repeated across dated articles (staff or branch counts), the most recently published one wins. A
question with an `as_of` date is answered from the last snapshot on or before that date; one dated
before the first snapshot gets "I don't know".

**Answering.** Rules answer what they can settle exactly: rate look-ups, yes/no questions about a
stated rate, rate-page links, branch hours and addresses. Everything else goes to a small language
model (one call, a second only if the reply is unusable), which reads a fact sheet of current values
plus passages found by keyword search. A second look, with passages found by meaning, runs at the
same time and is used only when the first finds nothing. The model's reply is checked in code:
every number must appear in what it was shown (a rate the member typed does not count), and where the
rules found the exact rate row for the whole question, the reply must state that row's value and no
other rate. A question plainly
about a change ("how has ... changed since May 12, 2026?") is answered from two snapshots.
`conflicts()` lists every contradiction found while cleaning, with all its values, the winner and the
reason.

## What it needs

- **Python 3.10 or newer** (developed and tested on 3.11), then `pip install -r requirements.txt`.
- **API key: `ANTHROPIC_API_KEY` (a paid API).** Questions that need wording are answered by
  Anthropic's Haiku 5.5 model. Export the key, or put `ANTHROPIC_API_KEY=...` in a `.env` file next
  to `solution.py` (a `.env` one folder up is also read, for the key only). Cost: 3 to 4 cents per 40
  questions. No other key or paid service is used.
- **First run only:** a small open-source embedding model (`BAAI/bge-small-en-v1.5`, 67 MB) is
  downloaded into `.cache/` in this folder, and the pages are prepared for it. That adds about three
  minutes to loading, once (the limit is 30). Later runs load in about 15 seconds. It runs locally
  and costs nothing. The download may print a harmless warning about unauthenticated requests. If
  this folder is read-only, the cache goes to the system's temporary folder.
- If the embedding library will not install on some platform, delete its line from
  `requirements.txt` and install again: the system then searches by keywords only.

## Nothing here is required for it to run

- No key: it answers from rules alone (0.69 on the practice set).
- Embedding model missing or blocked: keyword search only.
- A model call fails, times out or is rate-limited: that question gets the rules answer instead. A
  call is given at most 8 seconds and one answer at most 15 in all, and after three failures in a
  row the model is left alone for a minute, so a struggling API costs some accuracy but not speed.

## Running

From this folder:

```bash
python -m unittest discover tests  # no key or snapshots needed
```

From the organisers' kit folder (the one with `harness/`, `eval/` and `snapshots/`):

```bash
python harness/run_eval.py --submission /path/to/this/folder --eval eval/public_eval.jsonl
```

## Main settings (environment variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `KB_MODE` | `hybrid` | `rules`: never call the model. `model`: send every question to the model. |
| `KB_EMBED` | on whenever the model is used | `off`: no search by meaning. `on`: load it even when the model is not used. |
| `KB_CACHE_DIR` | `.cache/` here | Where the embedding model and saved vectors go. |
| `KB_ANSWER_SECONDS` | `15` | Time all model calls for one answer may take together. |
| `KB_TIMEOUT` | `8` | Time one model call may take. |
| `KB_LLM_CACHE` | unset | A folder that saves model replies, so a repeated development run costs nothing. |
