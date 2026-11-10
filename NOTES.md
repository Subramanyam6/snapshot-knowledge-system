# Notes

How to run it and what it needs. The design, the results and how they were measured are in
[README.md](README.md).

## What it needs

- **Python 3.10 or newer** (developed and tested on 3.11), then `pip install -r requirements.txt`.
- **API key: `ANTHROPIC_API_KEY`.** Questions that need wording are answered by Anthropic's Haiku 5.5
  model. Export the key, or put `ANTHROPIC_API_KEY=...` in a `.env` file next to `solution.py` (a
  `.env` one folder up is also read, for the key only). Cost: 3 to 4 cents per 40 questions.
- **First run only:** a small open-source embedding model (`BAAI/bge-small-en-v1.5`, 67 MB) is
  downloaded into `.cache/` in this folder, and the pages are prepared for it. That adds about three
  minutes to loading, once (the limit is 30). Later runs load in about 15 seconds. It runs locally
  and costs nothing. If this folder is read-only, the cache goes to the system's temporary folder.
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

## Settings (environment variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `KB_MODE` | `hybrid` | `rules`: never call the model. `model`: send every question to the model. |
| `KB_EMBED` | on whenever the model is used | `off`: no search by meaning. `on`: load it even when the model is not used. |
| `KB_CACHE_DIR` | `.cache/` here | Where the embedding model and saved vectors go. |
| `KB_ANSWER_SECONDS` | `15` | Time all model calls for one answer may take together. |
| `KB_TIMEOUT` | `8` | Time one model call may take. |
| `KB_LLM_CACHE` | unset | A folder that saves model replies, so a repeated development run costs nothing. |
