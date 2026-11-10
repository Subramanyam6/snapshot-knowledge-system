"""Development report: ask the practice questions and show each answer with its score, which path
answered it, how long it took, and what the model calls cost.

    KB_LLM_CACHE=.cache/llm python tools/dev_report.py [--kit=PATH] [--eval=FILE] [--ids=a,b | --n=5] [--only=fail]
        [--paraphrases=FILE] [--show-prompt]

--kit is the organisers' kit folder (harness/, eval/, snapshots/); by default the folder above this one.
--paraphrases is a JSON map from question id to a reworded question; --show-prompt prints the first prompt.

KB_LLM_CACHE saves model replies on disk, so repeating a run costs nothing. Prices are the list
prices per million tokens of the model named in kb/reader.py (prompts up to 100K tokens)."""
import json
import sys
import time
from pathlib import Path

opt = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
here = Path(__file__).resolve().parents[1]              # this folder: solution.py and kb/
kit = Path(opt.get("--kit", here.parent))               # the organisers' kit: harness/, eval/, snapshots/
sys.path.insert(0, str(kit / "harness"))
import run_eval as H  # noqa: E402
from scoring import score_item  # noqa: E402

PRICE = {"input": 0.10, "output": 0.50, "cache_write": 0.20, "cache_read": 0.01}   # $ per 1M tokens (one-hour cache writes)
items = [json.loads(l) for l in Path(opt.get("--eval", kit / "eval" / "public_eval.jsonl")).read_text().splitlines() if l.strip()]
if "--paraphrases" in opt:
    rew = json.loads(Path(opt["--paraphrases"]).read_text())
    for it in items:
        it["question"] = rew.get(it["id"], it["question"])
if "--ids" in opt:
    items = [it for it in items if it["id"] in opt["--ids"].split(",")]
elif "--n" in opt:
    items = items[: int(opt["--n"])]

snaps = H.load_snapshots(kit / "snapshots")
ks = H.load_submission(here)
for s in snaps:
    ks.ingest(s)
print("reader:", "ON" if ks.reader else "OFF (no key or KB_MODE=rules)", "| mode:", ks.mode)
if ks.reader is None:
    sys.exit(1)
from kb import reader as R  # noqa: E402
print("model:", R.MODEL, "| effort:", R.EFFORT, "| thinking:", R.THINKING, "| passages:", R.PASSAGES)

if "--show-prompt" in sys.argv:
    view = ks.views[-1]
    sheet_text, user, _ = ks.reader.build_prompt(view, items[0]["question"])
    print(f"\n--- instructions: {len(R.INSTRUCTIONS)} chars; sheet: {len(sheet_text)} chars; user message: {len(user)} chars")
    print(sheet_text[:1800], "\n...\n", user[:1500])

tot = wsum = 0.0
lat = []
for it in items:
    t = time.time()
    ans = H.to_answer(ks.answer(it["question"], it.get("as_of")))
    dt_s = time.time() - t
    lat.append(dt_s)
    route = ks.routes.get((" ".join(it["question"].split()).lower(), str(it["as_of"])[:10] if it.get("as_of") else None), "")
    r = score_item(it, ans)
    w = it.get("weight", 1.0)
    tot += r["score"] * w
    wsum += w
    if "--only" in opt and r["score"] >= 0.999:
        continue
    print(f"\n[{r['score']:.2f}] {it['id']} {it['category']} route={route} {dt_s:.2f}s src_hit={r['source_hit']}")
    print(f"   Q: {it['question']}" + (f"   (as_of {it['as_of']})" if it.get("as_of") else ""))
    print(f"   A: {ans.text[:500]}")
    if r["score"] < 0.999:
        print(f"   expected: {it.get('expected_answer','')[:220]}")
        print(f"   missed={r['missed']} traps={r['trap_hits']} must_not={it.get('must_not_include')}")
        if ans.conflict_note:
            print(f"   note: {ans.conflict_note[:200]}")
u = ks.reader.usage
cost = (u.input_tokens * PRICE["input"] + u.output_tokens * PRICE["output"] + u.cache_write_tokens * PRICE["cache_write"]
        + u.cache_read_tokens * PRICE["cache_read"]) / 1e6
xs = sorted(u.seconds)
print(f"\nweighted accuracy {tot/wsum:.4f} over {len(items)} items; fallbacks to rules: {ks.fallbacks}")
print(f"API calls {u.calls} (+{u.cached_calls} served from the local cache); tokens: {u.input_tokens} fresh input, "
      f"{u.cache_write_tokens} cache write, {u.cache_read_tokens} cache read, {u.output_tokens} output; cost ${cost:.4f}")
if xs:
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    print(f"model call seconds: median {p(0.5):.2f}, p95 {p(0.95):.2f}, max {xs[-1]:.2f}")
