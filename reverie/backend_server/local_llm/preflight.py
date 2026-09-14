"""
preflight.py -- prove the local-model pipeline works before spending hours on a run.

Six stages, each independently reported, in the order that failures cascade:

  1. Server reachable
  2. Configured models present on the server
  3. Chat path              (openai.ChatCompletion.create)
  4. Legacy completion path (openai.Completion.create -> chat, the CRSEC hot path)
  5. Embedding path         (openai.Embedding.create, dimension + padding + determinism)
  6. Prompt validity        (real CRSEC prompts through the real validators)

Stages 1-5 test the plumbing: can CRSEC talk to this server at all. Stage 6 is
the one that matters scientifically. A small model will pass 1-5 and can still
fail 6 badly, and if it does, the simulation will not crash -- it will quietly
substitute fail-safe defaults and produce a run that looks finished but is
mostly hardcoded values rather than model behaviour. Stage 6 is what catches
that, so read its number before trusting any result from a small model.

    python preflight.py
    python preflight.py --repeat 5      # validity is stochastic; average it

Exit code 0 if stages 1-5 pass. Stage 6 never fails the run; it reports a rate.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_backend  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []


def report(stage, status, detail=""):
    results.append((stage, status, detail))
    print("[%-4s] %-24s %s" % (status, stage, detail))


# --------------------------------------------------------------------------
# Stage 6 fixtures: real CRSEC prompts with their real validators.
# The clean_up/validate logic is copied verbatim from the nested closures in
# reverie/backend_server/persona/prompt_template/run_gpt_prompt.py, so a pass
# here means the same thing a pass means inside the simulation.
# --------------------------------------------------------------------------

def _v_wake_up_hour(resp):
    try:
        int(resp.strip().lower().split("am")[0])
        return True
    except Exception:
        return False


def _v_pronunciatio(resp):
    return len(resp.strip()) > 0


def _v_event_triple(resp):
    try:
        cr = resp.strip()
        parts = [i.strip() for i in cr.split(")")[0].split(",")]
        return len(parts) == 2
    except Exception:
        return False


def _v_yes_no(resp):
    return resp.strip().lower().split()[:1] in (["yes"], ["no"]) if resp.strip() else False


def _v_numbered_plan(resp):
    lines = [l for l in resp.strip().split("\n") if l.strip()]
    numbered = [l for l in lines if l.strip()[0].isdigit()]
    return len(numbered) >= 3


PROBES = [
    ("wake_up_hour",
     "Sam Moore is a 45 year old grocery store owner. He is organized and "
     "wakes early.\n\nIn general, Sam Moore goes to bed around 10pm and is an "
     "early riser.\nSam Moore's wake up hour:",
     {"max_tokens": 5, "stop": ["\n"]}, _v_wake_up_hour),

    ("event_triple",
     "Task: Turn the input into (subject, predicate, object).\n\n"
     "Input: Sam Moore is stocking shelves at the store\n"
     "Output: (Sam Moore, stock, shelves)\n---\n"
     "Input: Mary Smith is drafting a petition about emissions\n"
     "Output: (Mary Smith,",
     {"max_tokens": 30, "stop": ["\n"]}, _v_event_triple),

    ("pronunciatio",
     "Convert an action description to the most fitting emojis. "
     "Write only the emojis, at most 3.\n\n"
     "Action description: sleeping in bed\nEmoji:",
     {"max_tokens": 15, "stop": ["\n"]}, _v_pronunciatio),

    ("decide_to_talk",
     "Context: Mary Smith and Bob Johnson are both in Hobbs Cafe. "
     "They have argued about climate policy before.\n"
     "Question: Would Mary Smith initiate a conversation with Bob Johnson?\n"
     "Reasoning aside, answer in yes or no:",
     {"max_tokens": 10, "stop": ["\n"]}, _v_yes_no),

    ("daily_plan",
     "Today is Monday. Sam Moore is a grocery store owner who wakes at 6am.\n"
     "In broad strokes, list Sam Moore's plan for today as a numbered list of "
     "at least 5 items.\n1)",
     {"max_tokens": 300, "stop": None}, _v_numbered_plan),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the validity probes N times and average")
    ap.add_argument("--skip-validity", action="store_true")
    args = ap.parse_args()

    os.environ["CRSEC_LLM_BACKEND"] = "local"
    cfg = local_backend.config()

    print("=" * 68)
    print("CRSEC local-model preflight")
    print("  base_url    : %s" % cfg["base_url"])
    print("  chat_model  : %s" % cfg["chat_model"])
    print("  embed_model : %s" % cfg["embed_model"])
    print("=" * 68)

    # -- Stage 1 ------------------------------------------------------------
    models_url = cfg["base_url"].rstrip("/") + "/models"
    served = []
    try:
        with urllib.request.urlopen(models_url, timeout=15) as r:
            payload = json.loads(r.read().decode("utf-8"))
        served = [m.get("id", "") for m in payload.get("data", [])]
        report("1 server reachable", PASS, "%d model(s) served" % len(served))
    except Exception as exc:
        report("1 server reachable", FAIL, "%s -> %s" % (models_url, exc))
        print("\nThe server is not answering. Start it first "
              "(`ollama serve`, or vLLM/llama-server), then rerun.")
        return 1

    # -- Stage 2 ------------------------------------------------------------
    def present(name):
        return any(s == name or s.startswith(name.split(":")[0]) for s in served)

    missing = [m for m in (cfg["chat_model"], cfg["embed_model"]) if not present(m)]
    if missing:
        report("2 models present", WARN,
               "not listed: %s | served: %s" % (", ".join(missing), ", ".join(served[:6])))
    else:
        report("2 models present", PASS, "%s, %s" % (cfg["chat_model"], cfg["embed_model"]))

    local_backend.activate(force=True, verbose=False)
    import openai

    # -- Stage 3 ------------------------------------------------------------
    try:
        t0 = time.time()
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # deliberately the hardcoded name; shim must remap
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=10)
        text = resp["choices"][0]["message"]["content"]
        report("3 chat path", PASS,
               "%.1fs, %r" % (time.time() - t0, text.strip()[:40]))
    except Exception as exc:
        report("3 chat path", FAIL, repr(exc))
        return 1

    # -- Stage 4 ------------------------------------------------------------
    # This is the path most CRSEC prompts take, and the one most likely to
    # break: it asks for "text-davinci-003" and expects .choices[0].text back.
    try:
        t0 = time.time()
        resp = openai.Completion.create(
            model="text-davinci-003",
            prompt="Reply with the single word: ready",
            temperature=0.8, max_tokens=10, top_p=1,
            frequency_penalty=0, presence_penalty=0,
            stream=False, stop=["\n"])
        text = resp.choices[0].text
        report("4 legacy completion", PASS,
               "%.1fs, %r" % (time.time() - t0, text.strip()[:40]))
    except Exception as exc:
        report("4 legacy completion", FAIL, repr(exc))
        return 1

    # -- Stage 5 ------------------------------------------------------------
    try:
        a = openai.Embedding.create(input=["climate policy"],
                                    model="text-embedding-ada-002")
        b = openai.Embedding.create(input=["climate policy"],
                                    model="text-embedding-ada-002")
        va = a["data"][0]["embedding"]
        vb = b["data"][0]["embedding"]
        native = local_backend._state["embed_dim"]
        stable = (va == vb)
        detail = "native=%s padded=%s deterministic=%s" % (native, len(va), stable)
        if len(va) != cfg["pad_embeddings_to"] and cfg["pad_embeddings_to"]:
            report("5 embedding path", FAIL, detail + " (padding did not apply)")
            return 1
        report("5 embedding path", PASS if stable else WARN, detail)
        if not stable:
            print("      Non-deterministic embeddings mean retrieval will vary "
                  "run to run even with everything else fixed.")
    except Exception as exc:
        report("5 embedding path", FAIL, repr(exc))
        return 1

    # -- Stage 6 ------------------------------------------------------------
    if args.skip_validity:
        print("\nSkipped stage 6.")
        return 0

    print("\n" + "-" * 68)
    print("6  prompt validity (%d attempt(s) per probe)" % args.repeat)
    print("-" * 68)

    total = 0
    passed = 0
    per_probe = []
    for name, prompt, params, validator in PROBES:
        ok = 0
        sample = ""
        for _ in range(args.repeat):
            try:
                r = openai.Completion.create(
                    model="text-davinci-003", prompt=prompt,
                    temperature=0.8, top_p=1,
                    frequency_penalty=0, presence_penalty=0, stream=False,
                    max_tokens=params["max_tokens"], stop=params["stop"])
                out = r.choices[0].text
            except Exception as exc:
                out = ""
                sample = sample or ("ERROR " + repr(exc)[:60])
            if not sample:
                sample = out.strip().replace("\n", " / ")[:52]
            if validator(out):
                ok += 1
        total += args.repeat
        passed += ok
        per_probe.append((name, ok, args.repeat, sample))
        print("   %-16s %d/%d valid   %s" % (name, ok, args.repeat, sample))

    rate = 100.0 * passed / total if total else 0.0
    print("-" * 68)
    print("   overall prompt validity: %d/%d = %.0f%%" % (passed, total, rate))

    if rate >= 90:
        verdict = ("Usable. Comparable to what the OpenAI models achieve on "
                   "these prompts.")
    elif rate >= 60:
        verdict = ("Marginal. Expect a meaningful share of fail-safe "
                   "substitutions. Fine as a wiring proof, not as a source of "
                   "results. Move up a size class before running experiments.")
    else:
        verdict = ("Not usable for results. Most calls will exhaust the 5 "
                   "retries in safe_generate_response and return the hardcoded "
                   "fail-safe, so the simulation would complete while being "
                   "driven by defaults rather than by the model.")
    print("   verdict: %s" % verdict)
    print()
    print(local_backend.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
