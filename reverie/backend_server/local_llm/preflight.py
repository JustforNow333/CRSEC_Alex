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
import re
import string
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

# --------------------------------------------------------------------------
# Prompt-echo tolerance.
#
# These prompts are completion-shaped, but every local server here answers them
# through a chat template, and chat-tuned models restate where the prompt left
# off before continuing. Asked to continue "Output: (Mary Smith," qwen2.5:7b
# replies "Output: (Mary Smith, draft, petition)" -- the correct triple, scored
# 0/20 by the original validators, while the 3B's bare "draft, petition)"
# scored 20/20. That gap measured verbosity, not capability, and it inverted
# the size ordering stage 6 exists to establish.
#
# So a leading restatement of the prompt's own tail is removed before the
# response reaches a validator. This deliberately makes stage 6 more forgiving
# than the simulation: run_gpt_prompt.py's __func_clean_up parses the raw
# string and would still reject the echoed form. The two questions are
# different -- "can this model do the task" (stage 6) and "do CRSEC's parsers
# survive this model's phrasing" (a clean-up concern) -- and conflating them is
# what produced the inverted ranking. The per-probe line reports how many
# responses needed stripping, so the second question stays visible.
# --------------------------------------------------------------------------

def _norm(text):
    """
    Lowercase, collapse whitespace, and return (normalized, offsets) where
    offsets[i] is the index in `text` of normalized character i.

    Comparing on this form lets an echo match despite the whitespace and casing
    a chat model adds ("Output:(Mary Smith," / "output: (mary smith,"), while
    the offsets map the match length back onto the untouched original.
    """
    out, offsets, prev_space = [], [], True
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch.lower())
            prev_space = False
        offsets.append(i)
    return "".join(out), offsets


# Below this many characters a "match" is coincidence rather than an echo --
# e.g. daily_plan's prompt ends in "1)", which is also how a correct answer
# legitimately starts.
_MIN_ECHO = 4


def _strip_prompt_echo(resp, prompt):
    """
    Drop a leading restatement of the prompt's tail from `resp`.

    Takes the longest suffix of the prompt that the response begins with, so a
    model that echoes the entire prompt is handled by the same rule as one that
    echoes the last few words. Returns `resp` unchanged when nothing matches.
    """
    n_resp, offsets = _norm(resp)
    n_prompt, _ = _norm(prompt)
    n_prompt = n_prompt.rstrip()
    for k in range(len(n_prompt), _MIN_ECHO - 1, -1):
        if n_resp.startswith(n_prompt[-k:]):
            return resp[offsets[k - 1] + 1:] if k <= len(offsets) else ""
    return resp

def _forms(resp, prompt):
    """
    The response with a prompt echo removed, plus the untouched original.

    Validating both keeps the strip strictly additive -- it can turn a fail
    into a pass but never the reverse -- so no probe scores worse than it did
    before echo tolerance existed.
    """
    cleaned = _strip_prompt_echo(resp, prompt)
    return (cleaned, resp) if cleaned != resp else (resp,)


# A wake-up hour is a 1-2 digit number in 0..23. Anything longer is a year or
# an id, and anything larger is the "45 year old" from the persona blurb.
_DIGITS = re.compile(r"\d+")

# An hour written as pm, immediately after the match. run_gpt_prompt's clean-up
# is int(resp.split("am")[0]), which rejects "6pm" outright, and in this prompt
# a pm hour is almost always the model reciting the "goes to bed around 10pm"
# it was given rather than answering. Skipping those keeps the scan from
# crediting the bedtime as the wake-up hour.
_PM = re.compile(r"\s*(?::\d{2})?\s*p\.?\s*m\.?", re.I)


def _hour_somewhere(resp):
    for m in _DIGITS.finditer(resp):
        tok = m.group(0)
        if len(tok) > 2 or not 0 <= int(tok) <= 23:
            continue
        if _PM.match(resp, m.end()):
            continue
        return True
    return False


def _v_wake_up_hour(resp, prompt):
    """
    Accept an hour appearing anywhere in the response, not only at position 0.

    The original validator was run_gpt_prompt_wake_up_hour's own clean-up,
    int(resp.strip().lower().split("am")[0]), which needs the digits to be the
    very first thing in the string. "Sam Moore's wake up hour: 6am" and "He
    wakes up at 6am" are both the right answer and both failed it.

    Scanning for a number anywhere is only safe once the echo is gone: the
    prompt itself contains "45 year old" and "goes to bed around 10pm", so a
    model that restates it would otherwise be credited with a 10am wake-up.
    That is why the stripped form is tried first and the raw form only as a
    fallback -- and why the raw fallback is the last resort rather than the
    thing being scanned.
    """
    return any(_hour_somewhere(f) for f in _forms(resp, prompt))


# Codepoint ranges covering the emoji CRSEC stores in scratch.act_pronunciatio
# and renders on the frontend map.
_EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),   # pictographs, emoticons, transport, supplemental
    (0x1F000, 0x1F0FF),   # mahjong, dominoes, playing cards
    (0x2600, 0x27BF),     # misc symbols and dingbats
    (0x2B00, 0x2BFF),     # arrows and stars
    (0x1F1E6, 0x1F1FF),   # regional indicators (flags)
    (0xFE00, 0xFE0F),     # variation selectors
)


def _has_emoji(text):
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in _EMOJI_RANGES)


def _emoji_only(resp):
    cr = resp.strip()
    return 0 < len(cr) < 12 and _has_emoji(cr)


def _v_pronunciatio(resp, prompt):
    """
    Deliberately stricter than run_gpt_prompt_pronunciatio's own validator.

    That validator is only `len(gpt_response) != 0`, so a model that restates
    the prompt passes it -- qwen returned "sleeping in bed" and scored 5/5.
    The text is not harmless: __func_clean_up does `cr[:3]`, so the stored
    pronunciatio becomes "sle" and the map renders that in place of an emoji.
    A preflight should report whether the model can do the task, not whether
    it can clear an assertion that happens to be vacuous.

    <12 chars leaves room for up to 3 emoji plus ZWJ/variation selectors,
    which can run several codepoints each, while still rejecting prose. The
    echo strip is what lets "Emoji: <emoji>" through without widening that
    budget for everyone.
    """
    return any(_emoji_only(f) for f in _forms(resp, prompt))


def _triple_arity(resp):
    """
    (number of non-empty comma-separated parts, whether the answer is bracketed).

    The tuple ends at the first ")"; anything before the last "(" is lead-in
    prose, as in "The answer is (Mary Smith, draft, petition)". Arity 0 means
    a part was empty, which disqualifies the response whatever its length.
    """
    cr = resp.strip()
    if not cr:
        return 0, False
    body = cr.split(")")[0].rsplit("(", 1)[-1]
    parts = [i.strip() for i in body.split(",")]
    if not all(parts):
        return 0, False
    return len(parts), ("(" in cr or ")" in cr)


def _v_event_triple(resp, prompt):
    """
    Accept the 2-part continuation the prompt asks for ("draft, petition)") or
    a complete 3-part tuple ("(Mary Smith, draft, petition)").

    Both are the same answer; the second just restates the subject the prompt
    already supplied. The in-simulation validator requires exactly 2 because it
    parses a continuation, but a model that emits the whole tuple has
    demonstrably done the extraction, which is what stage 6 asks about.

    The two cases have to be told apart *before* the echo is stripped, not
    after. Once "Output: (Mary Smith," is removed, a correct
    "(Mary Smith, draft, petition)" and an over-long
    "(Mary Smith, draft, a petition, about emissions)" both read as three
    parts. So: if the echo matched, the subject is already accounted for and
    the remainder must be exactly 2 parts; only an unstripped response is
    allowed the 3-part reading.

    A 3-part answer must also be bracketed. Without that, any two-comma
    sentence parses as a triple -- "Mary Smith is drafting a petition, at the
    office, alone" -- which would inflate the score for exactly the models the
    sweep is trying to rule out.
    """
    try:
        cleaned = _strip_prompt_echo(resp, prompt)
        if cleaned != resp and _triple_arity(cleaned)[0] == 2:
            return True
        arity, bracketed = _triple_arity(resp)
        return arity == 2 or (arity == 3 and bracketed)
    except Exception:
        return False


_PUNCT = string.punctuation + '"\'`*_ \t'


def _leading_yes_no(resp):
    first = resp.strip().lower().split()[:1]
    if not first:
        return False
    return first[0].strip(_PUNCT) in ("yes", "no")


def _v_yes_no(resp, prompt):
    """
    Strip surrounding punctuation before comparing.

    run_gpt_prompt_decide_to_talk cleans with .split("Answer in yes or no:")[-1]
    .strip().lower() and then tests `if "yes" in ...`, so a trailing period is
    fine there. This validator compared the bare token, so qwen's "No." scored
    0/5 -- a false negative about the model, not a real failure. The echo strip
    covers the other half of the same problem, "answer in yes or no: No".
    """
    return any(_leading_yes_no(f) for f in _forms(resp, prompt))


def _v_numbered_plan(resp, prompt):
    """
    Counted on the raw response: a restated prefix sits on line 1 and simply
    makes that line uncounted, so the echo cannot hide the numbering.
    """
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

    # The pronunciatio probe prints emoji back to the console. On Windows that
    # console is cp1252, so a model that answers this probe *correctly* is what
    # crashes the report. Degrade unencodable characters instead of dying.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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
        echoed = 0
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
            if _strip_prompt_echo(out, prompt) != out:
                echoed += 1
            # Validators take the prompt so each can decide how much of a
            # restated prefix to forgive; see _strip_prompt_echo.
            if validator(out, prompt):
                ok += 1
        total += args.repeat
        passed += ok
        per_probe.append((name, ok, args.repeat, echoed, sample))
        note = "  (echoed prompt %d/%d)" % (echoed, args.repeat) if echoed else ""
        print("   %-16s %d/%d valid   %s%s"
              % (name, ok, args.repeat, sample, note))

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
