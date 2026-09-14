"""
fail_safe_monitor.py -- measure how much of a run is actually the model.

The problem this exists to solve:

    def safe_generate_response(prompt, gpt_parameter, repeat=5,
                               fail_safe_response="error", ...):
      for i in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        if func_validate(curr_gpt_response, prompt=prompt):
          return func_clean_up(curr_gpt_response, prompt=prompt)
      return fail_safe_response          # <-- silent

When a model cannot produce output in the shape a prompt demands, CRSEC retries
up to five times and then returns a hardcoded default: wake up at 8, "is idle",
the 😋 emoji. Nothing is logged. The simulation runs to completion and writes a
full set of output files.

With GPT-4o-mini this happens rarely enough to ignore. With a small open-weight
model it can be the majority of calls, and the result is a run that looks
finished but is largely the fail-safe constants talking to each other. Any
tipping-point measurement taken from such a run is measuring CRSEC's defaults,
not the model's social behaviour.

This module wraps every safe_* generator in gpt_structure and counts how often
each returns its fail-safe. Attach it once at startup; read the report at the
end. It changes no behaviour -- it only observes.

Usage, from reverie/backend_server:

    import local_llm.fail_safe_monitor as fsm
    fsm.attach()
    ...run the simulation...
    fsm.report()                       # prints
    fsm.dump("fail_safe_report.json")  # machine-readable

Author: Alexander Maiello (acm357@cornell.edu)
"""

import functools
import inspect
import json
import threading

WRAPPED = [
    "safe_generate_response",
    "ChatGPT_safe_generate_response",
    "ChatGPT_safe_generate_response_OLD",
    "GPT4_safe_generate_response",
    "GPT4_safe_generate_response_OLD",
    "GPT4_safe_generate_response_OLD_t1",
    "ChatGPT_safe_generate_response_OLD_t0",
]

_counts = {}
_lock = threading.Lock()
_attached = False


def _record(name, hit_fail_safe, prompt):
    with _lock:
        rec = _counts.setdefault(name, {"calls": 0, "fail_safe": 0, "examples": []})
        rec["calls"] += 1
        if hit_fail_safe:
            rec["fail_safe"] += 1
            if len(rec["examples"]) < 3:
                # First line of the prompt is usually the template name, which
                # is enough to identify which prompt is failing.
                rec["examples"].append(prompt.strip().split("\n")[0][:110])


def _fail_safe_arg(func, args, kwargs):
    """Recover the fail_safe_response argument however it was passed."""
    if "fail_safe_response" in kwargs:
        return kwargs["fail_safe_response"], True
    try:
        sig = inspect.signature(func)
        names = list(sig.parameters)
        idx = names.index("fail_safe_response")
        if idx < len(args):
            return args[idx], True
        default = sig.parameters["fail_safe_response"].default
        if default is not inspect.Parameter.empty:
            return default, True
    except (ValueError, TypeError):
        pass
    return None, False


def _prompt_arg(args, kwargs):
    if "prompt" in kwargs:
        return str(kwargs["prompt"])
    return str(args[0]) if args else ""


def attach(module=None):
    """Wrap the safe_* generators. Idempotent."""
    global _attached
    if _attached:
        return
    if module is None:
        from persona.prompt_template import gpt_structure as module

    for name in WRAPPED:
        original = getattr(module, name, None)
        if original is None or getattr(original, "_fsm_wrapped", False):
            continue

        def make(name, original):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                fail_safe, known = _fail_safe_arg(original, args, kwargs)
                result = original(*args, **kwargs)
                hit = known and result == fail_safe
                _record(name, hit, _prompt_arg(args, kwargs))
                return result
            wrapper._fsm_wrapped = True
            return wrapper

        setattr(module, name, make(name, original))

    _attached = True


def stats():
    with _lock:
        total = sum(r["calls"] for r in _counts.values())
        failed = sum(r["fail_safe"] for r in _counts.values())
        return {
            "total_calls": total,
            "fail_safe_returns": failed,
            "fail_safe_rate": (failed / total) if total else 0.0,
            "by_function": json.loads(json.dumps(_counts)),
        }


def report():
    s = stats()
    print("=" * 68)
    print("fail-safe report")
    print("=" * 68)
    if not s["total_calls"]:
        print("  no safe_* calls recorded (was attach() called before the run?)")
        return s
    for name, rec in sorted(s["by_function"].items()):
        rate = 100.0 * rec["fail_safe"] / rec["calls"] if rec["calls"] else 0
        print("  %-38s %5d calls  %5d fail-safe  %5.1f%%"
              % (name, rec["calls"], rec["fail_safe"], rate))
        for ex in rec["examples"]:
            print("        e.g. %s" % ex)
    print("-" * 68)
    print("  overall: %d/%d = %.1f%% of generations were fail-safe defaults"
          % (s["fail_safe_returns"], s["total_calls"], 100 * s["fail_safe_rate"]))
    rate = s["fail_safe_rate"]
    if rate < 0.05:
        print("  Comparable to a hosted-model run. Results are the model's.")
    elif rate < 0.20:
        print("  Elevated. Report this figure alongside any result from this run.")
    else:
        print("  High. A large share of this run is CRSEC's hardcoded defaults,")
        print("  not model behaviour. Do not draw conclusions from it.")
    print("=" * 68)
    return s


def dump(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats(), f, indent=2)
    return path
