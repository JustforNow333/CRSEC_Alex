#!/usr/bin/env python3
"""
bench.py v3  --  measures the speed changes WITHOUT needing a baseline run.

Run from  reverie/backend_server/ :

    python bench.py --origin base_the_ville_n10 --steps 200 --label run1
    python bench.py --origin base_the_ville_n10 --steps 200 --label run1 --keep

v3 changes vs v2:
  * drives ReverieServer.run_standalone() directly (same entry point as your
    run_headless.py) instead of faking answers to input() prompts. The old
    approach was fragile: reverie.py's __main__ also calls Create(), which
    prompts "Regenerate norms? (y or n)", so the canned answer list had to
    match the prompt order exactly. Calling run_standalone() skips every
    prompt. Note Create() with "n" is a no-op, so this is equivalent.
  * arguments are now flags, matching run_headless.py's interface.

WHY THERE'S NO BASELINE ARM
---------------------------
The original differs from the modified code in exactly two knowable ways:
  1. temp_sleep(0.1) once per request (it sits BEFORE the retry loop)
  2. no embedding cache, so every repeated text costs a real API call
So for any run we can reconstruct what the original would have cost on that
SAME workload. Only one workload is ever involved, which removes the
run-to-run workload lottery that makes raw A/B wall-clock comparisons useless.

Works unchanged on either branch:
  * on Alex_branch it adds the overhead back  -> reconstructs the ORIGINAL
  * on main        it takes the overhead away -> reconstructs the MODIFIED
Both should report the same improvement %.

Results append to bench_results.jsonl for aggregating across runs.
"""
import os, sys, time, json, shutil, argparse, atexit, subprocess, datetime

TEMP_SLEEP = 0.1     # gpt_structure.temp_sleep default (the original's per-request sleep)
SERVER_SLEEP = 0.1   # reverie's per-step sleep; identical on both branches


def _git_info():
    """
    Record exactly which code produced this result. Without this, a results
    file is unattributable a month later -- a label alone does not identify
    the code. `dirty` matters: with uncommitted edits the SHA does not fully
    describe what ran.
    """
    def run(*args):
        try:
            return subprocess.check_output(
                ["git"] + list(args), stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    return {
        "git_commit": run("rev-parse", "--short", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        # -uno: ignore untracked files. Helper scripts sitting in the directory
        # (aggregate.py, bench.py itself) don't change what the sim runs; only
        # modifications to tracked files mean the SHA is not the whole story.
        "git_dirty": bool(run("status", "--porcelain", "-uno")),
    }

# ---------------------------------------------------------------------------
# Patch BEFORE reverie is imported. Every consumer does
# `from persona.prompt_template.gpt_structure import *`, which binds names at
# import time, so they pick up the patched get_embedding only if we patch
# first. gpt_structure imports nothing that imports it back, so this is safe.
# ---------------------------------------------------------------------------
import openai
from persona.prompt_template import gpt_structure as G

C = {
    "gen_calls": 0, "gen_seconds": 0.0,
    "embed_calls": 0, "embed_seconds": 0.0,
    "embed_requests": 0,
    "cache_hits": 0,
    "dup_requests": 0,
    "sleep_seconds": 0.0,
}
_seen_texts = set()
HAS_CACHE = hasattr(G, "_embedding_cache")


def _time_api(fn, calls_key, secs_key):
    def inner(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            C[secs_key] += time.perf_counter() - t
            C[calls_key] += 1
    return inner


try:
    openai.ChatCompletion.create = _time_api(openai.ChatCompletion.create, "gen_calls", "gen_seconds")
    openai.Completion.create     = _time_api(openai.Completion.create, "gen_calls", "gen_seconds")
    openai.Embedding.create      = _time_api(openai.Embedding.create, "embed_calls", "embed_seconds")
except Exception as e:
    print(f"[bench] WARNING: could not wrap openai ({e}); counts will be wrong", file=sys.stderr)


def _norm_key(text):
    """Mirror gpt_structure.get_embedding's normalization exactly."""
    key = text.replace("\n", " ")
    if not key:
        key = "this is blank"
    return key


_real_get_embedding = G.get_embedding


def _counting_get_embedding(text, *a, **k):
    key = _norm_key(text)
    C["embed_requests"] += 1
    if key in _seen_texts:
        C["dup_requests"] += 1
    else:
        _seen_texts.add(key)
    if HAS_CACHE and key in G._embedding_cache:
        C["cache_hits"] += 1
    return _real_get_embedding(text, *a, **k)


G.get_embedding = _counting_get_embedding

_real_sleep = time.sleep


def _timed_sleep(s=0):
    C["sleep_seconds"] += (s or 0)
    _real_sleep(s)


time.sleep = _timed_sleep


def _fmt(sec):
    return f"{sec:9.2f}s ({sec/60:5.1f} min)"


def _report(label, steps, wall, origin):
    gen_calls = C["gen_calls"]
    embed_calls = C["embed_calls"]
    embed_req = C["embed_requests"]
    hits = C["cache_hits"] if HAS_CACHE else C["dup_requests"]

    e_lat = C["embed_seconds"] / embed_calls if embed_calls else 0.0
    g_lat = C["gen_seconds"] / gen_calls if gen_calls else 0.0

    total_requests = gen_calls + embed_req
    sleep_overhead = TEMP_SLEEP * total_requests
    embed_overhead = hits * e_lat

    if HAS_CACHE:
        mine = wall
        baseline = wall + sleep_overhead + embed_overhead
    else:
        baseline = wall
        mine = wall - sleep_overhead - embed_overhead

    saved = baseline - mine
    pct = (saved / baseline * 100) if baseline else 0.0

    branch = "MODIFIED (cache present)" if HAS_CACHE else "ORIGINAL (no cache)"
    print("\n" + "=" * 64)
    g = _git_info()
    dirty = "  *** UNCOMMITTED CHANGES ***" if g["git_dirty"] else ""
    print(f"  label / steps       : {label}  /  {steps}")
    print(f"  origin sim          : {origin}")
    print(f"  git                 : {g['git_branch']} @ {g['git_commit']}{dirty}")
    print(f"  code under test     : {branch}")
    print(f"  wall clock          : {_fmt(wall)}")
    print("  " + "-" * 60)
    print(f"  generation calls    : {gen_calls:6d}  {C['gen_seconds']:9.2f}s  avg {g_lat:.3f}s")
    print(f"  embedding API calls : {embed_calls:6d}  {C['embed_seconds']:9.2f}s  avg {e_lat:.3f}s")
    print(f"  embedding requests  : {embed_req:6d}")
    pct_hits = (hits / embed_req * 100) if embed_req else 0.0
    print(f"  cache hits          : {hits:6d}  ({pct_hits:.1f}% of requests)")
    print(f"  time in sleep       : {C['sleep_seconds']:9.2f}s")

    if HAS_CACHE:
        if C["cache_hits"] != C["dup_requests"]:
            print(f"  ! cache_hits({C['cache_hits']}) != repeated texts({C['dup_requests']}) "
                  "-- some embeddings errored and were not cached")
        expected_sleep = SERVER_SLEEP * steps
        if C["sleep_seconds"] > expected_sleep + 1:
            print(f"  ! sleep {C['sleep_seconds']:.2f}s exceeds expected {expected_sleep:.2f}s "
                  "-- backoff fired; timings include retry waits")
    else:
        expected_sleep = TEMP_SLEEP * total_requests + SERVER_SLEEP * steps
        drift = abs(C["sleep_seconds"] - expected_sleep)
        status = "OK" if drift < 1 else f"OFF BY {drift:.2f}s"
        print(f"  ! sleep model check : measured {C['sleep_seconds']:.2f}s vs "
              f"predicted {expected_sleep:.2f}s  [{status}]")

    print("  " + "-" * 60)
    print("  RECONSTRUCTED COMPARISON (same workload, both versions)")
    print(f"    generation time   : {C['gen_seconds']:9.2f}s  identical on both, cancels out")
    print(f"    temp_sleep        : {sleep_overhead:9.2f}s  0.1s x {total_requests} requests")
    print(f"    dup. embeddings   : {embed_overhead:9.2f}s  {hits} hits x {e_lat:.3f}s")
    print("  " + "-" * 60)
    print(f"    original   : {_fmt(baseline)}")
    print(f"    modified   : {_fmt(mine)}")
    print(f"    saved      : {_fmt(saved)}")
    print(f"    IMPROVEMENT: {pct:.1f}%")
    print("=" * 64)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": label, "steps": steps, "origin": origin, "has_cache": HAS_CACHE,
        **_git_info(),
        "wall": round(wall, 2), "gen_calls": gen_calls,
        "gen_seconds": round(C["gen_seconds"], 2), "gen_latency": round(g_lat, 4),
        "embed_calls": embed_calls, "embed_seconds": round(C["embed_seconds"], 2),
        "embed_latency": round(e_lat, 4), "embed_requests": embed_req,
        "cache_hits": hits, "sleep": round(C["sleep_seconds"], 2),
        "baseline": round(baseline, 2), "modified": round(mine, 2),
        "improvement_pct": round(pct, 2),
    }
    with open("bench_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print("  appended to bench_results.jsonl")


def main():
    p = argparse.ArgumentParser(description="Benchmark a CRSEC run; reconstructs the baseline.")
    p.add_argument("--origin", default="base_the_ville_n10", help="fork base sim_code")
    p.add_argument("--steps", type=int, default=200, help="number of steps")
    p.add_argument("--label", default="run", help="label for this run")
    p.add_argument("--keep", action="store_true", help="keep the sim folder afterwards")
    args = p.parse_args()

    from utils import fs_storage
    target = f"__bench_{args.label}"
    target_path = os.path.join(fs_storage, target)
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)

    # Imported AFTER patching, so the persona chain picks up the patched
    # get_embedding via its `import *`.
    from reverie import ReverieServer

    wall0 = time.perf_counter()

    @atexit.register
    def _on_exit():
        _report(args.label, args.steps, time.perf_counter() - wall0, args.origin)
        if not args.keep and os.path.isdir(target_path):
            shutil.rmtree(target_path)

    rs = ReverieServer(args.origin, target)
    rs.run_standalone(args.steps)


if __name__ == "__main__":
    main()
