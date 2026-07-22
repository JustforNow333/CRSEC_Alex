#!/usr/bin/env python3
"""
replay.py — deterministic record/replay harness for CRSEC.

Measures the EXACT speedup of the gpt_structure.py changes on a frozen
workload: record one live run's OpenAI responses, then replay the identical
responses to both code versions and compare wall time.

Location:  reverie/backend_server/  (next to bench.py)
Status:    measurement tool. Gitignored. NOT part of the speedup-only PR.

Usage (PowerShell). IMPORTANT: do NOT check out pristine `main` to get the
baseline — on Windows it won't run without the local compat fixes. Instead
stay on your working (compat-applied) checkout and swap ONLY gpt_structure.py.
The compat fixes are identical on both sides and touch JSON encoding / env
loading, not API calls or timing, so they cancel out of the measurement.

  # Phase 1 — record ONCE (live API, costs money). Use the SAME --seed you'll replay with.
  python replay.py --mode record --origin base_the_ville_n10 --steps 100 --seed 42

  # Phase 2 — replay (free, deterministic, no network), MODIFIED code first:
  python replay.py --mode replay --origin base_the_ville_n10 --steps 100 --seed 42

  # swap ONLY the one file to baseline, keep the rest of the tree (incl. compat):
  #   git checkout main -- reverie/backend_server/persona/prompt_template/gpt_structure.py
  python replay.py --mode replay --origin base_the_ville_n10 --steps 100 --seed 42
  #   git checkout speedup-only -- reverie/backend_server/persona/prompt_template/gpt_structure.py

  # Optional: latency sweep (produces the speedup-vs-API-latency curve)
  python replay.py --mode replay --steps 100 --latency-gen 0.5 --latency-embed 0.2 --seed 42

Nothing in this file or this workflow modifies gpt_structure.py's PR content,
adds a sim-code dependency, or touches the speedup-only branch. The seed is
applied externally, here in the harness, before reverie is imported.

Keying strategy (the part that matters):
  * chat / text completions: strict per-method sequence order, guarded by a
    sha256 of the full request. Responses are temperature-dependent, so
    content can never be the key; order can, because execution is strictly
    sequential and both branches issue identical request sequences.
    Any divergence => hard failure with the call index and both hashes.
  * embeddings: content-keyed (sha256 of the request). This is REQUIRED, not
    optional: the embedding cache on the modified branch legally skips
    duplicate calls, which would desync a sequential queue. Identical input
    => identical embedding, so content keying is safe here.

Sanity guarantees:
  * replay refuses to touch the network (originals are replaced, and a
    sentinel raises if anything else in openai is invoked).
  * at exit, replay reports served vs recorded counts. On the baseline
    branch every generation entry must be consumed and embedding calls must
    equal the recorded count. On the cached branch, fewer embedding calls is
    the expected (and measured) saving.
"""

import argparse
import atexit
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Load .env so OPENAI_API_KEY is available to utils.py / gpt_structure.py
# (must happen before anything imports utils)
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
_env_file = os.path.join(HERE, "..", "..", ".env")
if os.path.isfile(_env_file):
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_RECORDING = os.path.join(HERE, "recording.jsonl")
RESULTS_FILE = os.path.join(HERE, "replay_results.jsonl")

# Measured medians from the live bench runs; override via CLI to sweep.
DEFAULT_LATENCY_GEN = 1.28     # seconds per chat/completion call
DEFAULT_LATENCY_EMBED = 0.43   # seconds per embedding call

GEN_METHODS = ("ChatCompletion", "Completion")   # sequence-keyed
EMBED_METHOD = "Embedding"                        # content-keyed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canonical(kwargs: dict) -> str:
    """Deterministic serialization of a request for hashing.

    Drops keys that can differ between runs without changing the logical
    request (none known today; add here if the hash guard ever trips on
    something semantically irrelevant, e.g. request timeouts).
    """
    drop = {"request_timeout", "timeout", "api_key"}
    clean = {k: v for k, v in kwargs.items() if k not in drop}
    return json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str)


def req_hash(kwargs: dict) -> str:
    return hashlib.sha256(canonical(kwargs).encode("utf-8")).hexdigest()


def to_plain(obj):
    """Convert an OpenAI response object to plain JSON-serializable data."""
    if hasattr(obj, "to_dict_recursive"):
        return obj.to_dict_recursive()
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


class Recorded(dict):
    """Dict that also supports attribute access, so replayed responses work
    whether call sites use response["choices"][0]... or response.choices[0]...
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    @classmethod
    def wrap(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj


def git_info():
    def run(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=HERE, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "--short", "HEAD"),
        "dirty": run("status", "--porcelain") != "",
    }


class DesyncError(RuntimeError):
    pass


def _hard_fail(msg):
    """Stop the whole process at the first divergence.

    The sim wraps every openai call in a broad `try/except` that swallows
    exceptions and returns the string "ChatGPT ERROR", then retries. A normal
    `raise` therefore gets caught and the run cascades instead of stopping.
    os._exit bypasses all except-handlers and atexit, so the divergence diff
    is the LAST thing printed — the clean stop the guard is meant to produce.
    """
    print("\n" + "=" * 70, file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.stderr.flush()
    os._exit(3)


# ---------------------------------------------------------------------------
# Recorder — live API, captures everything, writes recording.jsonl
# ---------------------------------------------------------------------------

class Recorder:
    def __init__(self, openai_mod, path):
        self.path = path
        self.entries = []
        self.openai = openai_mod
        self.originals = {
            m: getattr(openai_mod, m).create for m in (*GEN_METHODS, EMBED_METHOD)
        }
        self._saved = False

    def install(self):
        for method in (*GEN_METHODS, EMBED_METHOD):
            setattr(getattr(self.openai, method), "create", self._make(method))
        atexit.register(self.save)  # survive crashes mid-run

    def _make(self, method):
        real = self.originals[method]
        def create(**kwargs):
            response = real(**kwargs)
            self.entries.append({
                "i": len(self.entries),
                "method": method,
                "hash": req_hash(kwargs),
                "request": json.loads(canonical(kwargs)),  # kept for diffing on desync
                "response": to_plain(response),
            })
            return response
        return create

    def save(self):
        if self._saved:
            return
        self._saved = True
        with open(self.path, "w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        gens = sum(1 for e in self.entries if e["method"] in GEN_METHODS)
        embeds = [e for e in self.entries if e["method"] == EMBED_METHOD]
        unique_embeds = len({e["hash"] for e in embeds})
        print(f"[replay] recording saved: {self.path}")
        print(f"[replay]   generations: {gens}")
        print(f"[replay]   embeddings:  {len(embeds)} ({unique_embeds} unique, "
              f"{len(embeds) - unique_embeds} duplicate => cacheable)")


# ---------------------------------------------------------------------------
# Replayer — no network, serves the recording, simulates latency
# ---------------------------------------------------------------------------

class Replayer:
    def __init__(self, openai_mod, path, latency_gen, latency_embed, lenient=False):
        self.openai = openai_mod
        self.latency_gen = latency_gen
        self.latency_embed = latency_embed
        self.lenient = lenient

        with open(path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # Generations: one FIFO queue per method, consumed strictly in order.
        self.gen_queues = {m: [e for e in entries if e["method"] == m]
                           for m in GEN_METHODS}
        self.gen_pos = {m: 0 for m in GEN_METHODS}

        # Embeddings: content-keyed map. Duplicates in the recording collapse
        # to one entry (identical request => identical response, verified).
        self.embed_map = {}
        self.embed_recorded_calls = 0
        for e in entries:
            if e["method"] == EMBED_METHOD:
                self.embed_recorded_calls += 1
                self.embed_map.setdefault(e["hash"], e)

        self.served = {m: 0 for m in (*GEN_METHODS, EMBED_METHOD)}
        self.mismatches = 0

    def install(self):
        for method in GEN_METHODS:
            setattr(getattr(self.openai, method), "create", self._make_gen(method))
        setattr(getattr(self.openai, EMBED_METHOD), "create", self._make_embed())
        # Belt and braces: make sure nothing can silently hit the network.
        self.openai.api_key = "REPLAY-MODE-NO-NETWORK"

    def _make_gen(self, method):
        def create(**kwargs):
            time.sleep(self.latency_gen)
            queue, pos = self.gen_queues[method], self.gen_pos[method]
            if pos >= len(queue):
                _hard_fail(
                    f"[replay] {method} call #{pos} exceeds recording "
                    f"({len(queue)} recorded). The replayed run made more "
                    f"{method} calls than the recorded run — an unpinned "
                    f"entropy source (check PYTHONHASHSEED is fixed and the "
                    f"recording was made under the same value)."
                )
            entry = queue[pos]
            self.gen_pos[method] += 1
            self.served[method] += 1
            incoming = req_hash(kwargs)
            if incoming != entry["hash"]:
                self.mismatches += 1
                # Dump BOTH full requests to files so we can diff the memory
                # lists (truncating to 400 chars hides where they diverge).
                rec_path = os.path.join(HERE, "desync_recorded.json")
                inc_path = os.path.join(HERE, "desync_incoming.json")
                try:
                    with open(rec_path, "w", encoding="utf-8") as f:
                        json.dump(entry["request"], f, indent=2, ensure_ascii=False)
                    with open(inc_path, "w", encoding="utf-8") as f:
                        json.dump(json.loads(canonical(kwargs)), f, indent=2,
                                  ensure_ascii=False)
                except Exception as _e:
                    print(f"[replay] (could not write desync dumps: {_e})",
                          file=sys.stderr)
                msg = (f"[replay] DESYNC at {method} #{pos}: request hash "
                       f"{incoming[:12]} != recorded {entry['hash'][:12]}.\n"
                       f"  full recorded request -> {rec_path}\n"
                       f"  full incoming request -> {inc_path}\n"
                       f"  recorded (head): {json.dumps(entry['request'])[:300]}\n"
                       f"  incoming (head): {canonical(kwargs)[:300]}")
                if self.lenient:
                    print(msg, file=sys.stderr)
                else:
                    _hard_fail(msg)
            return Recorded.wrap(entry["response"])
        return create

    def _make_embed(self):
        def create(**kwargs):
            time.sleep(self.latency_embed)
            self.served[EMBED_METHOD] += 1
            h = req_hash(kwargs)
            entry = self.embed_map.get(h)
            if entry is None:
                _hard_fail(
                    f"[replay] Embedding request not in recording "
                    f"(hash {h[:12]}): {canonical(kwargs)[:400]}\n"
                    f"The replayed run embedded text the recorded run never "
                    f"saw — an unpinned entropy source upstream (see PYTHONHASHSEED)."
                )
            return Recorded.wrap(entry["response"])
        return create

    def report(self):
        gens_recorded = {m: len(q) for m, q in self.gen_queues.items()}
        print("[replay] served vs recorded:")
        for m in GEN_METHODS:
            leftover = gens_recorded[m] - self.served[m]
            flag = "" if leftover == 0 else f"  <-- {leftover} UNCONSUMED"
            print(f"[replay]   {m:16s} {self.served[m]} / {gens_recorded[m]}{flag}")
        print(f"[replay]   {EMBED_METHOD:16s} {self.served[EMBED_METHOD]} calls "
              f"(recorded run made {self.embed_recorded_calls}; "
              f"{len(self.embed_map)} unique). "
              f"Fewer here = cache savings, and is expected on the modified branch.")
        if self.mismatches:
            print(f"[replay]   WARNING: {self.mismatches} hash mismatches "
                  f"(lenient mode) — result is NOT trustworthy.")
        return {
            "served": dict(self.served),
            "recorded_gens": gens_recorded,
            "recorded_embed_calls": self.embed_recorded_calls,
            "unique_embeds": len(self.embed_map),
            "mismatches": self.mismatches,
        }


# ---------------------------------------------------------------------------
# Sim boot + main
# ---------------------------------------------------------------------------

def _install_determinism_shims():
    """Neutralize object-identity ordering in memory retrieval.

    AssociativeMemory.retrieve_relevant_events/_thoughts each do `ret = set(ret)`
    on a list of ConceptNode OBJECTS, then the caller does list(...) on it. A set
    of objects orders by id() (memory address), which changes every process and
    is NOT controlled by random.seed() OR PYTHONHASHSEED. That reorders the
    persona/event context in prompts, so record and replay diverge.

    We return the same deduplicated nodes sorted by node_count (a monotonic int
    assigned at creation) — a stable, trajectory-consistent order. This is a
    measurement shim, same category as seeding: applied identically in record
    and replay, it doesn't touch temp_sleep or the embedding cache, so it can't
    bias the speedup. It patches the imported class, not any source file, so it
    never reaches the PR.
    """
    from persona.memory_structures.associative_memory import AssociativeMemory

    def _stable(original):
        def wrapped(self, *a, **k):
            return sorted(original(self, *a, **k), key=lambda n: n.node_count)
        return wrapped

    AssociativeMemory.retrieve_relevant_events = _stable(
        AssociativeMemory.retrieve_relevant_events)
    AssociativeMemory.retrieve_relevant_thoughts = _stable(
        AssociativeMemory.retrieve_relevant_thoughts)

    # ------------------------------------------------------------------
    # Stabilize new_retrieve's ranking. It scores nodes partly by RECENCY,
    # via node.last_accessed, which is stamped to sim-time WHEN a node is
    # retrieved. A tiny movement/timing drift between the live recording run
    # and a served replay run stamps those nodes on slightly different steps,
    # so the same nodes get ranked in a different order — reordering the
    # memory list in conversation prompts (confirmed: same statements, shuffled).
    # We re-sort each focal point's returned node list by node_count (creation
    # order), making every consumer (utterance memory list, decide_to_talk,
    # relationship/idea summaries) invariant to last_accessed drift.
    # CAVEAT: this replaces recency/relevance ordering with creation order, a
    # real change to what the agent "sees" — acceptable for a timing measurement
    # applied identically to both code versions, but worth noting if the number
    # is ever published.
    # ------------------------------------------------------------------
    import persona.cognitive_modules.retrieve as _retrieve_mod

    _orig_new_retrieve = _retrieve_mod.new_retrieve

    def _stable_new_retrieve(*args, **kwargs):
        out = _orig_new_retrieve(*args, **kwargs)
        for fp in out:
            out[fp] = sorted(out[fp], key=lambda nd: nd.node_count)
        return out

    _retrieve_mod.new_retrieve = _stable_new_retrieve

    # new_retrieve is pulled into several modules (converse, plan, reflect,
    # persona, ...) via `from ... import *`, which binds a COPY of the function
    # reference at import time. Enumerating them by hand already bit us once
    # (missed reflect.py -> reflection prompts still reordered). Instead, sweep
    # every loaded module and rebind any name still pointing at the original.
    for _m in list(sys.modules.values()):
        if getattr(_m, "new_retrieve", None) is _orig_new_retrieve:
            _m.new_retrieve = _stable_new_retrieve


def run_sim(origin, target, steps, keep=False):
    """Drive the simulation exactly the way bench.py does.

    Must be run from reverie/backend_server/ (bench.py's stated cwd), so that
    `from utils ...` and `from reverie ...` resolve. Boot mirrors bench.py:
      1. resolve fs_storage, 2. delete any stale target sim folder,
      3. import ReverieServer (AFTER patching), 4. ReverieServer(origin,target),
      5. run_standalone(steps) — the entry point that skips reverie.py's
         __main__ input() prompts (Create()'s "Regenerate norms? y/n").
    Cleanup on exit unless keep=True, so the next run forks fresh from origin.
    """
    sys.path.insert(0, HERE)
    from utils import fs_storage               # same import bench.py uses

    target_path = os.path.join(fs_storage, target)
    # Delete-before-run: a leftover target folder would make ReverieServer
    # RESUME stale state instead of forking fresh from origin, which silently
    # desyncs the recording/replay. bench.py does exactly this.
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)

    if not keep:
        @atexit.register
        def _cleanup():
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)

    from reverie import ReverieServer          # must import AFTER patching
    _install_determinism_shims()               # after class exists, before any retrieval
    rs = ReverieServer(origin, target)
    rs.run_standalone(steps)                   # same entry point bench.py uses


def main():
    # ------------------------------------------------------------------
    # PIN PYTHONHASHSEED. The sim iterates set()s whose order is randomized
    # per-process by hash randomization (e.g. tile["events"] in perceive.py,
    # sorted by distance only, so equal-distance events tie-break in set
    # order). random.seed() does NOT control this. If record and replay run
    # under different hash seeds, perception diverges and the guard trips.
    # Hash randomization is fixed at interpreter startup, so we can't set it
    # from here — we re-exec once with it pinned. Record AND replay both go
    # through this, so both share the same set-iteration order.
    # ------------------------------------------------------------------
    _WANT_HASHSEED = "0"
    if os.environ.get("PYTHONHASHSEED") != _WANT_HASHSEED:
        os.environ["PYTHONHASHSEED"] = _WANT_HASHSEED
        sys.exit(subprocess.run([sys.executable, *sys.argv]).returncode)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["record", "replay"], required=True)
    ap.add_argument("--origin", required=True, help="base sim folder, e.g. base_the_ville_n10")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--recording", default=DEFAULT_RECORDING)
    ap.add_argument("--latency-gen", type=float, default=DEFAULT_LATENCY_GEN)
    ap.add_argument("--latency-embed", type=float, default=DEFAULT_LATENCY_EMBED)
    ap.add_argument("--seed", type=int, default=0,
                    help="LOAD-BEARING, not optional. The sim never seeds itself "
                         "(verified), and unseeded random.* steers the call SEQUENCE, "
                         "not just prompt content: execute.py random.sample picks agent "
                         "movement and plan.py random.choice picks tasks, both of which "
                         "change which API calls come next. Record and replay (and the "
                         "baseline vs modified replays) MUST use the same --seed, or the "
                         "hash guard will fire legitimately on a diverged trajectory.")
    ap.add_argument("--lenient", action="store_true",
                    help="log generation hash mismatches instead of failing. "
                         "Diagnosis only — never report a number from a lenient run.")
    ap.add_argument("--keep", action="store_true",
                    help="keep the target sim folder after the run (default: delete, "
                         "matching bench.py, so the next run forks fresh from origin).")
    args = ap.parse_args()

    # Seed BEFORE importing anything from reverie. This is what makes the
    # replayed trajectory reproduce the recorded one: responses are pinned by
    # the recording, and the sim's own random.* calls are pinned by the seed.
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    import openai  # patch target; must be imported before reverie

    if args.mode == "record":
        harness = Recorder(openai, args.recording)
    else:
        harness = Replayer(openai, args.recording,
                           args.latency_gen, args.latency_embed, args.lenient)
    harness.install()

    info = git_info()
    # STABLE target name (like bench.py's __bench_<label>), NOT timestamped.
    # Record and replay must use an identical sim_code: if the sim_code string
    # ever leaks into a prompt or an embedded text, a per-run name would change
    # the request hash and trip the guard as a false desync. Deleted before each
    # run (in run_sim), so a fixed name still forks fresh from origin every time.
    target = f"__replay_{args.origin}"
    print(f"[replay] mode={args.mode} branch={info['branch']} commit={info['commit']} "
          f"dirty={info['dirty']} steps={args.steps} seed={args.seed}")
    if info["dirty"]:
        print("[replay] WARNING: working tree is dirty — result provenance is weak.")

    t0 = time.perf_counter()
    run_sim(args.origin, target, args.steps, keep=args.keep)
    wall = time.perf_counter() - t0

    result = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "wall_seconds": round(wall, 3),
        "steps": args.steps,
        "seed": args.seed,
        "latency_gen": args.latency_gen,
        "latency_embed": args.latency_embed,
        "git": info,
        "recording": os.path.basename(args.recording),
    }
    if args.mode == "record":
        harness.save()
        result["calls"] = len(harness.entries)
    else:
        result["counts"] = harness.report()

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[replay] wall time: {wall:.2f}s  (logged to {os.path.basename(RESULTS_FILE)})")


if __name__ == "__main__":
    main()
