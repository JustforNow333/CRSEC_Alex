#!/usr/bin/env python3
"""
replay.py — deterministic record/replay harness for CRSEC.

Record one live run, then replay the same responses to both code versions
and compare wall time. Generations are sequence-keyed (sha256 guard);
embeddings are content-keyed (cache may skip duplicates).

Usage:
  # Record (live API, costs money):
  python replay.py --mode record --origin base_the_ville_n10 --steps 100 --seed 42

  # Replay modified code, then swap gpt_structure.py to baseline and replay again:
  python replay.py --mode replay --origin base_the_ville_n10 --steps 100 --seed 42

  # Latency sweep:
  python replay.py --mode replay --steps 100 --latency-gen 0.5 --latency-embed 0.2 --seed 42
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

# Load .env before anything imports utils.
HERE = os.path.dirname(os.path.abspath(__file__))
_env_file = os.path.join(HERE, "..", "..", ".env")
if os.path.isfile(_env_file):
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

DEFAULT_RECORDING = os.path.join(HERE, "recording.jsonl")
RESULTS_FILE = os.path.join(HERE, "replay_results.jsonl")

# Measured medians from the live bench runs; override via CLI to sweep.
DEFAULT_LATENCY_GEN = 1.28
DEFAULT_LATENCY_EMBED = 0.43

GEN_METHODS = ("ChatCompletion", "Completion")
EMBED_METHOD = "Embedding"


def canonical(kwargs: dict) -> str:
    """Deterministic serialization of a request for hashing."""
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
    """Dict with attribute access for replayed OpenAI responses."""
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
    """os._exit to bypass the sim's broad try/except that swallows exceptions."""
    print("\n" + "=" * 70, file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.stderr.flush()
    os._exit(3)


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
        atexit.register(self.save)

    def _make(self, method):
        real = self.originals[method]
        def create(**kwargs):
            response = real(**kwargs)
            self.entries.append({
                "i": len(self.entries),
                "method": method,
                "hash": req_hash(kwargs),
                "request": json.loads(canonical(kwargs)),
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


class Replayer:
    def __init__(self, openai_mod, path, latency_gen, latency_embed, lenient=False):
        self.openai = openai_mod
        self.latency_gen = latency_gen
        self.latency_embed = latency_embed
        self.lenient = lenient

        with open(path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        self.gen_queues = {m: [e for e in entries if e["method"] == m]
                           for m in GEN_METHODS}
        self.gen_pos = {m: 0 for m in GEN_METHODS}

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
        self.openai.api_key = "REPLAY-MODE-NO-NETWORK"

    def _make_gen(self, method):
        def create(**kwargs):
            time.sleep(self.latency_gen)
            queue, pos = self.gen_queues[method], self.gen_pos[method]
            if pos >= len(queue):
                _hard_fail(
                    f"[replay] {method} call #{pos} exceeds recording "
                    f"({len(queue)} recorded). Check PYTHONHASHSEED."
                )
            entry = queue[pos]
            self.gen_pos[method] += 1
            self.served[method] += 1
            incoming = req_hash(kwargs)
            if incoming != entry["hash"]:
                self.mismatches += 1
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
                    f"[replay] Embedding not in recording "
                    f"(hash {h[:12]}): {canonical(kwargs)[:400]}"
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
              f"(recorded: {self.embed_recorded_calls}, "
              f"{len(self.embed_map)} unique)")
        if self.mismatches:
            print(f"[replay]   WARNING: {self.mismatches} hash mismatches (lenient mode)")
        return {
            "served": dict(self.served),
            "recorded_gens": gens_recorded,
            "recorded_embed_calls": self.embed_recorded_calls,
            "unique_embeds": len(self.embed_map),
            "mismatches": self.mismatches,
        }


def _install_determinism_shims():
    """Sort memory retrieval by node_count instead of object identity (set order)."""
    from persona.memory_structures.associative_memory import AssociativeMemory

    def _stable(original):
        def wrapped(self, *a, **k):
            return sorted(original(self, *a, **k), key=lambda n: n.node_count)
        return wrapped

    AssociativeMemory.retrieve_relevant_events = _stable(
        AssociativeMemory.retrieve_relevant_events)
    AssociativeMemory.retrieve_relevant_thoughts = _stable(
        AssociativeMemory.retrieve_relevant_thoughts)

    # Stabilize new_retrieve: sort by node_count to avoid last_accessed drift.
    import persona.cognitive_modules.retrieve as _retrieve_mod

    _orig_new_retrieve = _retrieve_mod.new_retrieve

    def _stable_new_retrieve(*args, **kwargs):
        out = _orig_new_retrieve(*args, **kwargs)
        for fp in out:
            out[fp] = sorted(out[fp], key=lambda nd: nd.node_count)
        return out

    _retrieve_mod.new_retrieve = _stable_new_retrieve

    # Rebind in all modules that imported it via `from ... import *`.
    for _m in list(sys.modules.values()):
        if getattr(_m, "new_retrieve", None) is _orig_new_retrieve:
            _m.new_retrieve = _stable_new_retrieve


def run_sim(origin, target, steps, keep=False):
    """Boot and run the sim the same way bench.py does."""
    sys.path.insert(0, HERE)
    from utils import fs_storage               # same import bench.py uses

    target_path = os.path.join(fs_storage, target)
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)

    if not keep:
        @atexit.register
        def _cleanup():
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)

    from reverie import ReverieServer  # after patching
    _install_determinism_shims()
    rs = ReverieServer(origin, target)
    rs.run_standalone(steps)


def main():
    # Pin PYTHONHASHSEED (must be set before interpreter startup, so re-exec).
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
                    help="RNG seed. Must match between record and all replays.")
    ap.add_argument("--lenient", action="store_true",
                    help="log hash mismatches instead of failing (diagnosis only)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the target sim folder after the run")
    args = ap.parse_args()

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    import openai

    if args.mode == "record":
        harness = Recorder(openai, args.recording)
    else:
        harness = Replayer(openai, args.recording,
                           args.latency_gen, args.latency_embed, args.lenient)
    harness.install()

    info = git_info()
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
