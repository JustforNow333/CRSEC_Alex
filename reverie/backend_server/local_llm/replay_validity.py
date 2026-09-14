#!/usr/bin/env python3
"""
replay_validity.py -- benchmark a local model against a recorded OpenAI run.

This reuses the recordings that replay.py already produces. A recording is a
JSONL file of every OpenAI request and response from one real simulation:

    {"i": 0, "method": "Completion", "hash": "...",
     "request": {"prompt": "...", "stop": ["\\n"], ...},
     "response": {"choices": [{"text": "7 am"}], ...}}

That file is the exact prompt distribution CRSEC issues, in order, paired with
what a hosted model answered. It is a far better validity benchmark than any
set of hand-written probe prompts, because it is the real workload.

What this measures:

  FORMAT AGREEMENT (generation calls)
    Derive a format signature from the recorded OpenAI answer -- integer,
    yes/no, parenthesised tuple, numbered list, emoji-only, line count -- and
    check whether the local model's answer to the same prompt carries the same
    signature. This approximates what func_validate will do inside the
    simulation without needing the validator closures, and it is grounded in
    what actually worked rather than in what a prompt appears to ask for.

  RETRIEVAL PERTURBATION (embedding calls)
    Take the texts that were embedded during the run, embed them with the
    local model, and compare the pairwise cosine-similarity matrices under
    both. Reported as Spearman rank correlation. This is the number that says
    whether swapping text-embedding-ada-002 for a local embedder reorders what
    the agents remember. It is easy to overlook because nothing crashes when
    retrieval degrades -- the agents just recall less relevant things.

Usage, from reverie/backend_server:

    python replay.py --mode record --origin base_the_ville_n10 --steps 100 --seed 42
    python local_llm/replay_validity.py --recording recording.jsonl --sample 150

Costs nothing beyond local compute; the recorded side is already paid for.

Author: Alexander Maiello (acm357@cornell.edu)
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_backend  # noqa: E402

GEN_METHODS = ("ChatCompletion", "Completion")
EMBED_METHOD = "Embedding"

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)


# --------------------------------------------------------------------------
# Format signatures
# --------------------------------------------------------------------------

def signature(text):
    """
    Reduce a response to the structural features CRSEC's validators care about.

    Deliberately coarse. The question is not whether the local model said the
    same thing as GPT-4o-mini -- for a social simulation it should not have to.
    The question is whether it produced something the parser can consume.
    """
    t = (text or "").strip()
    first = t.split("\n")[0].strip() if t else ""
    lines = [l for l in t.split("\n") if l.strip()]
    return {
        "empty": len(t) == 0,
        "int_parseable": _int_parseable(first),
        "yes_no": first.lower().split()[:1] in (["yes"], ["no"]) if first else False,
        "paren_tuple": _paren_tuple_arity(t),
        "numbered_list": sum(1 for l in lines if re.match(r"^\s*\d+[\.\)]", l)),
        "emoji_only": bool(t) and bool(_EMOJI.search(t)) and len(t) <= 12,
        "n_lines": len(lines),
        "len_bucket": _bucket(len(t)),
    }


def _int_parseable(s):
    try:
        int(s.lower().split("am")[0].strip())
        return True
    except Exception:
        return False


def _paren_tuple_arity(t):
    """Arity of a leading (a, b[, c]) construct, or 0."""
    m = re.search(r"\(([^)]*)\)", t)
    if not m:
        # run_gpt_prompt_event_triple splits on ')' without requiring '('
        head = t.split(")")[0]
        if ")" in t and "," in head:
            return len([p for p in head.split(",") if p.strip()])
        return 0
    return len([p for p in m.group(1).split(",") if p.strip()])


def _bucket(n):
    for edge in (0, 5, 20, 80, 300, 1000):
        if n <= edge:
            return edge
    return 9999


def compare(gold_sig, local_sig):
    """
    Does the local answer match the gold answer's format?

    Only the features the gold response actually exhibits are checked. If the
    hosted model answered with a bare integer, an integer is required; if it
    answered with free prose, only non-emptiness and a comparable length are.
    """
    reasons = []

    if gold_sig["empty"]:
        return True, []  # nothing to match
    if local_sig["empty"]:
        return False, ["empty response"]

    if gold_sig["int_parseable"] and not local_sig["int_parseable"]:
        reasons.append("expected parseable integer")
    if gold_sig["yes_no"] and not local_sig["yes_no"]:
        reasons.append("expected yes/no")
    if gold_sig["paren_tuple"] and local_sig["paren_tuple"] != gold_sig["paren_tuple"]:
        reasons.append("expected %d-part tuple, got %d"
                       % (gold_sig["paren_tuple"], local_sig["paren_tuple"]))
    if gold_sig["numbered_list"] >= 2 and local_sig["numbered_list"] < 2:
        reasons.append("expected numbered list")
    if gold_sig["emoji_only"] and not local_sig["emoji_only"]:
        reasons.append("expected short emoji")
    if gold_sig["len_bucket"] <= 20 and local_sig["len_bucket"] > 80:
        reasons.append("far longer than gold (parsers truncate or fail)")

    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------
# Recording access
# --------------------------------------------------------------------------

def load_recording(path):
    gens, embeds = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            (gens if e.get("method") in GEN_METHODS else
             embeds if e.get("method") == EMBED_METHOD else []).append(e)
    return gens, embeds


def prompt_of(entry):
    req = entry.get("request", {})
    if "prompt" in req:
        p = req["prompt"]
        return "".join(p) if isinstance(p, (list, tuple)) else str(p)
    return "\n".join(m.get("content", "") for m in req.get("messages", []))


def gold_of(entry):
    ch = (entry.get("response") or {}).get("choices") or [{}]
    c = ch[0]
    if "text" in c:
        return c["text"]
    return (c.get("message") or {}).get("content", "")


def texts_of(embed_entry):
    inp = (embed_entry.get("request") or {}).get("input", "")
    return inp if isinstance(inp, list) else [inp]


def gold_vector(embed_entry):
    data = (embed_entry.get("response") or {}).get("data") or []
    return data[0].get("embedding") if data else None


# --------------------------------------------------------------------------
# Retrieval comparison
# --------------------------------------------------------------------------

def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def pairwise(vectors):
    out = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            out.append(cos(vectors[i], vectors[j]))
    return out


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, idx in enumerate(order):
            r[idx] = float(pos)
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="recording.jsonl")
    ap.add_argument("--sample", type=int, default=100,
                    help="generation calls to replay (0 = all)")
    ap.add_argument("--embed-sample", type=int, default=60,
                    help="embedded texts to compare (0 = skip)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="", help="write per-call results to JSONL")
    args = ap.parse_args()

    if not os.path.exists(args.recording):
        print("No recording at %s.\n"
              "Produce one first:\n"
              "  python replay.py --mode record --origin base_the_ville_n10 "
              "--steps 100 --seed 42" % args.recording, file=sys.stderr)
        return 1

    os.environ["CRSEC_LLM_BACKEND"] = "local"
    local_backend.activate(force=True, verbose=False)
    import openai

    cfg = local_backend.config()
    gens, embeds = load_recording(args.recording)

    print("=" * 70)
    print("replay validity: %s" % os.path.basename(args.recording))
    print("  recorded    : %d generations, %d embedding calls"
          % (len(gens), len(embeds)))
    print("  local chat  : %s" % cfg["chat_model"])
    print("  local embed : %s" % cfg["embed_model"])
    print("=" * 70)

    rng = random.Random(args.seed)

    # ---- generations --------------------------------------------------
    picks = gens if not args.sample or args.sample >= len(gens) \
        else rng.sample(gens, args.sample)
    picks.sort(key=lambda e: e["i"])

    ok = 0
    failures = {}
    records = []
    t_start = time.time()

    for n, entry in enumerate(picks, 1):
        prompt = prompt_of(entry)
        gold = gold_of(entry)
        req = entry.get("request", {})
        try:
            r = openai.Completion.create(
                model="local", prompt=prompt,
                temperature=req.get("temperature", 0.8),
                top_p=req.get("top_p", 1),
                max_tokens=req.get("max_tokens", 150),
                frequency_penalty=req.get("frequency_penalty", 0),
                presence_penalty=req.get("presence_penalty", 0),
                stream=False, stop=req.get("stop"))
            local = r.choices[0].text
        except Exception as exc:
            local = ""
            failures.setdefault("api error", 0)
            failures["api error"] += 1

        gs, ls = signature(gold), signature(local)
        passed, reasons = compare(gs, ls)
        if passed:
            ok += 1
        for reason in reasons:
            key = reason.split(",")[0]
            failures[key] = failures.get(key, 0) + 1

        records.append({"i": entry["i"], "pass": passed, "reasons": reasons,
                        "gold": gold[:200], "local": local[:200]})

        if n % 25 == 0 or n == len(picks):
            elapsed = time.time() - t_start
            print("  %4d/%d  running agreement %.0f%%  (%.1fs, %.1fs/call)"
                  % (n, len(picks), 100.0 * ok / n, elapsed, elapsed / n))

    rate = 100.0 * ok / len(picks) if picks else 0.0
    print("-" * 70)
    print("  FORMAT AGREEMENT: %d/%d = %.1f%%" % (ok, len(picks), rate))
    if failures:
        print("  failure modes:")
        for reason, count in sorted(failures.items(), key=lambda kv: -kv[1]):
            print("    %-46s %4d" % (reason[:46], count))

    # ---- embeddings ---------------------------------------------------
    rho = None
    if args.embed_sample and embeds:
        pool = []
        seen = set()
        for e in embeds:
            v = gold_vector(e)
            for t in texts_of(e):
                if v and t and t not in seen:
                    seen.add(t)
                    pool.append((t, v))
        chosen = pool if args.embed_sample >= len(pool) \
            else rng.sample(pool, args.embed_sample)

        print("-" * 70)
        print("  embedding %d texts locally..." % len(chosen))
        gold_vecs, local_vecs = [], []
        for t, gv in chosen:
            try:
                lv = openai.Embedding.create(
                    input=[t], model="local")["data"][0]["embedding"]
            except Exception:
                continue
            gold_vecs.append(gv)
            local_vecs.append(lv)

        if len(gold_vecs) >= 3:
            rho = spearman(pairwise(gold_vecs), pairwise(local_vecs))
            print("  RETRIEVAL RANK CORRELATION (Spearman): %.3f" % rho)
            if rho >= 0.85:
                print("    Retrieval ordering is largely preserved.")
            elif rho >= 0.6:
                print("    Noticeably reordered. Agents will recall different")
                print("    memories than in the recorded run.")
            else:
                print("    Substantially different retrieval. Treat the local")
                print("    embedder as its own experimental condition, not as")
                print("    a drop-in substitution.")

    # ---- verdict ------------------------------------------------------
    print("=" * 70)
    if rate >= 90:
        print("  Usable for experiments on this workload.")
    elif rate >= 70:
        print("  Marginal. Run it, but report the fail-safe rate with results.")
    else:
        print("  Not usable for results. Most generations would be replaced by")
        print("  CRSEC's hardcoded fail-safes, and the run would still finish.")
    print(local_backend.summary())

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"summary": {
                "recording": args.recording,
                "chat_model": cfg["chat_model"],
                "embed_model": cfg["embed_model"],
                "sampled": len(picks),
                "format_agreement": rate,
                "retrieval_spearman": rho,
                "failure_modes": failures,
            }}) + "\n")
            for r in records:
                f.write(json.dumps(r) + "\n")
        print("  per-call results: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
