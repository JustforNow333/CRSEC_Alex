#!/usr/bin/env python3
"""
aggregate.py -- summarise bench_results.jsonl into a number you can quote.

Run from the same directory as bench.py (reverie/backend_server/):

    python aggregate.py
    python aggregate.py --steps 100        # only runs at this step count

Reports the median improvement and the spread. The spread is real variation
across workloads, not measurement error -- report it alongside the median.
"""
import json, argparse, statistics, os, sys


def load(path):
    if not os.path.exists(path):
        sys.exit(f"No {path} found. Run bench.py first.")
    rows = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  ! line {n} is corrupt (interleaved write?) - skipping", file=sys.stderr)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="bench_results.jsonl")
    p.add_argument("--steps", type=int, help="only include runs at this step count")
    args = p.parse_args()

    rows = load(args.file)
    if args.steps:
        rows = [r for r in rows if r.get("steps") == args.steps]
    if not rows:
        sys.exit("No matching runs.")

    print(f"\n{'label':<10} {'steps':>6} {'branch':<14} {'commit':<9} {'wall(m)':>8} "
          f"{'gen':>6} {'hits':>6} {'sleep':>7} {'impr%':>7}")
    print("-" * 82)
    for r in rows:
        flag = " *dirty" if r.get("git_dirty") else ""
        print(f"{r.get('label',''):<10} {r.get('steps',0):>6} "
              f"{str(r.get('git_branch','?')):<14} {str(r.get('git_commit','?')):<9} "
              f"{r.get('wall',0)/60:>8.1f} {r.get('gen_calls',0):>6} "
              f"{r.get('cache_hits',0):>6} {r.get('sleep',0):>7.2f} "
              f"{r.get('improvement_pct',0):>7.1f}{flag}")

    # --- validity warnings -------------------------------------------------
    problems = []
    steps_seen = {r.get("steps") for r in rows}
    if len(steps_seen) > 1:
        problems.append(f"mixed step counts {sorted(steps_seen)} -- improvement scales with "
                        "run length, so don't pool these. Use --steps N.")
    codes = {(r.get("git_commit"), r.get("git_branch")) for r in rows}
    if len(codes) > 1:
        problems.append(f"runs came from {len(codes)} different commits -- not one experiment.")
    if any(r.get("git_dirty") for r in rows):
        problems.append("some runs had uncommitted changes; their commit SHA doesn't "
                        "identify what actually ran.")
    for r in rows:
        expected = 0.1 * r.get("steps", 0)
        if r.get("has_cache") and r.get("sleep", 0) > expected + 1:
            problems.append(f"{r.get('label')}: sleep {r['sleep']:.2f}s > expected "
                            f"{expected:.2f}s -- backoff fired, timings include retry waits.")
    if problems:
        print("\nWARNINGS")
        for w in problems:
            print(f"  ! {w}")

    pcts = [r.get("improvement_pct", 0) for r in rows]
    print(f"\nn = {len(pcts)} runs")
    print(f"median improvement : {statistics.median(pcts):.1f}%")
    print(f"range              : {min(pcts):.1f}% - {max(pcts):.1f}%")
    if len(pcts) > 1:
        print(f"stdev              : {statistics.stdev(pcts):.1f}pp")
    steps = rows[0].get("steps")
    if len(steps_seen) == 1:
        print(f"\nQuote as: ~{statistics.median(pcts):.0f}% less time for the same work "
              f"({len(pcts)} runs at {steps} steps)")
    print()


if __name__ == "__main__":
    main()
