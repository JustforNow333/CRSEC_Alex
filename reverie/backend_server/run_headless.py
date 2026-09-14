"""
run_headless.py — Headless launcher for a single CRSEC simulation run.

Usage:
  python run_headless.py --origin <fork_sim_code> --target <new_sim_code> \
                         --steps <N> [--resume]

Must be run from reverie/backend_server/ (same requirement as reverie.py).
"""
import argparse
import os
import sys


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a single CRSEC generative-agent simulation headlessly."
    )
    parser.add_argument(
        "--origin", required=True,
        help="The fork base sim_code to copy from (ignored when --resume).",
    )
    parser.add_argument(
        "--target", required=True,
        help="The sim_code for the new (or resumed) simulation.",
    )
    parser.add_argument(
        "--steps", type=int, required=True,
        help="Number of simulation steps to run.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an existing simulation instead of forking a new one.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    # Force UTF-8 on the console. On Windows stdout/stderr default to cp1252, so
    # a single non-ASCII persona utterance raises UnicodeEncodeError and kills
    # the run; errors="replace" keeps a display problem from aborting it.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args(argv)

    # Import here so the module can be imported in tests without pulling in the
    # full simulation stack (tests monkeypatch ReverieServer before import).
    from reverie import ReverieServer
    from utils import fs_storage

    if args.resume:
        rs = ReverieServer(
            fork_sim_code=args.origin,
            sim_code=args.target,
            resume=True,
        )
    else:
        target_folder = os.path.join(fs_storage, args.target)
        if os.path.exists(target_folder):
            print(
                f"ERROR: target simulation folder already exists: {target_folder}\n"
                f"Use --resume to continue an existing run, or choose a different --target.",
                file=sys.stderr,
            )
            sys.exit(1)
        rs = ReverieServer(args.origin, args.target)

    rs.run_standalone(args.steps)


if __name__ == "__main__":
    main()
