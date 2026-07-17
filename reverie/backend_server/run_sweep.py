"""
run_sweep.py — Supervisor that runs multiple headless simulations in parallel.

Usage:
  python run_sweep.py --configs configs.json [--max-concurrent 2] [--retries 1]

configs.json format:
  [{"origin": "base_sim", "target": "run_01", "steps": 100}, ...]

Must be run from reverie/backend_server/ (same as run_headless.py).
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a sweep of headless CRSEC simulations in parallel."
    )
    parser.add_argument(
        "--configs", required=True,
        help="Path to a JSON file: list of {origin, target, steps} objects.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=2,
        help="Maximum number of simulations running at once (default: 2).",
    )
    parser.add_argument(
        "--retries", type=int, default=1,
        help="Number of times to retry a failed run (with --resume, default: 1).",
    )
    return parser.parse_args(argv)


def _run_one(cfg, retries, log_dir, run_subprocess):
    """
    Run a single simulation config, retrying on failure.

    Returns (target, final_exit_code, attempts).
    `run_subprocess` is the callable used to launch the process so it can be
    monkeypatched in tests without touching the rest of the logic.
    """
    origin = cfg["origin"]
    target = cfg["target"]
    steps  = str(cfg["steps"])

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{target}.log")

    attempts = 0
    exit_code = None

    with open(log_path, "w", encoding="utf-8") as log_fh:
        for attempt_num in range(1 + retries):
            attempts += 1
            cmd = [sys.executable, "run_headless.py",
                   "--origin", origin,
                   "--target", target,
                   "--steps",  steps]
            if attempt_num > 0:
                cmd.append("--resume")

            log_fh.write(f"\n--- attempt {attempt_num + 1} command: {' '.join(cmd)} ---\n")
            log_fh.flush()

            exit_code = run_subprocess(cmd, log_fh)

            log_fh.write(f"--- exit code: {exit_code} ---\n")
            log_fh.flush()

            if exit_code == 0:
                break

    return target, exit_code, attempts


def main(argv=None, _run_subprocess=None):
    args = _parse_args(argv)

    with open(args.configs, "r", encoding="utf-8") as f:
        configs = json.load(f)

    log_dir = os.path.join(_SCRIPT_DIR, "logs")

    # Default subprocess implementation — runs the real command.
    def default_run_subprocess(cmd, log_fh):
        proc = subprocess.run(
            cmd,
            cwd=_SCRIPT_DIR,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        return proc.returncode

    run_subprocess = _run_subprocess if _run_subprocess is not None else default_run_subprocess

    results = {}  # target -> (exit_code, attempts)

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {
            executor.submit(_run_one, cfg, args.retries, log_dir, run_subprocess): cfg["target"]
            for cfg in configs
        }
        for future in as_completed(futures):
            target, exit_code, attempts = future.result()
            results[target] = (exit_code, attempts)

    print("\n=== Sweep summary ===")
    for target, (code, attempts) in sorted(results.items()):
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  {target} -> {status}, {attempts} attempt(s)")


if __name__ == "__main__":
    main()
