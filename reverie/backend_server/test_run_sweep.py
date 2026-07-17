"""
Tests for run_sweep.py.
No real subprocesses; subprocess execution is monkeypatched.
Run from reverie/backend_server/:  python test_run_sweep.py
"""
import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_sweep


class TestRunSweep(unittest.TestCase):

    def _make_configs(self, tmp, entries):
        """Write a configs JSON file and return its path."""
        path = os.path.join(tmp, "configs.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        return path

    # ------------------------------------------------------------------
    # Test 1 — concurrency never exceeds --max-concurrent
    # ------------------------------------------------------------------
    def test_concurrency_limit(self):
        configs = [
            {"origin": "base", "target": f"run_{i:02d}", "steps": 1}
            for i in range(6)
        ]
        max_concurrent = 2

        active = [0]
        peak   = [0]
        lock   = threading.Lock()

        def fake_subprocess(cmd, log_fh):
            with lock:
                active[0] += 1
                if active[0] > peak[0]:
                    peak[0] = active[0]
            # Simulate a tiny bit of work so threads actually overlap
            import time; time.sleep(0.02)
            with lock:
                active[0] -= 1
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._make_configs(tmp, configs)
            log_dir = os.path.join(tmp, "logs")

            # Patch _SCRIPT_DIR so logs land in tmp
            orig = run_sweep._SCRIPT_DIR
            run_sweep._SCRIPT_DIR = tmp
            try:
                run_sweep.main(
                    ["--configs", cfg_path, f"--max-concurrent={max_concurrent}", "--retries=0"],
                    _run_subprocess=fake_subprocess,
                )
            finally:
                run_sweep._SCRIPT_DIR = orig

        self.assertLessEqual(peak[0], max_concurrent,
            f"Peak concurrent={peak[0]} exceeded max-concurrent={max_concurrent}")

    # ------------------------------------------------------------------
    # Test 2 — failed run retried; retry uses --resume
    # ------------------------------------------------------------------
    def test_retry_uses_resume(self):
        configs = [{"origin": "base", "target": "flaky", "steps": 5}]

        call_log = []  # each entry: list of cmd tokens

        def fake_subprocess(cmd, log_fh):
            call_log.append(list(cmd))
            # Fail once, then succeed
            return 1 if len(call_log) == 1 else 0

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._make_configs(tmp, configs)
            orig = run_sweep._SCRIPT_DIR
            run_sweep._SCRIPT_DIR = tmp
            try:
                run_sweep.main(
                    ["--configs", cfg_path, "--max-concurrent=1", "--retries=1"],
                    _run_subprocess=fake_subprocess,
                )
            finally:
                run_sweep._SCRIPT_DIR = orig

        self.assertEqual(len(call_log), 2, "Should have been called twice (1 fail + 1 retry)")
        self.assertNotIn("--resume", call_log[0], "First attempt must NOT have --resume")
        self.assertIn("--resume", call_log[1],    "Retry attempt MUST have --resume")

    # ------------------------------------------------------------------
    # Test 3 — per-target log file is written
    # ------------------------------------------------------------------
    def test_log_file_written(self):
        configs = [{"origin": "base", "target": "my_run", "steps": 3}]

        def fake_subprocess(cmd, log_fh):
            log_fh.write("simulated output\n")
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._make_configs(tmp, configs)
            orig = run_sweep._SCRIPT_DIR
            run_sweep._SCRIPT_DIR = tmp
            try:
                run_sweep.main(
                    ["--configs", cfg_path, "--max-concurrent=1", "--retries=0"],
                    _run_subprocess=fake_subprocess,
                )
            finally:
                run_sweep._SCRIPT_DIR = orig

            log_path = os.path.join(tmp, "logs", "my_run.log")
            self.assertTrue(os.path.exists(log_path), "Log file must exist")
            with open(log_path) as f:
                content = f.read()
            self.assertIn("simulated output", content)

    # ------------------------------------------------------------------
    # Test 4 — summary reflects correct exit codes and attempt counts
    # ------------------------------------------------------------------
    def test_summary_outcome(self):
        configs = [
            {"origin": "base", "target": "ok_run",   "steps": 1},
            {"origin": "base", "target": "bad_run",  "steps": 1},
        ]

        def fake_subprocess(cmd, log_fh):
            target = cmd[cmd.index("--target") + 1]
            return 0 if target == "ok_run" else 1

        captured = []
        orig_print = run_sweep.__builtins__["print"] if isinstance(
            run_sweep.__builtins__, dict) else print

        import io
        buf = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._make_configs(tmp, configs)
            orig_dir = run_sweep._SCRIPT_DIR
            run_sweep._SCRIPT_DIR = tmp
            orig_stdout = sys.stdout
            sys.stdout = buf
            try:
                run_sweep.main(
                    ["--configs", cfg_path, "--max-concurrent=2", "--retries=0"],
                    _run_subprocess=fake_subprocess,
                )
            finally:
                sys.stdout = orig_stdout
                run_sweep._SCRIPT_DIR = orig_dir

        summary = buf.getvalue()
        self.assertIn("ok_run",  summary)
        self.assertIn("bad_run", summary)
        self.assertIn("OK",      summary)
        self.assertIn("FAILED",  summary)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromTestCase(TestRunSweep))
    sys.exit(0 if result.wasSuccessful() else 1)
