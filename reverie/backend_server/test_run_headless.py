"""
Tests for run_headless.py.
No real API calls; ReverieServer is monkeypatched throughout.
Run from reverie/backend_server/:  python test_run_headless.py
"""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Minimal stubs so importing run_headless doesn't pull the real sim stack.
# ---------------------------------------------------------------------------

_mock_rs_class = MagicMock()

_reverie_mod = types.ModuleType("reverie")
_reverie_mod.ReverieServer = _mock_rs_class

_utils_mod = types.ModuleType("utils")

sys.modules.setdefault("reverie", _reverie_mod)
sys.modules.setdefault("utils", _utils_mod)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_headless  # noqa: E402


class TestRunHeadless(unittest.TestCase):

    def setUp(self):
        _mock_rs_class.reset_mock()
        self._standalone_calls = []
        self._init_kwargs = {}

        outer = self

        def fake_constructor(*args, **kwargs):
            # Normalise positional and keyword forms.
            fork  = kwargs.get("fork_sim_code", args[0] if len(args) > 0 else None)
            sim   = kwargs.get("sim_code",      args[1] if len(args) > 1 else None)
            resume = kwargs.get("resume",       args[2] if len(args) > 2 else False)
            outer._init_kwargs = dict(fork_sim_code=fork, sim_code=sim, resume=resume)

            instance = MagicMock()
            instance.run_standalone.side_effect = (
                lambda steps: outer._standalone_calls.append(steps)
            )
            return instance

        _mock_rs_class.side_effect = fake_constructor

    # ------------------------------------------------------------------
    # Test 1 — normal fork: correct args dispatched, run_standalone called
    # ------------------------------------------------------------------
    def test_fork_dispatches_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            _utils_mod.fs_storage = tmp  # target folder does NOT exist yet

            run_headless.main(["--origin", "base_sim", "--target", "new_sim",
                               "--steps", "50"])

            self.assertEqual(self._init_kwargs["fork_sim_code"], "base_sim")
            self.assertEqual(self._init_kwargs["sim_code"],      "new_sim")
            self.assertFalse(self._init_kwargs["resume"])
            self.assertEqual(self._standalone_calls, [50])

    # ------------------------------------------------------------------
    # Test 2 — pre-existing target without --resume must exit non-zero
    # ------------------------------------------------------------------
    def test_existing_target_without_resume_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "existing_sim")
            os.makedirs(existing)
            _utils_mod.fs_storage = tmp

            with self.assertRaises(SystemExit) as ctx:
                run_headless.main(["--origin", "base_sim", "--target", "existing_sim",
                                   "--steps", "10"])

            self.assertNotEqual(ctx.exception.code, 0)
            # ReverieServer must never have been constructed
            self.assertEqual(self._standalone_calls, [])

    # ------------------------------------------------------------------
    # Test 3 — --resume: resume=True forwarded, existence check skipped
    # ------------------------------------------------------------------
    def test_resume_flag_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Target folder absent — but --resume should skip the check
            _utils_mod.fs_storage = tmp

            run_headless.main(["--origin", "base_sim", "--target", "my_run",
                               "--steps", "5", "--resume"])

            self.assertEqual(self._init_kwargs["fork_sim_code"], "base_sim")
            self.assertEqual(self._init_kwargs["sim_code"],      "my_run")
            self.assertTrue(self._init_kwargs["resume"])
            self.assertEqual(self._standalone_calls, [5])

    # ------------------------------------------------------------------
    # Test 4 — --steps is parsed as int
    # ------------------------------------------------------------------
    def test_steps_parsed_as_int(self):
        with tempfile.TemporaryDirectory() as tmp:
            _utils_mod.fs_storage = tmp

            run_headless.main(["--origin", "o", "--target", "t", "--steps", "999"])

            self.assertEqual(self._standalone_calls, [999])
            self.assertIsInstance(self._standalone_calls[0], int)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromTestCase(TestRunHeadless))
    sys.exit(0 if result.wasSuccessful() else 1)
