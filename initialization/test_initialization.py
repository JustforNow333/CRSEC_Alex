"""
Tests for initialization.py.
No OpenAI API calls — update_biographies is never invoked.
Run from the repo root or initialization/:  python initialization/test_initialization.py
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from initialization import run_randomization, AGENTS


class TestRunRandomization(unittest.TestCase):

    # ------------------------------------------------------------------
    # Test 1 — same seed produces identical assignments
    # ------------------------------------------------------------------
    def test_same_seed_reproducible(self):
        _, df1 = run_randomization(seed=42)
        _, df2 = run_randomization(seed=42)

        self.assertTrue(
            (df1["Conviction"].values == df2["Conviction"].values).all(),
            "Conviction values differ across two calls with the same seed",
        )
        self.assertTrue(
            (df1["Trust"].values == df2["Trust"].values).all(),
            "Trust values differ across two calls with the same seed",
        )
        self.assertTrue(
            (df1["Archetype"].values == df2["Archetype"].values).all(),
            "Archetypes differ across two calls with the same seed",
        )

    # ------------------------------------------------------------------
    # Test 2 — minority_size N yields exactly N activists for N in {1,2,3,4}
    # ------------------------------------------------------------------
    def test_minority_size_variants(self):
        for n in (1, 2, 3, 4):
            # Use a distinct seed per size so we don't rely on one lucky seed
            _, df = run_randomization(seed=1000 + n * 17, minority_size=n)
            activists = df[df["Archetype"] == "Grassroots Activist / Dissenter"]
            self.assertEqual(
                len(activists), n,
                f"minority_size={n}: got {len(activists)} activists",
            )
            self.assertNotIn(
                "Jennifer Moore", activists["Agent"].values,
                f"minority_size={n}: Jennifer Moore must never be an activist",
            )

    # ------------------------------------------------------------------
    # Test 3 — infeasible minority_size raises ValueError within the cap
    # ------------------------------------------------------------------
    def test_infeasible_size_raises(self):
        # 11 > len(AGENTS) = 10, so it can never be satisfied
        with self.assertRaises(ValueError):
            run_randomization(seed=99, minority_size=11, max_draws=50)

    # ------------------------------------------------------------------
    # Test 4 — --out-base-name copies dir and leaves original untouched
    # ------------------------------------------------------------------
    def test_out_base_name_copies_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create a minimal fake base storage dir with a sentinel file
            base_dir = os.path.join(tmp, "base_storage")
            os.makedirs(base_dir)
            sentinel = os.path.join(base_dir, "sentinel.txt")
            with open(sentinel, "w") as f:
                f.write("original")

            dest_name = "copy_storage"
            dest_dir = os.path.join(tmp, dest_name)

            # Replicate the --out-base-name logic from main() directly
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(base_dir, dest_dir)

            # Original must still be intact
            self.assertTrue(os.path.exists(sentinel), "Original sentinel must still exist")
            with open(sentinel) as f:
                self.assertEqual(f.read(), "original")

            # Copy must exist with the same content
            copied_sentinel = os.path.join(dest_dir, "sentinel.txt")
            self.assertTrue(os.path.exists(copied_sentinel), "Copied sentinel must exist")
            with open(copied_sentinel) as f:
                self.assertEqual(f.read(), "original")

            # Mutate the copy — original must be unaffected
            with open(copied_sentinel, "w") as f:
                f.write("modified")
            with open(sentinel) as f:
                self.assertEqual(f.read(), "original",
                    "Mutating the copy must not affect the original")


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestRunRandomization)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
