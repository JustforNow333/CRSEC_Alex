"""
Standalone encoding tests for the file helpers touched by the utf-8 fix.

Simulates Windows by forcing every open() that does NOT pass an explicit
encoding to use cp1252 — so a regression (dropping encoding="utf-8") fails
here even when the suite runs on Linux/macOS.

Run from the repo root: python test_encoding_roundtrip.py
"""
import builtins, io, os, shutil, sys, tempfile, types, unittest

# Text that cp1252 cannot represent:
#   "é" encodes in cp1252 (single byte) but decodes wrongly as utf-8;
#   "🎉" has no cp1252 mapping at all and raises UnicodeEncodeError.
SAMPLE = "café 🎉 Mère Noël"

# ---------------------------------------------------------------------------
# Windows simulation: default-encoding open() becomes cp1252.
# ---------------------------------------------------------------------------
_real_open = builtins.open

def _cp1252_default_open(file, mode="r", *a, **kw):
    if "b" not in mode and kw.get("encoding") is None and len(a) < 2:
        kw["encoding"] = "cp1252"
    return _real_open(file, mode, *a, **kw)


class _Cp1252World:
    """Context manager: inside it, bare open() behaves like Windows."""
    def __enter__(self):
        builtins.open = _cp1252_default_open
        return self
    def __exit__(self, *exc):
        builtins.open = _real_open
        return False


# ---------------------------------------------------------------------------
# Stub utils so 'from utils import *' in global_methods doesn't fail.
# ---------------------------------------------------------------------------
utils_stub = types.ModuleType("utils")
utils_stub.openai_api_key = "test-key"
sys.modules.setdefault("utils", utils_stub)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "reverie/backend_server"))
import global_methods as gm

# simulation_logger.py instantiates a module-level SimulationLogger() at import
# time, which TRUNCATES ./simulation_output.txt in the current directory. Import
# it from a scratch cwd so running the tests never clobbers the repo's copy.
_import_cwd = os.getcwd()
_scratch = tempfile.mkdtemp(prefix="enc_test_import_")
try:
    os.chdir(_scratch)
    import simulation_logger as sl
finally:
    os.chdir(_import_cwd)
    shutil.rmtree(_scratch, ignore_errors=True)


class TestEncodingRoundTrip(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="enc_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Test 0 — the harness itself really does simulate cp1252
    # ------------------------------------------------------------------
    def test_harness_actually_breaks_on_cp1252(self):
        path = os.path.join(self.tmp, "control.txt")
        with _Cp1252World():
            with self.assertRaises(UnicodeEncodeError,
                    msg="control: a bare open() must fail under cp1252"):
                with open(path, "w") as f:     # no encoding -> cp1252
                    f.write(SAMPLE)
        print("PASS test_harness_actually_breaks_on_cp1252")

    # ------------------------------------------------------------------
    # Test 1 — csv write/read helpers round-trip under cp1252
    # ------------------------------------------------------------------
    def test_csv_helpers_roundtrip(self):
        path = os.path.join(self.tmp, "rows.csv")
        rows = [["name", "note"], ["Mère", SAMPLE]]
        with _Cp1252World():
            gm.write_list_of_list_to_csv(rows, path)
            back = gm.read_file_to_list(path, header=False)
        # NOTE: the csv helpers don't pass newline="", so on Windows the writer
        # emits \r\r\n and every other row reads back blank. That's a separate
        # pre-existing bug; filter it out so this test stays about encoding.
        back = [r for r in back if r]
        self.assertEqual(back, rows, "csv round-trip must preserve non-ASCII")
        print("PASS test_csv_helpers_roundtrip")

    # ------------------------------------------------------------------
    # Test 2 — incremental append helper round-trips under cp1252
    # ------------------------------------------------------------------
    def test_csv_append_line_roundtrip(self):
        path = os.path.join(self.tmp, "append.csv")
        with _Cp1252World():
            gm.write_list_to_csv_line(["header", "col"], path)
            gm.write_list_to_csv_line(["row", SAMPLE], path)
            back = [r for r in gm.read_file_to_list(path, header=False) if r]
        self.assertEqual(back[-1], ["row", SAMPLE],
            "appended non-ASCII row must read back intact")
        print("PASS test_csv_append_line_roundtrip")

    # ------------------------------------------------------------------
    # Test 3 — SimulationLogger: the live UnicodeEncodeError hazard
    #
    # original_stdout is swapped for a StringIO so this exercises the LOG FILE
    # write ('w' in __init__, 'a' in write) — the path this fix covers. The
    # real console stdout is cp1252 on Windows and still raises on emoji;
    # that is a separate fix (sys.stdout.reconfigure) and is NOT covered here.
    # ------------------------------------------------------------------
    def test_simulation_logger_writes_non_ascii(self):
        path = os.path.join(self.tmp, "sim_out.txt")
        with _Cp1252World():
            logger = sl.SimulationLogger(path)      # 'w' at __init__
            logger.original_stdout = io.StringIO()  # isolate the file path
            logger.write(SAMPLE + "\n")             # 'a' per write
            logger.flush()
        with _real_open(path, encoding="utf-8") as f:
            body = f.read()
        self.assertIn(SAMPLE, body,
            "persona utterances with emoji/accents must survive the log writer")
        print("PASS test_simulation_logger_writes_non_ascii")

    # ------------------------------------------------------------------
    # Test 4 — check_if_file_exists must not be fooled by encoding
    # ------------------------------------------------------------------
    def test_check_if_file_exists_on_non_ascii_content(self):
        path = os.path.join(self.tmp, "exists.txt")
        with _real_open(path, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        with _Cp1252World():
            self.assertTrue(gm.check_if_file_exists(path),
                "a utf-8 file with emoji must still report as existing")
        print("PASS test_check_if_file_exists_on_non_ascii_content")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEncodingRoundTrip)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
