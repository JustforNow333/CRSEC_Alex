"""
Standalone cache tests for get_embedding.
Monkeypatches openai.Embedding.create — no real API calls made.
Run from the repo root: python test_embedding_cache.py
"""
import sys, types, unittest

# ---------------------------------------------------------------------------
# Minimal stubs so gpt_structure.py imports without the real openai package
# or utils.py being present.
# ---------------------------------------------------------------------------

# Stub openai module with the error classes we need
openai_stub = types.ModuleType("openai")

class _RateLimitError(Exception): pass
class _ServiceUnavailableError(Exception): pass

error_mod = types.ModuleType("openai.error")
error_mod.RateLimitError = _RateLimitError
error_mod.ServiceUnavailableError = _ServiceUnavailableError
openai_stub.error = error_mod
openai_stub.api_key = None

# Stub Embedding.create — will be replaced per test
class _Embedding:
    @staticmethod
    def create(**kwargs):
        raise NotImplementedError("replace me in each test")

openai_stub.Embedding = _Embedding
sys.modules["openai"] = openai_stub
sys.modules["openai.error"] = error_mod

# Stub utils so 'from utils import *' doesn't fail
utils_stub = types.ModuleType("utils")
utils_stub.openai_api_key = "test-key"
sys.modules["utils"] = utils_stub

# ---------------------------------------------------------------------------
# Now import the module under test.
# ---------------------------------------------------------------------------
import importlib, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "reverie/backend_server/persona/prompt_template"))
import gpt_structure as gs


class TestEmbeddingCache(unittest.TestCase):

    def setUp(self):
        # Reset the shared cache before every test so tests are independent.
        gs._embedding_cache.clear()

    # ------------------------------------------------------------------
    # Test 1 — same string hits the cache on the second call
    # ------------------------------------------------------------------
    def test_same_string_cached(self):
        call_count = 0
        fixed_vector = [0.1] * 1536

        def mock_create(input, model):
            nonlocal call_count
            call_count += 1
            return {"data": [{"embedding": fixed_vector}]}

        openai_stub.Embedding.create = mock_create

        v1 = gs.get_embedding("hello world")
        v2 = gs.get_embedding("hello world")

        self.assertEqual(v1, fixed_vector, "first call should return fixed vector")
        self.assertEqual(v2, fixed_vector, "second call should return same vector")
        self.assertEqual(call_count, 1, "API should be called exactly once (second was cache hit)")
        print("PASS test_same_string_cached")

    # ------------------------------------------------------------------
    # Test 2 — newline-normalisation produces the same cache key
    # ------------------------------------------------------------------
    def test_normalization_same_key(self):
        call_count = 0
        fixed_vector = [0.2] * 1536

        def mock_create(input, model):
            nonlocal call_count
            call_count += 1
            return {"data": [{"embedding": fixed_vector}]}

        openai_stub.Embedding.create = mock_create

        v1 = gs.get_embedding("hello\nworld")   # contains newline
        v2 = gs.get_embedding("hello world")    # already has space

        self.assertEqual(v1, fixed_vector)
        self.assertEqual(v2, fixed_vector)
        self.assertEqual(call_count, 1,
            "both strings normalise to the same key — only one API call expected")
        print("PASS test_normalization_same_key")

    # ------------------------------------------------------------------
    # Test 3 — failure must NOT be cached; success after failure IS cached
    # ------------------------------------------------------------------
    def test_failure_not_cached_then_success_cached(self):
        call_count = 0
        real_vector = [0.3] * 1536

        # Phase A: always raise RateLimitError so retries exhaust
        def mock_always_fail(input, model):
            nonlocal call_count
            call_count += 1
            raise _RateLimitError("rate limited")

        openai_stub.Embedding.create = mock_always_fail

        fallback = gs.get_embedding("unique string xyz")

        self.assertEqual(fallback, [0.0] * 1536, "exhausted retries should return fallback")
        self.assertNotIn("unique string xyz", gs._embedding_cache,
            "fallback value must NOT be written to the cache")

        # Phase B: switch mock to succeed for the same key
        call_count = 0

        def mock_succeed(input, model):
            nonlocal call_count
            call_count += 1
            return {"data": [{"embedding": real_vector}]}

        openai_stub.Embedding.create = mock_succeed

        result = gs.get_embedding("unique string xyz")

        self.assertEqual(result, real_vector, "should return real vector on success")
        self.assertIn("unique string xyz", gs._embedding_cache,
            "real vector must now be written to the cache")
        self.assertEqual(gs._embedding_cache["unique string xyz"], real_vector)
        self.assertEqual(call_count, 1, "API called once in phase B")
        print("PASS test_failure_not_cached_then_success_cached")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEmbeddingCache)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
