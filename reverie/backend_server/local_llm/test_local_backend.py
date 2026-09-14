"""
test_local_backend.py -- offline tests for the local-model backend.

No server, no model, no network, no API key. Runs in about a second.

    python test_local_backend.py
    python -m unittest test_local_backend -v

The point of these is the safety property that matters most for the project:
with CRSEC_LLM_BACKEND unset, nothing is patched and the OpenAI path is
untouched. Astghik's runs cannot be affected by this code being present.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_backend  # noqa: E402


def reset():
    local_backend._state.update({"active": False, "config": None, "embed_dim": None})
    for k in local_backend.STATS:
        local_backend.STATS[k] = 0


class FakeChat(object):
    """Records what the shim actually sends upstream."""

    def __init__(self, content="ok"):
        self.calls = []
        self.content = content

    def create(self, *args, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"role": "assistant",
                                         "content": self.content}}]}


class FakeEmbed(object):
    def __init__(self, dim=768):
        self.calls = []
        self.dim = dim

    def create(self, *args, **kwargs):
        self.calls.append(kwargs)
        n = len(kwargs.get("input", [""]))
        return {"data": [{"embedding": [0.5] * self.dim, "index": i}
                         for i in range(n)]}


class OptIn(unittest.TestCase):

    def setUp(self):
        reset()
        os.environ.pop("CRSEC_LLM_BACKEND", None)

    def test_inert_without_env_var(self):
        """The whole point: absent the flag, nothing is patched."""
        import openai
        # classmethod access rebinds each time, so compare the underlying func
        before = openai.ChatCompletion.create.__func__
        self.assertFalse(local_backend.activate())
        self.assertIs(openai.ChatCompletion.create.__func__, before)
        self.assertFalse(local_backend.is_active())

    def test_wrong_env_value_is_inert(self):
        os.environ["CRSEC_LLM_BACKEND"] = "openai"
        self.assertFalse(local_backend.activate())
        os.environ.pop("CRSEC_LLM_BACKEND")


class Padding(unittest.TestCase):

    def test_zero_padding_preserves_cosine_similarity(self):
        """
        The justification for padding. Cosine similarity must be identical
        before and after, or retrieval scores would shift and the local runs
        would not be comparable to anything.
        """
        import math

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb)

        a = [0.3, -0.7, 0.1, 0.9]
        b = [-0.2, 0.4, 0.8, -0.5]
        pa = local_backend._pad(a, 1536)
        pb = local_backend._pad(b, 1536)
        self.assertEqual(len(pa), 1536)
        self.assertAlmostEqual(cos(a, b), cos(pa, pb), places=12)

    def test_padding_is_a_noop_when_already_wide_enough(self):
        v = [0.1] * 1536
        self.assertEqual(len(local_backend._pad(v, 1536)), 1536)

    def test_padding_disabled(self):
        v = [0.1] * 768
        self.assertEqual(len(local_backend._pad(v, 0)), 768)


class Patching(unittest.TestCase):

    def setUp(self):
        reset()
        os.environ["CRSEC_LLM_BACKEND"] = "local"
        os.environ["CRSEC_LOCAL_CHAT_MODEL"] = "tiny-chat"
        os.environ["CRSEC_LOCAL_EMBED_MODEL"] = "tiny-embed"
        os.environ["CRSEC_LOCAL_MAX_TOKENS_CAP"] = "1024"
        self.chat = FakeChat()
        self.embed = FakeEmbed(dim=768)
        import openai
        self.openai = openai
        self.saved = (openai.ChatCompletion.create, openai.Completion.create,
                      openai.Embedding.create, openai.api_base, openai.api_key)
        openai.ChatCompletion.create = self.chat.create
        openai.Embedding.create = self.embed.create
        local_backend.activate(verbose=False)

    def tearDown(self):
        (self.openai.ChatCompletion.create, self.openai.Completion.create,
         self.openai.Embedding.create, self.openai.api_base,
         self.openai.api_key) = self.saved
        for k in ("CRSEC_LLM_BACKEND", "CRSEC_LOCAL_CHAT_MODEL",
                  "CRSEC_LOCAL_EMBED_MODEL", "CRSEC_LOCAL_MAX_TOKENS_CAP"):
            os.environ.pop(k, None)
        reset()

    def test_hardcoded_chat_model_is_remapped(self):
        self.openai.ChatCompletion.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(self.chat.calls[-1]["model"], "tiny-chat")

    def test_dead_davinci_engine_is_remapped(self):
        """text-davinci-003 appears in two gpt_param dicts and no longer exists."""
        self.openai.Completion.create(
            model="text-davinci-003", prompt="hi", temperature=0.8,
            max_tokens=15, top_p=1, frequency_penalty=0, presence_penalty=0,
            stream=False, stop=["\n"])
        self.assertEqual(self.chat.calls[-1]["model"], "tiny-chat")

    def test_legacy_completion_returns_legacy_shape(self):
        """GPT_request reads response.choices[0].text by attribute."""
        r = self.openai.Completion.create(model="x", prompt="hi", stream=False)
        self.assertEqual(r.choices[0].text, "ok")
        self.assertEqual(r["choices"][0]["text"], "ok")

    def test_stop_sequences_survive(self):
        """CRSEC's validators depend on stop tokens; dropping them breaks parsing."""
        self.openai.Completion.create(model="x", prompt="hi", stop=["\n"],
                                      stream=False, temperature=0.8)
        self.assertEqual(self.chat.calls[-1]["stop"], ["\n"])

    def test_stream_is_stripped(self):
        self.openai.Completion.create(model="x", prompt="hi", stream=False)
        self.assertNotIn("stream", self.chat.calls[-1])

    def test_max_tokens_clamped(self):
        """creation.py asks for 4096, past a small model's usable window."""
        self.openai.ChatCompletion.create(
            model="gpt-4o-mini", messages=[], max_tokens=4096)
        self.assertEqual(self.chat.calls[-1]["max_tokens"], 1024)

    def test_max_tokens_under_cap_untouched(self):
        self.openai.ChatCompletion.create(
            model="gpt-4o-mini", messages=[], max_tokens=50)
        self.assertEqual(self.chat.calls[-1]["max_tokens"], 50)

    def test_embedding_model_remapped_and_padded(self):
        r = self.openai.Embedding.create(input=["hello"],
                                         model="text-embedding-ada-002")
        self.assertEqual(self.embed.calls[-1]["model"], "tiny-embed")
        v = r["data"][0]["embedding"]
        self.assertEqual(len(v), 1536)
        self.assertEqual(local_backend._state["embed_dim"], 768)

    def test_padded_width_matches_hardcoded_error_fallback(self):
        """
        gpt_structure.get_embedding returns [0.0] * 1536 when a call fails.
        If our width differed, one failed embedding would poison cos_sim for
        the rest of the run.
        """
        r = self.openai.Embedding.create(input=["hello"], model="whatever")
        self.assertEqual(len(r["data"][0]["embedding"]), 1536)

    def test_counters(self):
        self.openai.ChatCompletion.create(model="m", messages=[])
        self.openai.Completion.create(model="m", prompt="p")
        self.openai.Embedding.create(input=["e"], model="m")
        # The completion path calls the *original* chat function, so the two
        # counters stay separate and each call is counted exactly once.
        self.assertEqual(local_backend.STATS["chat_calls"], 1)
        self.assertEqual(local_backend.STATS["completion_calls"], 1)
        self.assertEqual(local_backend.STATS["embedding_calls"], 1)


class FailSafeMonitor(unittest.TestCase):

    def test_counts_fail_safe_returns(self):
        import types
        import fail_safe_monitor as fsm
        fsm._counts.clear()
        fsm._attached = False

        module = types.ModuleType("fake_gpt_structure")

        def safe_generate_response(prompt, gpt_parameter, repeat=5,
                                   fail_safe_response="error",
                                   func_validate=None, func_clean_up=None,
                                   verbose=False):
            # Simulate a model that only sometimes produces parseable output.
            if "good" in prompt:
                return 7
            return fail_safe_response

        module.safe_generate_response = safe_generate_response
        fsm.attach(module)

        module.safe_generate_response("good prompt", {}, 5, 8, None, None)
        module.safe_generate_response("bad prompt", {}, 5, 8, None, None)
        module.safe_generate_response("bad prompt 2", {}, 5, 8, None, None)

        s = fsm.stats()
        self.assertEqual(s["total_calls"], 3)
        self.assertEqual(s["fail_safe_returns"], 2)
        self.assertAlmostEqual(s["fail_safe_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
