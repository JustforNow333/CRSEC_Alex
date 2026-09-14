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
                  "CRSEC_LOCAL_EMBED_MODEL", "CRSEC_LOCAL_MAX_TOKENS_CAP",
                  "CRSEC_LOCAL_MAX_TOKENS_FLOOR", "CRSEC_LOCAL_STRIP_PROMPT_ECHO"):
            os.environ.pop(k, None)
        reset()

    def reactivate(self, **env):
        """
        Re-patch with different config.

        activate() captures whatever openai.ChatCompletion.create currently is
        as its upstream, so the fake has to be put back first; otherwise the
        new shim stacks on top of the old one and both adapt the same call.
        In a real run this cannot happen -- activate() returns early once
        _state["active"] is set -- but reset() here deliberately bypasses that.
        """
        os.environ.update(env)
        reset()
        self.openai.ChatCompletion.create = self.chat.create
        self.openai.Embedding.create = self.embed.create
        local_backend.activate(verbose=False)

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

    # -- response adapters -------------------------------------------------
    # Both change what CRSEC sees rather than only where the request goes, so
    # each one is pinned in both directions: that it fires when it should, and
    # that it leaves everything else alone.

    def test_max_tokens_floored(self):
        """
        run_gpt_prompt sets max_tokens=5 for wake_up_hour and 15 for eighteen
        other prompts. A chat-tuned model spends that on preamble.
        """
        self.openai.Completion.create(model="x", prompt="hi", max_tokens=5,
                                      stream=False)
        self.assertEqual(self.chat.calls[-1]["max_tokens"], 32)
        self.assertEqual(local_backend.STATS["token_floor_raises"], 1)

    def test_max_tokens_over_floor_untouched(self):
        self.openai.Completion.create(model="x", prompt="hi", max_tokens=150,
                                      stream=False)
        self.assertEqual(self.chat.calls[-1]["max_tokens"], 150)
        self.assertEqual(local_backend.STATS["token_floor_raises"], 0)

    def test_absent_max_tokens_is_not_invented(self):
        """A call that set no budget is not given one it never asked for."""
        self.openai.Completion.create(model="x", prompt="hi", stream=False)
        self.assertNotIn("max_tokens", self.chat.calls[-1])
        self.assertEqual(local_backend.STATS["token_floor_raises"], 0)

    def test_floor_disabled_by_config(self):
        self.reactivate(CRSEC_LOCAL_MAX_TOKENS_FLOOR="0")
        self.openai.Completion.create(model="x", prompt="hi", max_tokens=5,
                                      stream=False)
        self.assertEqual(self.chat.calls[-1]["max_tokens"], 5)

    def test_cap_still_wins_over_floor(self):
        """If the two ever cross, exceeding the context window is the worse
        failure, so the cap is applied last."""
        self.reactivate(CRSEC_LOCAL_MAX_TOKENS_CAP="8")
        self.openai.Completion.create(model="x", prompt="hi", max_tokens=5,
                                      stream=False)
        self.assertEqual(self.chat.calls[-1]["max_tokens"], 8)

    def test_prompt_echo_stripped(self):
        """The qwen2.5:7b failure: right answer, unparseable because echoed."""
        self.chat.content = "Output: (Mary Smith, draft, petition)"
        r = self.openai.Completion.create(
            model="x", prompt="Input: Mary Smith is drafting a petition\n"
                              "Output: (Mary Smith,", stream=False)
        self.assertEqual(r.choices[0].text, " draft, petition)")
        self.assertEqual(local_backend.STATS["echo_strips"], 1)
        # What run_gpt_prompt_event_triple's __func_clean_up does with it.
        parts = [i.strip() for i in r.choices[0].text.strip().split(")")[0].split(",")]
        self.assertEqual(parts, ["draft", "petition"])

    def test_echo_match_ignores_case_and_whitespace(self):
        self.chat.content = "output:(mary smith, draft, petition)"
        r = self.openai.Completion.create(
            model="x", prompt="Output: (Mary Smith,", stream=False)
        self.assertEqual(r.choices[0].text.strip(), "draft, petition)")

    def test_direct_answer_is_left_alone(self):
        """The common case must pass through byte-for-byte."""
        self.chat.content = " draft, petition)"
        r = self.openai.Completion.create(
            model="x", prompt="Output: (Mary Smith,", stream=False)
        self.assertEqual(r.choices[0].text, " draft, petition)")
        self.assertEqual(local_backend.STATS["echo_strips"], 0)

    def test_short_coincidental_overlap_is_not_stripped(self):
        """
        Some CRSEC prompts end in a bare "1)" that a correct answer also
        begins with. Below _MIN_ECHO that is treated as coincidence.
        """
        self.chat.content = "1) Wake up at 6am"
        r = self.openai.Completion.create(model="x", prompt="...list:\n1)",
                                          stream=False)
        self.assertEqual(r.choices[0].text, "1) Wake up at 6am")
        self.assertEqual(local_backend.STATS["echo_strips"], 0)

    def test_echo_strip_disabled_by_config(self):
        self.reactivate(CRSEC_LOCAL_STRIP_PROMPT_ECHO="0")
        self.chat.content = "Output: (Mary Smith, draft, petition)"
        r = self.openai.Completion.create(model="x", prompt="Output: (Mary Smith,",
                                          stream=False)
        self.assertEqual(r.choices[0].text, "Output: (Mary Smith, draft, petition)")
        self.assertEqual(local_backend.STATS["echo_strips"], 0)

    def test_chat_path_is_not_echo_stripped(self):
        """
        Only the completion path has a single prompt tail to match against.
        ChatCompletion callers (creation.py, the evaluator) pass structured
        messages and parse their own formats; guessing at an echo there would
        be a different change with different risks.
        """
        self.chat.content = "hi there"
        r = self.openai.ChatCompletion.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(r["choices"][0]["message"]["content"], "hi there")
        self.assertEqual(local_backend.STATS["echo_strips"], 0)

    def test_summary_reports_adaptation(self):
        self.chat.content = "Output: (Mary Smith, draft, petition)"
        self.openai.Completion.create(model="x", prompt="Output: (Mary Smith,",
                                      max_tokens=5, stream=False)
        line = local_backend.summary()
        self.assertIn("echo_strips=1", line)
        self.assertIn("token_floor_raises=1", line)

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
