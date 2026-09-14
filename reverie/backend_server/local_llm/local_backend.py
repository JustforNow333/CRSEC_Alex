"""
local_backend.py -- route CRSEC's LLM calls to a local, open-weight model.

CRSEC talks to OpenAI through the legacy `openai` 0.27.0 SDK in four places:

    reverie/backend_server/persona/prompt_template/gpt_structure.py
        openai.ChatCompletion.create   (5 call sites, model hardcoded "gpt-4o-mini")
        openai.Completion.create       (1 call site, model from gpt_parameter["engine"])
        openai.Embedding.create        (1 call site, "text-embedding-ada-002")
    reverie/backend_server/norm/creation.py
        openai.ChatCompletion.create   (norm creation)
    initialization/initialization.py
        new-SDK OpenAI() client, with a legacy fallback
    analysis pipeline/LLM-evaluator.py
        openai.ChatCompletion.create   (the measurement pipeline)

Every one of those reaches the SDK through the *module object* (`openai.X.create`),
never through `from openai import X`. So patching the module attributes once, early,
redirects all of them without editing any of the call sites.

ACTIVATION IS OPT-IN. If the environment variable CRSEC_LLM_BACKEND is not set to
"local", activate() returns immediately and the code path is byte-for-byte the
original OpenAI behaviour. Nothing here can change results on a run that does not
ask for it.

Usage:
    import local_backend
    local_backend.activate()          # no-op unless CRSEC_LLM_BACKEND=local

Author: Alexander Maiello (acm357@cornell.edu)
"""

import json
import os
import sys
import threading

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULTS = {
    # Any OpenAI-compatible server. Ollama defaults to 11434, vLLM and
    # llama.cpp's llama-server to 8000/8080. The /v1 suffix is required.
    "base_url": "http://localhost:11434/v1",
    "api_key": "local",  # required by the SDK, ignored by local servers

    # The open-weight model that answers every generation call.
    "chat_model": "qwen3.5:0.8b",

    # The open-weight model that answers every embedding call.
    "embed_model": "nomic-embed-text",

    # Zero-pad embeddings up to this width. See _pad() for why this is safe
    # and why it matters. Set to 0 to disable and use native width.
    "pad_embeddings_to": 1536,

    # Hard ceiling on max_tokens. creation.py asks for 4096, which exceeds the
    # context window of small models once the prompt is counted. 0 = no clamp.
    "max_tokens_cap": 1024,

    # Seconds. Local CPU inference is far slower than a hosted API.
    "request_timeout": 600,

    # Write one JSON line per call to this path. Empty string disables.
    "call_log": "",
}

CONFIG_ENV = "CRSEC_LOCAL_CONFIG"
ENABLE_ENV = "CRSEC_LLM_BACKEND"

_state = {"active": False, "config": None, "embed_dim": None}
_lock = threading.Lock()

# Counters, so a run can report what actually happened.
STATS = {
    "chat_calls": 0,
    "completion_calls": 0,
    "embedding_calls": 0,
    "errors": 0,
}


def load_config():
    """Config file, overridden by CRSEC_LOCAL_* environment variables."""
    cfg = dict(DEFAULTS)

    path = os.environ.get(CONFIG_ENV, "")
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "local_models.json")
        if os.path.exists(candidate):
            path = candidate
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    for key in DEFAULTS:
        env_key = "CRSEC_LOCAL_" + key.upper()
        if env_key in os.environ:
            raw = os.environ[env_key]
            if isinstance(DEFAULTS[key], int):
                cfg[key] = int(raw)
            else:
                cfg[key] = raw
    return cfg


def config():
    if _state["config"] is None:
        _state["config"] = load_config()
    return _state["config"]


def is_active():
    return _state["active"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _pad(vector, width):
    """
    Zero-pad an embedding to `width`.

    Cosine similarity is invariant under zero-padding: appending zeros changes
    neither the dot product nor either L2 norm, so cos_sim(pad(a), pad(b)) ==
    cos_sim(a, b) exactly. Retrieval scores are therefore unaffected.

    It is worth doing because gpt_structure.get_embedding() hardcodes a
    1536-wide zero vector as its error fallback. A 768-wide local model would
    otherwise mix widths inside one run the moment a single embedding call
    failed, and retrieve.cos_sim would raise on the mismatch mid-simulation.
    Padding to the same 1536 removes that failure mode.
    """
    if not width or len(vector) >= width:
        return vector
    return list(vector) + [0.0] * (width - len(vector))


def _clamp_tokens(kwargs, cap):
    if cap and kwargs.get("max_tokens") and kwargs["max_tokens"] > cap:
        kwargs["max_tokens"] = cap
    return kwargs


def _log_call(kind, model, ok, detail=""):
    path = config()["call_log"]
    if not path:
        return
    rec = {"kind": kind, "model": model, "ok": ok, "detail": detail}
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class _LegacyChoice(object):
    """Mimics the attribute access of a legacy Completion choice."""

    def __init__(self, text):
        self.text = text
        self.finish_reason = "stop"
        self.index = 0

    def __getitem__(self, key):
        return getattr(self, key)


class _LegacyCompletion(object):
    """Mimics openai.Completion.create()'s return shape (.choices[0].text)."""

    def __init__(self, text):
        self.choices = [_LegacyChoice(text)]

    def __getitem__(self, key):
        return getattr(self, key)


# --------------------------------------------------------------------------
# Activation
# --------------------------------------------------------------------------

def activate(force=False, verbose=True):
    """
    Patch the openai module to point at the local server.

    Returns True if the local backend is now active, False if it was skipped
    because CRSEC_LLM_BACKEND is not "local".
    """
    if not force and os.environ.get(ENABLE_ENV, "").strip().lower() != "local":
        return False
    if _state["active"]:
        return True

    cfg = config()

    try:
        import openai
    except ImportError:
        raise RuntimeError("local_backend: the openai package is not installed")

    if not hasattr(openai, "ChatCompletion"):
        raise RuntimeError(
            "local_backend: this is openai >= 1.0, but CRSEC pins openai==0.27.0. "
            "Run: pip install 'openai==0.27.0'"
        )

    openai.api_key = cfg["api_key"]
    openai.api_base = cfg["base_url"]
    # Covers initialization.py, which builds a modern OpenAI() client when the
    # new SDK happens to be importable. The modern client reads this variable.
    os.environ.setdefault("OPENAI_BASE_URL", cfg["base_url"])
    os.environ.setdefault("OPENAI_API_KEY", cfg["api_key"])

    orig_chat = openai.ChatCompletion.create
    orig_embed = openai.Embedding.create

    def patched_chat(*args, **kwargs):
        kwargs.pop("engine", None)
        kwargs["model"] = cfg["chat_model"]
        # The analysis pipeline's evaluator uses the newer parameter name.
        # Local servers speak the older one.
        if "max_completion_tokens" in kwargs:
            kwargs.setdefault("max_tokens", kwargs.pop("max_completion_tokens"))
            kwargs.pop("max_completion_tokens", None)
        _clamp_tokens(kwargs, cfg["max_tokens_cap"])
        kwargs.setdefault("request_timeout", cfg["request_timeout"])
        # Not universally supported by local servers; harmless to drop.
        kwargs.pop("logprobs", None)
        kwargs.pop("logit_bias", None)
        STATS["chat_calls"] += 1
        try:
            out = orig_chat(*args, **kwargs)
            _log_call("chat", cfg["chat_model"], True)
            return out
        except Exception as exc:
            STATS["errors"] += 1
            _log_call("chat", cfg["chat_model"], False, repr(exc))
            raise

    def patched_completion(*args, **kwargs):
        """
        Legacy text completion, re-expressed as a chat completion.

        Two reasons not to pass this through to /v1/completions. First, CRSEC
        still asks for "text-davinci-003" in two gpt_param dicts, a model that
        no longer exists anywhere. Second, Ollama applies the model's chat
        template to /v1/completions requests anyway, so the "raw completion"
        framing is not actually preserved by the local server. Routing through
        chat makes the behaviour explicit and identical across servers.

        `stop` is passed through unchanged. CRSEC's prompt validators depend on
        the stop sequences, so dropping them would break parsing.
        """
        prompt = kwargs.pop("prompt", "")
        if isinstance(prompt, (list, tuple)):
            prompt = "".join(str(p) for p in prompt)
        kwargs.pop("engine", None)
        kwargs.pop("model", None)
        kwargs.pop("stream", None)
        kwargs.pop("logprobs", None)

        passthrough = {}
        for key in ("temperature", "max_tokens", "top_p",
                    "frequency_penalty", "presence_penalty", "stop"):
            if key in kwargs and kwargs[key] is not None:
                passthrough[key] = kwargs[key]
        _clamp_tokens(passthrough, cfg["max_tokens_cap"])

        STATS["completion_calls"] += 1
        try:
            resp = orig_chat(
                model=cfg["chat_model"],
                messages=[{"role": "user", "content": prompt}],
                request_timeout=cfg["request_timeout"],
                **passthrough
            )
            text = resp["choices"][0]["message"]["content"]
            _log_call("completion", cfg["chat_model"], True)
            return _LegacyCompletion(text)
        except Exception as exc:
            STATS["errors"] += 1
            _log_call("completion", cfg["chat_model"], False, repr(exc))
            raise

    def patched_embedding(*args, **kwargs):
        kwargs.pop("engine", None)
        kwargs["model"] = cfg["embed_model"]
        kwargs.setdefault("request_timeout", cfg["request_timeout"])
        STATS["embedding_calls"] += 1
        try:
            resp = orig_embed(*args, **kwargs)
        except Exception as exc:
            STATS["errors"] += 1
            _log_call("embedding", cfg["embed_model"], False, repr(exc))
            raise

        width = cfg["pad_embeddings_to"]
        for item in resp["data"]:
            native = len(item["embedding"])
            if _state["embed_dim"] is None:
                _state["embed_dim"] = native
            item["embedding"] = _pad(item["embedding"], width)
        _log_call("embedding", cfg["embed_model"], True)
        return resp

    openai.ChatCompletion.create = patched_chat
    openai.Completion.create = patched_completion
    openai.Embedding.create = patched_embedding

    _state["active"] = True

    if verbose:
        sys.stderr.write(
            "[local_backend] active -> %s | chat=%s | embed=%s\n"
            % (cfg["base_url"], cfg["chat_model"], cfg["embed_model"])
        )
    return True


def summary():
    """One-line report of what this process sent to the local server."""
    cfg = config()
    return (
        "local_backend: chat=%d completion=%d embedding=%d errors=%d "
        "| chat_model=%s embed_model=%s native_embed_dim=%s"
        % (STATS["chat_calls"], STATS["completion_calls"],
           STATS["embedding_calls"], STATS["errors"],
           cfg["chat_model"], cfg["embed_model"], _state["embed_dim"])
    )
