"""
mock_server.py -- a fake OpenAI-compatible server.

Serves /v1/models, /v1/chat/completions, /v1/completions and /v1/embeddings
with the same JSON shapes Ollama and vLLM return, but with canned answers and
no model behind it.

Its purpose is to test the *plumbing* -- that CRSEC's calls reach a local
endpoint, in the right shape, and that the responses parse -- separately from
testing whether a small model is smart enough to answer well. Those are two
different failures and they are much easier to debug apart than together.

    python mock_server.py --port 11500

Then point the backend at it:

    $env:CRSEC_LOCAL_BASE_URL = "http://localhost:11500/v1"

Deterministic: the same prompt always produces the same answer, so it is also
usable as a fixed reference point when checking that a code change did not
alter the sequence of calls the simulation makes.
"""

import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

EMBED_DIM = 768  # deliberately not 1536, to exercise the padding path


def fake_embedding(text, dim=EMBED_DIM):
    """Deterministic pseudo-embedding derived from a hash of the text."""
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    counter = 0
    while len(vals) < dim:
        chunk = hashlib.sha256(seed + str(counter).encode()).digest()
        vals.extend((b - 127.5) / 127.5 for b in chunk)
        counter += 1
    return vals[:dim]


def fake_answer(prompt):
    """
    Canned answers keyed off recognisable CRSEC prompt shapes, so preflight can
    check that real responses parse. Everything else gets a generic reply.
    """
    low = prompt.lower()
    if "wake up hour" in low:
        return "7 am"
    if "convert an action description to a string of 3 emojis" in low or "emoji" in low:
        return "🛏️😴"
    if "output format: (<subject>, <predicate>, <object>)" in low or "predicate" in low:
        return "(sleeping, in bed)"
    if low.rstrip().endswith("answer in yes or no:") or "yes or no" in low:
        return "yes"
    return "This is a mock response."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send({"object": "list", "data": [
                {"id": "mock-chat", "object": "model", "owned_by": "mock"},
                {"id": "mock-embed", "object": "model", "owned_by": "mock"},
            ]})
        else:
            self._send({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send({"error": {"message": "bad json"}}, 400)
            return

        path = self.path.rstrip("/")
        model = req.get("model", "mock")

        if path.endswith("/chat/completions"):
            prompt = "\n".join(m.get("content", "") for m in req.get("messages", []))
            text = fake_answer(prompt)
            self._send({
                "id": "chatcmpl-mock", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        elif path.endswith("/completions"):
            text = fake_answer(req.get("prompt", ""))
            self._send({
                "id": "cmpl-mock", "object": "text_completion", "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        elif path.endswith("/embeddings"):
            inputs = req.get("input", "")
            if isinstance(inputs, str):
                inputs = [inputs]
            self._send({
                "object": "list", "model": model,
                "data": [{"object": "embedding", "index": i,
                          "embedding": fake_embedding(str(t))}
                         for i, t in enumerate(inputs)],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            })

        else:
            self._send({"error": {"message": "not found"}}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11500)
    args = ap.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print("mock OpenAI-compatible server on http://127.0.0.1:%d/v1" % args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
