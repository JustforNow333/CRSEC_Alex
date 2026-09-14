# Running CRSEC on local open-weight models

This replaces the OpenAI API with an open-weight model running on your own
machine. Nothing about the simulation changes: same prompts, same architecture,
same experiment. Only the thing answering the prompts changes.

The whole mechanism is opt-in. With the environment variable `CRSEC_LLM_BACKEND`
unset, every code path is byte-for-byte the original OpenAI behaviour, so this
cannot affect anyone else's runs.

---

## Read this part first

Getting the simulation to *run* locally is easy. Getting it to produce results
worth analysing is not, and the gap between those two things is where this
setup will mislead you if you let it.

CRSEC's `safe_generate_response` retries a prompt five times and, if the model
never produces output the validator accepts, returns a hardcoded default:

```python
for i in range(repeat):
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
return fail_safe_response          # silent
```

Nothing is logged. The simulation continues, finishes, and writes a complete
set of output files. A small model that cannot hold the required output format
will therefore produce a run that *looks* successful while being driven largely
by the fail-safe constants — wake up at 8, "is idle", the 😋 emoji — rather than
by any social reasoning.

So there are two separate questions, and they need two separate answers:

1. **Does the pipeline work?** Answered by `preflight.py` stages 1–5 and by
   `test_local_backend.py`. A 0.5B model is fine for this.
2. **Is the model good enough to run the experiment?** Answered by
   `preflight.py` stage 6 and by `fail_safe_monitor.py` on a real run. A 0.5B
   model is almost certainly *not* fine for this.

Prove (1) with the smallest model you can download. Establish (2) separately,
with a size sweep, before anything gets reported.

---

## 0. Where this fits in the repo

State of the fork as of this writing:

| Branch | Contents | Status |
|---|---|---|
| `main` | Astghik's baseline, `bcfad74` | Identical to upstream |
| `Alex_branch` | Working tree: speedup, tooling, Windows compat, `run_headless.py`, `run_sweep.py`, `.env` loading | Not a deliverable |
| `speedup-only` | `gpt_structure.py` + cache test | PR open |
| `tooling-replay-bench` | `replay.py`, `bench.py`, `aggregate.py` | PR open |

Upstream `main` is still at `bcfad74`, so neither PR has been merged. **Branch the
local backend off `main`, not off `Alex_branch`.** A local-model PR that also
carries the speedup and the tooling is three ideas in one branch, which is the
problem the branch-per-PR split already solved once.

Two pieces of existing tooling matter here and are used below rather than
duplicated:

- **`run_headless.py` / `run_sweep.py`** (`Alex_branch`) — a headless launcher
  and a parallel sweep supervisor driven by a `configs.json` of
  `{origin, target, steps}`. A model sweep should go through these.
- **`replay.py`** (`tooling-replay-bench`) — records every OpenAI request and
  response from a run to `recording.jsonl`. That recording turns out to be the
  best available benchmark for a local model, and `replay_validity.py` uses it.

### One conflict to know about

`replay.py` and `local_backend.py` patch the same three methods:

```python
setattr(getattr(openai, method), "create", ...)   # both do this
```

They compose correctly **in this order only**: `utils.py` activates the local
backend at import, then `replay.py` installs on top.

- *Recording a local run* works: the Recorder captures the shim's patched
  functions as its originals, so it records what the local server returned.
- *Replaying* works: the Replayer overwrites the shim entirely and serves from
  the recording with no network. The shim's counters go quiet, which is correct.
- *Activating the shim after the Replayer* would send replay traffic to the
  network. Do not import `local_backend` from inside a replay script.

---

## 1. What to download

### The inference server

Any server that speaks the OpenAI API works. Pick one:

| Option | Best for | Get it from |
|---|---|---|
| **Ollama** | Windows, first setup, CPU or single GPU | `https://ollama.com/download` |
| vLLM | Linux + NVIDIA GPU, running many sims in parallel | `pip install vllm` |
| llama.cpp (`llama-server`) | Maximum control over quantisation | `https://github.com/ggml-org/llama.cpp` |

**Use Ollama.** It is one installer on Windows, it manages model downloads, and
it exposes `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and
`/v1/models` on `http://localhost:11434/v1`, which is all CRSEC touches. Move to
vLLM later if throughput becomes the bottleneck; the shim does not care which
one is behind the port.

### The models

You need **two**: one that generates text, one that produces embeddings. CRSEC
uses both and they are not interchangeable.

Verify tags before pulling — the library moves. `ollama list` shows what you
have, `https://ollama.com/library` shows what is current.

**Generation model.** Start at the bottom for the proof of concept, then climb:

```powershell
ollama pull qwen2.5:0.5b      # ~400 MB  proof of concept only
ollama pull qwen3.5:0.8b      # ~1.0 GB  smallest with a real chance
ollama pull llama3.2:1b       # ~1.3 GB
ollama pull llama3.2:3b       # ~2.0 GB  first size that behaves
ollama pull qwen3.5:9b        # ~6 GB    realistic experiment floor
ollama pull gpt-oss:20b       # ~14 GB   needs ~16 GB VRAM
```

**Embedding model.** One is enough:

```powershell
ollama pull nomic-embed-text  # ~274 MB, 768 dimensions, the default
```

`mxbai-embed-large` and `qwen3-embedding:0.6b` are alternatives if retrieval
quality turns out to matter. Embedding models run fine on CPU; do not spend GPU
memory on them.

### Rough sizing

At 4-bit quantisation, budget about 0.6 GB of memory per billion parameters,
plus context overhead. Anything that does not fit in VRAM spills to system RAM
and runs several times slower — which matters here, because a single CRSEC day
is thousands of calls.

---

## 2. Install

```powershell
# 1. Install Ollama from https://ollama.com/download, then confirm it is serving
ollama --version
curl http://localhost:11434/v1/models

# 2. Models
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text

# 3. CRSEC dependencies, unchanged — the openai pin matters
cd C:\Users\burst\ResearchProject\crsec
pip install -r requirements.txt        # openai==0.27.0
```

The `openai==0.27.0` pin is load-bearing. CRSEC calls `openai.ChatCompletion`,
`openai.Completion` and `openai.Embedding`, all of which were removed in
openai 1.0. If you have 1.x in the environment the shim will say so and stop
rather than fail confusingly later.

---

## 3. Wire it up

Copy the `local_llm/` folder to `reverie/backend_server/local_llm/`, then make
one edit.

**Edit: `reverie/backend_server/utils.py`**, appended at the end.

Note that `Alex_branch` already modifies the top of this file to load `.env`
through `python-dotenv`. That is useful here: it means `CRSEC_LOCAL_*` settings
can live in `.env` alongside the API key rather than being re-exported into
every shell. (Separately: `python-dotenv` is imported there but is not in
`requirements.txt` — worth fixing, unrelated to this work.)

```python
# --- local open-weight backend (inert unless CRSEC_LLM_BACKEND=local) --------
try:
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_llm"))
    import local_backend
    local_backend.activate()
except Exception as _e:
    print("local_backend not activated:", _e)
```

That is the entire change to the simulation. It works because every backend
module — `gpt_structure.py`, `norm/creation.py`, `reverie.py`, and the rest —
reaches the SDK through the module object (`openai.ChatCompletion.create`) and
imports `from utils import *`. Patching the module attributes once, early,
redirects all of them without touching a single call site.

Two scripts sit outside that import graph and need the same block if you want
them local too:

- `initialization/initialization.py` — assigns the conviction/trust traits
- `analysis pipeline/LLM-evaluator.py` — scores dialogues for urgency and trust

**On the evaluator, think before you switch it.** It is the measuring
instrument. If you move the simulation to an open-weight model *and* move the
scorer to one at the same time, you have changed the phenomenon and the ruler in
the same step, and a shift in results will not be attributable to either. The
defensible default is to keep the evaluator on the same model as the original
study so the local runs stay comparable to the existing ones, and to validate a
local evaluator separately by scoring the *same* dialogues with both and
reporting the agreement.

---

## 4. Configure

`local_llm/local_models.json`:

```json
{
  "base_url": "http://localhost:11434/v1",
  "api_key": "local",
  "chat_model": "qwen3.5:0.8b",
  "embed_model": "nomic-embed-text",
  "pad_embeddings_to": 1536,
  "max_tokens_cap": 1024,
  "request_timeout": 600,
  "call_log": ""
}
```

Every key can be overridden by an environment variable named
`CRSEC_LOCAL_<KEY>`, e.g. `CRSEC_LOCAL_CHAT_MODEL`. That is what makes a model
sweep a loop rather than a series of file edits.

### What the shim does, and why

| Behaviour | Reason |
|---|---|
| Rewrites `model` on every call | CRSEC hardcodes `"gpt-4o-mini"` in five places and `"text-davinci-003"` in two `gpt_param` dicts. The davinci model no longer exists anywhere. |
| Routes `Completion.create` through chat | Ollama applies the chat template to `/v1/completions` regardless, so "raw completion" is not actually preserved. Routing through chat makes it explicit and identical across servers. Returns the legacy `.choices[0].text` shape that `GPT_request` expects. |
| Passes `stop` through untouched | CRSEC's validators depend on the stop sequences. Dropping them breaks parsing. |
| Clamps `max_tokens` to 1024 | `norm/creation.py` asks for 4096, which overruns a small model's usable window once the prompt is counted. |
| Translates `max_completion_tokens` | The evaluator uses the newer parameter name; local servers speak the older one. |
| Zero-pads embeddings to 1536 | See below. |

**The padding is not cosmetic.** `gpt_structure.get_embedding` returns
`[0.0] * 1536` when an embedding call fails — a width hardcoded for
`text-embedding-ada-002`. `nomic-embed-text` returns 768. Without padding, one
failed embedding mid-run would put two different widths into the same memory
store and `retrieve.cos_sim` would start raising on the mismatch. Padding to
1536 removes that failure mode, and it is free: appending zeros changes neither
the dot product nor either L2 norm, so cosine similarity is exactly preserved.
There is a test asserting that.

### One thing that is *not* a problem

The bundled base simulations ship with **empty** embedding stores:

```
storage/base_the_ville_n10/personas/*/bootstrap_memory/associative_memory/embeddings.json
```

All ten are `{}` in both `base_the_ville_n10` and `base_ville_n10_with_norm`.
So there are no leftover 1536-dimension ada-002 vectors to conflict with
locally-generated ones. Every embedding in a fresh run comes from whichever
model you configured. This is the thing that would otherwise have quietly
corrupted retrieval, and it happens not to apply here — but check it again if
anyone ever forks from a *completed* simulation rather than a base one.

---

## 5. Proof of concept

```powershell
cd C:\Users\burst\ResearchProject\crsec\reverie\backend_server

$env:CRSEC_LLM_BACKEND    = "local"
$env:CRSEC_LOCAL_CHAT_MODEL  = "qwen2.5:0.5b"
$env:CRSEC_LOCAL_EMBED_MODEL = "nomic-embed-text"

python local_llm/test_local_backend.py     # 16 offline tests, no server needed
python local_llm/preflight.py --repeat 5
```

`preflight.py` reports six stages:

```
[PASS] 1 server reachable       2 model(s) served
[PASS] 2 models present         qwen2.5:0.5b, nomic-embed-text
[PASS] 3 chat path              1.2s, 'ready'
[PASS] 4 legacy completion      0.9s, 'ready'
[PASS] 5 embedding path         native=768 padded=1536 deterministic=True
--------------------------------------------------------------------
6  prompt validity (5 attempt(s) per probe)
   wake_up_hour     5/5 valid   7 am
   event_triple     2/5 valid   (drafting, a petition)
   pronunciatio     5/5 valid   📝🌍
   decide_to_talk   4/5 valid   yes
   daily_plan       1/5 valid   Sam wakes up and ...
   overall prompt validity: 17/25 = 68%
```

Stages 1–5 are the pipeline. Stage 6 runs five real CRSEC prompts through the
real validators — the `func_validate` logic is copied verbatim out of the nested
closures in `run_gpt_prompt.py`, so a pass here means what a pass means inside
the simulation.

**Read stage 6, not stages 1–5.** Rough reading of the number: above 90% is
comparable to the hosted models; 60–90% means a meaningful share of the run will
be fail-safe defaults and it is a wiring proof rather than a result; below 60%
means the simulation would complete while being mostly hardcoded constants.

If you have no model downloaded yet, or want to test the pipeline in isolation,
there is a fake server that needs no model at all:

```powershell
python local_llm/mock_server.py --port 11500
$env:CRSEC_LOCAL_BASE_URL = "http://localhost:11500/v1"
python local_llm/preflight.py
```

### The better validity measurement: replay against a recording

`preflight.py` stage 6 uses five prompts I wrote by hand. That is a smoke test.
The real instrument reuses `replay.py`:

```powershell
# Once, against the hosted model. Costs money; this is the reference run.
python replay.py --mode record --origin base_the_ville_n10 --steps 100 --seed 42

# Then, free, for every local model you want to evaluate:
python local_llm/replay_validity.py --recording recording.jsonl --sample 150
```

`recording.jsonl` is every request and response from a real simulation, which
means it is the exact prompt distribution CRSEC issues, paired with what a
hosted model answered. Replaying those prompts through a local model measures
two things:

**Format agreement.** A format signature is derived from each recorded OpenAI
answer — integer, yes/no, parenthesised tuple, numbered list, emoji-only, length
band — and the local answer is checked against it. Only features the gold answer
actually exhibits are required, so a bare-integer answer demands an integer while
a prose answer demands only non-emptiness and a comparable length. This
approximates `func_validate` without needing the validator closures, and it is
grounded in what demonstrably worked rather than in what a prompt appears to ask
for. Content is deliberately not compared: two models should be free to disagree
about whether Mary talks to Bob, and a social simulation where they could not
would be measuring the wrong thing.

**Retrieval perturbation.** The texts embedded during the run are re-embedded
locally, and the pairwise cosine-similarity matrices are compared by Spearman
rank correlation. This is the number that says whether swapping
`text-embedding-ada-002` for `nomic-embed-text` reorders what the agents recall.
It is easy to miss because nothing crashes when retrieval degrades — the agents
simply remember less relevant things, and the simulation reads as subtly duller
rather than broken. Above ~0.85 the ordering is largely preserved; below ~0.6 the
local embedder is its own experimental condition rather than a substitution, and
should be reported as one.

Output:

```
  FORMAT AGREEMENT: 118/150 = 78.7%
  failure modes:
    expected numbered list                              19
    expected parseable integer                           8
    far longer than gold (parsers truncate or fail)      5
  RETRIEVAL RANK CORRELATION (Spearman): 0.71
```

The failure-mode breakdown is the useful part. "Expected numbered list" pointing
at `daily_planning_v6.txt` is a prompt-engineering problem with a plausible fix;
failures spread evenly across every prompt type are a model-capacity problem and
are not worth fixing at that size.

---

## 6. A real run

Use the headless runner that already exists rather than the interactive loop:

```powershell
cd C:\Users\burst\ResearchProject\crsec\reverie\backend_server
$env:CRSEC_LLM_BACKEND = "local"
$env:CRSEC_LOCAL_CHAT_MODEL = "llama3.2:3b"
python run_headless.py --origin base_the_ville_n10 --target local_llama3b_01 --steps 100
```

For a model sweep, go through `run_sweep.py`. It launches `run_headless.py` as
subprocesses, and subprocesses inherit the parent environment — so setting
`CRSEC_LOCAL_CHAT_MODEL` before the sweep applies it to every run in that sweep.
To vary the *model* across runs within one sweep, `_run_one` needs to pass an
env per config; a small addition, and the natural shape is an optional `"env"`
key in `configs.json`:

```json
[
  {"origin": "base_the_ville_n10", "target": "sweep_1b", "steps": 100,
   "env": {"CRSEC_LOCAL_CHAT_MODEL": "llama3.2:1b"}},
  {"origin": "base_the_ville_n10", "target": "sweep_3b", "steps": 100,
   "env": {"CRSEC_LOCAL_CHAT_MODEL": "llama3.2:3b"}}
]
```

Watch `--max-concurrent` against VRAM: two concurrent runs on one local server
means two models resident, or one model serving both, depending on whether the
tags differ.

Attach the monitor so the run reports how much of itself was real. The natural
place is `run_headless.py`, so every sweep run gets a report without touching
`reverie.py`:

```python
import local_llm.fail_safe_monitor as fsm
fsm.attach()
```

and where the run ends:

```python
fsm.report()
fsm.dump("fail_safe_report.json")
```

Output:

```
  safe_generate_response                  1842 calls    391 fail-safe   21.2%
        e.g. daily_planning_v6.txt
  ChatGPT_safe_generate_response            96 calls      4 fail-safe    4.2%
--------------------------------------------------------------------
  overall: 395/1938 = 20.4% of generations were fail-safe defaults
```

**Report that percentage next to any result from a local run.** It is the
single number that tells a reader whether they are looking at model behaviour
or at CRSEC's defaults, and no other artefact of the run contains it.

---

## 7. Expect it to be slow

A CRSEC day is thousands of sequential calls, and the agents' cognitive loop is
inherently serial — each step depends on the last. A 3B model on CPU at a few
tokens per second turns a run that took hours into one that takes days.

Things that actually help:

- **GPU.** The single biggest factor. A model that fits entirely in VRAM runs
  several times faster than one spilling to system RAM.
- **Keep the model resident.** Ollama unloads after five minutes idle and
  reloading costs seconds every time. `$env:OLLAMA_KEEP_ALIVE = "-1"`.
- **Parallel agents.** `$env:OLLAMA_NUM_PARALLEL = "4"`. Memory scales with
  this times the context length, so watch VRAM.
- **Parallel *simulations*.** Better use of a GPU than parallel agents, since
  runs are independent. This is what vLLM's continuous batching is for.
- **The embedding cache** already in `gpt_structure.py` (from the speedup PR)
  matters more locally than it did against the API, because every avoided call
  is now compute you own rather than a request you paid for.

---

## 8. What to hand the professor

The experiment is not "does it run locally." It is a comparison, and it needs a
control. Suggested shape:

1. **Establish the floor.** Run `preflight.py --repeat 20` across the size
   ladder (0.5B, 1B, 3B, 9B, 20B) and plot prompt validity against parameter
   count. This is cheap, takes minutes per model, and tells you the smallest
   model that can execute CRSEC's prompt formats at all. Everything below that
   line is unusable regardless of how interesting its social reasoning might be.
2. **Replicate the existing condition.** Re-run the current experiment with the
   smallest model that clears the floor, holding seeds, initialisation and the
   evaluator fixed. Report the fail-safe rate alongside the tipping-point result.
3. **Then vary the model.** Only once (1) and (2) are in hand does "does the
   tipping point depend on the model" become a well-posed question rather than a
   confound.

Two caveats worth stating explicitly in any writeup:

- **Prompt-format competence and social reasoning are different variables, and
  model size moves both.** If a 1B model shows no tipping point, that may be
  because it has no stable stance to tip, or because a fifth of its outputs were
  replaced by constants. The fail-safe rate is what lets you separate those.
- **The three sources of run-to-run randomness** traced earlier still apply, and
  local inference adds a fourth: most servers are not bitwise deterministic
  across batch sizes even at temperature 0. Pin `seed` where the server supports
  it, and treat single runs as single samples.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `APIRemovedInV1` | openai 1.x installed. `pip install "openai==0.27.0"`. |
| Preflight stage 1 fails | Server not running. `ollama serve`, then `curl http://localhost:11434/v1/models`. |
| Stage 2 warns "not listed" | Tag mismatch. `ollama list` and copy the tag exactly. |
| Stage 5 fails on padding | `pad_embeddings_to` was set to 0 or below the native width. |
| Everything passes, sim produces nonsense | This is the expected small-model outcome. Check the fail-safe rate before debugging anything else. |
| `openai.error.APIConnectionError` mid-run | Ollama unloaded the model or ran out of memory. Set `OLLAMA_KEEP_ALIVE=-1`, lower `OLLAMA_NUM_PARALLEL`. |
| Run is slower every hour | Context growth. Agent memories accumulate, so prompts lengthen through a simulated day. Expected, not a bug. |

---

## Files

```
local_llm/
  local_backend.py        the shim: redirects openai 0.27 to a local server
  local_models.json       configuration
  preflight.py            six-stage proof of concept (smoke test)
  replay_validity.py      validity + retrieval benchmark vs a replay.py recording
  fail_safe_monitor.py    counts fail-safe substitutions during a real run
  mock_server.py          fake OpenAI-compatible server, needs no model
  test_local_backend.py   16 offline tests, no server or network
run_local.ps1             sets the environment and runs preflight
```
