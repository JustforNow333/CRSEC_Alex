# Local model ladder: what the adapters buy, and what is left

Measured 2026-09-14, Ollama on localhost, `nomic-embed-text` for embeddings,
`preflight.py --repeat 20` (100 completions per model per configuration).

Stage 6's validators are verbatim copies of the `__func_validate` closures in
`persona/prompt_template/run_gpt_prompt.py`. They forgive nothing. Every
adaptation lives in `local_backend.py`, on the path the simulation also takes,
so a stage 6 number describes what a run will actually see.

Two adapters, both config-gated, both counted in `STATS` and reported by
`summary()`:

| adapter | default | env |
|---|---|---|
| `strip_prompt_echo` | on | `CRSEC_LOCAL_STRIP_PROMPT_ECHO` |
| `max_tokens_floor` | 32 | `CRSEC_LOCAL_MAX_TOKENS_FLOOR` |

## 1. Adapters off vs on (floor 32)

Cells are valid/20.

| probe | 0.5b | 1.5b | 3b | 7b |
|---|---|---|---|---|
| wake_up_hour | 0 = | 0 → 1 | 0 = | 0 = |
| event_triple | 3 → 7 | 20 = | 20 = | **0 → 20** |
| pronunciatio | 1 → 3 | 15 → 16 | 7 → 8 | 12 → 11 |
| decide_to_talk | 20 = | 20 = | 20 = | 19 → 20 |
| daily_plan | 18 → 19 | 20 = | 20 = | 20 = |
| **overall** | 42% → 49% | 75% → 77% | 67% → 68% | **51% → 71%** |

Only the 7B's +20pp is outside sampling noise at n=20, and it is one probe:
`event_triple` 0/20 → 20/20, with `echo_strips` confirming all 20 responses
were echoing. The 7B answered correctly every time and was scored zero because
it prefixed the answer with the prompt's own tail:

    Output: (Mary Smith, draft, petition)

Echoing gets *worse* with scale here — 20/100 responses for the 7B, 5/100 for
the 0.5B, none for the 1.5B or 3B — so the un-adapted harness inverted the size
ordering it exists to establish, ranking the 7B (51%) below the 3B (67%).

## 2. Floor 32 vs 64

| probe | 0.5b | 1.5b | 3b | 7b |
|---|---|---|---|---|
| wake_up_hour | 0 = | 1 → 0 | 0 = | 0 = |
| event_triple | 7 = | 20 = | 20 = | 20 = |
| pronunciatio | 3 → 2 | 16 → 17 | 8 → 10 | 11 → 13 |
| decide_to_talk | 20 = | 20 = | 20 = | 20 = |
| daily_plan | 19 = | 20 = | 20 = | 20 = |
| **overall** | 49% → 48% | 77% → 77% | 68% → 70% | 71% → 73% |

64 does not clearly win: +2pp on the 3B and 7B, -1pp on the 0.5B, flat on the
1.5B, all inside noise at n=20, and `wake_up_hour` stays at 0 everywhere.
**The default stays at 32.** Raising it costs tokens on all 27 of the 51
`gpt_param` budgets that sit under the floor, and buys nothing measurable.

## 3. The residual is paraphrase, not truncation

`wake_up_hour` is 0/20 on every model at both floors. That is not a capability
failure. Counting how often a parseable hour appears *anywhere* in the response
against how often `__func_clean_up` — `int(resp.strip().lower().split("am")[0])`
— actually parses it:

| model | floor 32: present / parsed | floor 64: present / parsed |
|---|---|---|
| 0.5b | 1/20 / 0 | 6/20 / 0 |
| 1.5b | 15/20 / 1 | 18/20 / 0 |
| 3b | 5/20 / 0 | 15/20 / 0 |
| 7b | 7/20 / 0 | 19/20 / 0 |

The 3B and 7B converted exactly as predicted: at 32 they were truncated before
reaching an hour (5/20 and 7/20 present), at 64 the hour is there almost always
(15/20 and 19/20) and is parsed exactly as often as before — never. Raising the
budget moved them out of truncation and into the failure the 1.5B already had.

The failure is that the model restates the prompt *in its own words* rather than
echoing it. The prompt ends:

    Sam Moore's wake up hour:

and the model answers:

    Sam Moore's wake up hour is 6:00 AM.

`strip_prompt_echo` cannot touch this. It matches a literal suffix of the
prompt, and `hour is` is not `hour:` — there is no echo, there is a paraphrase
that happens to contain the answer. Across all four models and both floors,
`wake_up_hour` produced **one** parseable response out of 160.

No extraction adapter was added, deliberately. Stripping removes text the model
added and the floor changes a request parameter; pulling `6:00 AM` out of a
sentence and asserting it was the answer is a different kind of intervention,
and it would put the harness in the business of deciding what the model meant.

### What this means for a run

Every persona gets the fail-safe wake hour of 8, on all four models, in every
run. `fail_safe_monitor.py` will show it. Any result that depends on when
personas wake is a result about the constant `8`, not about the model.

The honest fixes are upstream of the shim: rewrite
`persona/prompt_template/v2/wake_up_hour_v1.txt` so a chat-tuned model answers
in-format, or loosen that one `__func_clean_up`. Both change CRSEC itself, which
is a decision about the experiment rather than about the backend.

## Reproducing

```bash
export CRSEC_LLM_BACKEND=local
export CRSEC_LOCAL_CHAT_MODEL=qwen2.5:7b
export CRSEC_LOCAL_STRIP_PROMPT_ECHO=1     # 0 for the adapters-off column
export CRSEC_LOCAL_MAX_TOKENS_FLOOR=32     # 0 for the adapters-off column
python preflight.py --repeat 20
```

The present-vs-parsed counts in section 3 come from running `PROBES[0]` 20 times
and comparing `preflight._v_wake_up_hour(out)` against a scan for any 1-2 digit
number in 0..23 that is not immediately followed by `pm`:

```python
DIG = re.compile(r"\d+"); PM = re.compile(r"\s*(?::\d{2})?\s*p\.?\s*m\.?", re.I)
def hour_anywhere(t):
    return any(len(m.group(0)) <= 2 and 0 <= int(m.group(0)) <= 23
               and not PM.match(t, m.end()) for m in DIG.finditer(t))
```

Numbers are n=20 per probe at temperature 0.8. Treat anything under ~10pp as
noise; the 7B's `event_triple` result and the section 3 present-vs-parsed gap
are the only differences here large enough to carry weight.

---

# 4. Family comparison: llama3.2:3b vs qwen2.5:3b

Measured 2026-09-14, same session, same server, same protocol. Both models
rerun from scratch so nothing about the setup differs between them. Same
parameter count, different family and tuning recipe.

The question: every number in sections 1-3 comes from one model family, so the
paraphrase wall could be a property of how Qwen was tuned rather than of
chat-tuning in general. That confound needs closing before any of this is
written up.

**n = 20 per probe per configuration** (100 completions per model per
configuration), temperature 0.8. Nothing below about 5pp at this n is a
difference; where a cell moves 1-2 points below it is called noise and not
interpreted.

## 4.1 The ladder

Adapters off -> on, floor at the 32 default. Cells are valid/20.

| probe | llama3.2:3b | qwen2.5:3b |
|---|---|---|
| wake_up_hour | 1 → 0 | 0 = |
| event_triple | 1 = | 20 = |
| pronunciatio | 20 = | 10 → 9 |
| decide_to_talk | 20 = | 20 = |
| daily_plan | 20 = | 20 = |
| **overall** | 62% → 61% | 70% → 69% |

Both -1pp: noise, and in both cases the adapters did nothing measurable. The
reason is in the counters.

## 4.2 Echo: not a family difference at this size

`strip_prompt_echo` would fire on **0 of 100** responses for llama3.2:3b and
**0 of 100** for qwen2.5:3b, measured directly by running the matcher over raw
responses with the adapter disabled. The adapters-on preflight runs each
reported `echo_strips=1` out of 101 completions, which is the same number
within noise. `token_floor_raises=82` in both, identically, since the floor
depends on CRSEC's gpt_param values and not on the model.

So the answer to "does llama echo at 3B, where Qwen did not" is no. Neither
family echoes at 3B. Echoing remains something seen only in qwen2.5:7b (20/100)
and weakly in qwen2.5:0.5b (5/100), and with one family at one size above 3B
there is not enough here to attribute it to family, scale, or their
interaction. It is an open question, not a finding.

## 4.3 wake_up_hour: both families hit the same wall

Every failing response classified, on the echo-stripped text, n=20:

| class | llama3.2:3b | qwen2.5:3b |
|---|---|---|
| parses in-sim | 1 | 0 |
| paraphrase (hour present, behind a clause) | 9 | 1 |
| truncated before reaching an hour | 8 | 18 |
| no hour given (finished a sentence, no number) | 2 | 1 |

At floor 32 the mixes look different: llama reaches an hour within the budget
about half the time, qwen almost never. That difference is about verbosity, not
about the wall, and the headroom control separates the two.

**Headroom control.** The same prompt at `max_tokens` 32 and 256, n=20 each. If
a failure is truncation, removing the budget limit converts it into an hour
appearing; if it is paraphrase, hour-present is already high and parsing stays
at zero either way.

| model | 32 tok | 256 tok |
|---|---|---|
| llama3.2:3b | 11/20 present, 1 parsed | 15/20 present, 1 parsed |
| qwen2.5:3b | 3/20 present, 0 parsed | 19/20 present, 0 parsed |

qwen2.5:3b moves 3 -> 19: its floor-32 failures really were truncation, and the
8x budget converts them. llama3.2:3b moves 11 -> 15, inside noise at n=20: its
failures were never budget-bound.

Two families, opposite truncation behaviour, and the same endpoint. With the
budget removed, the hour is present in 15/20 and 19/20 of responses and is
parsed in 1 and 0 of them. **The paraphrase wall is a property of chat-tuning,
not of Qwen's recipe.** Both models answer the question and neither answers it
in a format `int(resp.strip().lower().split("am")[0])` can read:

    llama3.2:3b  "Based on the information provided, it can be inferred that
                  Sam Moore wakes up around 5:00 am, as..."
    qwen2.5:3b   "Based on the information provided, it seems that Sam Moore
                  is an early riser. Given that he wakes..."

### A category that did not survive

An earlier pass reported a `wrong_answer_pm` class — the model answering with
the 10pm bedtime the prompt handed it — at 7/20 for llama and 13/20 for qwen.
That was an artifact of classifier ordering, not a real failure mode. Those
responses were cut off mid-sentence while restating *"goes to bed around
10pm"*, and testing for a pm hour before testing for truncation labelled them
wrong answers:

    "Based on the information provided, Sam Moore is an early riser and
     typically goes to bed around 10pm. To calculate his wake-up hour, le"

Checking truncation first empties the class: **0 for both models**. It is
recorded here because it looked like a clean family signal and was not one.
(The 0.5B genuinely does assert a pm wake hour — *"Sam Moore's wake up hour is
typically around 10 PM."*, a completed sentence — but that is one model at one
size, not a pattern.)

## 4.4 Where the families actually differ

**event_triple — qwen 20/20, llama 1/20.** The largest gap in this comparison,
and not one the adapters can touch. llama does not echo the prompt and does not
run out of budget; it answers conversationally, and frequently emits no tuple
at all:

    llama3.2:3b   "Here are the transformed inputs:"
    llama3.2:3b   "I can help you break down the sentences into (subject,
                   predicate, object) format."
    llama3.2:3b   "I can help you convert the sentences into the (subject,
                   predicate, object) format."
    qwen2.5:3b    "draft, petition)"

qwen2.5:3b returns the bare continuation the prompt asks for, 20 times out of
20. This single probe accounts for the whole 8pp overall gap between the two
models. A run on llama3.2:3b would store `(name, "is", "idle")` for nearly
every event triple in the simulation.

**pronunciatio — llama 20/20, qwen 10/20, and the validator flatters qwen.**
llama returns clean emoji. qwen returns Chinese text with an emoji appended,
which is short enough to clear the `<12 chars and contains an emoji` test:

    llama3.2:3b   "🛏️😴"      "😴🛏️😌"
    qwen2.5:3b    "躺在床上睡了 🛌💤"   "躺在床上😴"

CRSEC stores `cr[:3]`. Applying that to the responses the validator passed, the
stored value contains an emoji in 20/20 of llama's and **5/20** of qwen's — the
rest put `躺在床` on the map. So the real gap on this probe is wider than the
scores show, and the probe's own validator does not fully discriminate. Passing
it is necessary but not sufficient; the stored-prefix check is the better
measure and is not currently part of stage 6.

**decide_to_talk and daily_plan — 20/20 for both.** No family difference.

## 4.5 What this changes

- The paraphrase wall is not a Qwen artifact. It reproduces in a second family
  with a different tuning recipe, under a control that rules out truncation.
  The confound is closed, and the write-up can state it for chat-tuned local
  models generally rather than hedging to one family.
- Neither adapter helps at 3B in either family, because neither 3B model
  echoes. The adapters earn their place on qwen2.5:7b and nowhere else so far.
- Family matters more than the adapters do at this size: 1/20 vs 20/20 on
  event_triple between two models of the same parameter count, which is larger
  than any effect measured in sections 1-3 except the 7B's echo.
- qwen2.5:3b is the better of the two for CRSEC despite llama's cleaner emoji,
  entirely on the strength of event_triple.

No adapters were added, no defaults changed, and `run_gpt_prompt.py` was not
touched. Measurement only.
