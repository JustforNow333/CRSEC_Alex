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
