# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CRSEC (Climate Response in Social Emergent Communities) is a research platform studying tipping points and norm emergence in LLM-powered multi-agent systems. Agents with varying climate change stances (conviction/trust dimensions) interact in a simulated environment. A committed minority ("norm entrepreneurs") attempt to spread social norms to influence others.

Built on the Generative Agents architecture (Park et al., IJCAI 2024).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
```

**Required:** `OPENAI_API_KEY` in environment. The project uses `openai==0.27.0` (legacy API).

## Running the Simulation

Three terminals are needed:

**Terminal 1 — Django frontend:**
```bash
cd environment/frontend_server
python manage.py runserver
# Live view: http://localhost:8000/simulator_home
```

**Terminal 2 — Simulation backend:**
```bash
cd reverie/backend_server
python reverie.py
# Prompts: fork sim name (blank = new), new sim name
# Optionally generate norms (enter entrepreneur agent name)
```

**Terminal 3 — Initialize agents (one-time or when resetting):**
```bash
python initialization/initialization.py
# Output: initialization/bio-initialization.txt
```

**Interactive commands in reverie.py prompt:**
```
run <N>                                     # Execute N simulation steps
save                                        # Persist state
fin / finish                                # Save and exit
exit                                        # Exit without saving
print persona schedule <name>
print persona associative memory (event|thought|chat) <name>
print persona spatial memory <name>
print current time
print all persona schedule
```

## Replay / Demo

```
http://localhost:8000/replay/<sim_code>/<step>/
http://localhost:8000/demo/<sim_code>/<step>/<speed>/
```

Simulation data lives under `environment/frontend_server/storage/<sim_code>/`.

## Analysis Pipeline

Run in order after a simulation completes:

```bash
# 1. Clean and deduplicate chat logs
python cleaning_data_unique_conversations_with_rounds.py

# 2. Score dialogues with LLM (urgency + trust, -1 to +1)
python "analysis pipeline/LLM-evaluator.py"

# 3. Visualize Trust-per-Round (TpR) and Urgency-per-Round (UpR)
python "analysis pipeline/LLM_results_visualization_TpR_UpR.py"

# Uniform distribution visualization (agent archetype sampling)
python uniform_distribution_visualization.py
```

## Architecture

### Two-Process Design

The backend (`reverie.py`) and frontend (`environment/frontend_server/`) are separate processes that communicate via JSON files written to `storage/`. The Django server reads these files to render movement and state for visualization.

### Agent Cognitive Loop (`persona/persona.py`)

Each simulation step, every agent runs through six cognitive modules in sequence:

1. **Perceive** — detect events within `vision_r=4` tiles
2. **Retrieve** — query associative memory for relevant context
3. **Plan** — generate schedule and decompose tasks
4. **Reflect** — consolidate memories into higher-level thoughts
5. **Execute** — perform action in the world
6. **Converse** — generate dialogue with nearby agents

### Three Memory Systems (per agent)

| Memory | File | Purpose |
|--------|------|---------|
| Spatial | `memory_structures/spatial_memory.py` | Tree of known world locations |
| Associative | `memory_structures/associative_memory.py` | Long-term event/thought/chat stream |
| Scratch | `memory_structures/scratch.py` | Short-term working memory + identity |

### Normative System (`norm/`)

Each agent holds a `NormDatabase` with two tiers:
- `norm_seed` — norms created by norm entrepreneurs
- `act_norm` — norms the agent has actively adopted

Norm lifecycle: **Creation** (LLM-generated from agent scratch) → **Retrieval** (surfaced during planning/conversation) → **Compliance** (adjusts behavior) → **Evaluation** (utility scoring determines adoption) → **Spread** (other agents receive and evaluate)

Key files: `norm/normDatabase.py`, `norm/creation.py`, `norm/norm_evaluate.py`, `norm/norm_comply.py`

### Agent Archetypes (initialization)

Agents are placed on two dimensions (`conviction` ∈ [-1,1], `trust` ∈ [-1,1]):

| Archetype | Conviction | Trust |
|-----------|-----------|-------|
| Apathetic Cynic | low | low |
| Passive Conformist | low | high |
| Grassroots Activist / Dissenter | high | low |
| Institutional Optimist / Policy Ally | high | high |

Two "Grassroots Activists" are always selected as the committed minority (norm entrepreneurs).

### LLM Integration

All LLM calls go through `persona/prompt_template/gpt_structure.py`. Prompts for each cognitive module live in `persona/prompt_template/v3_ChatGPT/`. Norm-specific prompts are in `norm/*_prompt/` subdirectories. Models used: `gpt-4o-mini` (default), `gpt-4o`.

### Storage Layout

```
environment/frontend_server/storage/<sim_code>/
├── movement/          # Agent positions per step (JSON)
├── meta.json          # Simulation metadata (start_date, sec_per_step)
└── personas/<name>/   # Per-agent memories, norms, scratch
```

`compressed_storage/` holds demo-ready compressed snapshots.

## Key Entry Points

| File | Role |
|------|------|
| `reverie/backend_server/reverie.py` | Main simulation controller and interactive CLI |
| `reverie/backend_server/persona/persona.py` | Agent class — ties all cognitive modules together |
| `reverie/backend_server/norm/normDatabase.py` | Norm storage and lifecycle management |
| `initialization/initialization.py` | Agent profile generation and archetype assignment |
| `reverie/backend_server/utils.py` | API key, storage paths, debug flag |
| `environment/frontend_server/translator/views.py` | Django views for demo/replay/live |
| `analysis pipeline/LLM-evaluator.py` | Post-hoc attitude scoring via LLM |
