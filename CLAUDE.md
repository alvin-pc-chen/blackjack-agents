# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Empirical blackjack strategy testing framework with pluggable AI agents. Uses `blackjack21` (PyPI) as the game engine, with custom `CardSource` implementations for reproducible shoe ordering.

## Commands

```bash
# Install (editable, with dev tools)
python3 -m pip install -e ".[dev]"

# Install with LLM agent support
python3 -m pip install -e ".[dev,llm]"

# Run tests
python3 -m pytest tests/ -v

# Run a single test
python3 -m pytest tests/test_shoe.py::TestSeededShoe::test_deterministic_order -v

# Run an experiment
python3 -m blackjack_agents run configs/example_experiment.yaml

# Summarize results
python3 -m blackjack_agents summarize results/<file>.json

# Lint
ruff check src/ tests/
```

## Architecture

**Game engine**: `blackjack21` handles all rules (hit, stand, split, double, surrender). We never reimplement blackjack logic.

**Key flow**: `ExperimentRunner` → `GameManager` → `blackjack21.Table` ← `Agent.decide(GameContext)`

- **`shoe.py`** — `SeededShoe(seed, num_decks)` and `PredeterminedShoe(cards)` implement the `CardSource` protocol (`draw_card()` + `__len__()`) for reproducible card ordering
- **`manager.py`** — `GameManager` drives the Table, builds `GameContext` snapshots, dispatches to agents, records via tracker. Keys agent lookup on player name strings (blackjack21 re-instantiates Player objects between rounds)
- **`state.py`** — Pydantic models (`RoundRecord`, `ActionRecord`, etc.) and `GameStateTracker` for Hi-Lo counting and round recording
- **`agents/base.py`** — `Agent` ABC with `decide(GameContext) -> Action`, plus `GameContext` frozen dataclass. Agents never see blackjack21 internals
- **`agents/`** — `RandomAgent`, `SimpleAgent`, `BasicStrategyAgent` (full chart), `CardCountingAgent` (Hi-Lo + Illustrious 18)
- **`agents/llm/`** — `BaseLLMAgent` with prompt formatting/parsing, `ClaudeAgent`, `OpenAIAgent`. Requires `[llm]` extras
- **`experiment.py`** — `ExperimentConfig` (Pydantic) + YAML loading, `AgentFactory`, `ExperimentRunner` with JSON/CSV output

## Key Design Decisions

- `Action` enum values are **lowercase**: `"hit"`, `"stand"`, `"double"`, `"split"`, `"surrender"`
- `blackjack21.EmptyDeckError()` takes **no arguments**
- `hand.result` is a plain attribute (not a property), set to `GameResult` after `_calculate_results()` runs
- `GameResult` values: `BLACKJACK`, `PLAYER_WIN`, `DEALER_BUST`, `PUSH`, `PLAYER_BUST`, `DEALER_WIN`, `SURRENDER`
- Shoe cards deal from left (popleft from deque), not pop from right
- `CardSuit`/`CardRank` are not re-exported from `blackjack21` top level — import from `blackjack21.deck`
