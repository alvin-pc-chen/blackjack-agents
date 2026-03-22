"""Experiment configuration, runner, and agent factory."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel

from .agents.base import Agent
from .agents.basic_strategy import BasicStrategyAgent
from .agents.card_counter import CardCountingAgent
from .agents.random_agent import RandomAgent
from .agents.simple_agent import SimpleAgent
from .manager import GameManager
from .shoe import SeededShoe
from .state import (
    ExperimentRecord,
    ExperimentSummary,
    PlayerSummary,
    RoundRecord,
)

logger = logging.getLogger(__name__)


class PlayerConfig(BaseModel):
    """Configuration for a single player at the table."""
    name: str
    agent_type: str
    bet: int = 100
    agent_params: dict[str, Any] = {}


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""
    experiment_name: str
    num_rounds: int
    num_decks: int = 6
    shoe_seed: int
    hit_soft_17: bool = False
    players: list[PlayerConfig]
    output_dir: str = "results"
    output_format: str = "json"


class AgentFactory:
    """Registry-based factory for creating Agent instances."""

    _registry: dict[str, type[Agent]] = {
        "random": RandomAgent,
        "simple": SimpleAgent,
        "basic_strategy": BasicStrategyAgent,
        "card_counter": CardCountingAgent,
    }

    @classmethod
    def create(cls, agent_type: str, **kwargs: Any) -> Agent:
        agent_cls = cls._registry.get(agent_type)
        if agent_cls is None:
            # Try importing LLM agents lazily
            if agent_type == "claude":
                from .agents.llm.claude_agent import ClaudeAgent
                return ClaudeAgent(**kwargs)
            if agent_type == "openai":
                from .agents.llm.openai_agent import OpenAIAgent
                return OpenAIAgent(**kwargs)
            if agent_type == "groq":
                from .agents.llm.groq_agent import GroqAgent
                return GroqAgent(**kwargs)
            raise ValueError(
                f"Unknown agent type: {agent_type!r}. "
                f"Available: {sorted(cls._registry)} + ['claude', 'openai', 'groq']"
            )
        return agent_cls(**kwargs)

    @classmethod
    def register(cls, name: str, agent_cls: type[Agent]) -> None:
        cls._registry[name] = agent_cls


def _compute_summary(
    rounds: list[RoundRecord],
    agent_type_map: dict[str, str],
) -> ExperimentSummary:
    """Compute aggregate statistics per player."""
    from collections import defaultdict

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total_hands": 0, "wins": 0, "losses": 0, "pushes": 0,
        "blackjacks": 0, "busts": 0, "surrenders": 0,
        "total_wagered": 0, "net_payout": 0,
    })

    for r in rounds:
        for pr in r.players:
            s = stats[pr.player_name]
            for h in pr.hands:
                s["total_hands"] += 1
                s["total_wagered"] += h.bet
                if h.result == "BLACKJACK":
                    s["blackjacks"] += 1
                    s["wins"] += 1
                    s["net_payout"] += int(h.bet * 1.5)
                elif h.result == "PLAYER_WIN":
                    s["wins"] += 1
                    s["net_payout"] += h.bet
                elif h.result == "DEALER_BUST":
                    s["wins"] += 1
                    s["net_payout"] += h.bet
                elif h.result == "PUSH":
                    s["pushes"] += 1
                elif h.result == "PLAYER_BUST":
                    s["busts"] += 1
                    s["losses"] += 1
                    s["net_payout"] -= h.bet
                elif h.result == "DEALER_WIN":
                    s["losses"] += 1
                    s["net_payout"] -= h.bet
                elif h.result == "SURRENDER":
                    s["surrenders"] += 1
                    s["losses"] += 1
                    s["net_payout"] -= int(h.bet * 0.5)

    summaries: list[PlayerSummary] = []
    for name, s in stats.items():
        total = s["total_hands"]
        summaries.append(PlayerSummary(
            player_name=name,
            agent_type=agent_type_map.get(name, "unknown"),
            total_hands=total,
            wins=s["wins"],
            losses=s["losses"],
            pushes=s["pushes"],
            blackjacks=s["blackjacks"],
            busts=s["busts"],
            surrenders=s["surrenders"],
            win_rate=s["wins"] / total if total else 0.0,
            net_units=s["net_payout"] / 100.0 if total else 0.0,  # normalize by base bet
        ))

    return ExperimentSummary(total_rounds=len(rounds), player_summaries=summaries)


class ExperimentRunner:
    """Configures and runs a complete blackjack experiment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def run(self) -> ExperimentRecord:
        shoe = SeededShoe(self._config.shoe_seed, num_decks=self._config.num_decks)

        player_agents: list[tuple[str, int, Agent]] = []
        for pc in self._config.players:
            agent = AgentFactory.create(pc.agent_type, **pc.agent_params)
            player_agents.append((pc.name, pc.bet, agent))

        manager = GameManager(
            player_agents=player_agents,
            shoe=shoe,
            hit_soft_17=self._config.hit_soft_17,
        )

        rounds = manager.play_rounds(self._config.num_rounds)

        agent_type_map = {name: agent.agent_type for name, _, agent in player_agents}
        summary = _compute_summary(rounds, agent_type_map)

        record = ExperimentRecord(
            experiment_id=str(uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            config=self._config.model_dump(),
            rounds=rounds,
            summary=summary,
        )

        self._save(record)
        return record

    def _save(self, record: ExperimentRecord) -> None:
        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        short_id = record.experiment_id[:8]
        name = self._config.experiment_name

        if self._config.output_format == "json":
            path = out_dir / f"{name}_{short_id}.json"
            path.write_text(record.model_dump_json(indent=2))
            logger.info("Results saved to %s", path)
        elif self._config.output_format == "csv":
            self._save_csv(record, out_dir, name, short_id)

    def _save_csv(
        self,
        record: ExperimentRecord,
        out_dir: Path,
        name: str,
        short_id: str,
    ) -> None:
        import csv

        # Summary CSV
        summary_path = out_dir / f"{name}_{short_id}_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "player_name", "agent_type", "total_hands", "wins", "losses",
                "pushes", "blackjacks", "busts", "surrenders", "win_rate", "net_units",
            ])
            for ps in record.summary.player_summaries:
                writer.writerow([
                    ps.player_name, ps.agent_type, ps.total_hands, ps.wins,
                    ps.losses, ps.pushes, ps.blackjacks, ps.busts, ps.surrenders,
                    f"{ps.win_rate:.4f}", f"{ps.net_units:.2f}",
                ])

        # Rounds CSV (one row per player per hand per round)
        rounds_path = out_dir / f"{name}_{short_id}_rounds.csv"
        with open(rounds_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "player_name", "agent_type", "hand_index",
                "cards", "total", "bust", "result", "bet",
                "dealer_total", "dealer_bust",
            ])
            for r in record.rounds:
                for pr in r.players:
                    for hi, h in enumerate(pr.hands):
                        cards_str = " ".join(f"{c.rank}{c.suit[0]}" for c in h.cards)
                        writer.writerow([
                            r.round_number, pr.player_name, pr.agent_type, hi,
                            cards_str, h.total, h.bust, h.result, h.bet,
                            r.dealer.final_total, r.dealer.bust,
                        ])

        logger.info("Results saved to %s and %s", summary_path, rounds_path)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)
