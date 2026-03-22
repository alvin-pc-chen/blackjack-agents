"""Tests for experiment runner."""

import json
import tempfile
from pathlib import Path

from blackjack_agents.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    AgentFactory,
    PlayerConfig,
    load_config,
)


class TestAgentFactory:
    def test_create_known_agents(self) -> None:
        for agent_type in ("random", "simple", "basic_strategy", "card_counter"):
            agent = AgentFactory.create(agent_type)
            assert agent is not None

    def test_unknown_agent_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.create("nonexistent")


class TestExperimentRunner:
    def test_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiment_name="test_run",
                num_rounds=10,
                num_decks=6,
                shoe_seed=42,
                players=[
                    PlayerConfig(name="Basic", agent_type="basic_strategy"),
                    PlayerConfig(name="Random", agent_type="random", agent_params={"seed": 1}),
                ],
                output_dir=tmpdir,
                output_format="json",
            )
            runner = ExperimentRunner(config)
            record = runner.run()

            assert record.summary.total_rounds > 0
            assert len(record.summary.player_summaries) == 2

            # Verify file was saved
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1

            # Verify JSON is valid
            data = json.loads(files[0].read_text())
            assert data["config"]["shoe_seed"] == 42

    def test_reproducibility(self) -> None:
        config = ExperimentConfig(
            experiment_name="repro_test",
            num_rounds=10,
            num_decks=6,
            shoe_seed=42,
            players=[
                PlayerConfig(name="Basic", agent_type="basic_strategy"),
            ],
            output_dir=tempfile.mkdtemp(),
        )
        r1 = ExperimentRunner(config).run()
        r2 = ExperimentRunner(config).run()

        results1 = [h.result for r in r1.rounds for pr in r.players for h in pr.hands]
        results2 = [h.result for r in r2.rounds for pr in r.players for h in pr.hands]
        assert results1 == results2

    def test_csv_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiment_name="csv_test",
                num_rounds=5,
                num_decks=6,
                shoe_seed=42,
                players=[PlayerConfig(name="Simple", agent_type="simple")],
                output_dir=tmpdir,
                output_format="csv",
            )
            ExperimentRunner(config).run()

            csv_files = list(Path(tmpdir).glob("*.csv"))
            assert len(csv_files) == 2  # summary + rounds


class TestLoadConfig:
    def test_load_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
experiment_name: test
num_rounds: 10
shoe_seed: 42
players:
  - name: Bot
    agent_type: simple
""")
            f.flush()
            config = load_config(f.name)
            assert config.experiment_name == "test"
            assert config.shoe_seed == 42
            assert len(config.players) == 1
