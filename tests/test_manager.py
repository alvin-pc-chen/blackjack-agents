"""Tests for GameManager."""

from blackjack21 import Action

from blackjack_agents.agents.base import Agent, GameContext
from blackjack_agents.manager import GameManager
from blackjack_agents.shoe import SeededShoe


class AlwaysStandAgent(Agent):
    def decide(self, context: GameContext) -> Action:
        return Action.STAND


class AlwaysHitAgent(Agent):
    def decide(self, context: GameContext) -> Action:
        if Action.HIT in context.available_actions:
            return Action.HIT
        return Action.STAND


class TestGameManager:
    def test_single_round(self) -> None:
        shoe = SeededShoe(seed=42, num_decks=6)
        mgr = GameManager(
            player_agents=[("Alice", 100, AlwaysStandAgent())],
            shoe=shoe,
        )
        record = mgr.play_round()
        assert record.round_number == 1
        assert len(record.players) == 1
        assert record.players[0].player_name == "Alice"
        assert record.players[0].agent_type == "AlwaysStandAgent"
        assert record.dealer.final_total > 0

    def test_multiple_rounds(self) -> None:
        shoe = SeededShoe(seed=42, num_decks=6)
        mgr = GameManager(
            player_agents=[("Alice", 100, AlwaysStandAgent())],
            shoe=shoe,
        )
        records = mgr.play_rounds(5)
        assert len(records) >= 1
        assert all(r.round_number == i + 1 for i, r in enumerate(records))

    def test_multiple_players(self) -> None:
        shoe = SeededShoe(seed=42, num_decks=6)
        mgr = GameManager(
            player_agents=[
                ("Alice", 100, AlwaysStandAgent()),
                ("Bob", 200, AlwaysHitAgent()),
            ],
            shoe=shoe,
        )
        record = mgr.play_round()
        assert len(record.players) == 2
        names = {p.player_name for p in record.players}
        assert names == {"Alice", "Bob"}

    def test_results_are_valid(self) -> None:
        shoe = SeededShoe(seed=42, num_decks=6)
        mgr = GameManager(
            player_agents=[("Alice", 100, AlwaysStandAgent())],
            shoe=shoe,
        )
        records = mgr.play_rounds(10)
        valid_results = {
            "BLACKJACK", "PLAYER_WIN", "DEALER_BUST", "PUSH",
            "PLAYER_BUST", "DEALER_WIN", "SURRENDER",
        }
        for r in records:
            for pr in r.players:
                for h in pr.hands:
                    assert h.result in valid_results

    def test_reproducibility(self) -> None:
        def run_experiment() -> list[str]:
            shoe = SeededShoe(seed=42, num_decks=6)
            mgr = GameManager(
                player_agents=[("Alice", 100, AlwaysStandAgent())],
                shoe=shoe,
            )
            records = mgr.play_rounds(10)
            return [h.result for r in records for pr in r.players for h in pr.hands]

        results1 = run_experiment()
        results2 = run_experiment()
        assert results1 == results2

    def test_shoe_depletion_stops_early(self) -> None:
        shoe = SeededShoe(seed=42, num_decks=1)  # Only 52 cards
        mgr = GameManager(
            player_agents=[
                ("Alice", 100, AlwaysStandAgent()),
                ("Bob", 100, AlwaysStandAgent()),
                ("Charlie", 100, AlwaysStandAgent()),
            ],
            shoe=shoe,
        )
        records = mgr.play_rounds(100)
        assert len(records) < 100  # Should stop early

    def test_actions_recorded(self) -> None:
        shoe = SeededShoe(seed=42, num_decks=6)
        mgr = GameManager(
            player_agents=[("Alice", 100, AlwaysHitAgent())],
            shoe=shoe,
        )
        record = mgr.play_round()
        # AlwaysHitAgent should have at least one action recorded
        assert len(record.players[0].actions_taken) >= 1

    def test_tracker_accessible(self) -> None:
        shoe = SeededShoe(seed=42, num_decks=6)
        mgr = GameManager(
            player_agents=[("Alice", 100, AlwaysStandAgent())],
            shoe=shoe,
        )
        mgr.play_round()
        assert len(mgr.tracker.rounds) == 1
        assert len(mgr.tracker.face_up_cards) > 0
