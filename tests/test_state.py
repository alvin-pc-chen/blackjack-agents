"""Tests for state tracking."""

from blackjack_agents.agents.base import CardView
from blackjack_agents.state import GameStateTracker, _HILO


class TestHiLoCount:
    def test_low_cards_positive(self) -> None:
        tracker = GameStateTracker()
        for rank in ["2", "3", "4", "5", "6"]:
            tracker.record_face_up_card(CardView("Hearts", rank, 0))
        assert tracker.running_count == 5

    def test_high_cards_negative(self) -> None:
        tracker = GameStateTracker()
        for rank in ["10", "J", "Q", "K", "A"]:
            tracker.record_face_up_card(CardView("Hearts", rank, 0))
        assert tracker.running_count == -5

    def test_neutral_cards_zero(self) -> None:
        tracker = GameStateTracker()
        for rank in ["7", "8", "9"]:
            tracker.record_face_up_card(CardView("Hearts", rank, 0))
        assert tracker.running_count == 0

    def test_true_count(self) -> None:
        tracker = GameStateTracker()
        for rank in ["2", "3", "4", "5", "6"]:
            tracker.record_face_up_card(CardView("Hearts", rank, 0))
        # Running count = 5, 2 decks remaining
        tc = tracker.true_count(2.0)
        assert tc == 2.5

    def test_true_count_zero_decks(self) -> None:
        tracker = GameStateTracker()
        tracker.record_face_up_card(CardView("Hearts", "2", 0))
        assert tracker.true_count(0.0) == 0.0
