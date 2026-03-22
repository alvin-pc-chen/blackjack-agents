"""Shared test fixtures."""

from __future__ import annotations

import pytest
from blackjack21 import Action, Card

from blackjack_agents.agents.base import Agent, CardView, GameContext
from blackjack_agents.shoe import PredeterminedShoe, SeededShoe


@pytest.fixture
def seeded_shoe() -> SeededShoe:
    return SeededShoe(seed=42, num_decks=6)


@pytest.fixture
def simple_context() -> GameContext:
    """A basic context: player has hard 15, dealer shows 10."""
    return GameContext(
        player_name="TestPlayer",
        hand_cards=[CardView("Hearts", "K", 10), CardView("Spades", "5", 5)],
        hand_total=15,
        hand_is_soft=False,
        hand_index=0,
        num_hands=1,
        dealer_upcard=CardView("Diamonds", "10", 10),
        available_actions=frozenset({Action.HIT, Action.STAND, Action.SURRENDER}),
        other_players=[],
        round_number=1,
        face_up_cards=[],
        running_count=0,
        true_count=0.0,
        shoe_remaining=200,
    )


def make_context(
    *,
    hand_cards: list[CardView] | None = None,
    hand_total: int = 15,
    hand_is_soft: bool = False,
    dealer_upcard: CardView | None = None,
    available_actions: frozenset[Action] | None = None,
    running_count: int = 0,
    true_count: float = 0.0,
) -> GameContext:
    """Helper to build GameContext with sensible defaults."""
    return GameContext(
        player_name="TestPlayer",
        hand_cards=hand_cards or [CardView("Hearts", "K", 10), CardView("Spades", "5", 5)],
        hand_total=hand_total,
        hand_is_soft=hand_is_soft,
        hand_index=0,
        num_hands=1,
        dealer_upcard=dealer_upcard or CardView("Diamonds", "10", 10),
        available_actions=available_actions or frozenset({Action.HIT, Action.STAND, Action.DOUBLE, Action.SURRENDER}),
        other_players=[],
        round_number=1,
        face_up_cards=[],
        running_count=running_count,
        true_count=true_count,
        shoe_remaining=200,
    )


def make_card(rank: str = "5", suit: str = "Hearts", value: int | None = None) -> Card:
    """Build a blackjack21 Card."""
    if value is None:
        vals = {"A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
                "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10}
        value = vals.get(rank, 10)
    return Card(suit, rank, value)
