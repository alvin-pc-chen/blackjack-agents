"""Agent interface and GameContext for blackjack decision-making."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from blackjack21 import Action


@dataclass(frozen=True)
class CardView:
    """Serializable view of a card."""
    suit: str
    rank: str
    value: int


@dataclass(frozen=True)
class OtherPlayerView:
    """What one player can see of another player's hand(s)."""
    name: str
    hands: list[HandView]


@dataclass(frozen=True)
class HandView:
    """Visible state of a hand."""
    cards: list[CardView]
    total: int
    bust: bool
    num_cards: int


@dataclass(frozen=True)
class GameContext:
    """Immutable snapshot of visible game state, provided to agents for decision-making."""
    player_name: str
    hand_cards: list[CardView]
    hand_total: int
    hand_is_soft: bool
    hand_index: int
    num_hands: int
    dealer_upcard: CardView
    available_actions: frozenset[Action]
    other_players: list[OtherPlayerView]
    round_number: int
    face_up_cards: list[CardView]
    running_count: int
    true_count: float
    shoe_remaining: int


class Agent(ABC):
    """Abstract base class for all blackjack player agents."""

    @abstractmethod
    def decide(self, context: GameContext) -> Action:
        """Given the current game context, return an action.

        The returned action must be in context.available_actions.
        """
        ...

    @property
    def agent_type(self) -> str:
        """Human-readable name for this agent type."""
        return self.__class__.__name__
