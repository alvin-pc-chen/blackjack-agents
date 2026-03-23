"""Custom CardSource implementations for reproducible blackjack experiments."""

from __future__ import annotations

import random as _random
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

from blackjack21 import DEFAULT_SUITS, Card, EmptyDeckError
from blackjack21.deck import CardSuit

DEFAULT_RANKS: dict[str, int] = {
    "A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10,
}


class SeededShoe:
    """A CardSource that shuffles a multi-deck shoe with a deterministic seed.

    Satisfies the blackjack21 CardSource protocol.
    """

    def __init__(
        self,
        seed: int,
        num_decks: int = 6,
        suits: Sequence[CardSuit] = DEFAULT_SUITS,
    ) -> None:
        self._seed = seed
        self._num_decks = num_decks
        self._suits = suits
        self._rng = _random.Random(seed)
        self._shuffle_count = 0

        self._build_and_shuffle()

    def _build_and_shuffle(self) -> None:
        """Build a fresh shoe from all decks and shuffle with the RNG."""
        cards: list[Card] = []
        for suit in self._suits:
            for rank, value in DEFAULT_RANKS.items():
                cards.append(Card(suit, rank, value))
        cards *= self._num_decks
        self._rng.shuffle(cards)

        self._cards = deque(cards)
        self._initial_order = list(self._cards)
        self._drawn_cards: list[Card] = []
        self._shuffle_count += 1

    def reshuffle(self) -> None:
        """Reshuffle all cards back into a fresh shoe.

        Uses the next state of the deterministic RNG, so the new card order
        is reproducible but different from the previous shuffle.
        """
        self._build_and_shuffle()

    def draw_card(self) -> Card:
        if not self._cards:
            raise EmptyDeckError()
        card = self._cards.popleft()
        self._drawn_cards.append(card)
        return card

    def __len__(self) -> int:
        return len(self._cards)

    @property
    def drawn_cards(self) -> list[Card]:
        return list(self._drawn_cards)

    @property
    def initial_order(self) -> list[Card]:
        """The full card order as shuffled (for logging/verification)."""
        return list(self._initial_order)

    @property
    def num_decks(self) -> int:
        return self._num_decks

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def shuffle_count(self) -> int:
        """How many times this shoe has been shuffled (including initial)."""
        return self._shuffle_count


class PredeterminedShoe:
    """A CardSource that deals cards in a fixed, predetermined order.

    Satisfies the blackjack21 CardSource protocol.
    """

    def __init__(self, cards: list[Card]) -> None:
        self._cards = deque(cards)
        self._drawn_cards: list[Card] = []

    def draw_card(self) -> Card:
        if not self._cards:
            raise EmptyDeckError()
        card = self._cards.popleft()
        self._drawn_cards.append(card)
        return card

    def __len__(self) -> int:
        return len(self._cards)

    @property
    def drawn_cards(self) -> list[Card]:
        return list(self._drawn_cards)


def create_shoe(
    *,
    seed: int | None = None,
    cards: list[Card] | None = None,
    num_decks: int = 6,
) -> SeededShoe | PredeterminedShoe:
    """Factory: provide either a seed (for reproducible shuffle) or an explicit card list."""
    if cards is not None:
        return PredeterminedShoe(cards)
    if seed is not None:
        return SeededShoe(seed, num_decks=num_decks)
    raise ValueError("Must provide either 'seed' or 'cards'")
