"""Random baseline agent — selects uniformly from available actions."""

from __future__ import annotations

import random as _random

from blackjack21 import Action

from .base import Agent, GameContext


class RandomAgent(Agent):
    """Selects uniformly at random from available actions."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = _random.Random(seed)

    def decide(self, context: GameContext) -> Action:
        return self._rng.choice(sorted(context.available_actions, key=lambda a: a.value))
