"""Simple heuristic agent — stand on 17+, hit below."""

from __future__ import annotations

from blackjack21 import Action

from .base import Agent, GameContext


class SimpleAgent(Agent):
    """Stand on 17 or higher, hit on anything below. Never splits/doubles/surrenders."""

    def __init__(self, stand_threshold: int = 17) -> None:
        self._threshold = stand_threshold

    def decide(self, context: GameContext) -> Action:
        if context.hand_total >= self._threshold:
            return Action.STAND
        return Action.HIT
