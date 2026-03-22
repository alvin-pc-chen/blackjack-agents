"""Card counting agent using Hi-Lo system with Illustrious 18 deviations."""

from __future__ import annotations

from blackjack21 import Action

from .base import Agent, GameContext
from .basic_strategy import BasicStrategyAgent


# Illustrious 18 deviations: (player_total, dealer_upcard_value, is_soft, deviation_action, threshold)
# If true_count >= threshold, deviate from basic strategy to deviation_action
# Sorted roughly by importance (EV impact)
ILLUSTRIOUS_18: list[tuple[int, int, bool, Action, float]] = [
    # Insurance is not modeled here (would need separate handling)
    (16, 10, False, Action.STAND, 0),     # Stand 16 vs 10 at TC >= 0 (instead of hit)
    (15, 10, False, Action.STAND, 4),     # Stand 15 vs 10 at TC >= +4
    (10, 10, False, Action.DOUBLE, 4),    # Double 10 vs 10 at TC >= +4
    (12,  3, False, Action.STAND, 2),     # Stand 12 vs 3 at TC >= +2
    (12,  2, False, Action.STAND, 3),     # Stand 12 vs 2 at TC >= +3
    (11, 11, False, Action.DOUBLE, 1),    # Double 11 vs A at TC >= +1
    (9,   2, False, Action.DOUBLE, 1),    # Double 9 vs 2 at TC >= +1
    (10, 11, False, Action.DOUBLE, 4),    # Double 10 vs A at TC >= +4
    (9,   7, False, Action.DOUBLE, 3),    # Double 9 vs 7 at TC >= +3
    (16,  9, False, Action.STAND, 5),     # Stand 16 vs 9 at TC >= +5
    (13,  2, False, Action.HIT, -1),      # Hit 13 vs 2 at TC <= -1
    (12,  4, False, Action.HIT, 0),       # Hit 12 vs 4 at TC < 0
    (12,  5, False, Action.HIT, -2),      # Hit 12 vs 5 at TC <= -2
    (12,  6, False, Action.HIT, -1),      # Hit 12 vs 6 at TC <= -1
    (13,  3, False, Action.HIT, -2),      # Hit 13 vs 3 at TC <= -2
]

# Deviations where we deviate BELOW the threshold (negative deviations)
_NEGATIVE_DEVIATIONS: set[int] = {10, 11, 12, 13, 14}  # indices in ILLUSTRIOUS_18


class CardCountingAgent(Agent):
    """Extends basic strategy with Hi-Lo card counting and Illustrious 18 deviations."""

    def __init__(self) -> None:
        self._basic = BasicStrategyAgent()

    def decide(self, context: GameContext) -> Action:
        tc = context.true_count

        # Check Illustrious 18 deviations
        for i, (total, dealer_val, is_soft, dev_action, threshold) in enumerate(ILLUSTRIOUS_18):
            if context.hand_total != total:
                continue
            if context.dealer_upcard.value != dealer_val:
                continue
            if context.hand_is_soft != is_soft:
                continue

            # Negative deviations: deviate when TC <= threshold
            if i in _NEGATIVE_DEVIATIONS:
                if tc <= threshold and dev_action in context.available_actions:
                    return dev_action
            else:
                # Positive deviations: deviate when TC >= threshold
                if tc >= threshold and dev_action in context.available_actions:
                    return dev_action

        # Fall back to basic strategy
        return self._basic.decide(context)

    @property
    def agent_type(self) -> str:
        return "CardCountingAgent"
