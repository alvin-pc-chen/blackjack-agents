"""Abstract base class for LLM-powered blackjack agents."""

from __future__ import annotations

import logging
from abc import abstractmethod

from blackjack21 import Action

from ..base import Agent, CardView, GameContext

logger = logging.getLogger(__name__)

_ACTION_KEYWORDS: dict[str, Action] = {
    "hit": Action.HIT,
    "stand": Action.STAND,
    "stay": Action.STAND,
    "double": Action.DOUBLE,
    "double down": Action.DOUBLE,
    "split": Action.SPLIT,
    "surrender": Action.SURRENDER,
}


def _format_card(card: CardView) -> str:
    return f"{card.rank} of {card.suit}"


def _format_hand(cards: list[CardView], total: int) -> str:
    card_str = ", ".join(_format_card(c) for c in cards)
    return f"[{card_str}] (total: {total})"


class BaseLLMAgent(Agent):
    """Base class for LLM-powered blackjack agents.

    Subclasses implement _call_llm() for their specific provider.
    """

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
        include_history: bool = False,
        include_card_counting: bool = False,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._include_history = include_history
        self._include_card_counting = include_card_counting
        self._temperature = temperature
        self._max_retries = max_retries

    def decide(self, context: GameContext) -> Action:
        prompt = self._format_prompt(context)
        for attempt in range(self._max_retries):
            try:
                raw_response = self._call_llm(prompt)
                parsed = self._parse_response(raw_response, context.available_actions)
                if parsed is not None:
                    logger.debug(
                        "LLM %s decided %s (attempt %d)", self.agent_type, parsed.value, attempt + 1
                    )
                    return parsed
                logger.warning(
                    "LLM response unparseable (attempt %d/%d): %s",
                    attempt + 1, self._max_retries, raw_response[:200],
                )
            except Exception:
                logger.exception("LLM call failed (attempt %d/%d)", attempt + 1, self._max_retries)

        logger.warning("All LLM attempts failed, falling back to STAND")
        return Action.STAND

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Send prompt to LLM and return raw text response."""
        ...

    def _format_prompt(self, context: GameContext) -> str:
        actions = ", ".join(a.value.upper() for a in sorted(context.available_actions, key=lambda a: a.value))

        lines = [
            "You are playing blackjack. Choose your action.",
            "",
            f"Your hand: {_format_hand(context.hand_cards, context.hand_total)}",
            f"Soft hand: {'yes' if context.hand_is_soft else 'no'}",
        ]

        if context.num_hands > 1:
            lines.append(f"This is hand {context.hand_index + 1} of {context.num_hands} (after a split)")

        lines.extend([
            f"Dealer's upcard: {_format_card(context.dealer_upcard)}",
            "",
        ])

        if context.other_players:
            lines.append("Other players at the table:")
            for op in context.other_players:
                for hv in op.hands:
                    lines.append(f"  {op.name}: {_format_hand(hv.cards, hv.total)}")
            lines.append("")

        if self._include_card_counting:
            lines.extend([
                f"Running count (Hi-Lo): {context.running_count}",
                f"True count: {context.true_count:.1f}",
                f"Cards remaining in shoe: ~{context.shoe_remaining}",
                "",
            ])

        lines.extend([
            f"Available actions: {actions}",
            "",
            f"Respond with exactly one action: {actions}",
        ])

        return "\n".join(lines)

    def _parse_response(
        self, response: str, valid_actions: frozenset[Action]
    ) -> Action | None:
        """Extract a valid action from the LLM's response text."""
        response_lower = response.strip().lower()

        # Try exact match first
        for keyword, action in _ACTION_KEYWORDS.items():
            if response_lower == keyword and action in valid_actions:
                return action

        # Try keyword search (longest match first to prefer "double down" over "double")
        for keyword, action in sorted(_ACTION_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if keyword in response_lower and action in valid_actions:
                return action

        return None

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are an expert blackjack player. When given a game situation, "
            "respond with exactly one action word from the available actions. "
            "Do not explain your reasoning. Just respond with the action."
        )
