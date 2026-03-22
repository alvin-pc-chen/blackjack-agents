"""Abstract base class for LLM-powered blackjack agents with structured outputs."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from blackjack21 import Action

from ..base import Agent, CardView, GameContext

logger = logging.getLogger(__name__)


def _format_card(card: CardView) -> str:
    return f"{card.rank} of {card.suit}"


def _format_hand(cards: list[CardView], total: int, is_soft: bool) -> str:
    card_str = ", ".join(_format_card(c) for c in cards)
    soft_str = " (soft)" if is_soft else ""
    return f"[{card_str}] = {total}{soft_str}"


# Few-shot examples: (user_message, correct_action)
# These demonstrate the format and teach the model basic blackjack reasoning.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # 1. Hard hand, should hit (low total vs strong dealer)
    (
        "Your hand: [9 of Clubs, 3 of Hearts] = 12\n"
        "Dealer upcard: 10 of Spades\n"
        "Other players: Bob has [K of Hearts, 7 of Diamonds] = 17\n"
        "Available actions: hit, stand",
        "hit",
    ),
    # 2. Hard hand, should stand (strong total vs weak dealer)
    (
        "Your hand: [10 of Diamonds, 8 of Clubs] = 18\n"
        "Dealer upcard: 6 of Hearts\n"
        "Other players: none\n"
        "Available actions: hit, stand",
        "stand",
    ),
    # 3. Soft hand, should double (soft 17 vs weak dealer)
    (
        "Your hand: [A of Spades, 6 of Hearts] = 17 (soft)\n"
        "Dealer upcard: 5 of Diamonds\n"
        "Other players: none\n"
        "Available actions: hit, stand, double",
        "double",
    ),
    # 4. Pair, should split (8s vs 10)
    (
        "Your hand: [8 of Hearts, 8 of Clubs] = 16\n"
        "Dealer upcard: 10 of Diamonds\n"
        "Other players: none\n"
        "Available actions: hit, stand, split",
        "split",
    ),
    # 5. Should surrender (16 vs Ace)
    (
        "Your hand: [10 of Spades, 6 of Hearts] = 16\n"
        "Dealer upcard: A of Clubs\n"
        "Other players: none\n"
        "Available actions: hit, stand, surrender",
        "surrender",
    ),
]

# The tool schema used for structured output (both Claude tool_use and OpenAI function calling)
DECIDE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "decide_action",
    "description": "Choose your blackjack action for this hand.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["hit", "stand", "double", "split", "surrender"],
                "description": "The action to take.",
            },
        },
        "required": ["action"],
    },
}

SYSTEM_PROMPT = (
    "You are playing blackjack at a casino table. Each turn you will be told your hand, "
    "the dealer's face-up card, what other players at the table have, and which actions "
    "are legal. Use the decide_action tool to submit your choice. Pick the action that "
    "gives the best expected outcome."
)

_ACTION_MAP: dict[str, Action] = {
    "hit": Action.HIT,
    "stand": Action.STAND,
    "double": Action.DOUBLE,
    "split": Action.SPLIT,
    "surrender": Action.SURRENDER,
}


class BaseLLMAgent(Agent):
    """Base class for LLM-powered blackjack agents.

    Uses few-shot prompting and structured outputs (tool use / function calling)
    so the model returns a validated action directly.

    Subclasses implement _call_structured() for their specific provider.
    """

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
        include_card_counting: bool = False,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt or SYSTEM_PROMPT
        self._include_card_counting = include_card_counting
        self._temperature = temperature
        self._max_retries = max_retries

    def decide(self, context: GameContext) -> Action:
        messages = self._build_messages(context)
        available = {a.value for a in context.available_actions}  # {"hit", "stand", ...}

        for attempt in range(self._max_retries):
            try:
                action_str = self._call_structured(messages, available)
                action = _ACTION_MAP.get(action_str)
                if action is not None and action in context.available_actions:
                    logger.debug(
                        "LLM %s decided %s (attempt %d)",
                        self.agent_type, action.value, attempt + 1,
                    )
                    return action
                logger.warning(
                    "LLM returned invalid action %r (attempt %d/%d, available: %s)",
                    action_str, attempt + 1, self._max_retries, available,
                )
            except Exception:
                logger.exception(
                    "LLM call failed (attempt %d/%d)", attempt + 1, self._max_retries,
                )

        logger.warning("All LLM attempts failed, falling back to STAND")
        return Action.STAND

    @abstractmethod
    def _call_structured(
        self,
        messages: list[dict[str, Any]],
        available_actions: set[str],
    ) -> str:
        """Call the LLM with tool use and return the action string from the tool call.

        Must return one of: "hit", "stand", "double", "split", "surrender".
        """
        ...

    def _build_messages(self, context: GameContext) -> list[dict[str, Any]]:
        """Build the full message list: few-shot examples + current situation."""
        messages: list[dict[str, Any]] = []

        # Few-shot examples as user/assistant turn pairs
        for user_msg, action in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": user_msg})
            messages.append({
                "role": "assistant",
                "tool_use": {"name": "decide_action", "action": action},
            })

        # Current game situation
        messages.append({"role": "user", "content": self._format_situation(context)})
        return messages

    def _format_situation(self, context: GameContext) -> str:
        """Format the current game state as a concise user message."""
        hand_str = _format_hand(context.hand_cards, context.hand_total, context.hand_is_soft)

        lines = [f"Your hand: {hand_str}"]

        if context.num_hands > 1:
            lines.append(f"(Hand {context.hand_index + 1} of {context.num_hands}, after split)")

        lines.append(f"Dealer upcard: {_format_card(context.dealer_upcard)}")

        if context.other_players:
            others = []
            for op in context.other_players:
                for hv in op.hands:
                    hand = _format_hand(hv.cards, hv.total, False)
                    status = " (bust)" if hv.bust else ""
                    others.append(f"{op.name} has {hand}{status}")
            lines.append(f"Other players: {'; '.join(others)}")
        else:
            lines.append("Other players: none")

        if self._include_card_counting:
            lines.append(
                f"Card count: running={context.running_count}, "
                f"true={context.true_count:.1f}, "
                f"~{context.shoe_remaining} cards remaining"
            )

        actions = ", ".join(sorted(a.value for a in context.available_actions))
        lines.append(f"Available actions: {actions}")

        return "\n".join(lines)
