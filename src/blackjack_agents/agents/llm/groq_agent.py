"""Groq LLM agent using response_format json_schema for structured output."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from blackjack21 import Action

from ..base import Agent, GameContext
from .base_llm_agent import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    _format_card,
    _format_hand,
    _ACTION_MAP,
)

logger = logging.getLogger(__name__)


class GroqAgent(Agent):
    """Blackjack agent powered by Groq's API with structured JSON output.

    Uses response_format with json_schema (strict mode) for guaranteed
    valid responses. Few-shot examples are provided as user/assistant
    pairs where the assistant replies with JSON.
    """

    def __init__(
        self,
        *,
        model: str = "openai/gpt-oss-20b",
        api_key: str | None = None,
        include_card_counting: bool = False,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY env var or pass api_key."
            )
        self._include_card_counting = include_card_counting
        self._temperature = temperature
        self._max_retries = max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import groq
            except ImportError:
                raise ImportError(
                    "groq package required. Install with: pip install groq"
                ) from None
            self._client = groq.Groq(api_key=self._api_key)
        return self._client

    def decide(self, context: GameContext) -> Action:
        available = {a.value for a in context.available_actions}
        messages = self._build_messages(context)
        response_format = self._build_response_format(available)

        for attempt in range(self._max_retries):
            try:
                action_str = self._call(messages, response_format)
                action = _ACTION_MAP.get(action_str)
                if action is not None and action in context.available_actions:
                    logger.debug(
                        "Groq decided %s (attempt %d)", action.value, attempt + 1,
                    )
                    return action
                logger.warning(
                    "Groq returned invalid action %r (attempt %d/%d, available: %s)",
                    action_str, attempt + 1, self._max_retries, available,
                )
            except Exception:
                logger.exception(
                    "Groq call failed (attempt %d/%d)", attempt + 1, self._max_retries,
                )

        logger.warning("All Groq attempts failed, falling back to STAND")
        return Action.STAND

    def _call(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any],
    ) -> str:
        client = self._get_client()

        response = client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=messages,
            response_format=response_format,
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed.get("action", "stand")

    def _build_response_format(self, available_actions: set[str]) -> dict[str, Any]:
        """Build a json_schema response_format with enum restricted to available actions."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "blackjack_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": sorted(available_actions),
                            "description": "The blackjack action to take.",
                        },
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }

    def _build_messages(self, context: GameContext) -> list[dict[str, Any]]:
        """Build messages: system prompt, few-shot JSON examples, current situation."""
        system = (
            f"{SYSTEM_PROMPT}\n\n"
            "Respond with a JSON object containing your chosen action. "
            'Example: {"action": "hit"}'
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
        ]

        # Few-shot examples as user/assistant pairs with JSON responses
        for user_msg, action in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": user_msg})
            messages.append({
                "role": "assistant",
                "content": json.dumps({"action": action}),
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

    @property
    def agent_type(self) -> str:
        return f"GroqAgent({self._model})"
