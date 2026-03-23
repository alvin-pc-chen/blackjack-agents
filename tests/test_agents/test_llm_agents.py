"""Tests for LLM agents with mocked API calls."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from blackjack21 import Action

from blackjack_agents.agents.base import CardView, GameContext
from blackjack_agents.agents.llm.base_llm_agent import (
    BaseLLMAgent,
    FEW_SHOT_EXAMPLES,
    _format_card,
    _format_hand,
)
from tests.conftest import make_context


class MockLLMAgent(BaseLLMAgent):
    """Test agent that returns a predetermined action."""

    def __init__(self, action: str = "hit", **kwargs: Any) -> None:
        super().__init__(model="mock-model", **kwargs)
        self._mock_action = action
        self.call_count = 0
        self.last_messages: list[dict[str, Any]] = []
        self.last_available: set[str] = set()

    def _call_structured(
        self,
        messages: list[dict[str, Any]],
        available_actions: set[str],
    ) -> str:
        self.call_count += 1
        self.last_messages = messages
        self.last_available = available_actions
        return self._mock_action

    @property
    def agent_type(self) -> str:
        return "MockLLMAgent"


class TestBaseLLMAgent:
    def test_decide_returns_valid_action(self) -> None:
        agent = MockLLMAgent(action="hit")
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        assert agent.decide(ctx) == Action.HIT

    def test_decide_stand(self) -> None:
        agent = MockLLMAgent(action="stand")
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        assert agent.decide(ctx) == Action.STAND

    def test_decide_double(self) -> None:
        agent = MockLLMAgent(action="double")
        ctx = make_context(
            available_actions=frozenset({Action.HIT, Action.STAND, Action.DOUBLE})
        )
        assert agent.decide(ctx) == Action.DOUBLE

    def test_decide_split(self) -> None:
        agent = MockLLMAgent(action="split")
        ctx = make_context(
            available_actions=frozenset({Action.HIT, Action.STAND, Action.SPLIT})
        )
        assert agent.decide(ctx) == Action.SPLIT

    def test_decide_surrender(self) -> None:
        agent = MockLLMAgent(action="surrender")
        ctx = make_context(
            available_actions=frozenset({Action.HIT, Action.STAND, Action.SURRENDER})
        )
        assert agent.decide(ctx) == Action.SURRENDER

    def test_invalid_action_falls_back_to_stand(self) -> None:
        agent = MockLLMAgent(action="fold")  # not a valid blackjack action
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        result = agent.decide(ctx)
        assert result == Action.STAND
        assert agent.call_count == 3  # retried max_retries times

    def test_exception_falls_back_to_stand(self) -> None:
        agent = MockLLMAgent(action="hit")
        agent._call_structured = MagicMock(side_effect=RuntimeError("API down"))  # type: ignore[method-assign]
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        result = agent.decide(ctx)
        assert result == Action.STAND

    def test_messages_include_few_shot_examples(self) -> None:
        agent = MockLLMAgent(action="stand")
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        agent.decide(ctx)

        messages = agent.last_messages
        # Should have: 5 few-shot pairs (user + assistant each) + 1 current situation
        # = 5*2 + 1 = 11 messages
        assert len(messages) == 11

        # First message should be a user message (few-shot)
        assert messages[0]["role"] == "user"
        # Second should be assistant with tool_use
        assert messages[1]["role"] == "assistant"
        assert "tool_use" in messages[1]

        # Last message should be current situation (user)
        assert messages[-1]["role"] == "user"

    def test_available_actions_passed_to_call(self) -> None:
        agent = MockLLMAgent(action="hit")
        ctx = make_context(
            available_actions=frozenset({Action.HIT, Action.STAND, Action.DOUBLE})
        )
        agent.decide(ctx)
        assert agent.last_available == {"hit", "stand", "double"}

    def test_situation_includes_hand_and_dealer(self) -> None:
        agent = MockLLMAgent(action="stand")
        ctx = make_context(
            hand_cards=[CardView("Hearts", "K", 10), CardView("Spades", "8", 8)],
            hand_total=18,
            dealer_upcard=CardView("Diamonds", "6", 6),
            available_actions=frozenset({Action.HIT, Action.STAND}),
        )
        agent.decide(ctx)

        situation = agent.last_messages[-1]["content"]
        assert "K of Hearts" in situation
        assert "8 of Spades" in situation
        assert "18" in situation
        assert "6 of Diamonds" in situation

    def test_situation_includes_card_counting_when_enabled(self) -> None:
        agent = MockLLMAgent(action="stand", include_card_counting=True)
        ctx = make_context(
            available_actions=frozenset({Action.HIT, Action.STAND}),
            running_count=5,
            true_count=2.5,
        )
        agent.decide(ctx)

        situation = agent.last_messages[-1]["content"]
        assert "running=5" in situation
        assert "true=2.5" in situation

    def test_situation_excludes_card_counting_by_default(self) -> None:
        agent = MockLLMAgent(action="stand")
        ctx = make_context(
            available_actions=frozenset({Action.HIT, Action.STAND}),
            running_count=5,
        )
        agent.decide(ctx)

        situation = agent.last_messages[-1]["content"]
        assert "running" not in situation


class TestFormatHelpers:
    def test_format_card(self) -> None:
        assert _format_card(CardView("Hearts", "K", 10)) == "K of Hearts"

    def test_format_hand_hard(self) -> None:
        cards = [CardView("Hearts", "K", 10), CardView("Spades", "5", 5)]
        result = _format_hand(cards, 15, False)
        assert result == "[K of Hearts, 5 of Spades] = 15"

    def test_format_hand_soft(self) -> None:
        cards = [CardView("Hearts", "A", 11), CardView("Spades", "6", 6)]
        result = _format_hand(cards, 17, True)
        assert result == "[A of Hearts, 6 of Spades] = 17 (soft)"


class TestGroqAgent:
    """Tests for GroqAgent with mocked Groq client."""

    def _make_agent(self, response_json: str = '{"action": "hit"}') -> Any:
        """Create a GroqAgent with a mocked client."""
        from blackjack_agents.agents.llm.groq_agent import GroqAgent

        agent = GroqAgent(api_key="fake-key", model="openai/gpt-oss-20b")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = response_json
        mock_client.chat.completions.create.return_value = mock_response
        agent._client = mock_client

        return agent, mock_client

    def test_decide_hit(self) -> None:
        agent, _ = self._make_agent('{"action": "hit"}')
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        assert agent.decide(ctx) == Action.HIT

    def test_decide_stand(self) -> None:
        agent, _ = self._make_agent('{"action": "stand"}')
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        assert agent.decide(ctx) == Action.STAND

    def test_decide_double(self) -> None:
        agent, _ = self._make_agent('{"action": "double"}')
        ctx = make_context(
            available_actions=frozenset({Action.HIT, Action.STAND, Action.DOUBLE})
        )
        assert agent.decide(ctx) == Action.DOUBLE

    def test_uses_json_schema_response_format(self) -> None:
        agent, mock_client = self._make_agent('{"action": "stand"}')
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        agent.decide(ctx)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        rf = call_kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        # Enum contains all 5 actions (static schema for Groq compatibility)
        schema = rf["json_schema"]["schema"]
        assert set(schema["properties"]["action"]["enum"]) == {
            "hit", "stand", "double", "split", "surrender",
        }

    def test_few_shot_examples_in_messages(self) -> None:
        agent, mock_client = self._make_agent('{"action": "hit"}')
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        agent.decide(ctx)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        # system + 5 few-shot pairs (user+assistant) + 1 current = 1 + 10 + 1 = 12
        assert len(messages) == 12
        assert messages[0]["role"] == "system"
        # Few-shot assistant messages should be JSON strings
        assert messages[2]["role"] == "assistant"
        import json
        parsed = json.loads(messages[2]["content"])
        assert "action" in parsed

    def test_invalid_json_retries(self) -> None:
        agent, mock_client = self._make_agent("not json")
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        result = agent.decide(ctx)
        assert result == Action.STAND  # fallback after retries
        assert mock_client.chat.completions.create.call_count == 3

    def test_model_is_gpt_oss(self) -> None:
        agent, mock_client = self._make_agent('{"action": "hit"}')
        ctx = make_context(available_actions=frozenset({Action.HIT, Action.STAND}))
        agent.decide(ctx)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-oss-20b"

    def test_agent_type(self) -> None:
        agent, _ = self._make_agent()
        assert agent.agent_type == "GroqAgent(openai/gpt-oss-20b)"


class TestFewShotExamples:
    def test_examples_are_well_formed(self) -> None:
        valid_actions = {"hit", "stand", "double", "split", "surrender"}
        for user_msg, action in FEW_SHOT_EXAMPLES:
            assert isinstance(user_msg, str)
            assert action in valid_actions
            assert "Your hand:" in user_msg
            assert "Dealer upcard:" in user_msg
            assert "Available actions:" in user_msg

    def test_examples_cover_all_actions(self) -> None:
        actions_covered = {action for _, action in FEW_SHOT_EXAMPLES}
        assert actions_covered == {"hit", "stand", "double", "split", "surrender"}
