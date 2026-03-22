"""Claude (Anthropic) LLM agent using tool_use for structured output."""

from __future__ import annotations

import os
from typing import Any

from .base_llm_agent import DECIDE_TOOL_SCHEMA, FEW_SHOT_EXAMPLES, BaseLLMAgent


class ClaudeAgent(BaseLLMAgent):
    """Blackjack agent powered by Anthropic's Claude.

    Uses tool_use to get structured action responses.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, **kwargs)  # type: ignore[arg-type]
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install 'blackjack-agents[llm]'"
                ) from None
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _call_structured(
        self,
        messages: list[dict[str, Any]],
        available_actions: set[str],
    ) -> str:
        """Call Claude with tool_use and extract the action from the tool call."""
        client = self._get_client()

        # Build the tool definition with enum restricted to available actions
        tool = {
            "name": DECIDE_TOOL_SCHEMA["name"],
            "description": DECIDE_TOOL_SCHEMA["description"],
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": sorted(available_actions),
                        "description": "The action to take.",
                    },
                },
                "required": ["action"],
            },
        }

        # Convert our generic message format to Anthropic's format
        anthropic_messages = self._to_anthropic_messages(messages)

        response = client.messages.create(
            model=self._model,
            max_tokens=200,
            temperature=self._temperature,
            system=self._system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": "decide_action"},
            messages=anthropic_messages,
        )

        # Extract action from the tool_use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "decide_action":
                return block.input.get("action", "stand")

        # Fallback: shouldn't happen with tool_choice forced
        return "stand"

    def _to_anthropic_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert generic message format to Anthropic's messages API format.

        Few-shot examples use tool_use content blocks in assistant messages
        and tool_result blocks in user messages (Anthropic requires a tool_result
        after each tool_use).
        """
        anthropic_msgs: list[dict[str, Any]] = []
        tool_use_id_counter = 0

        for msg in messages:
            if msg["role"] == "user" and "tool_use" not in msg:
                # Regular user message
                # Check if previous message was an assistant tool_use — if so,
                # we need to prepend a tool_result to this user turn
                if (anthropic_msgs
                        and anthropic_msgs[-1]["role"] == "assistant"
                        and any(
                            isinstance(b, dict) and b.get("type") == "tool_use"
                            for b in anthropic_msgs[-1].get("content", [])
                        )):
                    # Find the tool_use_id from the previous assistant message
                    prev_tool_id = None
                    for b in anthropic_msgs[-1]["content"]:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            prev_tool_id = b["id"]
                    if prev_tool_id:
                        anthropic_msgs.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": prev_tool_id,
                                    "content": "Action accepted.",
                                },
                                {"type": "text", "text": msg["content"]},
                            ],
                        })
                        continue

                anthropic_msgs.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg["content"]}],
                })

            elif msg["role"] == "assistant" and "tool_use" in msg:
                # Few-shot assistant response as a tool_use block
                tool_use_id = f"fewshot_{tool_use_id_counter}"
                tool_use_id_counter += 1
                anthropic_msgs.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": msg["tool_use"]["name"],
                            "input": {"action": msg["tool_use"]["action"]},
                        },
                    ],
                })

        return anthropic_msgs

    @property
    def agent_type(self) -> str:
        return f"ClaudeAgent({self._model})"
