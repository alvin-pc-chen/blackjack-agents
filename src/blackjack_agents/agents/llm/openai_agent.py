"""OpenAI LLM agent using function calling for structured output."""

from __future__ import annotations

import json
import os
from typing import Any

from .base_llm_agent import DECIDE_TOOL_SCHEMA, FEW_SHOT_EXAMPLES, BaseLLMAgent


class OpenAIAgent(BaseLLMAgent):
    """Blackjack agent powered by OpenAI's models.

    Uses function calling to get structured action responses.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, **kwargs)  # type: ignore[arg-type]
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install 'blackjack-agents[llm]'"
                ) from None
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def _call_structured(
        self,
        messages: list[dict[str, Any]],
        available_actions: set[str],
    ) -> str:
        """Call OpenAI with function calling and extract the action."""
        client = self._get_client()

        # Build function definition with enum restricted to available actions
        function_def = {
            "type": "function",
            "function": {
                "name": DECIDE_TOOL_SCHEMA["name"],
                "description": DECIDE_TOOL_SCHEMA["description"],
                "parameters": {
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
            },
        }

        # Convert our generic message format to OpenAI's format
        openai_messages = self._to_openai_messages(messages)

        response = client.chat.completions.create(
            model=self._model,
            max_tokens=200,
            temperature=self._temperature,
            messages=openai_messages,
            tools=[function_def],
            tool_choice={"type": "function", "function": {"name": "decide_action"}},
        )

        # Extract action from the function call
        choice = response.choices[0]
        if choice.message.tool_calls:
            call = choice.message.tool_calls[0]
            args = json.loads(call.function.arguments)
            return args.get("action", "stand")

        # Fallback: shouldn't happen with tool_choice forced
        return "stand"

    def _to_openai_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert generic message format to OpenAI's chat messages format.

        Few-shot examples use assistant function_call messages and tool response messages.
        """
        openai_msgs: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        call_id_counter = 0

        for msg in messages:
            if msg["role"] == "user" and "tool_use" not in msg:
                openai_msgs.append({
                    "role": "user",
                    "content": msg["content"],
                })

            elif msg["role"] == "assistant" and "tool_use" in msg:
                # Few-shot assistant response as a function call
                call_id = f"call_fewshot_{call_id_counter}"
                call_id_counter += 1
                action = msg["tool_use"]["action"]

                openai_msgs.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "decide_action",
                                "arguments": json.dumps({"action": action}),
                            },
                        },
                    ],
                })
                # OpenAI requires a tool response after each function call
                openai_msgs.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "Action accepted.",
                })

        return openai_msgs

    @property
    def agent_type(self) -> str:
        return f"OpenAIAgent({self._model})"
