"""Claude (Anthropic) LLM agent."""

from __future__ import annotations

import os

from .base_llm_agent import BaseLLMAgent


class ClaudeAgent(BaseLLMAgent):
    """Blackjack agent powered by Anthropic's Claude."""

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
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install 'blackjack-agents[llm]'"
                ) from None
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        message = client.messages.create(  # type: ignore[union-attr]
            model=self._model,
            max_tokens=50,
            temperature=self._temperature,
            system=self._system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text  # type: ignore[union-attr]

    @property
    def agent_type(self) -> str:
        return f"ClaudeAgent({self._model})"
