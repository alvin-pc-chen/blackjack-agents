"""OpenAI LLM agent."""

from __future__ import annotations

import os

from .base_llm_agent import BaseLLMAgent


class OpenAIAgent(BaseLLMAgent):
    """Blackjack agent powered by OpenAI's models."""

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
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install 'blackjack-agents[llm]'"
                ) from None
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model,
            max_tokens=50,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content  # type: ignore[union-attr]

    @property
    def agent_type(self) -> str:
        return f"OpenAIAgent({self._model})"
