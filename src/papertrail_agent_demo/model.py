"""OpenAI-compatible model client used by explicit LLM-required paths."""

from __future__ import annotations

import os


class AgentModel:
    def __init__(
        self, *, api_key: str = "", model: str = "", base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    @classmethod
    def from_env(cls) -> "AgentModel":
        return cls(
            api_key=(
                os.environ.get("PAPERTRAIL_AGENT_API_KEY")
                or os.environ.get("CUSTOM_API_KEY", "")
            ),
            model=(
                os.environ.get("PAPERTRAIL_AGENT_MODEL")
                or os.environ.get("CUSTOM_MODEL_ID", "")
            ),
            base_url=(
                os.environ.get("PAPERTRAIL_AGENT_BASE_URL")
                or os.environ.get("CUSTOM_BASE_URL", "")
            ),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    def require(self, operation: str) -> None:
        if not self.configured:
            raise RuntimeError(
                f"{operation} requires an LLM. Set PAPERTRAIL_AGENT_API_KEY "
                "and PAPERTRAIL_AGENT_MODEL; there is no deterministic fallback."
            )

    async def complete(self, prompt: str, *, operation: str = "This operation") -> str:
        self.require(operation)
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url or None,
            timeout=60,
            max_retries=0,
        )
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError(f"{operation} returned an empty model response")
            return text
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith(operation):
                raise
            raise RuntimeError(f"{operation} failed: {exc}") from exc
        finally:
            await client.close()
