from types import SimpleNamespace

import openai
import pytest

from papertrail_agent_demo.model import AgentModel


class FakeAsyncOpenAI:
    last_request: dict | None = None

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    async def create(self, **kwargs):
        type(self).last_request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        )

    async def close(self):
        return None


@pytest.fixture(autouse=True)
def fake_openai(monkeypatch):
    FakeAsyncOpenAI.last_request = None
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)


@pytest.mark.asyncio
async def test_complete_omits_temperature_by_default():
    model = AgentModel(api_key="test-key", model="test-model")

    assert await model.complete("prompt") == "answer"
    assert "temperature" not in FakeAsyncOpenAI.last_request


@pytest.mark.asyncio
async def test_complete_sends_explicit_environment_temperature(monkeypatch):
    monkeypatch.setenv("PAPERTRAIL_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("PAPERTRAIL_AGENT_MODEL", "test-model")
    monkeypatch.setenv("PAPERTRAIL_AGENT_TEMPERATURE", "0.25")
    model = AgentModel.from_env()

    assert await model.complete("prompt") == "answer"
    assert FakeAsyncOpenAI.last_request["temperature"] == 0.25


def test_invalid_environment_temperature_is_rejected(monkeypatch):
    monkeypatch.setenv("PAPERTRAIL_AGENT_TEMPERATURE", "cold")

    with pytest.raises(ValueError, match="must be a number"):
        AgentModel.from_env()
