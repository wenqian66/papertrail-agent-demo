import json

import pytest
from mcp import Client

from papertrail_agent_demo import server
from papertrail_agent_demo.runner import ChatRunner
from papertrail_agent_demo.store import ExportConfig, ExportStore


class FakeRunnerModel:
    def __init__(self, responses=(), configured=True):
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []
        self.configured = configured

    def require(self, operation: str) -> None:
        if not self.configured:
            raise RuntimeError(f"{operation} requires an LLM; no fallback is available")

    async def complete(self, prompt: str, *, operation: str = "operation") -> str:
        self.require(operation)
        self.prompts.append((operation, prompt))
        if not self.responses:
            raise AssertionError(f"Unexpected model call: {operation}")
        return self.responses.pop(0)


def _map_payload() -> dict:
    return {
        "document": "paper.pdf",
        "display_name": "Paper",
        "pages": 3,
        "summary": None,
        "summary_kind": None,
        "sections": [{
            "title": "Introduction",
            "page_start": 1,
            "page_end": 1,
            "highlights": [{
                "quote": "The deployment problem is expensive manual configuration.",
                "page": 1,
                "run_id": "run1",
                "run_name": "Motivation",
                "version": 1,
                "color": "#fbe3ab",
                "note": "Motivation",
                "source": "pipeline",
                "verdict": "",
            }],
        }, {
            "title": "Evaluation",
            "page_start": 2,
            "page_end": 3,
            "highlights": [{
                "quote": "The strongest baseline is selected for a conservative comparison.",
                "page": 2,
                "run_id": "run1",
                "run_name": "Motivation",
                "version": 1,
                "color": "#fbe3ab",
                "note": "Baseline rationale",
                "source": "pipeline",
                "verdict": "",
            }],
        }],
    }


@pytest.fixture
def runner_mcp(tmp_path, monkeypatch):
    map_path = tmp_path / "paper.map.json"
    map_path.write_text(json.dumps(_map_payload()), encoding="utf-8")
    monkeypatch.setattr(server, "store", ExportStore(ExportConfig(
        map_path=map_path,
        state_path=tmp_path / "state.json",
    )))
    return server.mcp


@pytest.mark.asyncio
async def test_ask_uses_selected_text_as_primary_and_returns_sources(runner_mcp):
    selected = "The strongest baseline is selected for a conservative comparison."
    response = json.dumps({
        "answer": "It provides a conservative comparison [Evaluation, p. 2].",
        "supporting_sources": [{
            "quote": selected,
            "section": "Evaluation",
            "page": 2,
        }],
    })
    model = FakeRunnerModel([response])
    async with Client(runner_mcp) as client:
        result = await ChatRunner(client, model).ask(
            "Why this baseline?", selected_text=selected,
        )
    assert result["supporting_sources"][0]["quote"] == selected
    assert result["grounding"]["primary"]["source"] == "selected_text"
    assert result["grounding"]["retrieval"].startswith("selected_text_primary")
    assert "Primary context" in model.prompts[0][1]


@pytest.mark.asyncio
async def test_ask_reports_unsupported_without_calling_model(runner_mcp):
    model = FakeRunnerModel([], configured=False)
    async with Client(runner_mcp) as client:
        result = await ChatRunner(client, model).ask(
            "xylophone quasar marmalade?",
        )
    assert result["answer"] == "The paper does not directly address this."
    assert result["supporting_sources"] == []
    assert model.prompts == []


@pytest.mark.asyncio
async def test_ask_with_evidence_requires_model(runner_mcp):
    model = FakeRunnerModel([], configured=False)
    async with Client(runner_mcp) as client:
        with pytest.raises(RuntimeError, match="requires an LLM"):
            await ChatRunner(client, model).ask("Why this baseline?")


@pytest.mark.asyncio
async def test_ask_model_may_reject_lexical_false_positive(runner_mcp):
    model = FakeRunnerModel([json.dumps({
        "answer": "The paper does not directly address this.",
        "supporting_sources": [],
    })])
    async with Client(runner_mcp) as client:
        result = await ChatRunner(client, model).ask("Why this baseline?")
    assert result["answer"] == "The paper does not directly address this."
    assert result["supporting_sources"] == []


@pytest.mark.asyncio
async def test_guide_generates_grounded_path_then_questions(runner_mcp):
    highlight = "The deployment problem is expensive manual configuration."
    path_response = json.dumps({"path": [{
        "section": "Introduction",
        "page": 1,
        "highlight": highlight,
        "reason": "Start with the concrete problem that motivates the work.",
    }]})
    questions_response = json.dumps({
        "suggested_questions": [
            "How is manual configuration measured?",
            "Which settings make deployment most expensive?",
            "How does the baseline address this cost?",
        ],
    })
    model = FakeRunnerModel([path_response, questions_response])
    async with Client(runner_mcp) as client:
        result = await ChatRunner(client, model).guide("Understand the argument")
    assert result["reading_path"][0]["section"] == "Introduction"
    assert result["reading_path"][0]["highlight"] == highlight
    assert len(result["suggested_questions"]) == 3
    assert result["context_policy"] == (
        "headings_and_highlights_plus_selective_gap_passages"
    )
    assert [operation for operation, _ in model.prompts] == [
        "Guide reading path", "Guide suggested questions",
    ]
    assert "Section headings and page ranges only" in model.prompts[0][1]


@pytest.mark.asyncio
async def test_selected_text_must_anchor_to_export(runner_mcp):
    model = FakeRunnerModel([])
    async with Client(runner_mcp) as client:
        with pytest.raises(ValueError, match="could not be anchored"):
            await ChatRunner(client, model).ask(
                "What is this?", selected_text="Text absent from this paper",
            )
