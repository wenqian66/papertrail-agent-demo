import json
from pathlib import Path

import pytest
from mcp import Client

from papertrail_agent_demo import server
from papertrail_agent_demo.runner import tool_value
from papertrail_agent_demo.store import ExportConfig, ExportStore


class FakeModel:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []
        self.model = "mock-model"
        self.configured = True

    def require(self, operation: str) -> None:
        return None

    async def complete(self, prompt: str, *, operation: str = "operation") -> str:
        self.prompts.append((operation, prompt))
        if not self.responses:
            raise AssertionError(f"Unexpected model call: {operation}")
        return self.responses.pop(0)


class MissingModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.model = ""
        self.configured = False

    async def complete(self, prompt: str, *, operation: str = "operation") -> str:
        raise RuntimeError(
            f"{operation} requires an LLM; there is no deterministic fallback."
        )


def _map_payload() -> dict:
    return {
        "document": "paper.pdf",
        "display_name": "Paper",
        "pages": 1,
        "summary": None,
        "summary_kind": None,
        "sections": [{
            "title": "Introduction",
            "page_start": 1,
            "page_end": 1,
            "highlights": [{
                "quote": "The system addresses a practical deployment problem.",
                "page": 1,
                "run_id": "run1",
                "run_name": "Motivation",
                "version": 1,
                "color": "#fbe3ab",
                "note": "States the practical motivation.",
                "source": "pipeline",
                "verdict": "",
            }, {
                "quote": "Implementation uses a custom runtime.",
                "page": 1,
                "run_id": "run1",
                "run_name": "Motivation",
                "version": 1,
                "color": "#fbe3ab",
                "note": "Implementation detail.",
                "source": "pipeline",
                "verdict": "",
            }],
        }],
    }


def _agreement_response() -> str:
    return json.dumps({
        "focus_raw": "Find the motivation.",
        "objectives": [{
            "id": "obj1",
            "source_text": "Find the motivation.",
            "facet": "motivation",
            "guidance": ["problem framing", "practical need"],
        }],
    })


def _tag_response() -> str:
    return json.dumps({"tags": [
        {
            "id": "0:0",
            "primary_objective_id": "obj1",
            "secondary_objective_ids": [],
            "salience": 0.95,
        },
        {
            "id": "0:1",
            "primary_objective_id": "other",
            "secondary_objective_ids": [],
            "salience": 0.2,
        },
    ]})


def _multi_agreement_response() -> str:
    return json.dumps({
        "focus_raw": "Find motivation and evaluation.",
        "objectives": [{
            "id": "obj1",
            "source_text": "Find motivation",
            "facet": "motivation",
            "guidance": ["problem framing"],
        }, {
            "id": "obj2",
            "source_text": "evaluation.",
            "facet": "evaluation",
            "guidance": ["reported evidence"],
        }],
    })


def _multi_tag_response() -> str:
    return json.dumps({"tags": [
        {
            "id": "0:0",
            "primary_objective_id": "obj1",
            "secondary_objective_ids": ["obj2"],
            "salience": 0.95,
        },
        {
            "id": "0:1",
            "primary_objective_id": "other",
            "secondary_objective_ids": [],
            "salience": 0.2,
        },
    ]})


@pytest.fixture
def offline_mcp(tmp_path, monkeypatch):
    map_path = tmp_path / "paper.map.json"
    map_path.write_text(json.dumps(_map_payload()), encoding="utf-8")
    store = ExportStore(ExportConfig(
        map_path=map_path,
        state_path=tmp_path / "state.json",
    ))
    monkeypatch.setattr(server, "store", store)
    return server.mcp


@pytest.mark.asyncio
async def test_mcp_exposes_the_exact_requested_surface(offline_mcp):
    async with Client(offline_mcp) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
    assert {tool.name for tool in tools.tools} == {
        "build_agreement", "update_agreement", "find_passages",
    }
    assert {str(resource.uri) for resource in resources.resources} == {
        "papertrail://paper/sections",
        "papertrail://highlights/current",
        "papertrail://agreement/current",
        "papertrail://session/info",
    }
    assert {prompt.name for prompt in prompts.prompts} == {
        "griswold_reading", "grounded_qa", "guided_reading",
        "critical_analysis", "paper_comparison",
    }


@pytest.mark.asyncio
async def test_build_saves_only_agreement_and_refine_tolerates_untagged_highlights(
    offline_mcp, monkeypatch,
):
    model = FakeModel([
        _agreement_response(),
        '{"operations":[{"op":"set_limit","value":1}]}',
    ])
    monkeypatch.setattr(server, "agent_model", model)
    async with Client(offline_mcp) as client:
        built = tool_value(await client.call_tool(
            "build_agreement", {"focus_text": "Find the motivation."},
        ))
        assert built == json.loads(_agreement_response())

        saved = await client.read_resource("papertrail://agreement/current")
        assert json.loads(saved.contents[0].text)["agreement"] == built

        refined = tool_value(await client.call_tool(
            "update_agreement", {"refinement": "Only keep the top one."},
        ))
        assert refined["re_extracted"] is False
        assert refined["status"] == "highlights_not_tagged"
        assert refined["highlight_filter"]["criteria_tagging_ran"] is False
        assert len(refined["filtered_highlights"]) == 2

        highlights = await client.read_resource("papertrail://highlights/current")
        current = json.loads(highlights.contents[0].text)
        visible = [
            item
            for section in current["sections"]
            for item in section["highlights"]
        ]
        assert current["source"] == "map_json_raw_pool"
        assert current["tagging"]["status"] == "not_tagged"
        assert len(visible) == 2
        assert not {
            "facet", "salience", "primary_objective_id",
            "secondary_objective_ids", "criterion_tags", "rank",
        } & set(visible[0])

        session = await client.read_resource("papertrail://session/info")
        session_payload = json.loads(session.contents[0].text)
        assert session_payload["highlight_tagging"]["status"] == "not_tagged"
        assert session_payload["highlight_filter"]["status"] == "highlights_not_tagged"
    assert [item[0] for item in model.prompts] == [
        "build_agreement", "update_agreement refinement translation",
    ]


@pytest.mark.asyncio
async def test_new_refine_criterion_does_not_tag_an_untagged_pool(
    offline_mcp, monkeypatch,
):
    model = FakeModel([
        _agreement_response(),
        '{"operations":[{"op":"add_exclude","criterion":"implementation_details"}]}',
    ])
    monkeypatch.setattr(server, "agent_model", model)
    async with Client(offline_mcp) as client:
        await client.call_tool(
            "build_agreement", {"focus_text": "Find the motivation."},
        )
        first = tool_value(await client.call_tool(
            "update_agreement", {"refinement": "Remove implementation details."},
        ))
        assert first["status"] == "highlights_not_tagged"
        assert first["highlight_filter"]["criteria_tagging_ran"] is False
        assert len(first["filtered_highlights"]) == 2
    assert [operation for operation, _ in model.prompts] == [
        "build_agreement", "update_agreement refinement translation",
    ]


@pytest.mark.asyncio
async def test_multi_objective_build_returns_only_agreement_without_tagging(
    offline_mcp, monkeypatch,
):
    model = FakeModel([
        _multi_agreement_response(),
    ])
    monkeypatch.setattr(server, "agent_model", model)
    async with Client(offline_mcp) as client:
        built = tool_value(await client.call_tool(
            "build_agreement",
            {"focus_text": "Find motivation and evaluation."},
        ))
        assert built == json.loads(_multi_agreement_response())
        highlights = await client.read_resource("papertrail://highlights/current")
        current = json.loads(highlights.contents[0].text)
        assert current["tagging"]["status"] == "not_tagged"
    assert [operation for operation, _ in model.prompts] == ["build_agreement"]


@pytest.mark.asyncio
async def test_build_errors_clearly_without_model(offline_mcp, monkeypatch):
    monkeypatch.setattr(server, "agent_model", MissingModel())
    async with Client(offline_mcp) as client:
        result = await client.call_tool(
            "build_agreement", {"focus_text": "Find the motivation."},
        )
    assert result.is_error is True
    assert "no deterministic fallback" in str(result.content)


@pytest.mark.asyncio
async def test_build_rejects_invalid_source_and_regenerates_once(
    offline_mcp, monkeypatch,
):
    invalid = json.dumps({
        "focus_raw": "Find the motivation.",
        "objectives": [{
            "id": "obj1",
            "source_text": "Explain the motivation.",
            "facet": "motivation",
            "guidance": ["problem framing"],
        }],
    })
    model = FakeModel([invalid, _agreement_response()])
    monkeypatch.setattr(server, "agent_model", model)
    async with Client(offline_mcp) as client:
        result = await client.call_tool(
            "build_agreement", {"focus_text": "Find the motivation."},
        )
    assert result.is_error is False
    assert [operation for operation, _ in model.prompts] == [
        "build_agreement", "build_agreement validation retry",
    ]


@pytest.mark.asyncio
async def test_find_passages_remains_deterministic(offline_mcp):
    async with Client(offline_mcp) as client:
        found = tool_value(await client.call_tool(
            "find_passages",
            {"question": "What practical problem is addressed?", "limit": 3},
        ))
    assert found["addressed"] is True
    assert found["passages"][0]["section"] == "Introduction"


def test_runtime_source_has_no_sibling_service_coupling():
    source_root = Path(__file__).resolve().parents[1] / "src"
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.rglob("*.py")
    )
    forbidden = ["papertrail-main", "ui/backend", "PAPERTRAIL_API_URL"]
    assert not [term for term in forbidden if term in combined]
