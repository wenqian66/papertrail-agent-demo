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
        {"id": "0:0", "objective_id": "obj1", "facet": "motivation", "salience": 0.95},
        {"id": "0:1", "objective_id": "other", "facet": "other", "salience": 0.2},
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
async def test_build_tags_once_then_refine_filters_without_extraction(
    offline_mcp, monkeypatch,
):
    model = FakeModel([
        _agreement_response(),
        _tag_response(),
        '{"operations":[{"op":"set_limit","value":1}]}',
    ])
    monkeypatch.setattr(server, "agent_model", model)
    async with Client(offline_mcp) as client:
        built = tool_value(await client.call_tool(
            "build_agreement", {"focus_text": "Find the motivation."},
        ))
        assert built["agreement"]["focus_raw"] == "Find the motivation."
        assert built["highlight_tagging"]["tagged"] == 2

        refined = tool_value(await client.call_tool(
            "update_agreement", {"refinement": "Only keep the top one."},
        ))
        assert refined["re_extracted"] is False
        assert refined["highlight_filter"]["criteria_tagging_ran"] is False
        assert len(refined["filtered_highlights"]) == 1

        highlights = await client.read_resource("papertrail://highlights/current")
        current = json.loads(highlights.contents[0].text)
        visible = [
            item
            for section in current["sections"]
            for item in section["highlights"]
        ]
        assert visible[0]["facet"] == "motivation"
        assert visible[0]["salience"] == 0.95
    assert [item[0] for item in model.prompts] == [
        "build_agreement", "highlight tagging",
        "update_agreement refinement translation",
    ]


@pytest.mark.asyncio
async def test_new_refine_criterion_gets_one_disclosed_tagging_pass(
    offline_mcp, monkeypatch,
):
    model = FakeModel([
        _agreement_response(),
        _tag_response(),
        '{"operations":[{"op":"add_exclude","facet":"implementation_details"}]}',
        json.dumps({"criteria": [
            {"id": "0:0", "matches": {"implementation_details": False}},
            {"id": "0:1", "matches": {"implementation_details": True}},
        ]}),
        '{"operations":[{"op":"remove_exclude","facet":"implementation_details"}]}',
        '{"operations":[{"op":"add_exclude","facet":"implementation_details"}]}',
    ])
    monkeypatch.setattr(server, "agent_model", model)
    async with Client(offline_mcp) as client:
        await client.call_tool(
            "build_agreement", {"focus_text": "Find the motivation."},
        )
        first = tool_value(await client.call_tool(
            "update_agreement", {"refinement": "Remove implementation details."},
        ))
        assert first["highlight_filter"]["criteria_tagging_ran"] is True
        assert len(first["filtered_highlights"]) == 1

        await client.call_tool(
            "update_agreement", {"refinement": "Allow implementation details."},
        )
        third = tool_value(await client.call_tool(
            "update_agreement", {"refinement": "Remove implementation details again."},
        ))
        assert third["highlight_filter"]["criteria_tagging_ran"] is False
        assert len(third["filtered_highlights"]) == 1
    assert sum(op == "new refinement criterion tagging" for op, _ in model.prompts) == 1


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
    model = FakeModel([invalid, _agreement_response(), _tag_response()])
    monkeypatch.setattr(server, "agent_model", model)
    async with Client(offline_mcp) as client:
        result = await client.call_tool(
            "build_agreement", {"focus_text": "Find the motivation."},
        )
    assert result.is_error is False
    assert [operation for operation, _ in model.prompts] == [
        "build_agreement", "build_agreement validation retry", "highlight tagging",
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
