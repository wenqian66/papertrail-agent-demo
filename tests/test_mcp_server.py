import json
from pathlib import Path

import pytest
from mcp import Client

from papertrail_agent_demo import server
from papertrail_agent_demo.store import ExportConfig, ExportStore


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


@pytest.fixture
def offline_mcp(tmp_path, monkeypatch):
    map_path = tmp_path / "paper.map.json"
    map_path.write_text(json.dumps(_map_payload()), encoding="utf-8")
    store = ExportStore(ExportConfig(
        map_path=map_path,
        state_path=tmp_path / "state.json",
    ))
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server.agent_model, "api_key", "")
    monkeypatch.setattr(server.agent_model, "model", "")
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
async def test_mcp_build_refine_filter_and_find_round_trip(offline_mcp):
    async with Client(offline_mcp) as client:
        built = await client.call_tool(
            "build_agreement", {"focus_text": "Find the motivation."},
        )
        assert built.is_error is False

        refined = await client.call_tool(
            "update_agreement",
            {"refinement": "Remove implementation details and only keep top 1."},
        )
        assert refined.is_error is False
        assert refined.structured_content["re_extracted"] is False

        highlights = await client.read_resource("papertrail://highlights/current")
        current = json.loads(highlights.contents[0].text)
        visible = [
            item
            for section in current["sections"]
            for item in section["highlights"]
        ]
        assert len(visible) == 1

        found = await client.call_tool(
            "find_passages",
            {"question": "What practical problem is addressed?", "limit": 3},
        )
        assert found.is_error is False
        assert found.structured_content["addressed"] is True
        assert found.structured_content["passages"][0]["section"] == "Introduction"


def test_runtime_source_has_no_sibling_service_coupling():
    source_root = Path(__file__).resolve().parents[1] / "src"
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.rglob("*.py")
    )
    assert "papertrail-main" not in combined
    assert "ui/backend" not in combined
    assert "PAPERTRAIL_API_URL" not in combined
