import asyncio
import json
from pathlib import Path

import pytest

from papertrail_agent_demo import agent_layer
from papertrail_agent_demo.model import AgentModel
from papertrail_agent_demo.store import ExportConfig, ExportStore


def _highlight(quote: str, page: int = 1, note: str = "") -> dict:
    return {
        "quote": quote,
        "page": page,
        "run_id": "run1",
        "run_name": "Reading focus",
        "version": 1,
        "color": "#fbe3ab",
        "note": note,
        "source": "pipeline",
        "verdict": "",
    }


def _map_payload() -> dict:
    return {
        "document": "paper.pdf",
        "display_name": "Paper",
        "pages": 2,
        "summary": None,
        "summary_kind": None,
        "sections": [{
            "title": "Introduction",
            "page_start": 1,
            "page_end": 1,
            "highlights": [
                _highlight("The system addresses a practical deployment problem."),
                _highlight("Implementation uses a batch optimizer and custom runtime."),
                _highlight("The strongest baseline exposes the central limitation."),
            ],
        }],
    }


def _store(tmp_path: Path) -> ExportStore:
    map_path = tmp_path / "paper.map.json"
    map_path.write_text(json.dumps(_map_payload()), encoding="utf-8")
    return ExportStore(ExportConfig(
        map_path=map_path,
        state_path=tmp_path / "state.json",
    ))


def _agreement() -> dict:
    return {
        "focus_raw": "I care about why they chose this dataset",
        "objectives": [{
            "id": "obj1",
            "source_text": "why they chose this dataset",
            "facet": "evaluation",
            "guidance": [
                "dataset choice",
                "dataset justification",
                "dataset limitations",
            ],
        }],
    }


def _tags() -> dict[str, dict]:
    return {
        "0:0": {"objective_id": "obj1", "facet": "evaluation", "salience": 0.9},
        "0:1": {"objective_id": "other", "facet": "other", "salience": 0.2},
        "0:2": {"objective_id": "obj1", "facet": "evaluation", "salience": 0.7},
    }


def test_single_goal_agreement_accepts_system_added_expertise():
    agreement = agent_layer.validate_agreement(
        _agreement(), focus_text=_agreement()["focus_raw"],
    )
    assert len(agreement["objectives"]) == 1
    assert agreement["objectives"][0]["guidance"] == [
        "dataset choice", "dataset justification", "dataset limitations",
    ]


def test_validation_rejects_paraphrase_reordering_and_dropped_content():
    focus = "Find the motivation and the evaluation evidence."
    paraphrase = {
        "focus_raw": focus,
        "objectives": [{
            "id": "obj1",
            "source_text": "Explain the motivation",
            "facet": "motivation",
            "guidance": ["problem framing"],
        }],
    }
    with pytest.raises(ValueError, match="exact ordered"):
        agent_layer.validate_agreement(paraphrase, focus_text=focus)

    dropped = {
        "focus_raw": focus,
        "objectives": [{
            "id": "obj1",
            "source_text": "the motivation",
            "facet": "motivation",
            "guidance": ["problem framing"],
        }],
    }
    with pytest.raises(ValueError, match="do not cover substantive"):
        agent_layer.validate_agreement(dropped, focus_text=focus)

    reordered = {
        "focus_raw": focus,
        "objectives": [{
            "id": "obj1",
            "source_text": "the evaluation evidence",
            "facet": "evaluation",
            "guidance": ["reported result"],
        }, {
            "id": "obj2",
            "source_text": "the motivation",
            "facet": "motivation",
            "guidance": ["problem framing"],
        }],
    }
    with pytest.raises(ValueError, match="exact ordered"):
        agent_layer.validate_agreement(reordered, focus_text=focus)


def test_build_prompt_explicitly_prohibits_forced_splitting():
    prompt = agent_layer.build_agreement_prompt("Explain one dataset choice")
    assert "exactly one objective" in prompt
    assert "Positive multi-goal example" in prompt
    assert "Negative example" in prompt


def test_missing_model_has_no_agreement_fallback():
    model = AgentModel()
    with pytest.raises(RuntimeError, match="there is no deterministic fallback"):
        model.require("build_agreement")
    assert not hasattr(agent_layer, "build_agreement_fallback")


def test_refinement_diff_accepts_only_fixed_operations():
    agreement = _agreement()
    result = agent_layer.refinement_diff_from_model(json.dumps({
        "operations": [
            {"op": "set_limit", "value": 5},
            {"op": "add_exclude", "facet": "Implementation Details"},
            {"op": "disable_facet", "id": "obj1"},
            {"op": "change_density", "facet": "evaluation", "level": "low"},
        ],
    }), agreement)
    assert result["operations"][1]["facet"] == "implementation_details"
    with pytest.raises(ValueError, match="unsupported refinement operation"):
        agent_layer.refinement_diff_from_model(
            '{"operations":[{"op":"rewrite_objective"}]}', agreement,
        )


def test_highlight_tags_validate_full_pool_and_objective_facet():
    response = json.dumps({"tags": [
        {"id": "0:0", "objective_id": "obj1", "facet": "evaluation", "salience": 0.8},
        {"id": "0:1", "objective_id": "other", "facet": "other", "salience": 0.1},
    ]})
    result = agent_layer.highlight_tags_from_model(
        response, ["0:0", "0:1"], _agreement(),
    )
    assert result["0:0"]["salience"] == 0.8
    with pytest.raises(ValueError, match="must match"):
        agent_layer.highlight_tags_from_model(json.dumps({"tags": [
            {"id": "0:0", "objective_id": "obj1", "facet": "method", "salience": 0.8},
        ]}), ["0:0"], _agreement())


def test_store_filters_cached_tags_deterministically_without_reextracting(tmp_path):
    store = _store(tmp_path)
    agreement = _agreement()
    asyncio.run(store.save_built_agreement(agreement, _tags(), model_name="mock"))
    report = asyncio.run(store.apply_refinement(
        agreement,
        [
            {"op": "add_exclude", "facet": "implementation_details"},
            {"op": "set_limit", "value": 1},
        ],
        "Remove implementation details and keep the top one",
        {
            "0:0": {"implementation_details": False},
            "0:1": {"implementation_details": True},
            "0:2": {"implementation_details": False},
        },
        model_name="mock",
    ))
    highlights = asyncio.run(store.ranked_highlights())
    assert report["re_extracted"] is False
    assert report["criteria_tagging_ran"] is True
    assert [item["quote"] for item in highlights] == [
        "The system addresses a practical deployment problem.",
    ]
    assert highlights[0]["facet"] == "evaluation"
    assert highlights[0]["salience"] == 0.9


def test_find_passages_returns_section_and_page():
    result = agent_layer.find_relevant_passages(
        "Why use the baseline?",
        [{
            "title": "Evaluation",
            "page_start": 7,
            "text": "We use the baseline because it is the strongest prior system.",
        }],
        [],
        3,
    )
    assert result["addressed"] is True
    assert result["passages"][0]["section"] == "Evaluation"
    assert result["passages"][0]["page"] == 7


def test_export_store_preserves_map_fields_and_reports_missing_pdf(tmp_path):
    store = _store(tmp_path)
    sections = asyncio.run(store.paper_sections())
    assert set(sections["sections"][0]) == {
        "title", "page_start", "page_end", "highlights",
    }
    assert any("No paired PDF" in item for item in sections["limitations"])


def test_bundled_pdf_recovers_searchable_section_text(tmp_path):
    project = Path(__file__).resolve().parents[1]
    store = ExportStore(ExportConfig(
        map_path=project / "fixtures" / "bundle" / "361011.361061.map.json",
        pdf_path=project / "fixtures" / "bundle" / "361011.361061_highlighted.pdf",
        state_path=tmp_path / "state.json",
    ))
    sections = asyncio.run(store.paper_sections())
    assert any(section.get("text") for section in sections["sections"])
    assert any("pypdf" in item for item in sections["limitations"])


def test_griswold_is_opt_in_and_routes_reasoning_tasks():
    prompt = agent_layer.griswold_prompt()
    assert len(agent_layer.GRISWOLD_AGREEMENT["objectives"]) == 5
    assert "questions 4, 7, and 8" in prompt
    assert "critical_analysis" in prompt
