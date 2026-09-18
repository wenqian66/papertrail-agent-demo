import asyncio
import json
from pathlib import Path

import pytest

from papertrail_agent_demo import agent_layer
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


def test_single_focus_stays_verbatim_and_single():
    focus = "Explain why the authors chose this baseline."
    agreement = agent_layer.build_agreement_fallback(focus)
    assert [item["original_text"] for item in agreement["objectives"]] == [focus]


def test_compound_focus_splits_into_exact_source_spans():
    focus = "What problem does the paper solve and what is its main result?"
    agreement = agent_layer.build_agreement_fallback(focus)
    assert [item["original_text"] for item in agreement["objectives"]] == [
        "What problem does the paper solve",
        "and what is its main result?",
    ]
    assert agent_layer.validate_agreement(agreement, focus_text=focus) == agreement


def test_multiline_focus_keeps_every_word_and_original_order():
    focus = "Read for:\n1. What problem matters?\n2. How is the system evaluated?"
    agreement = agent_layer.build_agreement_fallback(focus)
    originals = [item["original_text"] for item in agreement["objectives"]]
    assert originals == [
        "Read for:\n1. What problem matters?",
        "2. How is the system evaluated?",
    ]


def test_model_paraphrase_and_dropped_dimension_are_rejected():
    focus = "Find the motivation.\nFind the evaluation."
    paraphrased = {
        "objectives": [{
            "original_text": "Explain the motivation.",
            "extraction_guidance": {
                "look_in": ["introduction"],
                "signals": ["problem"],
                "exclude": ["results"],
                "edge_cases": "Keep implicit motivation.",
            },
        }]
    }
    with pytest.raises(ValueError, match="exact ordered focus substring"):
        agent_layer.validate_agreement(paraphrased, focus_text=focus)

    dropped = agent_layer.build_agreement_fallback("Find the motivation.")
    with pytest.raises(ValueError, match="dropped, added, or reordered"):
        agent_layer.validate_agreement(dropped, focus_text=focus)


def test_refinement_changes_guidance_and_freezes_objectives():
    focus = "Find the proposed solution."
    current = agent_layer.build_agreement_fallback(focus)
    updated = agent_layer.refine_agreement_fallback(
        current, "Remove implementation details and only keep top 5.",
    )
    assert updated["objectives"][0]["original_text"] == focus
    guidance = updated["objectives"][0]["extraction_guidance"]
    assert "implementation details" in guidance["exclude"]
    assert "at most 5" in guidance["edge_cases"]


def test_refine_filters_existing_pool_without_reextracting(tmp_path):
    store = _store(tmp_path)
    focus = "Find the proposed solution."
    current = agent_layer.build_agreement_fallback(focus)
    asyncio.run(store.save_built_agreement(current, focus))
    updated = agent_layer.refine_agreement_fallback(
        current, "Remove implementation details and only keep top 1.",
    )
    report = asyncio.run(store.refilter_with_agreement(
        updated, focus, "Remove implementation details and only keep top 1.",
    ))
    highlights = asyncio.run(store.current_highlights())
    visible = [
        item
        for section in highlights["sections"]
        for item in section["highlights"]
    ]
    assert report["re_extracted"] is False
    assert report["pool_size"] == 3
    assert report["kept"] == 1
    assert "Implementation uses" not in visible[0]["quote"]


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
