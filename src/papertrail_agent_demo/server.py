"""PaperTrail MCP server: resources, tools, and prompts for agent reading."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .agent_layer import (
    agreement_from_model,
    agreement_to_text,
    build_agreement_prompt,
    criterion_tagging_prompt,
    criterion_tags_from_model,
    critical_analysis_prompt,
    find_relevant_passages,
    griswold_prompt,
    grounded_qa_prompt,
    guided_reading_prompt,
    highlight_tagging_prompt,
    highlight_tags_from_model,
    paper_comparison_prompt,
    refinement_diff_from_model,
    refinement_prompt,
)
from .model import AgentModel
from .store import ExportStore


mcp = MCPServer(
    "PaperTrail Agent Layer",
    instructions=(
        "Read PaperTrail resources before answering. Use find_passages for paper questions, "
        "build_agreement for new focus text, and update_agreement for refinements. Never claim "
        "that a standalone Map JSON contains body text it does not contain."
    ),
)
store = ExportStore.from_env()
agent_model = AgentModel.from_env()


def _paper_list_for_agreement(sections_payload: dict[str, Any]) -> str:
    document = sections_payload.get("document") or sections_payload.get("display_name") or "paper"
    titles = [
        section.get("title") or section.get("header") or ""
        for section in sections_payload.get("sections") or []
    ]
    titles = [title for title in titles if title][:40]
    if not titles:
        return f"- {document}"
    return f"- {document}\n  Sections: " + "; ".join(titles)


async def _build(focus_text: str) -> dict[str, Any]:
    sections = await store.paper_sections()
    prompt = build_agreement_prompt(
        focus_text, _paper_list_for_agreement(sections),
    )
    raw = await agent_model.complete(
        prompt,
        operation="build_agreement",
    )
    try:
        return agreement_from_model(focus_text, raw)
    except Exception as first_error:
        correction = f"""{prompt}

Your previous response was rejected by the structural validator:
{first_error}

Previous response:
{raw}

Regenerate the JSON once. Preserve focus_raw byte for byte and correct only
the structural or source-span error described above.
"""
        retried = await agent_model.complete(
            correction,
            operation="build_agreement validation retry",
        )
        try:
            return agreement_from_model(focus_text, retried)
        except Exception as retry_error:
            raise RuntimeError(
                "build_agreement returned invalid agreements twice: "
                f"{retry_error}"
            ) from retry_error


@mcp.resource(
    "papertrail://paper/sections",
    name="paper_sections",
    title="Paper sections",
    description=(
        "The exported paper's section map and, when available, body text recovered from the "
        "paired PDF with pypdf. The original Map JSON section fields are preserved."
    ),
    mime_type="application/json",
)
async def paper_sections() -> dict[str, Any]:
    return await store.paper_sections()


@mcp.resource(
    "papertrail://highlights/current",
    name="current_highlights",
    title="Current highlights",
    description=(
        "The currently shown Map JSON highlights with cached objective, facet, and salience tags. "
        "Tags are created by one disclosed LLM enrichment pass; reads and filtering are deterministic."
    ),
    mime_type="application/json",
)
async def current_highlights() -> dict[str, Any]:
    return await store.current_highlights()


@mcp.resource(
    "papertrail://agreement/current",
    name="current_agreement",
    title="Current agreement",
    description=(
        "The active focus_raw/objectives agreement and deterministic view-filter state."
    ),
    mime_type="application/json",
)
async def current_agreement() -> dict[str, Any]:
    return await store.current_agreement()


@mcp.resource(
    "papertrail://session/info",
    name="session_info",
    title="Session information",
    description=(
        "The papers, runs, versions, and feedback that can be inferred from the Map JSON export."
    ),
    mime_type="application/json",
)
async def session_info() -> dict[str, Any]:
    return await store.session_info()


@mcp.tool(
    title="Build agreement",
    description=(
        "Use an LLM to build exact source-text objectives and system guidance, then tag every existing "
        "highlight once with facet and salience. Requires a configured model and has no fallback."
    ),
)
async def build_agreement(focus_text: str) -> dict[str, Any]:
    try:
        agreement = await _build(focus_text)
        highlight_inputs = store.highlight_inputs()
        raw_tags = await agent_model.complete(
            highlight_tagging_prompt(agreement, highlight_inputs),
            operation="highlight tagging",
        )
        tags = highlight_tags_from_model(
            raw_tags, [item["id"] for item in highlight_inputs], agreement,
        )
        await store.save_built_agreement(
            agreement, tags, model_name=agent_model.model,
        )
    except Exception as exc:
        raise ToolError(str(exc)) from exc
    return {
        "agreement": agreement,
        "agreement_text": agreement_to_text(agreement),
        "generation": "llm",
        "focus_preserved": True,
        "highlight_tagging": {
            "tagged": len(tags),
            "fields": ["objective_id", "facet", "salience"],
            "cached": True,
            "re_extracted": False,
        },
        "next_step": "The agreement and cached highlight tags are saved in local state.",
    }


@mcp.tool(
    title="Update agreement",
    description=(
        "Use an LLM only to translate a request into the fixed refinement operation set. Filtering "
        "then runs deterministically on cached facet and salience tags. A new semantic criterion "
        "causes one disclosed tagging pass. No extraction runs and no new highlight can appear."
    ),
)
async def update_agreement(refinement: str) -> dict[str, Any]:
    try:
        current = await store.current_agreement()
        agreement = current.get("agreement")
        if not agreement:
            raise ValueError(
                "No current agreement is available. Call build_agreement first; an LLM is required."
            )

        known_facets = store.known_facets(agreement)
        raw_diff = await agent_model.complete(
            refinement_prompt(agreement, refinement, known_facets),
            operation="update_agreement refinement translation",
        )
        diff = refinement_diff_from_model(raw_diff, agreement)
        criteria = store.criteria_needed(agreement, diff["operations"])
        criterion_updates = None
        if criteria:
            highlight_inputs = store.highlight_inputs()
            raw_criteria = await agent_model.complete(
                criterion_tagging_prompt(criteria, highlight_inputs),
                operation="new refinement criterion tagging",
            )
            criterion_updates = criterion_tags_from_model(
                raw_criteria,
                [item["id"] for item in highlight_inputs],
                criteria,
            )
        filter_report = await store.apply_refinement(
            agreement,
            diff["operations"],
            refinement,
            criterion_updates,
            model_name=agent_model.model,
        )
    except Exception as exc:
        raise ToolError(str(exc)) from exc
    filtered_highlights = await store.ranked_highlights()
    return {
        "agreement": agreement,
        "agreement_text": agreement_to_text(agreement),
        "filter_state": filter_report["filter_state"],
        "operations": diff["operations"],
        "generation": "llm_diff_then_deterministic_filter",
        "objectives_preserved": True,
        "filtered_highlights": filtered_highlights,
        "highlight_filter": filter_report,
        "re_extracted": False,
    }


@mcp.tool(
    title="Find passages",
    description=(
        "Search active paper text and current highlight quotes for a question. Returns ranked "
        "passages with specific section and page references and reports source limitations."
    ),
)
async def find_passages(question: str, limit: int = 8) -> dict[str, Any]:
    sections_payload = await store.paper_sections()
    result = find_relevant_passages(
        question,
        await store.normalized_sections(),
        await store.normalized_highlights(),
        limit,
    )
    result["document"] = sections_payload.get("document")
    result["limitations"] = sections_payload.get("limitations") or []
    if not result["addressed"]:
        result["message"] = "No relevant passage was found. The paper may not address this question."
    return result


@mcp.prompt(
    title="Griswold reading",
    description=(
        "Opt-in Griswold reading template with five extraction objectives and explicit routing of "
        "analysis questions to critical_analysis."
    ),
)
def griswold_reading() -> str:
    return griswold_prompt()


@mcp.prompt(
    title="Grounded paper question",
    description="Answer a paper question from find_passages with section and page citations.",
)
def grounded_qa(question: str) -> str:
    return grounded_qa_prompt(question)


@mcp.prompt(
    title="Guided reading",
    description="Build a reading path whose every step names a section, highlight, page, and reason.",
)
def guided_reading(goal: str = "Understand the paper") -> str:
    return guided_reading_prompt(goal)


@mcp.prompt(
    title="Critical analysis",
    description="Separate paper claims, inference, and agent opinion in a critical analysis.",
)
def critical_analysis(axis: str = "overall strengths and weaknesses") -> str:
    return critical_analysis_prompt(axis)


@mcp.prompt(
    title="Paper comparison",
    description="Compare session papers on an axis with a paper, section, and page citation per point.",
)
def paper_comparison(axis: str) -> str:
    return paper_comparison_prompt(axis)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
