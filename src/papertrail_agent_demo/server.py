"""PaperTrail MCP server: resources, tools, and prompts for agent reading."""

from __future__ import annotations

import os
from typing import Any

from mcp.server import MCPServer

from .agent_layer import (
    agreement_from_model,
    agreement_to_text,
    build_agreement_fallback,
    build_agreement_prompt,
    critical_analysis_prompt,
    find_relevant_passages,
    griswold_prompt,
    grounded_qa_prompt,
    guided_reading_prompt,
    paper_comparison_prompt,
    refine_agreement_fallback,
    refine_agreement_prompt,
    refined_agreement_from_model,
)
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


class AgentModel:
    """Optional OpenAI-compatible model used by MCP tools and the demo CLI."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("PAPERTRAIL_AGENT_API_KEY") or os.environ.get(
            "CUSTOM_API_KEY", ""
        )
        self.base_url = os.environ.get("PAPERTRAIL_AGENT_BASE_URL") or os.environ.get(
            "CUSTOM_BASE_URL", ""
        )
        self.model = os.environ.get("PAPERTRAIL_AGENT_MODEL") or os.environ.get(
            "CUSTOM_MODEL_ID", ""
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    async def complete(self, prompt: str) -> str:
        if not self.configured:
            raise RuntimeError("No agent model is configured")
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
            return (response.choices[0].message.content or "").strip()
        finally:
            await client.close()


agent_model = AgentModel()


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


async def _build(focus_text: str) -> tuple[dict[str, Any], str, str | None]:
    sections = await store.paper_sections()
    if agent_model.configured:
        try:
            raw = await agent_model.complete(
                build_agreement_prompt(focus_text, _paper_list_for_agreement(sections))
            )
            return agreement_from_model(focus_text, raw), "llm", None
        except Exception as exc:
            fallback = build_agreement_fallback(focus_text)
            return fallback, "deterministic_fallback", str(exc)
    return build_agreement_fallback(focus_text), "deterministic_fallback", None


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
        "The currently shown section-scoped highlights, selected only from the exact highlight "
        "records already present in the exported Map JSON."
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
        "The active structured extraction agreement, including exact focus text and per-objective "
        "guidance. Legacy plain-text agreements are labeled rather than silently converted."
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
        "Split a focus into exact, verbatim objectives and add extraction guidance. The tool rejects "
        "model output that rewrites, reorders, or drops focus words."
    ),
)
async def build_agreement(focus_text: str) -> dict[str, Any]:
    agreement, generation, fallback_reason = await _build(focus_text)
    await store.save_built_agreement(agreement, focus_text)
    return {
        "agreement": agreement,
        "agreement_text": agreement_to_text(agreement),
        "generation": generation,
        "fallback_reason": fallback_reason,
        "focus_preserved": True,
        "next_step": "The agreement is saved in this standalone demo's local state.",
    }


@mcp.tool(
    title="Update agreement",
    description=(
        "Translate a refinement request into guidance changes, preserve every objective verbatim, "
        "then re-filter and re-rank only the highlights already present in Map JSON. This isolated "
        "demo cannot re-run extraction or discover new evidence."
    ),
)
async def update_agreement(refinement: str) -> dict[str, Any]:
    current = await store.current_agreement()
    agreement = current.get("agreement")
    focus_text = current.get("focus_text") or ""
    if not agreement:
        if not focus_text:
            raise ValueError(
                "No structured current agreement is available. Call build_agreement with the focus first."
            )
        agreement, _generation, _reason = await _build(focus_text)

    generation = "deterministic_fallback"
    fallback_reason = None
    if agent_model.configured:
        try:
            raw = await agent_model.complete(refine_agreement_prompt(agreement, refinement))
            updated = refined_agreement_from_model(agreement, raw)
            generation = "llm"
        except Exception as exc:
            updated = refine_agreement_fallback(agreement, refinement)
            fallback_reason = str(exc)
    else:
        updated = refine_agreement_fallback(agreement, refinement)

    filter_report = await store.refilter_with_agreement(
        updated, focus_text, refinement,
    )
    return {
        "agreement": updated,
        "agreement_text": agreement_to_text(updated),
        "generation": generation,
        "fallback_reason": fallback_reason,
        "objectives_preserved": True,
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
