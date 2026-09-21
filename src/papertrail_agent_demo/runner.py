"""Client-side chat orchestration over the PaperTrail MCP surface."""

from __future__ import annotations

import json
from typing import Any

from .agent_layer import parse_json_object
from .model import AgentModel


def tool_value(result: Any) -> Any:
    if getattr(result, "is_error", False):
        messages = [
            getattr(block, "text", "")
            for block in getattr(result, "content", []) or []
            if getattr(block, "text", "")
        ]
        raise RuntimeError("\n".join(messages) or "MCP tool call failed")
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return None


async def read_resource(client: Any, uri: str) -> Any:
    result = await client.read_resource(uri)
    if not result.contents:
        return None
    text = getattr(result.contents[0], "text", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def read_prompt(client: Any, name: str, arguments: dict[str, str]) -> str:
    result = await client.get_prompt(name, arguments)
    return "\n\n".join(
        getattr(message.content, "text", "") for message in result.messages
    )


class ChatRunner:
    """The LLM-backed client; it is deliberately not an MCP component."""

    def __init__(self, client: Any, model: AgentModel):
        self.client = client
        self.model = model

    async def ask(
        self, question: str, *, selected_text: str | None = None,
    ) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("Ask needs a question")

        retrieval = tool_value(await self.client.call_tool(
            "find_passages", {"question": question, "limit": 8},
        ))
        passages = list((retrieval or {}).get("passages") or [])
        primary = None
        if selected_text:
            primary = await self._locate_selected_text(selected_text)
            passages = [
                passage for passage in passages
                if not _same_source(passage, primary)
            ]
        elif passages:
            primary = passages.pop(0)

        evidence = ([primary] if primary else []) + passages
        if not evidence:
            return {
                "answer": "The paper does not directly address this.",
                "supporting_sources": [],
                "grounding": {
                    "primary": None,
                    "supplementary": [],
                    "retrieval": "deterministic_lexical",
                },
            }

        self.model.require("Ask")
        template = await read_prompt(
            self.client, "grounded_qa", {"question": question},
        )
        prompt = f"""{template}

Primary context:
{json.dumps(primary, ensure_ascii=False, indent=2) if primary else "(none)"}

Supplementary evidence:
{json.dumps(passages, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "answer": "a cited answer",
  "supporting_sources": [
    {{"quote":"verbatim supplied quote","section":"exact section","page":1}}
  ]
}}

If the supplied evidence is insufficient, return the required unsupported
sentence as answer and an empty supporting_sources list.
"""
        raw = await self.model.complete(prompt, operation="Ask")
        response = _validated_ask_response(raw, evidence)
        response["grounding"] = {
            "primary": primary,
            "supplementary": passages,
            "retrieval": "selected_text_primary_plus_deterministic_lexical"
            if selected_text else "deterministic_lexical",
        }
        return response

    async def refine(self, request: str) -> dict[str, Any]:
        request = (request or "").strip()
        if not request:
            raise ValueError("Refine needs a request")
        return tool_value(await self.client.call_tool(
            "update_agreement", {"refinement": request},
        ))

    async def guide(self, goal: str = "Understand the paper") -> dict[str, Any]:
        goal = (goal or "Understand the paper").strip()
        sections_payload = await read_resource(
            self.client, "papertrail://paper/sections",
        )
        highlights_payload = await read_resource(
            self.client, "papertrail://highlights/current",
        )
        headings = [{
            "title": section.get("title") or "Untitled section",
            "page_start": section.get("page_start"),
            "page_end": section.get("page_end"),
        } for section in (sections_payload or {}).get("sections") or []]
        highlights = _compact_highlights(highlights_payload or {})
        if not highlights:
            return {
                "reading_path": [],
                "suggested_questions": [],
                "message": "No highlight-grounded reading path can be made.",
            }

        gaps = _missing_links(headings, highlights)
        gap_passages: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for gap in gaps[:3]:
            result = tool_value(await self.client.call_tool(
                "find_passages",
                {"question": f"{goal} {gap['title']}", "limit": 2},
            ))
            for passage in (result or {}).get("passages") or []:
                key = (
                    passage.get("section") or "",
                    int(passage.get("page") or 1),
                    passage.get("text") or "",
                )
                if key not in seen:
                    seen.add(key)
                    gap_passages.append(passage)

        self.model.require("Guide")
        template = await read_prompt(
            self.client, "guided_reading", {"goal": goal},
        )
        path_prompt = f"""{template}

Section headings and page ranges only:
{json.dumps(headings, ensure_ascii=False, indent=2)}

Current highlights:
{json.dumps(highlights, ensure_ascii=False, indent=2)}

Relevant text retrieved only for missing links:
{json.dumps(gap_passages, ensure_ascii=False, indent=2)}

Return JSON only:
{{"path":[{{"section":"exact section","page":1,"highlight":"verbatim current highlight","reason":"why this is the next step"}}]}}
"""
        raw_path = await self.model.complete(path_prompt, operation="Guide reading path")
        reading_path = _validated_reading_path(raw_path, highlights)

        questions_prompt = f"""\
Using only the reading path and grounded context below, suggest three short,
specific questions a reader could ask next about this paper. Do not answer
them and do not introduce facts absent from the context.

Reading path:
{json.dumps(reading_path, ensure_ascii=False, indent=2)}

Grounded highlights:
{json.dumps(highlights, ensure_ascii=False, indent=2)}

Return JSON only: {{"suggested_questions":["question one?","question two?"]}}
"""
        raw_questions = await self.model.complete(
            questions_prompt, operation="Guide suggested questions",
        )
        suggested = _validated_suggested_questions(raw_questions)
        return {
            "reading_path": reading_path,
            "suggested_questions": suggested,
            "gap_passages": gap_passages,
            "context_policy": "headings_and_highlights_plus_selective_gap_passages",
        }

    async def handle(
        self, interaction: str, text: str, *, selected_text: str | None = None,
    ) -> dict[str, Any]:
        if interaction == "ask":
            return await self.ask(text, selected_text=selected_text)
        if interaction == "refine":
            return await self.refine(text)
        if interaction == "guide":
            return await self.guide(text)
        raise ValueError(f"Unknown interaction: {interaction}")

    async def _locate_selected_text(self, selected_text: str) -> dict[str, Any]:
        selected_text = selected_text.strip()
        if not selected_text:
            raise ValueError("selected_text cannot be empty")
        sections = await read_resource(
            self.client, "papertrail://paper/sections",
        )
        needle = _normalized(selected_text)
        for section in (sections or {}).get("sections") or []:
            for page in section.get("text_pages") or []:
                if needle in _normalized(page.get("text") or ""):
                    return {
                        "text": selected_text,
                        "section": section.get("title") or "Untitled section",
                        "page": int(page.get("page") or section.get("page_start") or 1),
                        "source": "selected_text",
                    }
        highlights = await read_resource(
            self.client, "papertrail://highlights/current",
        )
        for section in (highlights or {}).get("sections") or []:
            for highlight in section.get("highlights") or []:
                if needle in _normalized(highlight.get("quote") or ""):
                    return {
                        "text": selected_text,
                        "section": section.get("title") or "Untitled section",
                        "page": int(highlight.get("page") or section.get("page_start") or 1),
                        "source": "selected_text",
                    }
        raise ValueError(
            "selected_text could not be anchored to a section and page in the active export"
        )


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _same_source(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("section") == right.get("section")
        and int(left.get("page") or 1) == int(right.get("page") or 1)
        and _normalized(left.get("text") or left.get("quote") or "")
        == _normalized(right.get("text") or right.get("quote") or "")
    )


def _validated_ask_response(
    raw: str, evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    data = parse_json_object(raw, "Ask response")
    answer = data.get("answer")
    sources = data.get("supporting_sources")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ask response needs a non-empty answer")
    if not isinstance(sources, list):
        raise ValueError("Ask response needs a supporting_sources list")
    if answer.strip() == "The paper does not directly address this.":
        if sources:
            raise ValueError("an unsupported Ask answer cannot cite supporting sources")
        return {"answer": answer.strip(), "supporting_sources": []}
    if not sources:
        raise ValueError("a supported Ask response needs at least one supporting source")
    allowed = {
        (
            _normalized(item.get("text") or item.get("quote") or ""),
            item.get("section") or "",
            int(item.get("page") or 1),
        )
        for item in evidence
    }
    checked: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("supporting sources must be objects")
        quote = source.get("quote")
        section = source.get("section")
        page = source.get("page")
        if not isinstance(quote, str) or not isinstance(section, str) or not isinstance(page, int):
            raise ValueError("each supporting source needs quote, section, and integer page")
        if (_normalized(quote), section, page) not in allowed:
            raise ValueError("Ask cited a source that was not supplied as grounding")
        if f"[{section}, p. {page}]" not in answer:
            raise ValueError("Ask answer must cite every supporting source as [Section, p. N]")
        checked.append({"quote": quote, "section": section, "page": page})
    return {"answer": answer.strip(), "supporting_sources": checked}


def _compact_highlights(payload: dict[str, Any], limit: int = 40) -> list[dict[str, Any]]:
    highlights = [{
        "section": section.get("title") or "Untitled section",
        "page": int(highlight.get("page") or section.get("page_start") or 1),
        "highlight": highlight.get("quote") or "",
        "reason_hint": highlight.get("note") or "",
        "facet": highlight.get("facet"),
        "salience": highlight.get("salience"),
        "rank": highlight.get("rank"),
    } for section in payload.get("sections") or []
      for highlight in section.get("highlights") or []
      if highlight.get("quote")]
    highlights.sort(key=lambda item: item.get("rank") or 10**9)
    return highlights[:limit]


def _missing_links(
    headings: list[dict[str, Any]], highlights: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    highlighted = {item["section"] for item in highlights}
    return [heading for heading in headings if heading["title"] not in highlighted]


def _validated_reading_path(
    raw: str, highlights: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    data = parse_json_object(raw, "Guide reading path")
    path = data.get("path")
    if not isinstance(path, list) or not path:
        raise ValueError("Guide needs a non-empty path")
    allowed = {
        (item["section"], item["page"], _normalized(item["highlight"]))
        for item in highlights
    }
    checked: list[dict[str, Any]] = []
    for step in path:
        if not isinstance(step, dict):
            raise ValueError("Guide path steps must be objects")
        section = step.get("section")
        page = step.get("page")
        highlight = step.get("highlight")
        reason = step.get("reason")
        if (
            not isinstance(section, str)
            or not isinstance(page, int)
            or not isinstance(highlight, str)
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("each Guide step needs section, page, highlight, and reason")
        if (section, page, _normalized(highlight)) not in allowed:
            raise ValueError("Guide used a highlight outside the supplied current highlights")
        checked.append({
            "section": section,
            "page": page,
            "highlight": highlight,
            "reason": reason.strip(),
        })
    return checked


def _validated_suggested_questions(raw: str) -> list[str]:
    data = parse_json_object(raw, "Guide suggested questions")
    questions = data.get("suggested_questions")
    if not isinstance(questions, list) or not 2 <= len(questions) <= 5:
        raise ValueError("Guide must return two to five suggested questions")
    if any(not isinstance(item, str) or not item.strip() for item in questions):
        raise ValueError("suggested questions must be non-empty strings")
    return [item.strip() for item in questions]
