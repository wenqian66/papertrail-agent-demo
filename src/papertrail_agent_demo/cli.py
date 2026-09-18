#!/usr/bin/env python3
"""Small interactive MCP client for the PaperTrail agent layer demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

def _configure(args: argparse.Namespace) -> None:
    if args.map:
        os.environ["PAPERTRAIL_MAP"] = str(args.map.resolve())
    if args.pdf:
        os.environ["PAPERTRAIL_PDF"] = str(args.pdf.resolve())
    if args.state:
        os.environ["PAPERTRAIL_AGENT_STATE"] = str(args.state.resolve())


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _tool_value(result: Any) -> Any:
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


async def _resource(client: Any, uri: str) -> Any:
    result = await client.read_resource(uri)
    if not result.contents:
        return None
    text = getattr(result.contents[0], "text", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def _prompt(client: Any, name: str, arguments: dict[str, str]) -> str:
    result = await client.get_prompt(name, arguments)
    return "\n\n".join(
        getattr(message.content, "text", "") for message in result.messages
    )


def _print_passages(result: dict[str, Any]) -> None:
    passages = result.get("passages") or []
    if not passages:
        print(result.get("message") or "The paper does not address this question in the available text.")
        return
    for index, passage in enumerate(passages, 1):
        print(f"{index}. [{passage['section']}, p. {passage['page']}]")
        print(f"   {passage['text']}")
    limitations = result.get("limitations") or []
    if limitations:
        print("\nSource limits:")
        for limitation in limitations:
            print(f"- {limitation}")


def _fallback_path(highlights: dict[str, Any], limit: int = 6) -> str:
    steps: list[tuple[str, int, str, str]] = []
    for section in highlights.get("sections") or []:
        title = section.get("title") or section.get("section_header") or "Untitled section"
        items = section.get("highlights") if "highlights" in section else section.get("items")
        for item in items or []:
            spans = item.get("spans") or []
            page = item.get("page") or (spans[0].get("page") if spans else None) or 1
            quote = item.get("quote") or ""
            reason = item.get("note") or "This passage anchors the section's main evidence."
            if quote:
                steps.append((title, int(page), quote, reason))
                break
        if len(steps) == limit:
            break
    if not steps:
        return "No highlight-grounded path can be made because the active paper has no highlights."
    lines = []
    for index, (section, page, quote, reason) in enumerate(steps, 1):
        lines.append(f"{index}. {section}, p. {page}\n   Highlight: {quote}\n   Why: {reason}")
    return "\n\n".join(lines)


async def _ask(client: Any, question: str, agent_model: Any) -> None:
    result = _tool_value(await client.call_tool(
        "find_passages", {"question": question, "limit": 8},
    ))
    prompt = await _prompt(client, "grounded_qa", {"question": question})
    if agent_model.configured:
        answer = await agent_model.complete(
            f"{prompt}\n\nRetrieved passages:\n{_json_text(result)}"
        )
        print(answer)
    else:
        print("No agent model is configured, so the demo is showing the grounded retrieval result.\n")
        _print_passages(result)


async def _guide(client: Any, goal: str, agent_model: Any) -> None:
    highlights = await _resource(client, "papertrail://highlights/current")
    prompt = await _prompt(client, "guided_reading", {"goal": goal})
    if agent_model.configured:
        sections = await _resource(client, "papertrail://paper/sections")
        answer = await agent_model.complete(
            f"{prompt}\n\nPaper sections:\n{_json_text(sections)}"
            f"\n\nCurrent highlights:\n{_json_text(highlights)}"
        )
        print(answer)
    else:
        print(_fallback_path(highlights))


HELP = """\
Commands:
  ask QUESTION       Search and answer with section and page evidence
  refine REQUEST     Update guidance and re-filter existing exported highlights
  guide [GOAL]       Build a section and highlight reading path
  build FOCUS        Build a new structured agreement
  agreement          Show the current agreement resource
  sections           Show the paper sections resource
  highlights         Show the current highlights resource
  session            Show session information
  help               Show this list
  quit               Exit
"""


async def run_demo() -> None:
    from mcp import Client
    from .server import agent_model, mcp

    async with Client(mcp) as client:
        print("PaperTrail MCP agent demo. Type help for commands.")
        while True:
            try:
                raw = await asyncio.to_thread(input, "papertrail> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            command, _, argument = raw.strip().partition(" ")
            if command in ("quit", "exit"):
                return
            if command == "help" or not command:
                print(HELP)
            elif command == "ask":
                await _ask(client, argument, agent_model)
            elif command == "guide":
                await _guide(client, argument or "Understand the paper", agent_model)
            elif command == "refine":
                result = _tool_value(await client.call_tool(
                    "update_agreement", {"refinement": argument},
                ))
                print(_json_text(result))
            elif command == "build":
                result = _tool_value(await client.call_tool(
                    "build_agreement", {"focus_text": argument},
                ))
                print(_json_text(result))
            elif command in ("agreement", "sections", "highlights", "session"):
                uris = {
                    "agreement": "papertrail://agreement/current",
                    "sections": "papertrail://paper/sections",
                    "highlights": "papertrail://highlights/current",
                    "session": "papertrail://session/info",
                }
                print(_json_text(await _resource(client, uris[command])))
            else:
                print("Unknown command. Type help.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, help="Map JSON export; defaults to bundled fixture")
    parser.add_argument("--pdf", type=Path, help="paired highlighted or original PDF")
    parser.add_argument("--state", type=Path, help="local agreement and filter state")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    _configure(arguments)
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
