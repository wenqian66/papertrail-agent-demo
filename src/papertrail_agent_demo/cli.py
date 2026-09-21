#!/usr/bin/env python3
"""Interactive client-side chat runner for the PaperTrail MCP demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .model import AgentModel
from .runner import ChatRunner, read_resource, tool_value


def _configure(args: argparse.Namespace) -> None:
    if args.map:
        os.environ["PAPERTRAIL_MAP"] = str(args.map.resolve())
    if args.pdf:
        os.environ["PAPERTRAIL_PDF"] = str(args.pdf.resolve())
    if args.state:
        os.environ["PAPERTRAIL_AGENT_STATE"] = str(args.state.resolve())


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


HELP = """\
Commands:
  ask QUESTION                 Answer with grounded section/page citations
  ask-selected QUESTION || TEXT
                               Use selected paper text as primary context
  refine REQUEST              Apply an LLM-parsed diff, then filter cached tags
  guide [GOAL]                Generate a cited reading path and questions
  build FOCUS                 Build an agreement and tag the highlight pool
  agreement                   Show the current agreement resource
  sections                    Show the paper sections resource
  highlights                  Show the current highlights resource
  session                     Show session information
  help                        Show this list
  quit                        Exit

Build, Refine, Ask with evidence, and Guide require a configured LLM.
"""


async def run_demo() -> None:
    from mcp import Client
    from .server import mcp

    async with Client(mcp) as client:
        runner = ChatRunner(client, AgentModel.from_env())
        print("PaperTrail MCP agent demo. Type help for commands.")
        while True:
            try:
                raw = await asyncio.to_thread(input, "papertrail> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            command, _, argument = raw.strip().partition(" ")
            try:
                if command in ("quit", "exit"):
                    return
                if command == "help" or not command:
                    print(HELP)
                elif command == "ask":
                    print(_json_text(await runner.ask(argument)))
                elif command == "ask-selected":
                    question, separator, selected = argument.partition("||")
                    if not separator:
                        raise ValueError(
                            "Use: ask-selected QUESTION || SELECTED PAPER TEXT"
                        )
                    print(_json_text(await runner.ask(
                        question.strip(), selected_text=selected.strip(),
                    )))
                elif command == "guide":
                    print(_json_text(await runner.guide(
                        argument or "Understand the paper",
                    )))
                elif command == "refine":
                    print(_json_text(await runner.refine(argument)))
                elif command == "build":
                    result = tool_value(await client.call_tool(
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
                    print(_json_text(await read_resource(client, uris[command])))
                else:
                    print("Unknown command. Type help.")
            except (RuntimeError, ValueError) as exc:
                print(f"Error: {exc}")


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
