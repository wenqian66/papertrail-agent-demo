# PaperTrail Agent Demo

This is a fully standalone MCP server and interactive CLI for a PaperTrail
export bundle. It reads Map JSON plus an optional paired highlighted PDF. It
does not import from, modify, call, or require a running PaperTrail service.
No `papertrail-main` checkout is required at build time or runtime.

The bundled default fixture contains:

- `fixtures/bundle/361011.361061.map.json`
- `fixtures/bundle/361011.361061_highlighted.pdf`

The highlight-only fixture is
`fixtures/highlight-only/2901318.2901341.map.json`. Map JSON contains section
titles, page ranges, and highlights. It does not contain section body text,
tables, figures, PDF coordinates, confidence, agreements, or complete session
metadata. When a paired PDF is available, this project uses the generic
`pypdf` library to recover searchable page and section text.

## Run the demo

From this directory:

```bash
uv sync --group test
uv run python -m papertrail_agent_demo
```

The CLI supports:

```text
ask QUESTION
build FOCUS
refine REQUEST
guide [GOAL]
agreement
sections
highlights
session
quit
```

Run against the highlight-only fixture:

```bash
uv run python -m papertrail_agent_demo \
  --map fixtures/highlight-only/2901318.2901341.map.json
```

Run the MCP server over stdio:

```bash
uv run papertrail-agent-mcp
```

The server exposes the same fixed surface:

- Resources: `paper_sections`, `current_highlights`, `current_agreement`,
  `session_info`
- Tools: `build_agreement`, `update_agreement`, `find_passages`
- Prompts: `griswold_reading`, `grounded_qa`, `guided_reading`,
  `critical_analysis`, `paper_comparison`

Set `PAPERTRAIL_MAP`, `PAPERTRAIL_PDF`, and `PAPERTRAIL_AGENT_STATE` to select
other local files. `PAPERTRAIL_AGENT_API_KEY`, `PAPERTRAIL_AGENT_MODEL`, and
optional `PAPERTRAIL_AGENT_BASE_URL` enable an OpenAI-compatible model. Without
them, agreement build and refine use deterministic fallbacks that preserve the
same no-rewrite invariant.

## Agreement before and after

For the focus:

> What problem does the paper solve and what is its main result?

The former agreement behavior produced a prose paraphrase beginning:

> You are looking for the core motivation of the research and the primary
> claim the authors make about their success.

The standalone agent produces structured objectives whose `original_text`
values are exact source spans:

```json
{
  "objectives": [
    {
      "original_text": "What problem does the paper solve",
      "extraction_guidance": {
        "look_in": ["abstract", "introduction", "related work"],
        "signals": ["however", "limitation", "fails to", "we ask", "challenge"],
        "exclude": ["method details with no stated rationale", "results with no problem framing"],
        "edge_cases": "Motivation can be implicit. Keep a passage only when the authors connect a prior limitation or practical need to the question they pursue."
      }
    },
    {
      "original_text": "and what is its main result?",
      "extraction_guidance": {
        "look_in": ["evaluation", "experiments", "results", "discussion"],
        "signals": ["baseline", "compared with", "improves", "decreases", "ablation", "limitation"],
        "exclude": ["experimental setup with no result", "unsupported performance claims"],
        "edge_cases": "Keep the comparison conditions with the reported outcome."
      }
    }
  ]
}
```

Validation requires each objective to be an exact, ordered substring of the
focus and requires all focus words to remain covered. A one-goal focus remains
one objective. Griswold is available only through the opt-in
`griswold_reading` prompt.

## Refine is filtering, not extraction

This project has no extraction model or highlighting pipeline. Therefore
`update_agreement` cannot rerun extraction and cannot find new quotes. It:

1. changes only extraction guidance while freezing every `original_text`;
2. applies new exclude rules to the existing Map JSON highlight pool;
3. re-ranks the survivors and applies requests such as "only keep the top 5";
4. stores the resulting visible highlight ids in local state.

Every tool result and filter report says `re_extracted: false`. This is the
deliberate cost of making the demo fully isolated. Delete
`.papertrail-agent-state.json` to reset the default fixture to its unfiltered
highlight pool.

## Tests

```bash
uv run pytest -q
```

The tests verify exact focus preservation, deterministic refinement,
highlight-pool filtering, PDF text recovery, the complete MCP surface, and the
absence of runtime references to the sibling service source tree.
