# PaperTrail Agent Demo

This is a standalone MCP server plus client-side chat runner for a PaperTrail
export bundle. It reads Map JSON and an optional paired highlighted PDF. It
does not import from, modify, call, or require the PaperTrail backend or its
source tree. All paper data and state stay local to this project.

The default fixture is self-contained:

- `fixtures/bundle/361011.361061.map.json`
- `fixtures/bundle/361011.361061_highlighted.pdf`

The highlight-only fixture is
`fixtures/highlight-only/2901318.2901341.map.json`. Map JSON provides section
titles, page ranges, and highlights. It omits section body text, so the bundled
PDF is read with the generic `pypdf` library when available.

## Run

From this directory:

```bash
uv sync --group test
export PAPERTRAIL_AGENT_API_KEY=...
export PAPERTRAIL_AGENT_MODEL=...
uv run python -m papertrail_agent_demo
```

`PAPERTRAIL_AGENT_BASE_URL` is optional and can target a compatible local or
hosted endpoint. Build, Refine, Ask when evidence exists, and Guide require a
model. They fail clearly when one is not configured; there are no silent or
rule-based answer/agreement fallbacks.

The interactive client supports:

```text
ask QUESTION
ask-selected QUESTION || SELECTED PAPER TEXT
build FOCUS
refine REQUEST
guide [GOAL]
agreement
sections
highlights
session
quit
```

Run with Map JSON only:

```bash
uv run python -m papertrail_agent_demo \
  --map fixtures/highlight-only/2901318.2901341.map.json
```

Run the MCP server over stdio:

```bash
uv run papertrail-agent-mcp
```

Other local inputs can be selected with `--map`, `--pdf`, and `--state`, or
with `PAPERTRAIL_MAP`, `PAPERTRAIL_PDF`, and `PAPERTRAIL_AGENT_STATE`.

## Architecture and MCP surface

The MCP server exposes protocol components in the sense defined by the
official MCP specifications for
[Resources](https://modelcontextprotocol.io/specification/draft/server/resources),
[Tools](https://modelcontextprotocol.io/specification/draft/server/tools), and
[Prompts](https://modelcontextprotocol.io/specification/draft/server/prompts).

- Resources: `paper_sections`, `current_highlights`, `current_agreement`,
  `session_info`
- Tools: `build_agreement`, `update_agreement`, `find_passages`
- Prompts: `griswold_reading`, `grounded_qa`, `guided_reading`,
  `critical_analysis`, `paper_comparison`

The chat runner is deliberately client-side, not another MCP component. The
CLI routes a turn, reads only the needed resources, calls tools, obtains the
matching prompt, supplies the grounded context to the model, and validates
the result.

- Ask treats optional selected text as primary context and runs
  `find_passages` only for supplementary evidence. The answer must cite
  supplied quotes with section and page. With no evidence it returns, "The
  paper does not directly address this."
- Refine invokes `update_agreement` and displays its filtered highlight view.
- Guide sends headings and current highlights, not the whole paper, then uses
  `find_passages` for a few missing links. It returns an ordered reading path
  plus a small suggested-question list.

## Agreement before and after

For this focus:

> I care about why they chose this dataset

A paraphrase-style agreement might replace the user's text with:

> Identify and explain the authors' rationale for selecting the dataset used
> in their evaluation.

`build_agreement` instead requires an LLM to preserve user-authored spans and
add expertise in separate fields:

```json
{
  "focus_raw": "I care about why they chose this dataset",
  "objectives": [
    {
      "id": "obj1",
      "source_text": "why they chose this dataset",
      "facet": "evaluation",
      "guidance": [
        "dataset choice",
        "dataset justification",
        "dataset limitations"
      ]
    }
  ]
}
```

Validation requires `focus_raw` to match the input byte for byte. Every
`source_text` must be an exact, ordered substring, and the source spans must
cover all substantive focus text. `facet` and `guidance` are system-added
expertise and are intentionally not subject to substring validation. A
single reading goal produces one objective; splitting is only for genuinely
distinct goals. Griswold remains opt-in through `griswold_reading`.

There is no deterministic agreement fallback. A structurally invalid response
gets one LLM regeneration attempt with the validation error; a missing key,
failed call, or second invalid response is an explicit error.

## Highlight enrichment and Refine

Map JSON highlights contain `quote`, `page`, `run_id`, `run_name`, `version`,
`color`, `note`, `source`, and `verdict`; they do not contain a facet or
salience score. After each successful agreement build, one disclosed LLM pass
tags every existing highlight with an agreement objective/facet (or `other`)
and a salience value from 0 to 1. Those tags are cached in local state and are
exposed by `current_highlights`.

This enrichment is not extraction. It classifies only the already exported
highlight pool and never searches the PDF for new highlights.

For Refine, the model has one limited job: translate the request into these
operations:

- `set_limit <N>`
- `add_exclude <facet>` or `remove_exclude <facet>`
- `enable_facet <id>` or `disable_facet <id>`
- `change_density <facet> <low|medium|high>`

State mutation, salience ranking, and filtering then run in deterministic
Python over the cached tags. If a request introduces an uncached semantic
criterion, such as `implementation_details`, one additional disclosed LLM
pass labels the complete existing pool against that criterion. The labels are
cached, so later refinements reuse them.

Every update reports `re_extracted: false`. Refine can only narrow, restore,
or reorder highlights from the original exported pool; it can never create or
surface evidence that was absent from that pool.

## Grounding limitation

`find_passages` is deterministic keyword-overlap retrieval over recovered PDF
text plus current highlights. It is the weakest grounding component: relevant
passages with different wording can be missed. A future upgrade could use
semantic retrieval while remaining reproducible under a fixed embedding
model, index, corpus, and retrieval configuration. This demo does not build
that upgrade.

## Tests

```bash
uv run pytest -q
```

Tests mock every LLM call, so they require no key or network. They cover
verbatim source validation, the no-fallback error, full-pool tagging, fixed
Refine operations, deterministic filtering, new-criterion caching, selected
text grounding, guided reading, PDF recovery, the exact MCP surface, and the
absence of runtime coupling to the sibling service.
