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
- Refine invokes `update_agreement`. With untagged highlights it reports that
  no filtering was applied and leaves the raw exported pool unchanged.
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
expertise and are intentionally not subject to substring validation. Facet is
a display label only; deterministic slicing uses the stable objective `id`.
A single reading goal produces one objective; splitting is only for genuinely
distinct goals. Griswold remains opt-in through `griswold_reading`.

There is no deterministic agreement fallback. A structurally invalid response
gets one LLM regeneration attempt with the validation error; a missing key,
failed call, or second invalid response is an explicit error.

## Highlights and Refine

Map JSON highlights contain `quote`, `page`, `run_id`, `run_name`, `version`,
`color`, `note`, `source`, and `verdict`; they do not contain a facet or
salience score. `build_agreement` stops after validating and saving the
agreement: it never reads, tags, filters, or modifies highlights. Consequently,
the normal build flow leaves `current_highlights` as the raw Map JSON pool and
reports `not_tagged`; no facet, salience, or objective-id fields are added.

The standalone tagging and filtering functions remain available for cached
tag state and direct testing, but agreement building does not invoke them.
Any such enrichment classifies only the already exported highlight pool and
is not extraction; it never searches the PDF for new highlights.

For Refine, the model has one limited job: translate the request into these
operations:

- `set_limit <N>`
- `add_exclude <objective_id>` or `remove_exclude <objective_id>`
- `enable_objective <objective_id>` or `disable_objective <objective_id>`
- `change_density <objective_id> <low|medium|high>`

When complete cached tags exist, state mutation, salience ranking, and
filtering run in deterministic Python over the cached objective ids and tags.
A highlight assigned to several objectives remains visible while at least one
of those objectives is enabled; it is hidden by objective disabling only when
none remain enabled. Free-form facet text is never a filter key, so duplicate
facet labels cannot cross-wire objective filters. When highlights are
untagged, Refine returns `highlights_not_tagged`, applies no filter, and leaves
the raw pool unchanged rather than raising.

If a request introduces an uncached semantic criterion, such as
`implementation_details`, `add_exclude`, `remove_exclude`, or
`change_density` uses an explicit `criterion` field instead of an objective
id. If complete base tags already exist, one additional disclosed LLM pass
labels the complete existing pool against that criterion. The labels are
cached, so later refinements reuse them. No criterion pass runs for an
untagged pool.

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
verbatim source validation, the no-fallback error, agreement-only builds,
untagged highlight and Refine behavior, direct multi-objective tagging,
objective-id filtering, shared-highlight disable behavior, selected-text
grounding, guided reading, PDF recovery, the exact MCP surface, and the
absence of runtime coupling to the sibling service.
