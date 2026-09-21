"""Pure agreement, tagging, refinement, retrieval, and prompt logic."""

from __future__ import annotations

import json
import re
from typing import Any


MAX_OBJECTIVES = 32
MAX_HIGHLIGHTS = 500
DENSITY_LEVELS = {"low", "medium", "high"}
REFINEMENT_OPERATIONS = {
    "set_limit",
    "add_exclude",
    "remove_exclude",
    "enable_objective",
    "disable_objective",
    "change_density",
}

BUILD_AGREEMENT_PROMPT = """\
Build a structured extraction agreement from the user's reading focus.

The user's words are authoritative. Copy focus_raw byte for byte. Every
source_text must be one exact, contiguous substring of focus_raw. Do not
paraphrase, improve, complete, or replace the user's words. The source_text
spans must remain in their original order and together retain every
substantive part of the focus.

Split only when the focus contains genuinely distinct reading goals. A focus
about one point gets exactly one objective, even when that point has useful
sub-aspects. facet is only a short display label; filtering uses the stable
objective id instead. facet is not a taxonomy that determines the objectives.
guidance is system-added expertise, so it does not need to occur in the user's
words.

Positive single-goal example:
Focus: I care about why they chose this dataset
Output has exactly one objective whose source_text is
"why they chose this dataset", facet might be "evaluation", and guidance may
include "dataset justification" and "dataset limitations".

Positive multi-goal example:
Focus: What problem does it solve, and what is the main result?
Output has two objectives, one sourced from "What problem does it solve" and
one sourced from "what is the main result?".

Negative example:
Do not split "Explain why they chose this dataset" into separate objectives
for dataset choice, justification, and limitations. Those are guidance for
one user goal, not three goals.

Return JSON only in this exact shape:
{{
  "focus_raw": "the exact complete focus",
  "objectives": [
    {{
      "id": "obj1",
      "source_text": "an exact focus substring",
      "facet": "short_display_label",
      "guidance": ["concrete search phrase", "nearby concept"]
    }}
  ]
}}

Use obj1, obj2, and so on in order. Keep guidance concise and operational.
The paper names and headings below are context only; do not claim findings.

Focus:
<focus>{focus}</focus>

Paper context:
{papers}
"""

REFINE_OPERATIONS_PROMPT = """\
Translate one refinement request into a small structured diff. Do not rewrite
the agreement, judge individual quotes, or invent any operation outside the
fixed set below.

Allowed operations and exact JSON fields:
- {{"op":"set_limit","value":5}}
- {{"op":"add_exclude","objective_id":"obj1"}}
- {{"op":"remove_exclude","objective_id":"obj1"}}
- {{"op":"enable_objective","objective_id":"obj1"}}
- {{"op":"disable_objective","objective_id":"obj1"}}
- {{"op":"change_density","objective_id":"obj1","level":"low"}}

All operations about an existing agreement objective must use its id. Never
use its free-form facet label as a filter key. Density levels are low, medium,
and high.

If the request names a new semantic criterion that is not an agreement
objective, keep the same operation name but use a concise lowercase
snake_case criterion instead of objective_id:
- {{"op":"add_exclude","criterion":"implementation_details"}}
- {{"op":"remove_exclude","criterion":"implementation_details"}}
- {{"op":"change_density","criterion":"implementation_details","level":"low"}}

The caller will run one disclosed tagging pass for an uncached criterion.
Return JSON only as {{"operations":[...]}}.

Agreement:
{agreement}

Known cached semantic criteria:
{known_criteria}

Refinement request:
{request}
"""

_WORDS = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_MULTI_GOAL_SEPARATOR = re.compile(
    r"(?:[,;:/\n.!?]|\b(?:and|also|plus|then|versus|vs|while)\b)",
    re.IGNORECASE,
)
_COVERAGE_GLUE = {
    "a", "about", "also", "and", "care", "focus", "for", "i", "interested",
    "looking", "me", "need", "on", "please", "plus", "read", "show", "tell",
    "then", "to", "us", "want", "we", "would",
}
_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "paper",
    "that", "the", "their", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "with",
}


def _json_object(value: str | dict[str, Any], label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = (value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return loaded


def parse_json_object(value: str, label: str = "model response") -> dict[str, Any]:
    """Parse a model JSON object, accepting one surrounding JSON code fence."""
    return _json_object(value, label)


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} entries must be non-empty strings")
        result.append(item.strip())
    return result


def _focus_tokens(text: str) -> list[str]:
    return [match.group(0) for match in _WORDS.finditer(text)]


def _validate_source_coverage(focus_raw: str, sources: list[str]) -> None:
    """Require ordered source spans and reject uncovered substantive words."""
    cursor = 0
    spans: list[tuple[int, int]] = []
    for source in sources:
        position = focus_raw.find(source, cursor)
        if position < 0:
            raise ValueError("source_text is not an exact ordered focus_raw substring")
        spans.append((position, position + len(source)))
        cursor = position + len(source)

    if len(spans) > 1:
        for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
            boundary = focus_raw[
                max(0, previous_end - 1):min(len(focus_raw), next_start + 1)
            ]
            if not _MULTI_GOAL_SEPARATOR.search(boundary):
                raise ValueError(
                    "a single-goal focus must yield exactly one objective"
                )

    uncovered: list[str] = []
    for match in _WORDS.finditer(focus_raw):
        covered = any(start <= match.start() and match.end() <= end for start, end in spans)
        token = match.group(0)
        if not covered and not token.isdigit() and token.casefold() not in _COVERAGE_GLUE:
            uncovered.append(token)
    if uncovered:
        raise ValueError(
            "agreement objectives do not cover substantive focus_raw text: "
            + ", ".join(uncovered[:8])
        )


def validate_agreement(
    value: str | dict[str, Any], *, focus_text: str | None = None,
) -> dict[str, Any]:
    """Validate the new agreement shape without rewriting user source text."""
    data = _json_object(value, "agreement")
    focus_raw = data.get("focus_raw")
    if not isinstance(focus_raw, str) or not focus_raw.strip():
        raise ValueError("focus_raw must be a non-empty string")
    if focus_text is not None and focus_raw != focus_text:
        raise ValueError("focus_raw must match the user's focus byte for byte")

    objectives = data.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("agreement needs at least one objective")
    if len(objectives) > MAX_OBJECTIVES:
        raise ValueError(f"agreement has more than {MAX_OBJECTIVES} objectives")

    normalized: list[dict[str, Any]] = []
    sources: list[str] = []
    for index, objective in enumerate(objectives, 1):
        if not isinstance(objective, dict):
            raise ValueError("each objective must be an object")
        objective_id = objective.get("id")
        expected_id = f"obj{index}"
        if objective_id != expected_id:
            raise ValueError(f"objective {index} id must be {expected_id}")
        source_text = objective.get("source_text")
        if not isinstance(source_text, str) or not source_text:
            raise ValueError("source_text must be a non-empty string")
        facet = objective.get("facet")
        if not isinstance(facet, str) or not facet.strip():
            raise ValueError("facet must be a non-empty string")
        normalized.append({
            "id": objective_id,
            "source_text": source_text,
            "facet": facet.strip(),
            "guidance": _string_list(objective.get("guidance"), "guidance"),
        })
        sources.append(source_text)

    _validate_source_coverage(focus_raw, sources)
    return {"focus_raw": focus_raw, "objectives": normalized}


def agreement_from_model(focus_text: str, response_text: str) -> dict[str, Any]:
    return validate_agreement(response_text, focus_text=focus_text)


def agreement_to_text(agreement: dict[str, Any]) -> str:
    return json.dumps(validate_agreement(agreement), ensure_ascii=False, indent=2)


def build_agreement_prompt(focus_text: str, papers: str = "(none)") -> str:
    if not isinstance(focus_text, str) or not focus_text.strip():
        raise ValueError("focus text cannot be empty")
    return BUILD_AGREEMENT_PROMPT.format(focus=focus_text, papers=papers or "(none)")


def refinement_prompt(
    agreement: dict[str, Any], request: str, known_criteria: list[str],
) -> str:
    if not isinstance(request, str) or not request.strip():
        raise ValueError("refinement request cannot be empty")
    return REFINE_OPERATIONS_PROMPT.format(
        agreement=agreement_to_text(agreement),
        known_criteria=json.dumps(sorted(set(known_criteria)), ensure_ascii=False),
        request=request,
    )


def _criterion_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("criterion must be a non-empty string")
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not normalized:
        raise ValueError("criterion must contain a letter or number")
    return normalized


def _operation_target(
    operation: dict[str, Any], objective_ids: set[str],
) -> dict[str, str]:
    has_objective = "objective_id" in operation
    has_criterion = "criterion" in operation
    if has_objective == has_criterion:
        raise ValueError(
            "operation must name exactly one objective_id or semantic criterion"
        )
    if has_objective:
        objective_id = operation.get("objective_id")
        if objective_id not in objective_ids:
            raise ValueError(f"unknown objective id: {objective_id}")
        return {"objective_id": objective_id}
    return {"criterion": _criterion_name(operation.get("criterion"))}


def refinement_diff_from_model(
    response_text: str, agreement: dict[str, Any],
) -> dict[str, Any]:
    data = _json_object(response_text, "refinement diff")
    operations = data.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("refinement diff needs at least one operation")
    objective_ids = {item["id"] for item in validate_agreement(agreement)["objectives"]}
    normalized: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("each refinement operation must be an object")
        name = operation.get("op")
        if name not in REFINEMENT_OPERATIONS:
            raise ValueError(f"unsupported refinement operation: {name}")
        if name == "set_limit":
            value = operation.get("value")
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 500:
                raise ValueError("set_limit value must be an integer from 1 to 500")
            normalized.append({"op": name, "value": value})
        elif name in {"enable_objective", "disable_objective"}:
            objective_id = operation.get("objective_id")
            if objective_id not in objective_ids:
                raise ValueError(f"unknown objective id: {objective_id}")
            normalized.append({"op": name, "objective_id": objective_id})
        elif name in {"add_exclude", "remove_exclude"}:
            normalized.append({"op": name, **_operation_target(operation, objective_ids)})
        else:
            level = operation.get("level")
            if level not in DENSITY_LEVELS:
                raise ValueError("density level must be low, medium, or high")
            normalized.append({
                "op": name,
                **_operation_target(operation, objective_ids),
                "level": level,
            })
    return {"operations": normalized}


def highlight_tagging_prompt(
    agreement: dict[str, Any], highlights: list[dict[str, Any]],
) -> str:
    if len(highlights) > MAX_HIGHLIGHTS:
        raise ValueError(f"highlight pool exceeds the demo cap of {MAX_HIGHLIGHTS}")
    compact = [{
        "id": item["id"],
        "section": item.get("section"),
        "page": item.get("page"),
        "quote": item.get("quote"),
        "note": item.get("note"),
    } for item in highlights]
    return f"""\
Enrich an existing highlight pool. This is classification only, not extraction.
For every highlight id, choose the primary agreement objective it best serves
and an ordered list of any additional objectives it also serves. Use "other"
as primary_objective_id only when it serves no agreement objective; in that
case secondary_objective_ids must be empty. Give one salience score from 0.0
to 1.0 for the quote's overall importance. Do not alter, drop, add, or rewrite
any highlight. Do not return facet; the caller derives it from the primary
objective for display.

Return JSON only:
{{"tags":[{{"id":"0:0","primary_objective_id":"obj1","secondary_objective_ids":["obj2"],"salience":0.8}}]}}

Agreement:
{agreement_to_text(agreement)}

Existing highlights:
{json.dumps(compact, ensure_ascii=False)}
"""


def highlight_tags_from_model(
    response_text: str, expected_ids: list[str], agreement: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    data = _json_object(response_text, "highlight tags")
    tags = data.get("tags")
    if not isinstance(tags, list):
        raise ValueError("highlight tags must contain a tags list")
    checked = validate_agreement(agreement)
    objective_facets = {item["id"]: item["facet"] for item in checked["objectives"]}
    objective_ids = set(objective_facets)
    expected = set(expected_ids)
    result: dict[str, dict[str, Any]] = {}
    for tag in tags:
        if not isinstance(tag, dict):
            raise ValueError("each highlight tag must be an object")
        highlight_id = str(tag.get("id") or "")
        if highlight_id not in expected or highlight_id in result:
            raise ValueError(f"unexpected or duplicate highlight tag id: {highlight_id}")
        primary_objective_id = tag.get("primary_objective_id")
        if primary_objective_id not in objective_ids | {"other"}:
            raise ValueError(
                "unknown primary_objective_id in highlight tags: "
                f"{primary_objective_id}"
            )
        secondary_objective_ids = tag.get("secondary_objective_ids", [])
        if not isinstance(secondary_objective_ids, list):
            raise ValueError("secondary_objective_ids must be a list")
        if any(not isinstance(value, str) for value in secondary_objective_ids):
            raise ValueError("secondary_objective_ids entries must be strings")
        if len(secondary_objective_ids) != len(set(secondary_objective_ids)):
            raise ValueError("secondary_objective_ids must not contain duplicates")
        if primary_objective_id in secondary_objective_ids:
            raise ValueError("primary objective cannot also be a secondary objective")
        unknown_secondary = set(secondary_objective_ids) - objective_ids
        if unknown_secondary:
            raise ValueError(
                "unknown secondary_objective_ids in highlight tags: "
                + ", ".join(sorted(unknown_secondary))
            )
        if primary_objective_id == "other" and secondary_objective_ids:
            raise ValueError("an other highlight cannot have secondary objectives")
        facet = (
            "other" if primary_objective_id == "other"
            else objective_facets[primary_objective_id]
        )
        salience = tag.get("salience")
        if not isinstance(salience, (int, float)) or isinstance(salience, bool):
            raise ValueError("salience must be numeric")
        if not 0 <= float(salience) <= 1:
            raise ValueError("salience must be between 0 and 1")
        result[highlight_id] = {
            "primary_objective_id": primary_objective_id,
            "secondary_objective_ids": list(secondary_objective_ids),
            "facet": facet,
            "salience": round(float(salience), 4),
        }
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise ValueError(f"highlight tags omitted ids: {', '.join(missing[:8])}")
    return result


def criterion_tagging_prompt(
    criteria: list[str], highlights: list[dict[str, Any]],
) -> str:
    compact = [{
        "id": item["id"],
        "section": item.get("section"),
        "page": item.get("page"),
        "quote": item.get("quote"),
        "note": item.get("note"),
    } for item in highlights]
    return f"""\
Label the existing highlights against new filtering criteria. This is one
disclosed enrichment pass, not extraction. Do not add, remove, rank, or rewrite
quotes. For every id, return one boolean for every criterion.

Return JSON only:
{{"criteria":[{{"id":"0:0","matches":{{"implementation_details":true}}}}]}}

Criteria:
{json.dumps(criteria, ensure_ascii=False)}

Existing highlights:
{json.dumps(compact, ensure_ascii=False)}
"""


def criterion_tags_from_model(
    response_text: str, expected_ids: list[str], criteria: list[str],
) -> dict[str, dict[str, bool]]:
    data = _json_object(response_text, "criterion tags")
    rows = data.get("criteria")
    if not isinstance(rows, list):
        raise ValueError("criterion tags must contain a criteria list")
    expected = set(expected_ids)
    expected_criteria = set(criteria)
    result: dict[str, dict[str, bool]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each criterion tag must be an object")
        highlight_id = str(row.get("id") or "")
        if highlight_id not in expected or highlight_id in result:
            raise ValueError(f"unexpected or duplicate criterion tag id: {highlight_id}")
        matches = row.get("matches")
        if not isinstance(matches, dict) or set(matches) != expected_criteria:
            raise ValueError("criterion matches must name every requested criterion exactly")
        if any(not isinstance(value, bool) for value in matches.values()):
            raise ValueError("criterion matches must be booleans")
        result[highlight_id] = dict(matches)
    if set(result) != expected:
        raise ValueError("criterion tags must cover the complete highlight pool")
    return result


def _search_terms(text: str) -> list[str]:
    return [
        token.casefold() for token in _focus_tokens(text)
        if len(token) > 2 and token.casefold() not in _STOP_WORDS
    ]


def _passage_chunks(text: str, max_chars: int = 900) -> list[tuple[int, str]]:
    paragraphs = [match for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S)]
    if not paragraphs:
        paragraphs = [match for match in re.finditer(r"\S.*?(?=\Z)", text, re.S)]
    result: list[tuple[int, str]] = []
    for paragraph in paragraphs:
        raw = " ".join(paragraph.group(0).split())
        if not raw:
            continue
        if len(raw) <= max_chars:
            result.append((paragraph.start(), raw))
            continue
        sentences = re.split(r"(?<=[.!?])\s+", raw)
        chunk = ""
        offset = paragraph.start()
        for sentence in sentences:
            if chunk and len(chunk) + len(sentence) + 1 > max_chars:
                result.append((offset, chunk))
                offset += len(chunk) + 1
                chunk = sentence
            else:
                chunk = f"{chunk} {sentence}".strip()
        if chunk:
            result.append((offset, chunk))
    return result


def _score_passage(question: str, passage: str) -> float:
    terms = _search_terms(question)
    if not terms:
        return 0.0
    lowered = passage.casefold()
    counts = sum(min(lowered.count(term), 3) for term in terms)
    coverage = sum(1 for term in set(terms) if term in lowered) / len(set(terms))
    phrase_bonus = 2.0 if question.casefold() in lowered else 0.0
    return round(counts + coverage * 5.0 + phrase_bonus, 4)


def find_relevant_passages(
    question: str, sections: list[dict[str, Any]],
    highlights: list[dict[str, Any]], limit: int = 8,
) -> dict[str, Any]:
    """Deterministically rank paper text and highlights with lexical overlap."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question cannot be empty")
    limit = max(1, min(int(limit), 20))
    candidates: list[dict[str, Any]] = []
    for section in sections:
        title = section.get("title") or section.get("header") or "Untitled section"
        page_start = section.get("page_start") or section.get("page") or 1
        page_texts = section.get("text_pages") or []
        text_sources = [
            (entry.get("page") or page_start, entry.get("text") or "")
            for entry in page_texts
        ] if page_texts else [(page_start, section.get("text") or "")]
        for page, text in text_sources:
            for _offset, passage in _passage_chunks(text):
                score = _score_passage(question, passage)
                if score:
                    candidates.append({
                        "text": passage,
                        "section": title,
                        "page": int(page),
                        "source": "paper_text",
                        "score": score,
                    })
    for highlight in highlights:
        quote = highlight.get("quote") or ""
        note = highlight.get("note") or ""
        score = _score_passage(question, f"{quote} {note}")
        if score:
            candidates.append({
                "text": quote,
                "section": highlight.get("section") or "Untitled section",
                "page": int(highlight.get("page") or 1),
                "source": "current_highlight",
                "score": round(score + 1.5, 4),
                "note": note,
                "run_id": highlight.get("run_id"),
            })
    candidates.sort(key=lambda item: (-item["score"], item["page"], item["section"]))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for candidate in candidates:
        key = (candidate["section"], candidate["page"], candidate["text"].casefold())
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) == limit:
            break
    return {"question": question, "addressed": bool(selected), "passages": selected}


_GRISWOLD_SOURCES = [
    "Motivations: What is the people problem? What is the technical problem? Why are prior solutions inadequate? What is the research question?",
    "Proposed solution: What is the hypothesis? Why should it work? How is it achieved? Keep the solution separate from its results.",
    "Evaluation: What argument or experiment is used? What benefits and problems are found?",
    "Contributions: What concrete contributions does the paper claim?",
    "Future directions: What future work or next steps do the authors identify?",
]
GRISWOLD_AGREEMENT = validate_agreement({
    "focus_raw": "\n".join(_GRISWOLD_SOURCES),
    "objectives": [
        {"id": "obj1", "source_text": _GRISWOLD_SOURCES[0], "facet": "motivation", "guidance": ["people affected", "technical gap", "prior-solution limitation", "research question"]},
        {"id": "obj2", "source_text": _GRISWOLD_SOURCES[1], "facet": "proposed_solution", "guidance": ["hypothesis", "mechanism", "design rationale", "separate claims from results"]},
        {"id": "obj3", "source_text": _GRISWOLD_SOURCES[2], "facet": "evaluation", "guidance": ["experiment or argument", "baseline and metric", "reported benefits", "reported problems"]},
        {"id": "obj4", "source_text": _GRISWOLD_SOURCES[3], "facet": "contributions", "guidance": ["explicit contribution list", "claimed novelty", "concrete artifact or finding"]},
        {"id": "obj5", "source_text": _GRISWOLD_SOURCES[4], "facet": "future_directions", "guidance": ["future work", "limitations implying next steps", "open extensions"]},
    ],
})


def griswold_prompt() -> str:
    return f"""\
Use this opt-in Griswold reading agreement for extraction:

{agreement_to_text(GRISWOLD_AGREEMENT)}

Griswold questions 1, 2, 3, 5, and 6 are represented as extraction
objectives. Route questions 4, 7, and 8 to critical_analysis because the
reader's analysis, open questions, and take-away message require reasoning
that must remain distinct from what the paper explicitly states.
"""


def grounded_qa_prompt(question: str) -> str:
    return f"""\
Answer this question about the active paper: {question}

Use only the primary context and supplementary passages supplied by the
client. Every answer claim must be supported by a verbatim quote and cite its
specific section and page as [Section title, p. N]. Include a short Supporting
sources list containing those quotes. If the supplied evidence does not
support an answer, say exactly: "The paper does not directly address this."
Never fill an evidence gap from general knowledge.
"""


def guided_reading_prompt(goal: str = "Understand the paper") -> str:
    return f"""\
Create a concrete reading path for this goal: {goal}

Use only the section headings, current highlights, and gap passages supplied
by the client. Every step must name one specific section, quote one specific
highlight, give its page, and explain why that exact highlight is the next
useful step. Do not give generic reading advice. If the supplied highlights
cannot support a path, say so rather than inventing one.
"""


def critical_analysis_prompt(axis: str = "overall strengths and weaknesses") -> str:
    return f"""\
Critically analyze the active paper on this axis: {axis}

Use find_passages and the PaperTrail resources. Organize the answer under
three labels: Paper states, Inference, and Agent opinion. Paper states must be
verbatim-grounded and cite [Section title, p. N]. Inference must state the
evidence and the reasoning step. Agent opinion must give the evaluation
criterion. Never present an inference or opinion as an author claim.
"""


def paper_comparison_prompt(axis: str) -> str:
    return f"""\
Compare every paper in the active PaperTrail session on this axis: {axis}

Use session_info to enumerate the papers, then retrieve evidence for each.
Every comparison point must cite paper name, section, and page in the form
[Paper, Section title, p. N]. Use a side-by-side structure with a final account
of similarities, differences, and missing evidence. If a paper does not
address the axis, say so instead of inferring an answer.
"""
