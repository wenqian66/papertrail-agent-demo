"""Shared logic for PaperTrail's MCP agent layer.

This module has no MCP or FastAPI dependency. The FastAPI agreement step, the
MCP tools, and the demo CLI all use the same validation, serialization,
fallback guidance, passage search, and prompt text.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any


MAX_OBJECTIVES = 32

BUILD_AGREEMENT_PROMPT = """\
Turn the user's reading focus into a structured extraction agreement.

You have exactly two jobs:
1. Split the focus only when it contains more than one distinct reading goal.
2. Add practical extraction guidance for each resulting objective.

Preservation is absolute. Every original_text value must be copied as one
exact, contiguous substring of the user's focus. Keep the user's spelling,
capitalization, punctuation, qualifiers, numbering, and wording. Do not
paraphrase, summarize, improve, complete, or map the focus to a taxonomy. Do
not drop any word from the focus. A focus with one goal stays one objective.

Guidance should say where in an academic paper to look, which words or
patterns are useful signals, what near misses to exclude, and how to handle
ambiguity. Guidance may use the paper list below, but must not claim that the
papers contain a finding.

Return JSON only, with this exact shape:
{{
  "objectives": [
    {{
      "original_text": "an exact substring of the focus",
      "extraction_guidance": {{
        "look_in": ["section type"],
        "signals": ["word or pattern"],
        "exclude": ["near miss"],
        "edge_cases": "how to handle ambiguity"
      }}
    }}
  ]
}}

The user's focus:
<focus>
{question}
</focus>

The papers this run will read:
{papers}
"""

REFINE_AGREEMENT_PROMPT = """\
Update a structured extraction agreement from a user's refinement request.

Change only extraction_guidance. Keep the objectives in the same order and
copy every original_text value byte for byte from the current agreement. Do
not add, remove, merge, split, or rewrite objectives. Translate the request
into concrete look_in, signals, exclude, or edge_cases guidance. A request for
a result limit belongs in edge_cases. A request to remove a category belongs
in exclude.

Return JSON only in the same shape as the current agreement.

Current agreement:
{agreement}

Refinement request:
{request}
"""


_WORDS = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_INLINE_NUMBER = re.compile(r"(?=\s+(?:\(\d+\)|\d+[.)])\s+)")
_CONJUNCTION_GOAL = re.compile(
    r"(?=\s+and\s+(?:what|why|how|which|where|when|who)\b)",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[-*+]\s+|\(?\d+[.)]\s+)")
_TOP_N = re.compile(
    r"\b(?:top|best|strongest|only\s+keep|keep\s+only|at\s+most)\s+(\d+)\b",
    re.IGNORECASE,
)
_EXCLUDE_REQUEST = re.compile(
    r"\b(?:remove|exclude|omit|drop|without)\s+(.+?)(?:[.!?]|$)",
    re.IGNORECASE,
)

_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "paper",
    "that", "the", "their", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "with",
}


def agreement_to_text(agreement: dict[str, Any]) -> str:
    """Canonical, editable text stored in the existing agreement column."""
    checked = validate_agreement(agreement)
    return json.dumps(checked, ensure_ascii=False, indent=2)


def _json_object(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = (value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("agreement must be a JSON object")
    return loaded


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    out: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"{field} entries must be non-empty strings")
        out.append(entry.strip())
    return out


def _focus_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _WORDS.finditer(text)]


def _validate_focus_coverage(focus_text: str, originals: list[str]) -> None:
    """Require exact, ordered source spans and complete word coverage."""
    focus = focus_text.strip()
    cursor = 0
    for original in originals:
        pos = focus.find(original, cursor)
        if pos < 0:
            raise ValueError("original_text is not an exact ordered focus substring")
        cursor = pos + len(original)
    covered = [token for original in originals for token in _focus_tokens(original)]
    if covered != _focus_tokens(focus):
        raise ValueError("agreement dropped, added, or reordered focus words")


def validate_agreement(
    value: str | dict[str, Any], *, focus_text: str | None = None,
    original_agreement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize an agreement without changing original text."""
    data = _json_object(value)
    objectives = data.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("agreement needs at least one objective")
    if len(objectives) > MAX_OBJECTIVES:
        raise ValueError(f"agreement has more than {MAX_OBJECTIVES} objectives")

    normalized: list[dict[str, Any]] = []
    originals: list[str] = []
    for objective in objectives:
        if not isinstance(objective, dict):
            raise ValueError("each objective must be an object")
        original = objective.get("original_text")
        if not isinstance(original, str) or not original:
            raise ValueError("original_text must be a non-empty string")
        guidance = objective.get("extraction_guidance")
        if not isinstance(guidance, dict):
            raise ValueError("extraction_guidance must be an object")
        edge_cases = guidance.get("edge_cases")
        if not isinstance(edge_cases, str):
            raise ValueError("edge_cases must be a string")
        normalized.append({
            "original_text": original,
            "extraction_guidance": {
                "look_in": _string_list(guidance.get("look_in"), "look_in"),
                "signals": _string_list(guidance.get("signals"), "signals"),
                "exclude": _string_list(guidance.get("exclude"), "exclude"),
                "edge_cases": edge_cases.strip(),
            },
        })
        originals.append(original)

    if focus_text is not None:
        _validate_focus_coverage(focus_text, originals)
    if original_agreement is not None:
        before = validate_agreement(original_agreement)
        expected = [obj["original_text"] for obj in before["objectives"]]
        if originals != expected:
            raise ValueError("a refinement changed the agreement objectives")
    return {"objectives": normalized}


def agreement_from_model(focus_text: str, response_text: str) -> dict[str, Any]:
    """Parse model output and enforce the no-rewrite, no-drop contract."""
    return validate_agreement(response_text, focus_text=focus_text)


def refined_agreement_from_model(
    current: dict[str, Any], response_text: str,
) -> dict[str, Any]:
    """Parse model refinement output and freeze every objective text."""
    return validate_agreement(response_text, original_agreement=current)


def _focus_parts(focus_text: str) -> list[str]:
    """Conservative offline split used only when no model is configured."""
    focus = focus_text.strip()
    if not focus:
        raise ValueError("focus text cannot be empty")

    lines = list(re.finditer(r"[^\r\n]+", focus))
    nonempty = [m for m in lines if m.group(0).strip()]
    bullet_indexes = [i for i, m in enumerate(nonempty) if _BULLET.match(m.group(0))]
    if len(bullet_indexes) >= 2:
        starts = [nonempty[i].start() for i in bullet_indexes]
        starts[0] = 0
        return [
            focus[start:(starts[i + 1] if i + 1 < len(starts) else len(focus))].strip()
            for i, start in enumerate(starts)
        ]
    if len(nonempty) >= 2:
        return [m.group(0).strip() for m in nonempty]

    numbered = [part.strip() for part in _INLINE_NUMBER.split(focus) if part.strip()]
    if len(numbered) >= 2:
        return numbered
    compound = [part.strip() for part in _CONJUNCTION_GOAL.split(focus) if part.strip()]
    if len(compound) >= 2:
        return compound
    return [focus]


def _keywords(text: str, limit: int = 6) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in _focus_tokens(text):
        lowered = token.casefold()
        if len(lowered) < 3 or lowered in _STOP_WORDS or lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
        if len(out) == limit:
            break
    return out


def _guidance_for(text: str) -> dict[str, Any]:
    lowered = text.casefold()
    look_in: list[str]
    signals: list[str]
    exclude: list[str]
    edge_cases: str
    if any(term in lowered for term in ("motivat", "problem", "gap", "prior")):
        look_in = ["abstract", "introduction", "related work"]
        signals = ["however", "limitation", "fails to", "we ask", "challenge"]
        exclude = ["method details with no stated rationale", "results with no problem framing"]
        edge_cases = (
            "Motivation can be implicit. Keep a passage only when the authors connect "
            "a prior limitation or practical need to the question they pursue."
        )
    elif any(term in lowered for term in ("method", "solution", "approach", "system", "design")):
        look_in = ["abstract", "method", "approach", "system design"]
        signals = ["we propose", "our approach", "consists of", "algorithm", "architecture"]
        exclude = ["evaluation results", "future work", "background methods not adopted"]
        edge_cases = (
            "Separate the claimed mechanism and rationale from evidence that it worked. "
            "Keep implementation detail only when it explains how the proposed solution is achieved."
        )
    elif any(term in lowered for term in ("evaluat", "result", "baseline", "experiment", "benefit")):
        look_in = ["evaluation", "experiments", "results", "discussion"]
        signals = ["baseline", "compared with", "improves", "decreases", "ablation", "limitation"]
        exclude = ["experimental setup with no result", "unsupported performance claims"]
        edge_cases = (
            "Keep the comparison conditions with the reported outcome. Treat an author explanation "
            "as interpretation unless the same passage supplies supporting evidence."
        )
    elif any(term in lowered for term in ("contribut", "novel", "advance")):
        look_in = ["abstract", "introduction", "conclusion"]
        signals = ["we contribute", "our contributions", "first", "novel", "we introduce"]
        exclude = ["broad impact claims with no concrete contribution", "future work"]
        edge_cases = (
            "Prefer explicit author claims, but keep a concrete contribution stated without a label. "
            "Do not invent novelty by comparing unrelated passages."
        )
    elif any(term in lowered for term in ("future", "open question", "next step")):
        look_in = ["discussion", "limitations", "conclusion", "future work"]
        signals = ["future work", "remains", "open question", "could", "next"]
        exclude = ["work already completed in the paper", "generic field-wide speculation"]
        edge_cases = (
            "Distinguish directions the authors actually propose from limitations that merely imply "
            "a direction. Label an implication as such in the extraction note."
        )
    else:
        look_in = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        signals = _keywords(text) or ["explicit answer", "definition", "reported evidence"]
        exclude = ["keyword matches that do not answer the objective", "references to other work only"]
        edge_cases = (
            "When the answer is distributed across passages, keep each independently useful verbatim "
            "passage and explain the connection in its note rather than inferring a new quote."
        )
    return {
        "look_in": look_in,
        "signals": signals,
        "exclude": exclude,
        "edge_cases": edge_cases,
    }


def build_agreement_fallback(focus_text: str) -> dict[str, Any]:
    """Build a safe local agreement for demos without an agent model."""
    agreement = {
        "objectives": [
            {"original_text": part, "extraction_guidance": _guidance_for(part)}
            for part in _focus_parts(focus_text)
        ]
    }
    return validate_agreement(agreement, focus_text=focus_text)


def _append_unique(values: list[str], value: str) -> None:
    if value and value.casefold() not in {entry.casefold() for entry in values}:
        values.append(value)


def refine_agreement_fallback(
    current: dict[str, Any], request: str,
) -> dict[str, Any]:
    """Translate common refine requests without changing objective text."""
    request = (request or "").strip()
    if not request:
        raise ValueError("refinement request cannot be empty")
    updated = copy.deepcopy(validate_agreement(current))
    limit_match = _TOP_N.search(request)
    exclusion_match = _EXCLUDE_REQUEST.search(request)
    for objective in updated["objectives"]:
        guidance = objective["extraction_guidance"]
        if limit_match:
            limit = int(limit_match.group(1))
            sentence = (
                f"Across the rerun, return at most {limit} of the strongest passages for this "
                "objective, preferring direct and specific evidence."
            )
            existing = guidance["edge_cases"]
            guidance["edge_cases"] = f"{existing} {sentence}".strip()
        if exclusion_match:
            exclusion = re.split(
                r"\s+and\s+(?:only\s+keep|keep\s+only|top|at\s+most)\b",
                exclusion_match.group(1).strip(),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            _append_unique(guidance["exclude"], exclusion)
        if not limit_match and not exclusion_match:
            existing = guidance["edge_cases"]
            guidance["edge_cases"] = f"{existing} User refinement: {request}".strip()
    return validate_agreement(updated, original_agreement=current)


def build_agreement_prompt(focus_text: str, papers: str = "(none)") -> str:
    return BUILD_AGREEMENT_PROMPT.format(question=focus_text, papers=papers or "(none)")


def refine_agreement_prompt(current: dict[str, Any], request: str) -> str:
    return REFINE_AGREEMENT_PROMPT.format(
        agreement=agreement_to_text(current), request=request,
    )


def _search_terms(text: str) -> list[str]:
    return [
        token.casefold() for token in _focus_tokens(text)
        if len(token) > 2 and token.casefold() not in _STOP_WORDS
    ]


def _passage_chunks(text: str, max_chars: int = 900) -> list[tuple[int, str]]:
    paragraphs = [m for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S)]
    if not paragraphs:
        paragraphs = [m for m in re.finditer(r"\S.*?(?=\Z)", text, re.S)]
    out: list[tuple[int, str]] = []
    for paragraph in paragraphs:
        raw = " ".join(paragraph.group(0).split())
        if not raw:
            continue
        if len(raw) <= max_chars:
            out.append((paragraph.start(), raw))
            continue
        sentences = re.split(r"(?<=[.!?])\s+", raw)
        chunk = ""
        offset = paragraph.start()
        for sentence in sentences:
            if chunk and len(chunk) + len(sentence) + 1 > max_chars:
                out.append((offset, chunk))
                offset += len(chunk) + 1
                chunk = sentence
            else:
                chunk = f"{chunk} {sentence}".strip()
        if chunk:
            out.append((offset, chunk))
    return out


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
    """Rank paper text and highlight quotes with section and page anchors."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question cannot be empty")
    limit = max(1, min(int(limit), 20))
    candidates: list[dict[str, Any]] = []
    for section in sections:
        title = section.get("title") or section.get("header") or "Untitled section"
        page_start = section.get("page_start") or section.get("page") or 1
        page_texts = section.get("text_pages") or []
        if page_texts:
            text_sources = [
                (entry.get("page") or page_start, entry.get("text") or "")
                for entry in page_texts
            ]
        else:
            text_sources = [(page_start, section.get("text") or "")]
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
    return {
        "question": question,
        "addressed": bool(selected),
        "passages": selected,
    }


GRISWOLD_AGREEMENT = validate_agreement({
    "objectives": [
        {
            "original_text": (
                "Motivations: What is the people problem? What is the technical problem? "
                "Why are prior solutions inadequate? What is the research question?"
            ),
            "extraction_guidance": {
                "look_in": ["abstract", "introduction", "related work"],
                "signals": ["however", "challenge", "limitation", "we ask", "existing approaches"],
                "exclude": ["solution detail without problem framing", "results without motivation"],
                "edge_cases": (
                    "The people problem may be described as consequences or stakeholders rather than "
                    "named directly. Keep implicit motivation only when the authors connect it to the "
                    "technical problem."
                ),
            },
        },
        {
            "original_text": (
                "Proposed solution: What is the hypothesis? Why should it work? How is it achieved? "
                "Keep the solution separate from its results."
            ),
            "extraction_guidance": {
                "look_in": ["abstract", "approach", "methods", "system design"],
                "signals": ["we propose", "our hypothesis", "we design", "consists of", "because"],
                "exclude": ["evaluation outcomes", "background techniques the authors do not adopt"],
                "edge_cases": (
                    "Keep rationale and mechanism together when they are in one passage. Do not treat an "
                    "observed result as part of the proposed solution."
                ),
            },
        },
        {
            "original_text": (
                "Evaluation: What argument or experiment is used? What benefits and problems are found?"
            ),
            "extraction_guidance": {
                "look_in": ["evaluation", "experiments", "results", "discussion", "limitations"],
                "signals": ["baseline", "dataset", "metric", "compared with", "improves", "fails"],
                "exclude": ["setup detail with no bearing on validity", "unmeasured benefit claims"],
                "edge_cases": (
                    "Keep enough context to identify the comparison and metric. Separate reported outcomes "
                    "from the authors' explanation of why they occurred."
                ),
            },
        },
        {
            "original_text": "Contributions: What concrete contributions does the paper claim?",
            "extraction_guidance": {
                "look_in": ["abstract", "introduction", "conclusion"],
                "signals": ["we contribute", "our contributions", "we introduce", "first", "novel"],
                "exclude": ["generic impact claims", "work credited only to prior research"],
                "edge_cases": (
                    "Prefer explicit contribution lists, but keep an unlabeled concrete contribution when "
                    "the authors clearly claim ownership of it."
                ),
            },
        },
        {
            "original_text": "Future directions: What future work or next steps do the authors identify?",
            "extraction_guidance": {
                "look_in": ["discussion", "limitations", "conclusion", "future work"],
                "signals": ["future work", "remains", "next", "could be extended", "open"],
                "exclude": ["directions proposed only by cited work", "completed follow-up experiments"],
                "edge_cases": (
                    "Distinguish explicit future directions from limitations that only imply one. Keep the "
                    "latter only with a note that the direction is implied."
                ),
            },
        },
    ]
})


def griswold_prompt() -> str:
    return f"""\
Use this opt-in Griswold reading agreement for extraction:

{agreement_to_text(GRISWOLD_AGREEMENT)}

Objectives 1, 2, 3, 5, and 6 are extraction tasks. After highlights exist,
handle Griswold questions 4, 7, and 8 with the critical_analysis prompt:
the reader's analysis, open questions, and take-away message require reasoning
that must remain distinct from what the paper explicitly states.
"""


def grounded_qa_prompt(question: str) -> str:
    return f"""\
Answer this question about the active PaperTrail paper:

{question}

Call find_passages with the question before answering. Use only returned paper
text or current highlights as evidence. Every substantive claim must cite a
specific section and page in the form [Section title, p. N]. If the retrieved
passages do not answer the question, say that the paper does not address it.
Do not fill the gap from general knowledge without labeling it as outside the
paper.
"""


def guided_reading_prompt(goal: str = "Understand the paper") -> str:
    return f"""\
Create a concrete reading path for this goal: {goal}

Read papertrail://paper/sections and papertrail://highlights/current. Return an
ordered path. Every step must name one specific section, quote one current
highlight from that section, give its page, and explain why that exact passage
is the next useful step. Do not give generic advice or name a section without
a highlight. If there are no highlights, say that a highlight-grounded path
cannot yet be made.
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
