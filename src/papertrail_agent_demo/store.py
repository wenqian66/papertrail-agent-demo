"""Offline storage and search adapters for exported PaperTrail bundles."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_layer import _focus_tokens, validate_agreement


_LIMIT = re.compile(r"\bat most\s+(\d+)\b", re.IGNORECASE)
_FILTER_STOP_WORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "keep", "no", "not", "of", "on", "only", "or", "the",
    "their", "to", "with", "without",
}
_CATEGORY_TERMS = {
    "implementation detail": {
        "implementation", "implemented", "code", "hyperparameter", "optimizer",
        "batch", "epoch", "hardware", "runtime", "architecture",
    },
    "method detail": {
        "method", "algorithm", "implementation", "procedure", "architecture",
    },
    "results number": {
        "accuracy", "score", "percent", "result", "improvement", "table",
    },
}


@dataclass(frozen=True)
class ExportConfig:
    map_path: Path
    pdf_path: Path | None = None
    state_path: Path | None = None

    @classmethod
    def from_env(cls) -> "ExportConfig":
        project_root = Path(__file__).resolve().parents[2]
        default_map = project_root / "fixtures" / "bundle" / "361011.361061.map.json"
        default_pdf = (
            project_root / "fixtures" / "bundle" / "361011.361061_highlighted.pdf"
        )
        map_value = os.environ.get("PAPERTRAIL_MAP")
        pdf_value = os.environ.get("PAPERTRAIL_PDF")
        state_value = os.environ.get("PAPERTRAIL_AGENT_STATE")
        map_path = Path(map_value).expanduser().resolve() if map_value else default_map
        if pdf_value:
            pdf_path: Path | None = Path(pdf_value).expanduser().resolve()
        elif map_value:
            stem = str(map_path)
            if stem.endswith(".map.json"):
                stem = stem[:-9]
            candidates = (Path(stem + "_highlighted.pdf"), Path(stem + ".pdf"))
            pdf_path = next((path for path in candidates if path.is_file()), None)
        else:
            pdf_path = default_pdf
        state_path = (
            Path(state_value).expanduser().resolve()
            if state_value
            else Path.cwd() / ".papertrail-agent-state.json"
        )
        return cls(map_path=map_path, pdf_path=pdf_path, state_path=state_path)


class ExportStore:
    """Read one Map JSON export and optional paired PDF, with local demo state."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self._map_cache: dict[str, Any] | None = None
        self._sections_cache: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> "ExportStore":
        return cls(ExportConfig.from_env())

    def _load_map(self) -> dict[str, Any]:
        if not self.config.map_path.is_file():
            raise RuntimeError(
                f"Map JSON not found: {self.config.map_path}. Set PAPERTRAIL_MAP."
            )
        if self._map_cache is None:
            loaded = json.loads(self.config.map_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or not isinstance(loaded.get("sections"), list):
                raise ValueError("Map JSON must be an object with a sections list")
            self._map_cache = loaded
        return self._map_cache

    def _read_state(self) -> dict[str, Any]:
        path = self.config.state_path
        if path is None or not path.is_file():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _write_state(self, state: dict[str, Any]) -> None:
        path = self.config.state_path
        if path is None:
            raise RuntimeError("No local state path is configured")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        temporary.replace(path)

    def _highlight_pool(self) -> list[dict[str, Any]]:
        pool: list[dict[str, Any]] = []
        for section_index, section in enumerate(self._load_map().get("sections") or []):
            for highlight_index, highlight in enumerate(section.get("highlights") or []):
                pool.append({
                    "id": f"{section_index}:{highlight_index}",
                    "section_index": section_index,
                    "highlight_index": highlight_index,
                    "section": section.get("title") or "",
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                    "highlight": highlight,
                })
        return pool

    def _visible_ids(self) -> set[str]:
        state = self._read_state()
        stored = state.get("visible_highlight_ids")
        if isinstance(stored, list):
            return {str(value) for value in stored}
        return {item["id"] for item in self._highlight_pool()}

    async def paper_sections(self) -> dict[str, Any]:
        if self._sections_cache is not None:
            return self._sections_cache
        exported = self._load_map()
        sections = [dict(section) for section in exported.get("sections") or []]
        limitations = [
            "Map JSON has no section body text, tables, figures, coordinates, or confidence."
        ]
        pdf_path = self.config.pdf_path
        if pdf_path and pdf_path.is_file():
            try:
                recovered = await asyncio.to_thread(_sections_from_pdf, sections, pdf_path)
                for section, text_fields in zip(sections, recovered):
                    section.update(text_fields)
                limitations.append(
                    "Section text was recovered from the bundled PDF with pypdf; it is not a Map JSON field."
                )
            except Exception as exc:
                limitations.append(f"The paired PDF could not be read: {exc}")
        else:
            limitations.append(
                "No paired PDF was found, so passage search is limited to exported highlights."
            )
        self._sections_cache = {
            "source": "map_json_with_optional_pdf",
            "document": exported.get("document"),
            "display_name": exported.get("display_name"),
            "page_count": exported.get("pages") or 0,
            "summary": exported.get("summary"),
            "summary_kind": exported.get("summary_kind"),
            "sections": sections,
            "limitations": limitations,
        }
        return self._sections_cache

    async def current_highlights(self) -> dict[str, Any]:
        exported = self._load_map()
        visible = self._visible_ids()
        sections: list[dict[str, Any]] = []
        for section_index, section in enumerate(exported.get("sections") or []):
            highlights = [
                highlight
                for highlight_index, highlight in enumerate(section.get("highlights") or [])
                if f"{section_index}:{highlight_index}" in visible
            ]
            sections.append({
                "title": section.get("title"),
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "highlights": highlights,
            })
        state = self._read_state()
        return {
            "source": "map_json_filtered_view",
            "document": exported.get("document"),
            "sections": sections,
            "filter_report": state.get("filter_report") or {
                "mode": "unfiltered_export",
                "pool_size": len(self._highlight_pool()),
                "kept": len(visible),
                "removed": len(self._highlight_pool()) - len(visible),
                "re_extracted": False,
            },
        }

    async def normalized_highlights(self) -> list[dict[str, Any]]:
        payload = await self.current_highlights()
        return [
            {**highlight, "section": section.get("title") or ""}
            for section in payload.get("sections") or []
            for highlight in section.get("highlights") or []
        ]

    async def normalized_sections(self) -> list[dict[str, Any]]:
        payload = await self.paper_sections()
        return [{
            "title": section.get("title") or "",
            "page_start": section.get("page_start") or 1,
            "page_end": section.get("page_end") or section.get("page_start") or 1,
            "text": section.get("text") or "",
            "text_pages": section.get("text_pages") or [],
        } for section in payload.get("sections") or []]

    async def save_built_agreement(
        self, agreement: dict[str, Any], focus_text: str,
    ) -> None:
        checked = validate_agreement(agreement, focus_text=focus_text)
        state = self._read_state()
        history = list(state.get("history") or [])
        if state.get("current_agreement"):
            history.append(_history_entry(state))
        pool_ids = [item["id"] for item in self._highlight_pool()]
        self._write_state({
            "schema_version": 1,
            "focus_text": focus_text,
            "current_agreement": checked,
            "event": "build_agreement",
            "updated_at": _now(),
            "history": history,
            "visible_highlight_ids": pool_ids,
            "filter_report": {
                "mode": "unfiltered_export",
                "pool_size": len(pool_ids),
                "kept": len(pool_ids),
                "removed": 0,
                "re_extracted": False,
            },
        })

    async def refilter_with_agreement(
        self, agreement: dict[str, Any], focus_text: str, refinement: str,
    ) -> dict[str, Any]:
        checked = validate_agreement(agreement, focus_text=focus_text)
        state = self._read_state()
        history = list(state.get("history") or [])
        if state.get("current_agreement"):
            history.append(_history_entry(state))

        exclusions = _agreement_exclusions(checked)
        limit = _agreement_limit(checked)
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        removed_by_rule: list[dict[str, str]] = []
        objectives = checked["objectives"]
        for order, item in enumerate(self._highlight_pool()):
            highlight = item["highlight"]
            searchable = " ".join(str(value or "") for value in (
                item["section"], highlight.get("quote"), highlight.get("note"),
                highlight.get("source"),
            ))
            matched = next((rule for rule in exclusions if _matches_exclusion(searchable, rule)), None)
            if matched:
                removed_by_rule.append({"id": item["id"], "rule": matched})
                continue
            score = _highlight_score(searchable, highlight, objectives)
            ranked.append((score, order, item))

        ranked.sort(key=lambda entry: (-entry[0], entry[1]))
        removed_by_limit = 0
        if limit is not None and len(ranked) > limit:
            removed_by_limit = len(ranked) - limit
            ranked = ranked[:limit]
        visible_ids = [entry[2]["id"] for entry in ranked]
        pool_size = len(self._highlight_pool())
        report = {
            "mode": "existing_highlight_pool_refilter",
            "pool_size": pool_size,
            "kept": len(visible_ids),
            "removed": pool_size - len(visible_ids),
            "removed_by_exclude": len(removed_by_rule),
            "removed_by_limit": removed_by_limit,
            "exclude_rules": exclusions,
            "limit": limit,
            "refinement": refinement,
            "re_extracted": False,
            "limitation": (
                "This standalone demo only filters and re-ranks quotes already present in Map JSON. "
                "It cannot discover new evidence or rerun PaperTrail extraction."
            ),
        }
        self._write_state({
            "schema_version": 1,
            "focus_text": focus_text,
            "current_agreement": checked,
            "event": "update_agreement",
            "updated_at": _now(),
            "history": history,
            "visible_highlight_ids": visible_ids,
            "filter_report": report,
        })
        return report

    async def current_agreement(self) -> dict[str, Any]:
        state = self._read_state()
        return {
            "source": "offline_state",
            "focus_text": state.get("focus_text") or "",
            "agreement": state.get("current_agreement"),
            "updated_at": state.get("updated_at"),
            "history_count": len(state.get("history") or []),
            "message": (
                None if state.get("current_agreement")
                else "The export has no agreement. Call build_agreement to create one."
            ),
        }

    async def session_info(self) -> dict[str, Any]:
        exported = self._load_map()
        runs: dict[str, dict[str, Any]] = {}
        verdicts = {"up": 0, "down": 0, "unmarked": 0}
        for item in self._highlight_pool():
            highlight = item["highlight"]
            run_id = highlight.get("run_id") or ""
            if run_id:
                runs.setdefault(run_id, {
                    "run_id": run_id,
                    "name": highlight.get("run_name") or "",
                    "version": highlight.get("version") or 1,
                    "color": highlight.get("color") or "",
                })
            verdict = highlight.get("verdict") or "unmarked"
            verdicts[verdict if verdict in verdicts else "unmarked"] += 1
        state = self._read_state()
        return {
            "source": "offline_map_json",
            "papers": [{
                "filename": exported.get("document"),
                "display_name": exported.get("display_name"),
                "total_pages": exported.get("pages"),
                "summary": exported.get("summary"),
                "summary_kind": exported.get("summary_kind"),
            }],
            "runs": list(runs.values()),
            "feedback_counts": verdicts,
            "highlight_filter": state.get("filter_report"),
            "limitations": [
                "Map JSON has no session id, focus, agreement, token counts, or full run history."
            ],
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _history_entry(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "agreement": state.get("current_agreement"),
        "focus_text": state.get("focus_text") or "",
        "event": state.get("event"),
        "replaced_at": _now(),
    }


def _agreement_exclusions(agreement: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for objective in agreement["objectives"]:
        for rule in objective["extraction_guidance"]["exclude"]:
            key = rule.casefold()
            if key not in seen:
                seen.add(key)
                result.append(rule)
    return result


def _agreement_limit(agreement: dict[str, Any]) -> int | None:
    limits: list[int] = []
    for objective in agreement["objectives"]:
        match = _LIMIT.search(objective["extraction_guidance"]["edge_cases"])
        if match:
            limits.append(max(1, int(match.group(1))))
    return min(limits) if limits else None


def _matches_exclusion(text: str, rule: str) -> bool:
    lowered = text.casefold()
    normalized_rule = rule.casefold().strip()
    if normalized_rule and normalized_rule in lowered:
        return True
    rule_terms = {
        token.casefold() for token in _focus_tokens(rule)
        if len(token) > 2 and token.casefold() not in _FILTER_STOP_WORDS
    }
    expansions: set[str] = set()
    for category, terms in _CATEGORY_TERMS.items():
        if category in normalized_rule:
            expansions.update(terms)
    candidates = rule_terms | expansions
    if not candidates:
        return False
    text_terms = {token.casefold() for token in _focus_tokens(text)}
    required = 1 if expansions else min(2, len(rule_terms))
    return len(candidates & text_terms) >= required


def _highlight_score(
    text: str, highlight: dict[str, Any], objectives: list[dict[str, Any]],
) -> float:
    lowered = text.casefold()
    terms: set[str] = set()
    for objective in objectives:
        terms.update(
            token.casefold() for token in _focus_tokens(objective["original_text"])
            if len(token) > 2 and token.casefold() not in _FILTER_STOP_WORDS
        )
        for signal in objective["extraction_guidance"]["signals"]:
            terms.update(
                token.casefold() for token in _focus_tokens(signal)
                if len(token) > 2 and token.casefold() not in _FILTER_STOP_WORDS
            )
    score = float(sum(min(lowered.count(term), 3) for term in terms))
    if highlight.get("verdict") == "up":
        score += 4.0
    elif highlight.get("verdict") == "down":
        score -= 4.0
    if highlight.get("note"):
        score += 0.25
    return score


def _normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    offsets: list[int] = []
    pending_space = False
    for index, char in enumerate(text):
        if char.isalnum():
            if pending_space and normalized and normalized[-1] != " ":
                normalized.append(" ")
                offsets.append(index)
            normalized.append(char.casefold())
            offsets.append(index)
            pending_space = False
        else:
            pending_space = True
    return "".join(normalized), offsets


def _find_title(text: str, title: str, start: int = 0) -> int:
    normalized_text, offsets = _normalized_with_offsets(text[start:])
    normalized_title, _ = _normalized_with_offsets(title)
    if not normalized_title:
        return start
    found = normalized_text.find(normalized_title)
    if found < 0 or found >= len(offsets):
        return -1
    return start + offsets[found]


def _sections_from_pdf(
    sections: list[dict[str, Any]], pdf_path: Path,
) -> list[dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    page_text = {
        page_number: (page.extract_text() or "")
        for page_number, page in enumerate(reader.pages, 1)
    }
    cursors: dict[int, int] = {}
    starts: list[tuple[int, int]] = []
    for section in sections:
        page = max(1, int(section.get("page_start") or 1))
        text = page_text.get(page, "")
        cursor = cursors.get(page, 0)
        offset = _find_title(text, str(section.get("title") or ""), cursor)
        if offset < 0:
            offset = cursor
        cursors[page] = max(cursor, offset + 1)
        starts.append((page, offset))

    output: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        start_page, start_offset = starts[index]
        end_page = max(start_page, int(section.get("page_end") or start_page))
        next_start = starts[index + 1] if index + 1 < len(starts) else None
        if next_start and next_start[0] <= end_page:
            final_page, final_offset = next_start
        else:
            final_page, final_offset = end_page, len(page_text.get(end_page, ""))
        text_pages: list[dict[str, Any]] = []
        for page in range(start_page, final_page + 1):
            raw = page_text.get(page, "")
            left = start_offset if page == start_page else 0
            right = final_offset if page == final_page else len(raw)
            body = " ".join(raw[left:right].split())
            if body:
                text_pages.append({"page": page, "text": body})
        output.append({
            "text": "\n\n".join(item["text"] for item in text_pages),
            "text_pages": text_pages,
        })
    return output
