"""Deterministic storage, PDF recovery, and tagged-highlight filtering."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_layer import validate_agreement


_DENSITY_FLOORS = {"low": 0.80, "medium": 0.50, "high": 0.20}


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
    """Read one export bundle and persist only local agent-layer state."""

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
                    "order": len(pool),
                    "section_index": section_index,
                    "highlight_index": highlight_index,
                    "section": section.get("title") or "",
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                    "highlight": highlight,
                })
        return pool

    def highlight_inputs(self) -> list[dict[str, Any]]:
        """Compact, immutable pool sent to disclosed LLM tagging passes."""
        return [{
            "id": item["id"],
            "section": item["section"],
            "page": item["highlight"].get("page"),
            "quote": item["highlight"].get("quote") or "",
            "note": item["highlight"].get("note") or "",
        } for item in self._highlight_pool()]

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

    def _visible_ids(self, state: dict[str, Any]) -> list[str]:
        stored = state.get("visible_highlight_ids")
        if isinstance(stored, list):
            return [str(value) for value in stored]
        return [item["id"] for item in self._highlight_pool()]

    def _enriched_highlight(
        self, item: dict[str, Any], state: dict[str, Any], rank: int | None,
    ) -> dict[str, Any]:
        base_tag = (state.get("highlight_tags") or {}).get(item["id"]) or {}
        criterion_map = (state.get("criterion_tags") or {}).get(item["id"]) or {}
        return {
            **item["highlight"],
            "objective_id": base_tag.get("objective_id"),
            "facet": base_tag.get("facet"),
            "salience": base_tag.get("salience"),
            "criterion_tags": dict(criterion_map),
            "rank": rank,
        }

    async def current_highlights(self) -> dict[str, Any]:
        exported = self._load_map()
        state = self._read_state()
        visible_ids = self._visible_ids(state)
        visible = set(visible_ids)
        ranks = {highlight_id: index + 1 for index, highlight_id in enumerate(visible_ids)}
        sections: list[dict[str, Any]] = []
        for section_index, section in enumerate(exported.get("sections") or []):
            highlights: list[dict[str, Any]] = []
            for highlight_index, _highlight in enumerate(section.get("highlights") or []):
                highlight_id = f"{section_index}:{highlight_index}"
                if highlight_id not in visible:
                    continue
                item = next(
                    pool_item for pool_item in self._highlight_pool()
                    if pool_item["id"] == highlight_id
                )
                highlights.append(self._enriched_highlight(item, state, ranks[highlight_id]))
            highlights.sort(key=lambda item: item.get("rank") or 10**9)
            sections.append({
                "title": section.get("title"),
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "highlights": highlights,
            })
        pool_size = len(self._highlight_pool())
        return {
            "source": "map_json_filtered_view",
            "document": exported.get("document"),
            "sections": sections,
            "tagging": state.get("tagging") or {
                "status": "not_tagged",
                "note": "Call build_agreement to run the required one-time LLM enrichment.",
                "re_extracted": False,
            },
            "filter_report": state.get("filter_report") or {
                "mode": "unfiltered_export",
                "pool_size": pool_size,
                "kept": len(visible_ids),
                "removed": pool_size - len(visible_ids),
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

    async def ranked_highlights(self) -> list[dict[str, Any]]:
        highlights = await self.normalized_highlights()
        return sorted(highlights, key=lambda item: item.get("rank") or 10**9)

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
        self, agreement: dict[str, Any], tags: dict[str, dict[str, Any]],
        *, model_name: str = "",
    ) -> None:
        checked = validate_agreement(agreement)
        pool_ids = [item["id"] for item in self._highlight_pool()]
        if set(tags) != set(pool_ids):
            raise ValueError("highlight tags must cover the complete exported pool")
        previous = self._read_state()
        history = list(previous.get("history") or [])
        if previous.get("current_agreement"):
            history.append(_history_entry(previous))
        filter_state = _default_filter_state(checked)
        self._write_state({
            "schema_version": 2,
            "focus_raw": checked["focus_raw"],
            "current_agreement": checked,
            "event": "build_agreement",
            "updated_at": _now(),
            "history": history,
            "highlight_tags": tags,
            "criterion_tags": {},
            "filter_state": filter_state,
            "visible_highlight_ids": pool_ids,
            "tagging": {
                "status": "tagged",
                "base_tagged_at": _now(),
                "model": model_name or None,
                "criteria": [],
                "re_extracted": False,
                "note": "Existing Map JSON highlights were enriched once with facet and salience tags.",
            },
            "filter_report": {
                "mode": "unfiltered_tagged_pool",
                "pool_size": len(pool_ids),
                "kept": len(pool_ids),
                "removed": 0,
                "re_extracted": False,
            },
        })

    def known_facets(self, agreement: dict[str, Any]) -> list[str]:
        state = self._read_state()
        known = {"other"}
        known.update(_canonical_facet(item["facet"]) for item in agreement["objectives"])
        known.update(
            _canonical_facet(tag.get("facet") or "other")
            for tag in (state.get("highlight_tags") or {}).values()
        )
        for matches in (state.get("criterion_tags") or {}).values():
            known.update(matches)
        return sorted(known)

    def criteria_needed(
        self, agreement: dict[str, Any], operations: list[dict[str, Any]],
    ) -> list[str]:
        known = set(self.known_facets(agreement))
        needed = {
            operation["facet"]
            for operation in operations
            if operation["op"] in {"add_exclude", "change_density"}
            and operation["facet"] not in known
        }
        return sorted(needed)

    async def apply_refinement(
        self, agreement: dict[str, Any], operations: list[dict[str, Any]],
        refinement: str,
        criterion_updates: dict[str, dict[str, bool]] | None = None,
        *, model_name: str = "",
    ) -> dict[str, Any]:
        checked = validate_agreement(agreement)
        state = self._read_state()
        if state.get("current_agreement") != checked:
            raise RuntimeError("The cached agreement changed; rebuild before refining")
        pool = self._highlight_pool()
        pool_ids = [item["id"] for item in pool]
        base_tags = state.get("highlight_tags") or {}
        if set(base_tags) != set(pool_ids):
            raise RuntimeError("Highlights are not tagged. Run build_agreement first.")

        criterion_tags = {
            highlight_id: dict(matches)
            for highlight_id, matches in (state.get("criterion_tags") or {}).items()
        }
        if criterion_updates:
            if set(criterion_updates) != set(pool_ids):
                raise ValueError("criterion tags must cover the complete highlight pool")
            for highlight_id, matches in criterion_updates.items():
                criterion_tags.setdefault(highlight_id, {}).update(matches)

        filter_state = _normalized_filter_state(
            state.get("filter_state"), checked,
        )
        for operation in operations:
            _apply_operation(filter_state, operation)

        ranked, reasons = _filter_pool(pool, base_tags, criterion_tags, filter_state)
        visible_ids = [item["id"] for item in ranked]
        previous_history = list(state.get("history") or [])
        previous_history.append(_history_entry(state))
        tagging = dict(state.get("tagging") or {})
        criteria = sorted({
            criterion
            for matches in criterion_tags.values()
            for criterion in matches
        })
        tagging.update({
            "status": "tagged",
            "criteria": criteria,
            "re_extracted": False,
        })
        if criterion_updates:
            tagging["criteria_tagged_at"] = _now()
            tagging["criteria_model"] = model_name or None
        report = {
            "mode": "cached_tag_filter",
            "pool_size": len(pool),
            "kept": len(visible_ids),
            "removed": len(pool) - len(visible_ids),
            "removed_by_reason": reasons,
            "operations": operations,
            "filter_state": filter_state,
            "refinement": refinement,
            "criteria_tagging_ran": bool(criterion_updates),
            "re_extracted": False,
            "limitation": (
                "Filtering can only narrow or reorder the existing Map JSON highlight pool; "
                "it cannot discover new evidence."
            ),
        }
        self._write_state({
            **state,
            "schema_version": 2,
            "focus_raw": checked["focus_raw"],
            "current_agreement": checked,
            "event": "update_agreement",
            "updated_at": _now(),
            "history": previous_history,
            "criterion_tags": criterion_tags,
            "filter_state": filter_state,
            "visible_highlight_ids": visible_ids,
            "tagging": tagging,
            "filter_report": report,
        })
        return report

    async def current_agreement(self) -> dict[str, Any]:
        state = self._read_state()
        agreement = state.get("current_agreement")
        if agreement:
            try:
                agreement = validate_agreement(agreement)
            except (ValueError, json.JSONDecodeError):
                agreement = None
        return {
            "source": "offline_state",
            "agreement": agreement,
            "filter_state": state.get("filter_state"),
            "updated_at": state.get("updated_at"),
            "history_count": len(state.get("history") or []),
            "message": (
                None if agreement
                else "No current v2 agreement. Call build_agreement; an LLM is required."
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
            "highlight_tagging": state.get("tagging"),
            "highlight_filter": state.get("filter_report"),
            "limitations": [
                "Map JSON has no session id, focus, agreement, token counts, or full run history."
            ],
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_facet(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "other"


def _default_filter_state(agreement: dict[str, Any]) -> dict[str, Any]:
    return {
        "limit": None,
        "excluded_facets": [],
        "enabled_objective_ids": [item["id"] for item in agreement["objectives"]],
        "density_by_facet": {},
    }


def _normalized_filter_state(
    value: Any, agreement: dict[str, Any],
) -> dict[str, Any]:
    default = _default_filter_state(agreement)
    if not isinstance(value, dict):
        return default
    objective_ids = {item["id"] for item in agreement["objectives"]}
    enabled = value.get("enabled_objective_ids")
    if not isinstance(enabled, list):
        enabled = default["enabled_objective_ids"]
    return {
        "limit": value.get("limit") if isinstance(value.get("limit"), int) else None,
        "excluded_facets": sorted({
            _canonical_facet(item) for item in value.get("excluded_facets") or []
            if isinstance(item, str)
        }),
        "enabled_objective_ids": [item for item in enabled if item in objective_ids],
        "density_by_facet": {
            _canonical_facet(key): level
            for key, level in (value.get("density_by_facet") or {}).items()
            if isinstance(key, str) and level in _DENSITY_FLOORS
        },
    }


def _apply_operation(filter_state: dict[str, Any], operation: dict[str, Any]) -> None:
    name = operation["op"]
    if name == "set_limit":
        filter_state["limit"] = operation["value"]
    elif name == "add_exclude":
        excluded = set(filter_state["excluded_facets"])
        excluded.add(operation["facet"])
        filter_state["excluded_facets"] = sorted(excluded)
    elif name == "remove_exclude":
        filter_state["excluded_facets"] = [
            item for item in filter_state["excluded_facets"]
            if item != operation["facet"]
        ]
    elif name == "enable_facet":
        enabled = set(filter_state["enabled_objective_ids"])
        enabled.add(operation["id"])
        filter_state["enabled_objective_ids"] = sorted(enabled)
    elif name == "disable_facet":
        filter_state["enabled_objective_ids"] = [
            item for item in filter_state["enabled_objective_ids"]
            if item != operation["id"]
        ]
    elif name == "change_density":
        filter_state["density_by_facet"][operation["facet"]] = operation["level"]


def _matches_facet(
    highlight_id: str, facet: str, base_tag: dict[str, Any],
    criterion_tags: dict[str, dict[str, bool]],
) -> bool:
    return (
        _canonical_facet(str(base_tag.get("facet") or "other")) == facet
        or bool((criterion_tags.get(highlight_id) or {}).get(facet))
    )


def _filter_pool(
    pool: list[dict[str, Any]], base_tags: dict[str, dict[str, Any]],
    criterion_tags: dict[str, dict[str, bool]], filter_state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    enabled = set(filter_state["enabled_objective_ids"])
    excluded = set(filter_state["excluded_facets"])
    density = filter_state["density_by_facet"]
    reasons = {"disabled_objective": 0, "excluded_facet": 0, "density": 0, "limit": 0}
    kept: list[dict[str, Any]] = []
    for item in pool:
        tag = base_tags[item["id"]]
        objective_id = tag.get("objective_id")
        if objective_id != "other" and objective_id not in enabled:
            reasons["disabled_objective"] += 1
            continue
        if any(
            _matches_facet(item["id"], facet, tag, criterion_tags)
            for facet in excluded
        ):
            reasons["excluded_facet"] += 1
            continue
        salience = float(tag.get("salience") or 0.0)
        failed_density = any(
            _matches_facet(item["id"], facet, tag, criterion_tags)
            and salience < _DENSITY_FLOORS[level]
            for facet, level in density.items()
        )
        if failed_density:
            reasons["density"] += 1
            continue
        kept.append(item)
    kept.sort(key=lambda item: (-float(base_tags[item["id"]]["salience"]), item["order"]))
    limit = filter_state.get("limit")
    if isinstance(limit, int) and len(kept) > limit:
        reasons["limit"] = len(kept) - limit
        kept = kept[:limit]
    return kept, reasons


def _history_entry(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "agreement": state.get("current_agreement"),
        "filter_state": state.get("filter_state"),
        "event": state.get("event"),
        "replaced_at": _now(),
    }


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
