"""Mind2Web replay-environment helpers.

The environment is intentionally offline and deterministic:
- each episode replays a single Mind2Web task
- each step exposes a compact OpenClaw-style interactive snapshot
- the agent acts with Mind2Web-aligned operations: CLICK / TYPE / SELECT

This keeps training/evaluation stable while still matching the dataset's
action semantics and aggressively reducing token-heavy raw HTML.
"""

from __future__ import annotations

from copy import deepcopy
import json
from html.parser import HTMLParser
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import threading
from typing import Any

import ijson


ACTION_REPR_PATTERN = re.compile(
    r"^\[(?P<role>[^\]]*)\]\s*(?P<target>.*?)\s*->\s*"
    r"(?P<op>CLICK|TYPE|SELECT|HOVER|ENTER)"
    r"(?:\s*:\s*(?P<value>.*))?$",
    re.IGNORECASE,
)
MODEL_ACTION_PATTERN = re.compile(
    r"^\s*(?P<op>CLICK|TYPE|SELECT|HOVER|ENTER)\s+"
    r"(?P<ref>e\d+)"
    r"(?:\s*(?:\||:)\s*(?P<value>.*))?\s*$",
    re.IGNORECASE | re.DOTALL,
)

INTERACTIVE_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "link",
    "menuitem",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
}
TAG_TO_ROLE = {
    "a": "link",
    "button": "button",
    "option": "option",
    "select": "combobox",
    "summary": "button",
    "textarea": "textbox",
}
INPUT_TYPE_TO_ROLE = {
    "button": "button",
    "checkbox": "checkbox",
    "email": "textbox",
    "image": "button",
    "number": "spinbutton",
    "password": "textbox",
    "radio": "radio",
    "range": "slider",
    "reset": "button",
    "search": "searchbox",
    "submit": "button",
    "tel": "textbox",
    "text": "textbox",
    "url": "textbox",
}
ROLE_EQUIVALENTS = {
    "searchbox": {"textbox", "searchbox"},
    "textbox": {"textbox", "searchbox"},
    "combobox": {"combobox", "listbox", "option"},
    "link": {"link"},
    "button": {"button"},
    "checkbox": {"checkbox"},
    "radio": {"radio"},
    "option": {"option", "combobox"},
}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


_singleflight_registry_lock = threading.Lock()
_singleflight_locks: dict[tuple[Any, ...], threading.Lock] = {}


def _get_singleflight_lock(*key_parts: Any) -> threading.Lock:
    key = tuple(key_parts)
    with _singleflight_registry_lock:
        lock = _singleflight_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _singleflight_locks[key] = lock
        return lock


def _task_cache_paths(source_file: str) -> tuple[Path, Path]:
    path = Path(source_file).expanduser().resolve()
    cache_dir = path.parent / ".webcache"
    return (
        cache_dir / f"{path.stem}.tasks.jsonl",
        cache_dir / f"{path.stem}.offsets.json",
    )


def _task_cache_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def _load_offsets_metadata(offsets_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(offsets_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _task_cache_is_valid(source_file: str) -> bool:
    path = Path(source_file).expanduser().resolve()
    cache_jsonl_path, offsets_path = _task_cache_paths(str(path))
    if not cache_jsonl_path.exists() or not offsets_path.exists():
        return False

    payload = _load_offsets_metadata(offsets_path)
    if not payload:
        return False

    signature = _task_cache_signature(path)
    if int(payload.get("source_size", -1)) != signature["source_size"]:
        return False
    if int(payload.get("source_mtime_ns", -1)) != signature["source_mtime_ns"]:
        return False
    return True


def _ensure_task_cache(source_file: str) -> tuple[Path, Path]:
    path = Path(source_file).expanduser().resolve()
    cache_jsonl_path, offsets_path = _task_cache_paths(str(path))
    if _task_cache_is_valid(str(path)):
        return cache_jsonl_path, offsets_path

    with _get_singleflight_lock("task_cache_build", str(path)):
        if _task_cache_is_valid(str(path)):
            return cache_jsonl_path, offsets_path

        cache_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_jsonl_path = cache_jsonl_path.with_suffix(cache_jsonl_path.suffix + ".tmp")
        tmp_offsets_path = offsets_path.with_suffix(offsets_path.suffix + ".tmp")

        offsets: list[tuple[int, int]] = []
        position = 0
        with path.open("rb") as src, tmp_jsonl_path.open("wb") as dst:
            for task in ijson.items(src, "item"):
                line = (json.dumps(_compact_task(task), ensure_ascii=False) + "\n").encode("utf-8")
                start = position
                dst.write(line)
                position += len(line)
                offsets.append((start, position))

        payload = {
            **_task_cache_signature(path),
            "count": len(offsets),
            "offsets": offsets,
        }
        tmp_offsets_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_jsonl_path.replace(cache_jsonl_path)
        tmp_offsets_path.replace(offsets_path)

    return cache_jsonl_path, offsets_path


@dataclass
class DOMNode:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["DOMNode"] = field(default_factory=list)
    text_chunks: list[str] = field(default_factory=list)

    def add_text(self, text: str) -> None:
        normalized = normalize_text(text)
        if normalized:
            self.text_chunks.append(normalized)


def normalize_text(text: str | None, *, lower: bool = False) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text.lower() if lower else text


def normalize_value(text: str | None) -> str:
    return normalize_text(text, lower=True)


def normalize_op(op: str | None) -> str:
    op = normalize_text(op, lower=True).upper()
    if op in {"HOVER", "ENTER"}:
        return "CLICK"
    return op


def _safe_json_loads(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def parse_action_repr(action_repr: str) -> dict[str, str]:
    raw = normalize_text(action_repr)
    if not raw:
        return {"role": "", "target_text": "", "op": "", "value": ""}

    match = ACTION_REPR_PATTERN.match(raw)
    if not match:
        return {"role": "", "target_text": raw, "op": "", "value": ""}

    target_text = normalize_text(match.group("target")).strip(" .")
    return {
        "role": normalize_text(match.group("role"), lower=True),
        "target_text": target_text,
        "op": normalize_op(match.group("op")),
        "value": normalize_text(match.group("value")),
    }


def normalize_model_action(action: Any) -> dict[str, str]:
    if isinstance(action, dict):
        return {
            "op": normalize_op(action.get("op")),
            "ref": normalize_text(action.get("ref")),
            "value": normalize_text(action.get("value")),
        }

    raw = normalize_text(str(action or ""))
    if not raw:
        return {"op": "", "ref": "", "value": ""}

    try:
        parsed_json = json.loads(raw)
    except Exception:
        parsed_json = None
    if isinstance(parsed_json, dict):
        return normalize_model_action(parsed_json)

    match = MODEL_ACTION_PATTERN.match(raw)
    if match:
        return {
            "op": normalize_op(match.group("op")),
            "ref": normalize_text(match.group("ref")),
            "value": normalize_text(match.group("value")),
        }

    return {"op": "", "ref": "", "value": raw}


class _DOMBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = DOMNode("document")
        self.stack: list[DOMNode] = [self.root]

    def handle_decl(self, decl: str) -> None:
        return

    def handle_comment(self, data: str) -> None:
        return

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = DOMNode(
            tag=normalize_text(tag, lower=True),
            attrs={
                normalize_text(key, lower=True): normalize_text(value)
                for key, value in attrs
                if key is not None
            },
        )
        self.stack[-1].children.append(node)
        if node.tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = normalize_text(tag, lower=True)
        for idx in range(len(self.stack) - 1, 0, -1):
            if self.stack[idx].tag == tag:
                del self.stack[idx:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].add_text(data)


def _parse_html_document(html_text: str) -> DOMNode:
    builder = _DOMBuilder()
    try:
        builder.feed(html_text)
        builder.close()
    except Exception:
        return DOMNode("html")

    for child in builder.root.children:
        if child.tag == "html":
            return child
    return builder.root.children[0] if builder.root.children else DOMNode("html")


def _iter_text(node: DOMNode):
    for text in node.text_chunks:
        yield text
    for child in node.children:
        yield from _iter_text(child)


def _element_text(elem: DOMNode) -> str:
    return normalize_text(" ".join(_iter_text(elem)))


def _infer_role(elem: DOMNode) -> str:
    attrib = elem.attrs
    role = normalize_text(attrib.get("role"), lower=True)
    if role in INTERACTIVE_ROLES:
        return role

    tag = normalize_text(elem.tag, lower=True)
    if tag == "input":
        input_type = normalize_text(attrib.get("type") or "text", lower=True)
        if input_type == "hidden":
            return ""
        return INPUT_TYPE_TO_ROLE.get(input_type, "textbox")
    if tag in TAG_TO_ROLE:
        return TAG_TO_ROLE[tag]

    if normalize_text(attrib.get("aria_role"), lower=True) in INTERACTIVE_ROLES:
        return normalize_text(attrib.get("aria_role"), lower=True)
    if normalize_text(attrib.get("is_clickable"), lower=True) == "true":
        return "button"
    if attrib.get("onclick") is not None or attrib.get("tabindex") is not None:
        return "button"
    return ""


def _primary_name(elem: DOMNode) -> str:
    attrib = elem.attrs
    candidates = [
        attrib.get("aria_label"),
        attrib.get("aria_description"),
        attrib.get("label"),
        attrib.get("title"),
        attrib.get("placeholder"),
        attrib.get("name"),
        attrib.get("text_value"),
        attrib.get("input_value"),
        attrib.get("value"),
        _element_text(elem),
    ]
    seen: set[str] = set()
    for value in candidates:
        normalized = normalize_text(value)
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        if lowered in {"true", "false", "null", "undefined"}:
            continue
        return " ".join(normalized.split()[:14])
    return ""


def _detail_bits(elem: DOMNode, *, primary_name: str) -> list[str]:
    attrib = elem.attrs
    details: list[str] = []
    for key in [
        "placeholder",
        "value",
        "input_value",
        "text_value",
        "title",
        "aria_label",
        "type",
    ]:
        value = normalize_text(attrib.get(key))
        if not value:
            continue
        if normalize_value(value) == normalize_value(primary_name):
            continue
        if key == "type" and value.lower() in {"text", "button", "submit"}:
            continue
        details.append(f'{key}="{ " ".join(value.split()[:10]) }"')

    if normalize_text(attrib.get("input_checked"), lower=True) == "true":
        details.append("[checked]")
    if normalize_text(attrib.get("option_selected"), lower=True) == "true":
        details.append("[selected]")
    return details[:3]


def _fallback_tree_repr(elem: DOMNode) -> str:
    parts: list[str] = []

    def walk(node: DOMNode, depth: int) -> None:
        if len(parts) >= 64:
            return
        text = _element_text(node)
        text = " ".join(text.split()[:12])
        attrs = []
        backend_node_id = normalize_text(node.attrs.get("backend_node_id"))
        if backend_node_id:
            attrs.append(f"id={backend_node_id}")
        role = normalize_text(node.attrs.get("role"), lower=True)
        if role:
            attrs.append(f"role={role}")
        line = f"{'  ' * depth}<{node.tag}"
        if attrs:
            line += " " + " ".join(attrs)
        line += ">"
        if text:
            line += f" {text}"
        parts.append(line)
        for child in node.children:
            walk(child, depth + 1)

    walk(elem, 0)
    return normalize_text("\n".join(parts))[:2000]


def build_openclaw_style_snapshot(
    html_text: str,
    *,
    positive_candidate_ids: set[str] | None = None,
    max_depth: int = 20,
) -> tuple[str, dict[str, dict[str, Any]]]:
    root = _parse_html_document(html_text)
    positive_candidate_ids = set(positive_candidate_ids or set())
    refs: dict[str, dict[str, Any]] = {}
    lines: list[str] = []
    name_counts: dict[str, int] = {}

    def walk(elem: DOMNode, depth: int) -> None:
        if depth > max_depth:
            return

        node_id = normalize_text(elem.attrs.get("backend_node_id"))
        role = _infer_role(elem)
        force_include = node_id in positive_candidate_ids
        if role or force_include:
            role_name = role or "element"
            name = _primary_name(elem)
            details = _detail_bits(elem, primary_name=name)
            ref = f"e{len(refs) + 1}"
            dedupe_key = f"{role_name}:{normalize_value(name)}"
            nth = name_counts.get(dedupe_key, 0)
            name_counts[dedupe_key] = nth + 1

            line = f"- {role_name}"
            if name:
                line += f' "{name}"'
            line += f" [ref={ref}]"
            if nth > 0:
                line += f" [nth={nth}]"
            if details:
                line += " " + " ".join(details)
            lines.append(line)

            refs[ref] = {
                "backend_node_id": node_id,
                "role": role_name,
                "name": name,
                "tag": normalize_text(elem.tag, lower=True),
                "details": details,
                "search_text": normalize_text(
                    " ".join([role_name, name, *details, normalize_text(elem.tag, lower=True)]),
                    lower=True,
                ),
            }

        for child in elem.children:
            walk(child, depth + 1)

    walk(root, 0)
    if lines:
        return "\n".join(lines), refs
    return f"(no interactive elements)\n{_fallback_tree_repr(root)}", refs


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    operation = action.get("operation") or {}
    return {
        "cleaned_html": str(action.get("cleaned_html") or ""),
        "raw_html": str(action.get("raw_html") or ""),
        "pos_candidates": [
            {"backend_node_id": normalize_text(candidate.get("backend_node_id"))}
            for candidate in (action.get("pos_candidates") or [])
            if normalize_text(candidate.get("backend_node_id"))
        ],
        "operation": {
            "op": normalize_text(operation.get("op")),
            "value": normalize_text(operation.get("value")),
        },
    }


def _compact_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_id": normalize_text(task.get("annotation_id")),
        "confirmed_task": normalize_text(task.get("confirmed_task")),
        "website": normalize_text(task.get("website")),
        "domain": normalize_text(task.get("domain")),
        "subdomain": normalize_text(task.get("subdomain")),
        "action_reprs": [normalize_text(value) for value in (task.get("action_reprs") or [])],
        "actions": [_compact_action(action) for action in (task.get("actions") or [])],
    }


@lru_cache(maxsize=64)
def _load_file_offsets_cached(source_file: str) -> tuple[tuple[int, int], ...]:
    path = Path(source_file).expanduser().resolve()
    _, offsets_path = _ensure_task_cache(str(path))
    payload = _load_offsets_metadata(offsets_path)
    offsets = payload.get("offsets") or []
    normalized: list[tuple[int, int]] = []
    for item in offsets:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        normalized.append((int(item[0]), int(item[1])))
    if not normalized:
        raise ValueError(f"No cached task offsets found for Mind2Web file: {path}")
    return tuple(normalized)


def _load_file_offsets(source_file: str) -> tuple[tuple[int, int], ...]:
    resolved_source_file = str(Path(source_file).expanduser().resolve())
    with _get_singleflight_lock("file_offsets", resolved_source_file):
        return _load_file_offsets_cached(resolved_source_file)


@lru_cache(maxsize=4096)
def _load_task_from_offsets_cached(source_file: str, task_index: int) -> dict[str, Any]:
    path = Path(source_file).expanduser().resolve()
    cache_jsonl_path, _ = _ensure_task_cache(str(path))
    offsets = _load_file_offsets(str(path))
    if task_index < 0 or task_index >= len(offsets):
        raise IndexError(f"task_index out of range: {task_index}")

    start, end = offsets[task_index]
    with cache_jsonl_path.open("rb") as fh:
        fh.seek(start)
        payload = fh.read(end - start)

    task = json.loads(payload.decode("utf-8"))
    if not isinstance(task, dict):
        raise ValueError(f"Mind2Web task must be a JSON object: {path}#{task_index}")
    return _compact_task(task)


def _load_task_from_offsets(source_file: str, task_index: int) -> dict[str, Any]:
    resolved_source_file = str(Path(source_file).expanduser().resolve())
    resolved_task_index = int(task_index)
    with _get_singleflight_lock("task_record", resolved_source_file, resolved_task_index):
        return _load_task_from_offsets_cached(resolved_source_file, resolved_task_index)


def load_task(source_file: str, task_index: int) -> dict[str, Any]:
    return dict(_load_task_from_offsets(str(Path(source_file).expanduser().resolve()), int(task_index)))


def resolve_task(payload: dict[str, Any], *, data_root: str | None = None) -> dict[str, Any]:
    label = payload.get("label") if isinstance(payload.get("label"), dict) else payload
    source_file = label.get("source_file") or label.get("task_source_file")
    task_index = label.get("task_index")

    if source_file is not None and task_index is not None:
        resolved_source_file = str(Path(str(source_file)).expanduser().resolve())
        resolved_task_index = int(task_index)
        task = load_task(resolved_source_file, resolved_task_index)
        task["_resolved_source_file"] = resolved_source_file
        task["_resolved_task_index"] = resolved_task_index
        return task

    annotation_id = normalize_text(label.get("annotation_id"))
    if annotation_id and data_root:
        for path in sorted(Path(data_root).expanduser().resolve().rglob("*.json")):
            offsets = _load_file_offsets(str(path))
            for idx in range(len(offsets)):
                record = _load_task_from_offsets(str(path), idx)
                if normalize_text(record.get("annotation_id")) == annotation_id:
                    task = dict(record)
                    task.setdefault("_resolved_source_file", str(path))
                    task.setdefault("_resolved_task_index", idx)
                    return task

    raise ValueError(
        "Unable to resolve Mind2Web task. Provide label.source_file and label.task_index, "
        "or pass annotation_id together with --data-root."
    )


def _choose_html_field(action: dict[str, Any]) -> str:
    return "cleaned_html" if action.get("pos_candidates") else "raw_html"


def _build_step_observation_uncached(
    task: dict[str, Any],
    step_index: int,
    *,
    max_depth: int = 20,
) -> dict[str, Any]:
    actions = list(task.get("actions") or [])
    if step_index < 0 or step_index >= len(actions):
        raise IndexError(f"step_index out of range: {step_index}")

    action = actions[step_index]
    positive_ids = {
        normalize_text(candidate.get("backend_node_id"))
        for candidate in action.get("pos_candidates") or []
        if normalize_text(candidate.get("backend_node_id"))
    }
    html_field = _choose_html_field(action)
    html_text = str(action.get(html_field) or "")
    snapshot, refs = build_openclaw_style_snapshot(
        html_text,
        positive_candidate_ids=positive_ids,
        max_depth=max_depth,
    )

    if not refs and html_field != "raw_html" and action.get("raw_html"):
        html_field = "raw_html"
        html_text = str(action.get(html_field) or "")
        snapshot, refs = build_openclaw_style_snapshot(
            html_text,
            positive_candidate_ids=positive_ids,
            max_depth=max_depth,
        )

    expected_repr = parse_action_repr((task.get("action_reprs") or [""])[step_index])
    return {
        "snapshot": snapshot,
        "refs": refs,
        "html_field": html_field,
        "step_index": step_index,
        "expected_repr": expected_repr,
        "previous_actions": list(task.get("action_reprs") or [])[:step_index],
    }


@lru_cache(maxsize=8192)
def _build_step_observation_cached(
    source_file: str,
    task_index: int,
    step_index: int,
    max_depth: int,
) -> dict[str, Any]:
    task = dict(_load_task_from_offsets(source_file, task_index))
    task["_resolved_source_file"] = source_file
    task["_resolved_task_index"] = int(task_index)
    return _build_step_observation_uncached(
        task,
        int(step_index),
        max_depth=int(max_depth),
    )


def build_step_observation(
    task: dict[str, Any],
    step_index: int,
    *,
    max_depth: int = 20,
) -> dict[str, Any]:
    resolved_source_file = normalize_text(task.get("_resolved_source_file"))
    resolved_task_index = task.get("_resolved_task_index")
    if resolved_source_file and resolved_task_index not in (None, ""):
        cache_key = (
            "step_observation",
            resolved_source_file,
            int(resolved_task_index),
            int(step_index),
            int(max_depth),
        )
        with _get_singleflight_lock(*cache_key):
            return deepcopy(
                _build_step_observation_cached(
                    resolved_source_file,
                    int(resolved_task_index),
                    int(step_index),
                    int(max_depth),
                )
            )
    return _build_step_observation_uncached(task, int(step_index), max_depth=int(max_depth))


def _role_matches(expected_role: str, actual_role: str) -> bool:
    expected_role = normalize_text(expected_role, lower=True)
    actual_role = normalize_text(actual_role, lower=True)
    if not expected_role:
        return True
    if expected_role == actual_role:
        return True
    return actual_role in ROLE_EQUIVALENTS.get(expected_role, {expected_role})


def _text_matches(expected_text: str, actual_text: str) -> bool:
    expected_text = normalize_value(expected_text)
    actual_text = normalize_value(actual_text)
    if not expected_text:
        return True
    if expected_text in actual_text or actual_text in expected_text:
        return True
    return SequenceMatcher(None, expected_text, actual_text).ratio() >= 0.72


def expected_refs_for_step(
    task: dict[str, Any],
    step_index: int,
    ref_map: dict[str, dict[str, Any]],
) -> list[str]:
    actions = list(task.get("actions") or [])
    if step_index < 0 or step_index >= len(actions):
        return []

    expected_action = actions[step_index]
    positive_ids = {
        normalize_text(candidate.get("backend_node_id"))
        for candidate in expected_action.get("pos_candidates") or []
        if normalize_text(candidate.get("backend_node_id"))
    }
    if positive_ids:
        return [
            ref
            for ref, chosen in ref_map.items()
            if normalize_text(chosen.get("backend_node_id")) in positive_ids
        ]

    expected_repr = parse_action_repr((task.get("action_reprs") or [""])[step_index])
    return [
        ref
        for ref, chosen in ref_map.items()
        if _role_matches(expected_repr.get("role", ""), chosen.get("role", ""))
        and _text_matches(expected_repr.get("target_text", ""), chosen.get("search_text", ""))
    ]


def validate_action_for_step(
    task: dict[str, Any],
    step_index: int,
    action: Any,
    ref_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = normalize_model_action(action)
    actions = list(task.get("actions") or [])
    expected_action = actions[step_index]
    expected_op = normalize_op(expected_action.get("operation", {}).get("op"))
    expected_value = normalize_text(expected_action.get("operation", {}).get("value"))
    expected_repr = parse_action_repr((task.get("action_reprs") or [""])[step_index])

    if not normalized["op"] or not normalized["ref"]:
        return {
            "ok": False,
            "reason": "invalid_action_format",
            "parsed_action": normalized,
            "match_mode": "none",
        }

    chosen = ref_map.get(normalized["ref"])
    if chosen is None:
        return {
            "ok": False,
            "reason": "unknown_ref",
            "parsed_action": normalized,
            "match_mode": "none",
        }

    op_ok = normalize_op(normalized["op"]) == expected_op
    positive_ids = {
        normalize_text(candidate.get("backend_node_id"))
        for candidate in expected_action.get("pos_candidates") or []
        if normalize_text(candidate.get("backend_node_id"))
    }

    if positive_ids:
        target_ok = normalize_text(chosen.get("backend_node_id")) in positive_ids
        match_mode = "pos_candidates"
    else:
        target_ok = _role_matches(expected_repr.get("role", ""), chosen.get("role", "")) and (
            _text_matches(expected_repr.get("target_text", ""), chosen.get("search_text", ""))
        )
        match_mode = "action_repr_fuzzy"

    if expected_op == "CLICK":
        value_ok = True
    else:
        gold_value = expected_value or normalize_text(expected_repr.get("value"))
        value_ok = normalize_value(normalized.get("value")) == normalize_value(gold_value)

    ok = op_ok and target_ok and value_ok
    return {
        "ok": ok,
        "reason": "ok" if ok else "mismatch",
        "parsed_action": normalized,
        "chosen": chosen,
        "match_mode": match_mode,
        "op_ok": op_ok,
        "target_ok": target_ok,
        "value_ok": value_ok,
        "expected_op": expected_op,
        "expected_value": expected_value or normalize_text(expected_repr.get("value")),
        "expected_repr": expected_repr,
    }
