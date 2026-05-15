"""Mind2Web replay environment HTTP server.

This server intentionally uses only the Python standard library so the web
experiment does not depend on Flask/FastAPI being installed.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import traceback
from urllib.parse import urlparse
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.web.mind2web_env import (
    build_step_observation,
    resolve_task,
    validate_action_for_step,
)


_episodes: dict[str, dict] = {}
_episodes_lock = threading.Lock()
_data_root: str | None = None
_max_depth = 20
_request_gate: threading.BoundedSemaphore | None = None


class _ConcurrentHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 4096


def _now() -> float:
    return time.monotonic()


def _action_schema_text() -> str:
    return (
        'Use JSON inside <action>...</action> with one of:\n'
        '- {"op":"CLICK","ref":"e12"}\n'
        '- {"op":"TYPE","ref":"e7","value":"text to type"}\n'
        '- {"op":"SELECT","ref":"e9","value":"option text"}'
    )


def _observation_payload(task: dict, obs: dict) -> dict:
    actions = list(task.get("actions") or [])
    return {
        "obs": obs["snapshot"],
        "previous_actions": obs["previous_actions"],
        "action_schema": _action_schema_text(),
        "html_field": obs["html_field"],
        "ref_count": len(obs["refs"]),
        "task_description": str(task.get("confirmed_task") or ""),
        "website": str(task.get("website") or ""),
        "domain": str(task.get("domain") or ""),
        "subdomain": str(task.get("subdomain") or ""),
        "total_steps": len(actions),
    }


def _handle_health() -> tuple[int, dict]:
    with _episodes_lock:
        active = len(_episodes)
    return 200, {"status": "ok", "active_episodes": active, "max_depth": _max_depth}


def _handle_reset(payload: dict) -> tuple[int, dict]:
    try:
        task = resolve_task(payload, data_root=_data_root)
        obs = build_step_observation(task, 0, max_depth=_max_depth)
    except Exception as exc:
        return 400, {"ok": False, "error": str(exc)}

    episode_id = str(uuid.uuid4())
    with _episodes_lock:
        _episodes[episode_id] = {
            "task": task,
            "step_index": 0,
            "ref_map": obs["refs"],
            "last_used_at": _now(),
        }

    return 200, {
        "ok": True,
        "episode_id": episode_id,
        "step": 0,
        **_observation_payload(task, obs),
    }


def _handle_step(payload: dict) -> tuple[int, dict]:
    episode_id = str(payload.get("episode_id") or "").strip()
    action = payload.get("action")

    with _episodes_lock:
        episode = _episodes.get(episode_id)

    if episode is None:
        return 404, {"ok": False, "error": f"Unknown episode_id: {episode_id}"}

    episode["last_used_at"] = _now()
    task = episode["task"]
    step_index = int(episode["step_index"])
    validation = validate_action_for_step(task, step_index, action, episode["ref_map"])

    if not validation["ok"]:
        with _episodes_lock:
            _episodes.pop(episode_id, None)
        return 200, {
            "ok": True,
            "obs": "The action did not match the required page interaction. Episode terminated.",
            "previous_actions": list(task.get("action_reprs") or [])[:step_index],
            "action_schema": _action_schema_text(),
            "html_field": "",
            "ref_count": 0,
            "reward": 0.0,
            "done": True,
            "won": False,
            "step": step_index + 1,
            "action_correct": False,
            "match_mode": validation.get("match_mode", "none"),
        }

    next_step = step_index + 1
    actions = list(task.get("actions") or [])
    if next_step >= len(actions):
        with _episodes_lock:
            _episodes.pop(episode_id, None)
        return 200, {
            "ok": True,
            "obs": "Task completed successfully.",
            "previous_actions": list(task.get("action_reprs") or []),
            "action_schema": _action_schema_text(),
            "html_field": "",
            "ref_count": 0,
            "reward": 1.0,
            "done": True,
            "won": True,
            "step": next_step,
            "action_correct": True,
            "match_mode": validation.get("match_mode", "pos_candidates"),
        }

    next_obs = build_step_observation(task, next_step, max_depth=_max_depth)
    with _episodes_lock:
        if episode_id in _episodes:
            _episodes[episode_id]["step_index"] = next_step
            _episodes[episode_id]["ref_map"] = next_obs["refs"]
            _episodes[episode_id]["last_used_at"] = _now()

    return 200, {
        "ok": True,
        "reward": 1.0,
        "done": False,
        "won": False,
        "step": next_step,
        "action_correct": True,
        "match_mode": validation.get("match_mode", "pos_candidates"),
        **_observation_payload(task, next_obs),
    }


def _handle_close(payload: dict) -> tuple[int, dict]:
    episode_id = str(payload.get("episode_id") or "").strip()
    with _episodes_lock:
        removed = _episodes.pop(episode_id, None)
    return 200, {"ok": True, "success": removed is not None}


def dispatch_request(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    payload = payload or {}
    if method == "GET" and path == "/health":
        return _handle_health()
    if method == "POST" and path == "/reset":
        return _handle_reset(payload)
    if method == "POST" and path == "/step":
        return _handle_step(payload)
    if method == "POST" and path == "/close":
        return _handle_close(payload)
    return 404, {"ok": False, "error": f"Unknown route: {method} {path}"}


def _dispatch_with_guard(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    if method == "GET" and path == "/health":
        return dispatch_request(method, path, payload)

    gate = _request_gate
    if gate is None:
        return dispatch_request(method, path, payload)

    with gate:
        return dispatch_request(method, path, payload)


class _RequestHandler(BaseHTTPRequestHandler):
    def _write_json(self, status_code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        try:
            status, payload = _dispatch_with_guard("GET", urlparse(self.path).path)
        except Exception as exc:
            traceback.print_exc()
            status, payload = 500, {"ok": False, "error": f"internal_server_error: {exc}"}
        self._write_json(status, payload)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            payload = {}
        try:
            status, body = _dispatch_with_guard("POST", urlparse(self.path).path, payload)
        except Exception as exc:
            traceback.print_exc()
            status, body = 500, {"ok": False, "error": f"internal_server_error: {exc}"}
        self._write_json(status, body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    global _data_root, _max_depth, _request_gate

    parser = argparse.ArgumentParser(description="Run the Mind2Web replay env server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Optional fallback root for annotation_id-based task resolution.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help=(
            "Maximum DOM depth kept in the compact interactive snapshot. "
            "This is intentionally wider than OpenClaw's aria-tree depth because "
            "Mind2Web provides raw HTML rather than an accessibility tree."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="Maximum number of concurrent HTTP requests processed by the env server.",
    )
    args = parser.parse_args()

    _data_root = args.data_root or None
    _max_depth = max(1, int(args.max_depth))
    _request_gate = threading.BoundedSemaphore(max(1, int(args.max_workers)))

    server = _ConcurrentHTTPServer((args.host, args.port), _RequestHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
