import argparse
import gc
import os
import threading
import time
import uuid
import logging

import yaml
from flask import Flask, jsonify, request

try:
    from env.alfworld.textworld_env_cache import (
        cleanup_env_registration as _cleanup_env_registration,
        create_env_from_game_file as _shared_create_env_from_game_file,
    )
except ModuleNotFoundError:
    from textworld_env_cache import (
        cleanup_env_registration as _cleanup_env_registration,
        create_env_from_game_file as _shared_create_env_from_game_file,
    )

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.logger.disabled = True

_config: dict = {}
_alf_config_path: str = ""
_worker_id: str = "worker"

_episodes: dict = {}
_episodes_lock = threading.Lock()
_env_op_lock = threading.Lock()

MAX_ACTIVE_EPISODES = 256
EPISODE_IDLE_TIMEOUT_SECONDS = 10 * 60
EPISODE_CLEANUP_INTERVAL_SECONDS = 60

_cleanup_stop = threading.Event()
_cleanup_thread: threading.Thread | None = None


def _load_alf_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_game_file(path: str) -> str:
    path = os.path.expanduser(path)
    if path.endswith(".tw-pddl"):
        return path
    return os.path.join(path.rstrip("/"), "game.tw-pddl")


def _create_env_from_game_file(game_file: str):
    return _shared_create_env_from_game_file(_config, game_file)


def _first_scalar(x):
    while isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    return x


def _normalize_obs(obs) -> str:
    return str(_first_scalar(obs))


def _normalize_reward(scores) -> float:
    return float(_first_scalar(scores))


def _normalize_done(dones) -> bool:
    return bool(_first_scalar(dones))


def _extract_task_description(obs_text) -> str:
    obs_text = str(obs_text)
    marker = "Your task is to: "
    idx = obs_text.find(marker)
    if idx != -1:
        rest = obs_text[idx + len(marker):]
        for end_char in [".\n", "\n", "."]:
            end_idx = rest.find(end_char)
            if end_idx != -1:
                return rest[:end_idx].strip()
        return rest.strip()
    return ""


def _extract_gamefile_task_type(infos: dict) -> str:
    gamefile = ""
    for key in ("extra.gamefile", "gamefile"):
        val = infos.get(key)
        if val:
            if isinstance(val, list):
                gamefile = val[0] if val else ""
            else:
                gamefile = str(val)
            break

    gamefile_lower = gamefile.lower()
    task_types = [
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
        "pick_and_place",
    ]
    for tt in task_types:
        if tt in gamefile_lower:
            return tt
    return "unknown"


def _format_admissible(commands) -> list[str]:
    if not commands:
        return []
    if isinstance(commands, tuple):
        commands = list(commands)
    elif not isinstance(commands, list):
        commands = [commands]
    return [str(c) for c in commands if str(c) != "help"]


def _unwrap_singleton_info_values(infos: dict) -> dict:
    if not isinstance(infos, dict):
        return {}

    infos = dict(infos)
    for k in list(infos.keys()):
        v = infos[k]
        while isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        infos[k] = v
    return infos


def _safe_close_env(env):
    if env is None:
        return
    try:
        with _env_op_lock:
            env.close()
    except Exception:
        pass
    finally:
        _cleanup_env_registration(env)
        gc.collect()


def _now() -> float:
    return time.monotonic()


def _touch_episode(episode: dict):
    episode["last_used_at"] = _now()


def _pop_episode(episode_id: str):
    with _episodes_lock:
        return _episodes.pop(episode_id, None)


def _close_removed_episode(
    episode_id: str,
    episode: dict | None,
    reason: str,
    *,
    already_locked: bool = False,
) -> bool:
    if episode is None:
        return False

    if already_locked:
        _safe_close_env(episode.get("env"))
    else:
        with episode["lock"]:
            _safe_close_env(episode.get("env"))

    if reason == "idle_timeout":
        idle_for = max(0.0, _now() - episode.get("last_used_at", _now()))
        print(
            f"[worker {_worker_id}] auto-closed idle episode {episode_id} "
            f"after {idle_for:.1f}s inactivity",
            flush=True,
        )
    else:
        print(f"[worker {_worker_id}] closed episode {episode_id} ({reason})", flush=True)

    return True


def _acquire_live_episode(episode_id: str):
    with _episodes_lock:
        episode = _episodes.get(episode_id)

    if episode is None:
        return None

    episode["lock"].acquire()
    with _episodes_lock:
        if _episodes.get(episode_id) is not episode:
            episode["lock"].release()
            return None

    return episode


def _cleanup_idle_episodes_once():
    deadline = _now() - EPISODE_IDLE_TIMEOUT_SECONDS

    with _episodes_lock:
        items = list(_episodes.items())

    for episode_id, episode in items:
        if episode.get("last_used_at", 0.0) >= deadline:
            continue

        if not episode["lock"].acquire(blocking=False):
            continue

        try:
            with _episodes_lock:
                current = _episodes.get(episode_id)
                if current is not episode:
                    continue
                if episode.get("last_used_at", 0.0) >= deadline:
                    continue
                removed = _episodes.pop(episode_id, None)

            _close_removed_episode(
                episode_id,
                removed,
                reason="idle_timeout",
                already_locked=True,
            )
        finally:
            episode["lock"].release()


def _cleanup_idle_episodes_loop():
    while not _cleanup_stop.wait(EPISODE_CLEANUP_INTERVAL_SECONDS):
        try:
            _cleanup_idle_episodes_once()
        except Exception:
            app.logger.exception("idle cleanup failed")


@app.route("/health", methods=["GET"])
def health():
    with _episodes_lock:
        n = len(_episodes)
    return jsonify({
        "status": "ok",
        "worker_id": _worker_id,
        "active_episodes": n,
    })


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True) or {}
    game_file_path: str = data.get("game_file_path", "")
    env = None
    game_file = None

    try:
        if not game_file_path:
            return jsonify({
                "error": "game_file_path is required; task_seed reset path has been removed",
                "error_type": "ValueError",
                "worker_id": _worker_id,
            }), 400

        with _episodes_lock:
            active = len(_episodes)
        if active >= MAX_ACTIVE_EPISODES:
            return jsonify({
                "error": f"Too many active episodes: {active} >= {MAX_ACTIVE_EPISODES}"
            }), 503

        with _env_op_lock:
            game_file = _resolve_game_file(game_file_path)
            if not os.path.exists(game_file):
                raise FileNotFoundError(f"Game file not found: {game_file}")

            env = _create_env_from_game_file(game_file)
            obs_list, infos = env.reset()

        obs_text = _normalize_obs(obs_list)
        infos = _unwrap_singleton_info_values(infos)

        admissible = _format_admissible(infos.get("admissible_commands", []))
        task_description = _extract_task_description(obs_text)
        task_type = _extract_gamefile_task_type(infos)

        episode_id = str(uuid.uuid4())
        with _episodes_lock:
            _episodes[episode_id] = {
                "env": env,
                "step": 0,
                "won": False,
                "lock": threading.Lock(),
                "last_used_at": _now(),
                "task_type": task_type,
            }

        return jsonify({
            "episode_id": episode_id,
            "obs": obs_text,
            "admissible_commands": admissible,
            "task_description": task_description,
            "task_type": task_type,
            "step": 0,
            "worker_id": _worker_id,
        })

    except Exception as e:
        _safe_close_env(env)
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "game_file_path": game_file_path,
            "resolved_game_file": game_file,
            "worker_id": _worker_id,
        }), 500


@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(force=True) or {}
    episode_id: str = data.get("episode_id", "")
    action: str = data.get("action", "")

    episode = _acquire_live_episode(episode_id)
    if episode is None:
        return jsonify({"error": f"Unknown episode_id: {episode_id}"}), 404

    try:
        env = episode["env"]
        _touch_episode(episode)

        with _env_op_lock:
            obs_list, scores, dones, infos = env.step([action])

        obs_text = _normalize_obs(obs_list)
        reward = _normalize_reward(scores)
        done = _normalize_done(dones)

        infos = _unwrap_singleton_info_values(infos)
        won = bool(_first_scalar(infos.get("won", False)))
        admissible = _format_admissible(infos.get("admissible_commands", []))

        episode["step"] += 1
        episode["won"] = won
        _touch_episode(episode)
        current_step = episode["step"]

        if done:
            removed = _pop_episode(episode_id)
            _close_removed_episode(
                episode_id,
                removed,
                reason="done",
                already_locked=True,
            )

        return jsonify({
            "obs": obs_text,
            "admissible_commands": admissible,
            "reward": reward,
            "done": done,
            "won": won,
            "step": current_step,
            "worker_id": _worker_id,
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "episode_id": episode_id,
            "action": action,
            "worker_id": _worker_id,
        }), 500
    finally:
        episode["lock"].release()


@app.route("/close", methods=["POST"])
def close():
    data = request.get_json(force=True) or {}
    episode_id: str = data.get("episode_id", "")

    removed = _pop_episode(episode_id)
    success = _close_removed_episode(episode_id, removed, reason="api_close")

    return jsonify({"success": success, "worker_id": _worker_id})


def main():
    global _config, _alf_config_path, _worker_id, _cleanup_thread

    parser = argparse.ArgumentParser(description="ALFWorld worker HTTP server")
    parser.add_argument("--alf-config-path", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--worker-id", type=str, required=True)
    args = parser.parse_args()

    _alf_config_path = args.alf_config_path
    _config = _load_alf_config(_alf_config_path)
    _worker_id = args.worker_id
    _cleanup_stop.clear()
    _cleanup_thread = threading.Thread(
        target=_cleanup_idle_episodes_loop,
        name=f"{_worker_id}-idle-cleanup",
        daemon=True,
    )
    _cleanup_thread.start()

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        _cleanup_stop.set()
        if _cleanup_thread is not None:
            _cleanup_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
