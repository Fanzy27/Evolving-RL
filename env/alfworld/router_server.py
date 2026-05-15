import argparse
import asyncio
import logging
import time
import traceback

from aiohttp import web, ClientSession, ClientTimeout, TCPConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_workers = []
_episode_to_worker = {}
_episode_last_used_at = {}
_rr = 0
_session: ClientSession = None

REQUEST_TIMEOUT = ClientTimeout(total=3600)
HEALTH_TIMEOUT = ClientTimeout(total=10)
EPISODE_IDLE_TIMEOUT_SECONDS = 10 * 60
EPISODE_CLEANUP_INTERVAL_SECONDS = 60


def _now() -> float:
    return time.monotonic()


def _register_episode(episode_id: str, worker: dict):
    _episode_to_worker[episode_id] = worker
    _episode_last_used_at[episode_id] = _now()


def _touch_episode(episode_id: str):
    if episode_id in _episode_to_worker:
        _episode_last_used_at[episode_id] = _now()


def _pop_episode(episode_id: str):
    worker = _episode_to_worker.pop(episode_id, None)
    _episode_last_used_at.pop(episode_id, None)
    return worker


def _collect_stale_episodes():
    deadline = _now() - EPISODE_IDLE_TIMEOUT_SECONDS
    stale = []

    for episode_id, worker in list(_episode_to_worker.items()):
        last_used_at = _episode_last_used_at.get(episode_id, 0.0)
        if last_used_at >= deadline:
            continue
        stale.append((episode_id, worker, last_used_at))
        _episode_to_worker.pop(episode_id, None)
        _episode_last_used_at.pop(episode_id, None)

    return stale


async def _cleanup_idle_episodes(app):
    while True:
        await asyncio.sleep(EPISODE_CLEANUP_INTERVAL_SECONDS)
        stale = _collect_stale_episodes()

        for episode_id, worker, last_used_at in stale:
            try:
                async with _session.post(
                    f"{worker['url']}/close",
                    json={"episode_id": episode_id},
                ) as r:
                    await r.read()
            except Exception:
                logging.exception(
                    "[router] failed to auto-close idle episode %s on %s",
                    episode_id,
                    worker.get("url"),
                )

            idle_for = max(0.0, _now() - last_used_at)
            logging.info(
                "[router] auto-closed idle episode %s on %s after %.1fs inactivity",
                episode_id,
                worker.get("id", worker.get("url", "worker")),
                idle_for,
            )


def _choose_worker():
    global _rr
    idx = _rr % len(_workers)
    _rr += 1
    return _workers[idx]


async def health(request):
    worker_stats = []
    for w in _workers:
        try:
            async with _session.get(f"{w['url']}/health", timeout=HEALTH_TIMEOUT) as r:
                data = await r.json()
                worker_stats.append({"worker": w, "ok": True, "data": data})
        except Exception as e:
            worker_stats.append({"worker": w, "ok": False, "error": str(e)})

    return web.json_response({
        "status": "ok",
        "active_episodes": len(_episode_to_worker),
        "workers": worker_stats,
    })


async def reset(request):
    data = await request.json()
    worker = _choose_worker()

    try:
        async with _session.post(f"{worker['url']}/reset", json=data) as r:
            resp = await r.json()
            if r.status == 200 and "episode_id" in resp:
                _register_episode(resp["episode_id"], worker)
            return web.json_response(resp, status=r.status)
    except Exception as e:
        traceback.print_exc()
        return web.json_response({
            "error": str(e),
            "error_type": type(e).__name__,
            "worker": worker,
        }, status=500)


async def step(request):
    data = await request.json()
    episode_id = data.get("episode_id", "")
    worker = _episode_to_worker.get(episode_id)

    if worker is None:
        return web.json_response({"error": f"Unknown episode_id: {episode_id}"}, status=404)

    try:
        _touch_episode(episode_id)
        async with _session.post(f"{worker['url']}/step", json=data) as r:
            resp = await r.json()
            if r.status == 200:
                if resp.get("done"):
                    _pop_episode(episode_id)
                else:
                    _touch_episode(episode_id)
            elif r.status == 404:
                _pop_episode(episode_id)
            return web.json_response(resp, status=r.status)
    except Exception as e:
        traceback.print_exc()
        return web.json_response({
            "error": str(e),
            "error_type": type(e).__name__,
            "episode_id": episode_id,
            "worker": worker,
        }, status=500)


async def close(request):
    data = await request.json()
    episode_id = data.get("episode_id", "")
    worker = _pop_episode(episode_id)

    if worker is None:
        return web.json_response({"success": False}, status=404)

    try:
        async with _session.post(f"{worker['url']}/close", json=data) as r:
            resp = await r.json()
            return web.json_response(resp, status=r.status)
    except Exception as e:
        traceback.print_exc()
        return web.json_response({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "worker": worker,
        }, status=500)


async def on_startup(app):
    global _session
    # limit=0: 不限制连接数，适合高并发转发场景
    connector = TCPConnector(limit=0)
    _session = ClientSession(connector=connector, timeout=REQUEST_TIMEOUT)
    app["idle_cleanup_task"] = asyncio.create_task(_cleanup_idle_episodes(app))


async def on_cleanup(app):
    cleanup_task = app.get("idle_cleanup_task")
    if cleanup_task is not None:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)
    await _session.close()


def main():
    global _workers

    parser = argparse.ArgumentParser(description="ALFWorld router HTTP server (async)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--workers",
        type=str,
        required=True,
        help='Comma-separated worker URLs, e.g. "http://127.0.0.1:9101,http://127.0.0.1:9102"',
    )
    args = parser.parse_args()

    urls = [x.strip() for x in args.workers.split(",") if x.strip()]
    _workers = [{"id": f"w{i}", "url": u} for i, u in enumerate(urls)]

    print(f"[router] serving on {args.host}:{args.port}")
    print(f"[router] workers = {_workers}")

    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/health", health)
    app.router.add_post("/reset", reset)
    app.router.add_post("/step", step)
    app.router.add_post("/close", close)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
