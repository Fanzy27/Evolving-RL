"""Launch the Mind2Web web environment server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _wait_for_health(url: str, timeout: float = 60.0, interval: float = 0.5, proc=None):
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"env server exited before health check passed: code={proc.returncode}")
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(interval)
    raise RuntimeError(f"health check failed for {url}, last_err={last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Mind2Web web env server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--env-script",
        default="env/web/env_server.py",
        help="Path to the environment server script.",
    )
    parser.add_argument("--data-root", default="")
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--pid-file", default="web_env_server_pids.json")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    args = parser.parse_args()

    stdout = None
    stderr = None
    opened_file = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, "env_server.log")
        opened_file = open(log_path, "w")
        stdout = opened_file
        stderr = subprocess.STDOUT

    cmd = [
        args.python,
        str((PROJECT_ROOT / args.env_script).resolve()) if not os.path.isabs(args.env_script) else args.env_script,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-depth",
        str(args.max_depth),
        "--max-workers",
        str(args.max_workers),
    ]
    if args.data_root:
        cmd.extend(["--data-root", args.data_root])

    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, cwd=str(PROJECT_ROOT))
    health_url = f"http://127.0.0.1:{args.port}/health"
    _wait_for_health(health_url, timeout=args.startup_timeout, proc=proc)

    with open(args.pid_file, "w", encoding="utf-8") as fh:
        json.dump({"env_server": {"pid": proc.pid, "host": args.host, "port": args.port}}, fh, indent=2)

    if opened_file is not None:
        opened_file.close()

    print(f"[web/launcher] env server ready at {health_url} (pid={proc.pid})")


if __name__ == "__main__":
    main()
