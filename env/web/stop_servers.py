"""Stop the Mind2Web web environment server."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess


SCRIPT_MATCHES = (
    "env/web/launch_servers.py",
    "env/web/env_server.py",
)


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int, name: str) -> None:
    if not _is_alive(pid):
        return
    print(f"[web/stop] killing {name} pid={pid} ...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _list_web_processes() -> dict[int, str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    pids: dict[int, str] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, cmdline = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if any(match in cmdline for match in SCRIPT_MATCHES):
            pids[pid] = cmdline
    return pids


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop the Mind2Web web env server.")
    parser.add_argument("--pid-file", default="web_env_server_pids.json")
    _ = parser.parse_args()

    matches = _list_web_processes()
    if not matches:
        print("[web/stop] no web env process found")
        return

    for pid, cmdline in sorted(matches.items()):
        _terminate_pid(pid, cmdline)


if __name__ == "__main__":
    main()
