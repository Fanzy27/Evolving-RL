import argparse
import signal
import subprocess
import os

ALFWORLD_SCRIPT_MATCHES = (
    "env/alfworld/launch_servers.py",
    "env/alfworld/router_server.py",
    "env/alfworld/worker_server.py",
)


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int, name: str):
    if not _is_alive(pid):
        return

    print(f"[stop] killing {name} pid={pid} ...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _list_alfworld_processes() -> dict[int, str]:
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
        if any(match in cmdline for match in ALFWORLD_SCRIPT_MATCHES):
            pids[pid] = cmdline
    return pids


def main():
    parser = argparse.ArgumentParser(description="Stop ALFWorld router + workers")
    parser.add_argument(
        "--pid-file",
        type=str,
        default="alfworld_server_pids.json",
        help="Unused compatibility argument; shutdown is now port-based",
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=9000,
        help="Fallback: kill router by port",
    )
    parser.add_argument(
        "--worker-start-port",
        type=int,
        default=9100,
        help="Fallback: starting worker port",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=256,
        help="Fallback: number of workers",
    )
    args = parser.parse_args()

    matches = _list_alfworld_processes()
    if not matches:
        print("[stop] no ALFWorld process found")
        return

    for pid, cmdline in sorted(matches.items()):
        _terminate_pid(pid, cmdline)


if __name__ == "__main__":
    main()
