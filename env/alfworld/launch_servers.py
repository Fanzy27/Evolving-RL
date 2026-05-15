import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from urllib.request import urlopen, Request
from urllib.error import URLError


def _terminate_process(proc: subprocess.Popen, name: str, timeout: float = 5.0):
    if proc.poll() is not None:
        return

    print(f"[launcher] stopping {name} (pid={proc.pid}) ...")
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[launcher] force killing {name} (pid={proc.pid}) ...")
        proc.kill()
        proc.wait(timeout=timeout)
    except Exception as e:
        print(f"[launcher] failed stopping {name}: {e}")


def _wait_for_health(
    url: str,
    timeout: float = 60.0,
    interval: float = 0.5,
    proc: subprocess.Popen | None = None,
    name: str | None = None,
):
    deadline = time.time() + timeout
    last_err = None

    while time.time() < deadline:
        if proc is not None:
            code = proc.poll()
            if code is not None:
                proc_name = name or "process"
                raise RuntimeError(
                    f"{proc_name} exited before health check passed for {url} "
                    f"(exit_code={code})"
                )
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return True
        except Exception as e:
            last_err = e
        time.sleep(interval)

    raise RuntimeError(f"health check failed for {url}, last_err={last_err}")


def _write_pid_file(pid_file: str, payload: dict):
    os.makedirs(os.path.dirname(pid_file) or ".", exist_ok=True)
    with open(pid_file, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Launch ALFWorld router + multiple workers")
    parser.add_argument(
        "--alf-config-path",
        type=str,
        required=True,
        help="Path to ALFWorld YAML config",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes to launch",
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=9000,
        help="External router port",
    )
    parser.add_argument(
        "--worker-host",
        type=str,
        default="127.0.0.1",
        help="Host for internal worker servers",
    )
    parser.add_argument(
        "--worker-start-port",
        type=int,
        default=9101,
        help="Starting port for worker servers",
    )
    parser.add_argument(
        "--router-host",
        type=str,
        default="0.0.0.0",
        help="Host for router server",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use",
    )
    parser.add_argument(
        "--worker-script",
        type=str,
        default="worker_server.py",
        help="Path to worker server script",
    )
    parser.add_argument(
        "--router-script",
        type=str,
        default="router_server.py",
        help="Path to router server script",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Optional directory to write worker/router logs",
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default="alfworld_server_pids.json",
        help="Path to pid file for stop script",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for each service health check",
    )
    args = parser.parse_args()

    processes = []
    opened_files = []

    def cleanup():
        for name, proc in reversed(processes):
            _terminate_process(proc, name)
        for f in opened_files:
            try:
                f.close()
            except Exception:
                pass

    atexit.register(cleanup)

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    worker_urls = []
    pid_payload = {
        "router": None,
        "workers": [],
        "meta": {
            "router_host": args.router_host,
            "router_port": args.router_port,
            "worker_host": args.worker_host,
            "worker_start_port": args.worker_start_port,
            "num_workers": args.num_workers,
        }
    }

    print(f"[launcher] starting {args.num_workers} workers ...")
    for i in range(args.num_workers):
        worker_id = f"w{i}"
        port = args.worker_start_port + i
        worker_url = f"http://{args.worker_host}:{port}"
        worker_urls.append(worker_url)

        cmd = [
            args.python,
            args.worker_script,
            "--alf-config-path", args.alf_config_path,
            "--host", args.worker_host,
            "--port", str(port),
            "--worker-id", worker_id,
        ]

        stdout = None
        stderr = None
        if args.log_dir:
            log_path = os.path.join(args.log_dir, f"{worker_id}.log")
            f = open(log_path, "w")
            opened_files.append(f)
            stdout = f
            stderr = subprocess.STDOUT
            print(f"[launcher] {worker_id} log -> {log_path}")

        print(f"[launcher] starting worker {worker_id} on {worker_url}")
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        processes.append((f"worker-{worker_id}", proc))
        pid_payload["workers"].append({
            "worker_id": worker_id,
            "pid": proc.pid,
            "host": args.worker_host,
            "port": port,
            "url": worker_url,
        })

    # wait workers ready
    for w in pid_payload["workers"]:
        health_url = f"{w['url']}/health"
        print(f"[launcher] waiting for worker {w['worker_id']} health: {health_url}")
        worker_proc = next(
            proc for name, proc in processes if name == f"worker-{w['worker_id']}"
        )
        _wait_for_health(
            health_url,
            timeout=args.startup_timeout,
            proc=worker_proc,
            name=f"worker-{w['worker_id']}",
        )

    workers_arg = ",".join(worker_urls)
    router_cmd = [
        args.python,
        args.router_script,
        "--host", args.router_host,
        "--port", str(args.router_port),
        "--workers", workers_arg,
    ]

    router_stdout = None
    router_stderr = None
    if args.log_dir:
        log_path = os.path.join(args.log_dir, "router.log")
        f = open(log_path, "w")
        opened_files.append(f)
        router_stdout = f
        router_stderr = subprocess.STDOUT
        print(f"[launcher] router log -> {log_path}")

    print(f"[launcher] starting router on {args.router_host}:{args.router_port}")
    print(f"[launcher] router workers = {workers_arg}")
    router_proc = subprocess.Popen(router_cmd, stdout=router_stdout, stderr=router_stderr)
    processes.append(("router", router_proc))

    pid_payload["router"] = {
        "pid": router_proc.pid,
        "host": args.router_host,
        "port": args.router_port,
    }
    _write_pid_file(args.pid_file, pid_payload)

    router_health_url = f"http://127.0.0.1:{args.router_port}/health"
    print(f"[launcher] waiting for router health: {router_health_url}")
    _wait_for_health(
        router_health_url,
        timeout=args.startup_timeout,
        proc=router_proc,
        name="router",
    )

    print()
    print("[launcher] all services started")
    print(f"[launcher] public endpoint: http://127.0.0.1:{args.router_port}")
    print(f"[launcher] pid file: {args.pid_file}")
    if args.log_dir:
        print(f"[launcher] logs dir: {args.log_dir}")
    print("[launcher] press Ctrl+C to stop all")

    try:
        while True:
            time.sleep(1)
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(f"{name} exited unexpectedly with code {code}")
    except KeyboardInterrupt:
        print("\n[launcher] received Ctrl+C, shutting down ...")
    except Exception as e:
        print(f"\n[launcher] error: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
