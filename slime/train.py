import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action


ALFWORLD_SCRIPT_MATCHES = (
    "env/alfworld/launch_servers.py",
    "env/alfworld/router_server.py",
    "env/alfworld/worker_server.py",
)


def add_project_custom_arguments(parser):
    """Training hyperparams via CLI; environment config via --custom-config-path."""
    parser.add_argument("--n-experiences", type=int, default=None)
    parser.add_argument("--num-repeat", type=int, default=1)
    parser.add_argument("--retrieval-topk", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--solver-max-response-len", type=int, default=512)
    parser.add_argument("--solver-temperature", type=float, default=None)
    parser.add_argument("--solver-top-p", type=float, default=0.9)
    parser.add_argument("--extractor-reward-weight", type=float, default=None)
    parser.add_argument("--solver-reward-weight", type=float, default=None)
    parser.add_argument("--skill-format-penalty", type=float, default=None)
    parser.add_argument("--solver-entropy-coef", type=float, default=None)
    parser.add_argument("--extractor-entropy-coef", type=float, default=None)
    parser.add_argument("--solver-kl-loss-coef", type=float, default=None)
    parser.add_argument("--extractor-kl-loss-coef", type=float, default=None)
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_alfworld_training(args) -> bool:
    for attr in (
        "custom_generate_function_path",
        "custom_reward_post_process_path",
        "custom_rollout_log_function_path",
        "custom_eval_rollout_log_function_path",
    ):
        value = getattr(args, attr, None)
        if isinstance(value, str) and "src.alfworld" in value:
            return True
    return False


def _use_alfworld_new_servers(args) -> bool:
    for attr in (
        "custom_generate_function_path",
        "custom_reward_post_process_path",
        "custom_rollout_log_function_path",
        "custom_eval_rollout_log_function_path",
    ):
        value = getattr(args, attr, None)
        if isinstance(value, str) and "src.alfworld_new" in value:
            return True
    return False


def _list_alfworld_server_processes() -> dict[int, str]:
    ps_output = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    current_pid = os.getpid()
    matches: dict[int, str] = {}
    for line in ps_output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, cmdline = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if any(match in cmdline for match in ALFWORLD_SCRIPT_MATCHES):
            matches[pid] = cmdline
    return matches


def _terminate_pid(pid: int, name: str) -> None:
    try:
        os.kill(pid, 0)
    except OSError:
        return

    print(f"[train] killing {name} pid={pid}", flush=True)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _kill_alfworld_processes_now() -> None:
    matches = _list_alfworld_server_processes()
    if not matches:
        print("[train] no matched ALFWorld server processes found to kill", flush=True)
        return
    for pid, cmdline in sorted(matches.items()):
        _terminate_pid(pid, cmdline)


def _wait_for_alfworld_processes_to_exit(args) -> None:
    timeout = float(getattr(args, "alfworld_shutdown_timeout", 3))
    deadline = time.time() + timeout
    while time.time() < deadline:
        matches = _list_alfworld_server_processes()
        if not matches:
            return
        time.sleep(0.2)

    matches = _list_alfworld_server_processes()
    if not matches:
        return

    for pid, cmdline in matches.items():
        print(f"[train] killing leftover pid={pid} cmd={cmdline}", flush=True)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    time.sleep(0.5)
    matches = _list_alfworld_server_processes()
    if matches:
        raise RuntimeError(
            "ALFWorld servers are still running after shutdown: "
            + ", ".join(f"{pid}={cmdline}" for pid, cmdline in matches.items())
        )


def _wait_for_alfworld_router(args) -> None:
    timeout = float(getattr(args, "alfworld_startup_timeout", 180))
    router_port = int(getattr(args, "alfworld_router_port", 9000))
    deadline = time.time() + timeout
    last_err = None
    health_url = f"http://127.0.0.1:{router_port}/health"

    while time.time() < deadline:
        try:
            with urlopen(Request(health_url, method="GET"), timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(1.0)

    raise RuntimeError(
        f"Timed out waiting for ALFWorld router health at {health_url}, last_err={last_err}"
    )


def _stop_alfworld_servers(args) -> None:
    print("[train] stopping ALFWorld servers before restart", flush=True)
    _kill_alfworld_processes_now()
    _wait_for_alfworld_processes_to_exit(args)


def _start_alfworld_servers(args) -> None:
    repo_root = _repo_root()
    log_dir_name = str(getattr(args, "alfworld_log_dir", "log/env_log"))
    num_workers = str(getattr(args, "alfworld_num_workers", 256))
    log_dir = repo_root / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_log_path = repo_root / "tmp.log"

    if _use_alfworld_new_servers(args):
        router_port = str(getattr(args, "alfworld_router_port", 9000))
        worker_host = str(getattr(args, "alfworld_worker_host", "127.0.0.1"))
        worker_start_port = str(getattr(args, "alfworld_worker_start_port", 9101))
        router_host = str(getattr(args, "alfworld_router_host", "0.0.0.0"))
        pid_file = str(getattr(args, "alfworld_pid_file", "alfworld_server_pids.json"))
        startup_timeout = str(getattr(args, "alfworld_startup_timeout", 180))
        cmd = [
            sys.executable,
            "env/alfworld/launch_servers.py",
            "--alf-config-path",
            "configs/alfworld.yaml",
            "--num-workers",
            num_workers,
            "--router-port",
            router_port,
            "--worker-host",
            worker_host,
            "--worker-start-port",
            worker_start_port,
            "--router-host",
            router_host,
            "--python",
            sys.executable,
            "--log-dir",
            log_dir_name,
            "--pid-file",
            pid_file,
            "--startup-timeout",
            startup_timeout,
            "--worker-script",
            "env/alfworld/worker_server.py",
            "--router-script",
            "env/alfworld/router_server.py",
        ]
    else:
        cmd = [
            sys.executable,
            "env/alfworld/launch_servers.py",
            "--alf-config-path",
            "configs/alfworld.yaml",
            "--num-workers",
            num_workers,
            "--log-dir",
            log_dir_name,
            "--worker-script",
            "env/alfworld/worker_server.py",
            "--router-script",
            "env/alfworld/router_server.py",
        ]

    print("[train] starting ALFWorld servers after restart", flush=True)
    with open(tmp_log_path, "ab") as tmp_log:
        subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=tmp_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    _wait_for_alfworld_router(args)


def _restart_alfworld_servers(args) -> None:
    _stop_alfworld_servers(args)
    _start_alfworld_servers(args)
    print("[train] ALFWorld servers restarted successfully", flush=True)


def train(args):
    configure_logger()
    enable_alfworld_restart = _is_alfworld_training(args)
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    alfworld_restart_interval = int(getattr(args, "alfworld_restart_interval", 25))
    if enable_alfworld_restart:
        print(
            f"[train] ALFWorld server auto-restart enabled: every {alfworld_restart_interval} rollouts",
            flush=True,
        )

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

        if (
            enable_alfworld_restart
            and rollout_id < args.num_rollout - 1
            and (rollout_id + 1) % alfworld_restart_interval == 0
        ):
            _restart_alfworld_servers(args)

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_project_custom_arguments)
    train(args)
