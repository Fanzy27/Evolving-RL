import hashlib
import os


def _config_data_path(config: dict, train_eval: str) -> str:
    if train_eval == "train":
        return os.path.expandvars(config["dataset"]["data_path"])
    if train_eval == "eval_in_distribution":
        return os.path.expandvars(config["dataset"]["eval_id_data_path"])
    if train_eval == "eval_out_of_distribution":
        return os.path.expandvars(config["dataset"]["eval_ood_data_path"])
    raise ValueError(f"Unsupported train_eval: {train_eval}")


def _config_signature(config: dict, train_eval: str) -> str:
    training_method = config["general"]["training_method"]
    if training_method == "dqn":
        max_steps = config["rl"]["training"]["max_nb_steps_per_episode"]
    elif training_method == "dagger":
        max_steps = config["dagger"]["training"]["max_nb_steps_per_episode"]
    else:
        raise NotImplementedError(f"Unsupported training method: {training_method}")

    signature = (
        train_eval,
        _config_data_path(config, train_eval),
        tuple(config.get("env", {}).get("task_types", [])),
        bool(config.get("env", {}).get("domain_randomization", False) and train_eval == "train"),
        config.get("env", {}).get("expert_type", ""),
        training_method,
        max_steps,
    )
    return repr(signature)


def _stable_name(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _env_id_for_name(name: str) -> str:
    return f"tw-{name}-v0"


def _mark_env(env, *, env_id: str, unregister_on_close: bool):
    try:
        env._alfworld_registry_env_id = env_id
        env._alfworld_unregister_on_close = unregister_on_close
    except Exception:
        pass
    return env


def cleanup_env_registration(env):
    env_id = getattr(env, "_alfworld_registry_env_id", None)
    if not env_id or not getattr(env, "_alfworld_unregister_on_close", False):
        return

    try:
        import textworld.gym

        textworld.gym.registry.pop(env_id, None)
    except Exception:
        pass


def create_env_from_game_file(config: dict, game_file: str):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    max_steps = (
        config.get("dagger", {}).get("training", {}).get("max_nb_steps_per_episode", 50)
        or config.get("rl", {}).get("training", {}).get("max_nb_steps_per_episode", 50)
        or 50
    )
    domain_randomization = config.get("env", {}).get("domain_randomization", False)

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        extras=["gamefile"],
    )
    wrappers = [AlfredDemangler(shuffle=domain_randomization), AlfredInfos]

    config_sig = _config_signature(config, train_eval="train")
    game_key = f"{config_sig}|{os.path.abspath(game_file)}"
    name = _stable_name("alfworld-game", game_key)
    env_id = _env_id_for_name(name)

    if env_id not in textworld.gym.registry:
        textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=max_steps,
            wrappers=wrappers,
            name=name,
        )

    env = textworld.gym.make(env_id)
    return _mark_env(env, env_id=env_id, unregister_on_close=True)
