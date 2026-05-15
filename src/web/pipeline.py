"""Entry points for Mind2Web web training."""

from __future__ import annotations


def _solver_raw_reward(sample) -> float | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    group_index = sample.group_index
    if group_index is None or int(group_index) < 1:
        return None

    for key in ("raw_env_reward", "raw_step_reward"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue

    reward = sample.reward
    if isinstance(reward, (int, float)):
        return float(reward)
    return None


async def web_pipeline(
    args,
    sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> list:
    if evaluation:
        from src.web.generate.eval import run_eval

        return await run_eval(args, sample, sampling_params)

    from src.web.generate.rollout import run_rollout

    return await run_rollout(args, sample, sampling_params, evaluation)


def web_reward_post_process(
    args,
    samples: list,
) -> tuple[list[float], list[float]]:
    rewards = [sample.get_reward_value(args) for sample in samples]
    per_sample_solver_raw_rewards = [_solver_raw_reward(sample) for sample in samples]
    solver_raw_rewards = [reward for reward in per_sample_solver_raw_rewards if reward is not None]
    rollout_raw_reward = (
        float(sum(solver_raw_rewards) / len(solver_raw_rewards))
        if solver_raw_rewards
        else (float(sum(rewards) / len(rewards)) if rewards else 0.0)
    )
    raw_rewards = [
        (reward if reward is not None else rollout_raw_reward)
        for reward in per_sample_solver_raw_rewards
    ]
    return raw_rewards, rewards


web_stage2_pipeline = web_pipeline
web_stage2_reward_post_process = web_reward_post_process
