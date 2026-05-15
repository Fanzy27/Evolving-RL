"""Entry points for ALFWorld training."""

from __future__ import annotations


async def alfworld_pipeline(
    args,
    sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> list:
    if evaluation:
        from src.alfworld.generate.eval import run_eval

        return await run_eval(args, sample, sampling_params)

    from src.alfworld.generate.rollout import run_rollout

    return await run_rollout(args, sample, sampling_params, evaluation)


def alfworld_reward_post_process(
    args,
    samples: list,
) -> tuple[list[float], list[float]]:
    rewards = [sample.get_reward_value(args) for sample in samples]
    return rewards, rewards


alfworld_stage2_pipeline = alfworld_pipeline
alfworld_stage2_reward_post_process = alfworld_reward_post_process
