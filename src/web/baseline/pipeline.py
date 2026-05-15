"""SLIME-compatible entry point for plain GRPO on the Mind2Web web env."""

from __future__ import annotations

from slime.utils.types import Sample

from ..generate.episode import run_episode
from ..reward.functions import normalize_step_rewards
from ..utils.common import ensure_train_placeholder_sample


WEB_GRPO_BASELINE_CONFIGS: dict[str, object] = {
    "web_env_url": "http://127.0.0.1:9010",
    "web_concurrency": 64,
    "max_episode_steps": 50,
    "solver_max_response_len": 4096,
    "partial_rollout": False,
}


def _ensure_defaults(args) -> None:
    for key, default_value in WEB_GRPO_BASELINE_CONFIGS.items():
        if not hasattr(args, key):
            setattr(args, key, default_value)


def _build_failed_sample(sample: Sample) -> Sample:
    failed = Sample(status=Sample.Status.FAILED)
    failed.prompt = sample.prompt
    failed.label = sample.label
    failed.reward = 0.0
    failed.tokens = []
    failed.loss_mask = []
    failed.response = ""
    failed.response_length = 0
    failed.metadata = {
        "won": False,
        "done": False,
        "num_steps": 0,
        "action_accuracy": 0.0,
    }
    return ensure_train_placeholder_sample(failed, reason="baseline_pipeline_exception")


async def web_grpo_pipeline(
    args,
    sample: Sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Run one plain solver episode for standard GRPO training/eval."""
    _ensure_defaults(args)

    try:
        return await run_episode(
            args,
            sample=sample,
            sampling_params=sampling_params,
            evaluation=evaluation,
        )
    except Exception as exc:
        print(f"[web/baseline] run_episode error: {exc}")
        failed = _build_failed_sample(sample)
        return failed if evaluation else [failed]


def web_grpo_reward_post_process(
    args,
    samples: list[Sample],
) -> tuple[list[float], list[float]]:
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    normalize_step_rewards(samples)
    rewards = [sample.get_reward_value(args) for sample in samples]
    return raw_rewards, rewards
