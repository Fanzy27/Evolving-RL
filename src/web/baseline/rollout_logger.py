"""Custom rollout logger for the Mind2Web plain-GRPO baseline."""

from __future__ import annotations

import logging

import numpy as np
from argparse import Namespace

from slime.utils import logging_utils
from slime.utils.metric_utils import (
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _meta_float(sample: Sample, key: str, default: float | None = None) -> float | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _reward(sample: Sample) -> float:
    value = _meta_float(sample, "raw_training_reward")
    if value is not None:
        return value
    value = _meta_float(sample, "raw_step_reward")
    if value is not None:
        return value
    value = _meta_float(sample, "raw_won_reward")
    if value is not None:
        return value
    try:
        return float(sample.reward)
    except Exception:
        return 0.0


def _solver_raw_reward(sample: Sample) -> float:
    value = _meta_float(sample, "raw_env_reward")
    if value is not None:
        return value
    value = _meta_float(sample, "raw_step_reward")
    if value is not None:
        return value
    try:
        return float(sample.reward)
    except Exception:
        return 0.0


def _won(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if "episode_won" in metadata:
        return bool(metadata.get("episode_won"))
    if "won" in metadata:
        return bool(metadata.get("won"))
    return _reward(sample) >= 5.0


def _label_value(sample: Sample, key: str, default: str = "unknown") -> str:
    label = sample.label if isinstance(sample.label, dict) else {}
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = label.get(key)
    if value in (None, ""):
        value = metadata.get(key, default)
    return str(value if value not in (None, "") else default)


def _task_key(sample: Sample, fallback_idx: int) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    label = sample.label if isinstance(sample.label, dict) else {}

    episode_id = str(metadata.get("episode_id") or "").strip()
    if episode_id:
        return f"episode:{episode_id}"

    annotation_id = str(label.get("annotation_id") or metadata.get("annotation_id") or "").strip()
    if annotation_id:
        group_index = sample.group_index if sample.group_index is not None else "none"
        return f"annotation:{annotation_id}:group:{group_index}"

    source_file = str(label.get("source_file") or metadata.get("source_file") or "").strip()
    task_index = str(label.get("task_index") or metadata.get("task_index") or "").strip()
    if source_file and task_index:
        group_index = sample.group_index if sample.group_index is not None else "none"
        return f"file:{source_file}:task:{task_index}:group:{group_index}"

    return f"fallback:{fallback_idx}"


def _task_success_values(samples: list[Sample]) -> list[float]:
    by_task: dict[str, float] = {}
    for idx, sample in enumerate(samples):
        key = _task_key(sample, idx)
        by_task[key] = max(by_task.get(key, 0.0), 1.0 if _won(sample) else 0.0)
    return list(by_task.values())


def _log_group_metrics(
    log_dict: dict,
    prefix: str,
    samples: list[Sample],
) -> None:
    if not samples:
        return

    rewards = [_reward(sample) for sample in samples]
    sample_success_flags = [1.0 if _won(sample) else 0.0 for sample in samples]
    task_success_flags = _task_success_values(samples)
    num_steps = [float(_meta_float(sample, "num_steps", 0.0) or 0.0) for sample in samples]
    action_accuracy = [
        float(_meta_float(sample, "action_accuracy", 0.0) or 0.0)
        for sample in samples
    ]

    log_dict |= dict_add_prefix(compute_statistics(rewards), f"{prefix}/training_reward/")
    log_dict |= dict_add_prefix(compute_statistics(num_steps), f"{prefix}/num_steps/")
    log_dict |= dict_add_prefix(compute_statistics(action_accuracy), f"{prefix}/action_accuracy/")
    log_dict[f"{prefix}/count"] = len(samples)
    log_dict[f"{prefix}/task_count"] = len(task_success_flags)
    log_dict[f"{prefix}/sample_success_rate"] = (
        float(np.mean(sample_success_flags)) if sample_success_flags else 0.0
    )
    log_dict[f"{prefix}/task_success_rate"] = (
        float(np.mean(task_success_flags)) if task_success_flags else 0.0
    )
    log_dict[f"{prefix}/win_rate"] = log_dict[f"{prefix}/task_success_rate"]


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    samples: list[Sample],
    rollout_extra_metrics: dict | None,
    rollout_time: float,
) -> bool:
    """Log rollout metrics for solver-only GRPO training."""
    log_dict: dict = {}
    if rollout_extra_metrics:
        log_dict.update(rollout_extra_metrics)

    if samples:
        rewards = [_reward(sample) for sample in samples]
        raw_rewards = [_solver_raw_reward(sample) for sample in samples]
        sample_success_flags = [1.0 if _won(sample) else 0.0 for sample in samples]
        task_success_flags = _task_success_values(samples)
        response_lengths = [sample.response_length for sample in samples]
        effective_lengths = [sample.effective_response_length for sample in samples]
        num_steps = [float(_meta_float(sample, "num_steps", 0.0) or 0.0) for sample in samples]
        action_accuracy = [
            float(_meta_float(sample, "action_accuracy", 0.0) or 0.0)
            for sample in samples
        ]

        log_dict["rollout/raw_reward"] = float(np.mean(raw_rewards)) if raw_rewards else 0.0
        log_dict |= dict_add_prefix(compute_statistics(rewards), "rollout/solver/training_reward/")
        log_dict |= dict_add_prefix(compute_statistics(response_lengths), "rollout/solver/response_len/")
        log_dict |= dict_add_prefix(
            compute_statistics(effective_lengths),
            "rollout/solver/effective_response_len/",
        )
        log_dict |= dict_add_prefix(compute_statistics(num_steps), "rollout/solver/num_steps/")
        log_dict |= dict_add_prefix(
            compute_statistics(action_accuracy),
            "rollout/solver/action_accuracy/",
        )
        log_dict["rollout/solver/count"] = len(samples)
        log_dict["rollout/solver/task_count"] = len(task_success_flags)
        log_dict["rollout/solver/sample_success_rate"] = (
            float(np.mean(sample_success_flags)) if sample_success_flags else 0.0
        )
        log_dict["rollout/solver/task_success_rate"] = (
            float(np.mean(task_success_flags)) if task_success_flags else 0.0
        )
        log_dict["rollout/solver/win_rate"] = log_dict["rollout/solver/task_success_rate"]
        log_dict["rollout/solver/truncated_ratio"] = float(
            np.mean([int(sample.status == Sample.Status.TRUNCATED) for sample in samples])
        )
        log_dict["rollout/solver/repetition_frac"] = float(
            np.mean([int(has_repetition(sample.response)) for sample in samples])
        )

        task_type_groups: dict[str, list[Sample]] = {}
        domain_groups: dict[str, list[Sample]] = {}
        for sample in samples:
            task_type_groups.setdefault(_label_value(sample, "task_type"), []).append(sample)
            domain_groups.setdefault(_label_value(sample, "domain"), []).append(sample)

        for task_type, task_samples in sorted(task_type_groups.items()):
            _log_group_metrics(log_dict, f"rollout/solver/task_type/{task_type}", task_samples)
        for domain, domain_samples in sorted(domain_groups.items()):
            _log_group_metrics(log_dict, f"rollout/solver/domain/{domain}", domain_samples)

    log_dict["perf/rollout_time"] = rollout_time

    all_response_lengths = [sample.response_length for sample in samples]
    all_effective_lengths = [sample.effective_response_length for sample in samples]
    non_generation_time = [sample.non_generation_time for sample in samples]

    if non_generation_time and max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(
            compute_statistics(non_generation_time),
            "perf/non_generation_time/",
        )

    if getattr(args, "rollout_num_gpus", None):
        log_dict["perf/tokens_per_gpu_per_sec"] = (
            sum(all_response_lengths) / rollout_time / args.rollout_num_gpus
        )
        log_dict["perf/effective_tokens_per_gpu_per_sec"] = (
            sum(all_effective_lengths) / rollout_time / args.rollout_num_gpus
        )

    if all_response_lengths:
        log_dict["perf/longest_sample_tokens_per_sec"] = max(all_response_lengths) / rollout_time

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step

    logger.info("rollout %s: %s", rollout_id, log_dict)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True
