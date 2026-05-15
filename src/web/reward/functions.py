"""Reward computation for Mind2Web web training."""

from __future__ import annotations

from typing import Any

import numpy as np
from slime.utils.types import Sample

from src.web.utils.common import (
    ensure_metadata,
    ensure_train_placeholder_sample,
    maybe_print_random_sample,
    require_arg,
)


def grpo_normalize(samples: list[Sample]) -> None:
    if len(samples) <= 1:
        return
    rewards = [s.reward if s.reward is not None else 0.0 for s in samples]
    mean_r = sum(rewards) / len(rewards)
    variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = variance ** 0.5
    for sample, reward in zip(samples, rewards):
        sample.reward = (reward - mean_r) / (std_r + 1e-8)


def _downstream_episode_key(sample: Sample) -> tuple[Any, Any, str]:
    metadata = ensure_metadata(sample)
    episode_id = str(metadata.get("episode_id") or "").strip()
    if episode_id:
        return ("episode", episode_id, "")

    label = sample.label if isinstance(sample.label, dict) else {}
    annotation_id = str(label.get("annotation_id") or metadata.get("annotation_id") or "").strip()
    if annotation_id:
        group_index = sample.group_index if sample.group_index is not None else "none"
        sample_index = sample.index if sample.index is not None else "none"
        return ("annotation", group_index, f"{annotation_id}:{sample_index}")

    source_file = str(label.get("source_file") or metadata.get("source_file") or "").strip()
    task_index = str(label.get("task_index") or metadata.get("task_index") or "").strip()
    if source_file and task_index:
        group_index = sample.group_index if sample.group_index is not None else "none"
        return ("file", group_index, f"{source_file}:{task_index}")

    group_index = sample.group_index if sample.group_index is not None else "none"
    sample_index = sample.index if sample.index is not None else "none"
    return ("fallback", group_index, str(sample_index))


def _set_normalized_downstream_reward(
    sample: Sample,
    *,
    normalized_reward: float,
    step_index: int,
    group_size: int,
    strategy: str,
) -> None:
    sample.reward = float(normalized_reward)
    metadata = ensure_metadata(sample)
    metadata["raw_training_reward"] = float(normalized_reward)
    metadata["normalized_training_reward"] = float(normalized_reward)
    metadata["reward_normalization_scope"] = "downstream_step_index"
    metadata["reward_normalization_group_step_index"] = int(step_index)
    metadata["reward_normalization_group_size"] = int(group_size)
    metadata["reward_normalization_strategy"] = str(strategy)


def normalize_step_rewards(samples: list[Sample]) -> None:
    if not samples:
        return

    step_groups: dict[int, list[Sample]] = {}
    normalized_by_episode_step: dict[tuple[Any, Any, str, int], float] = {}

    for sample in samples:
        metadata = ensure_metadata(sample)
        step_index = int(metadata.get("step_index", 0) or 0)
        step_groups.setdefault(step_index, []).append(sample)

    for step_index in sorted(step_groups):
        group = step_groups[step_index]

        if len(group) > 1:
            rewards = [
                float(sample.reward) if isinstance(sample.reward, (int, float)) else 0.0
                for sample in group
            ]
            mean_r = sum(rewards) / len(rewards)
            variance = sum((reward - mean_r) ** 2 for reward in rewards) / len(rewards)
            std_r = variance**0.5

            for sample, reward in zip(group, rewards):
                normalized_reward = (reward - mean_r) / (std_r + 1e-8)
                _set_normalized_downstream_reward(
                    sample,
                    normalized_reward=normalized_reward,
                    step_index=step_index,
                    group_size=len(group),
                    strategy="step_index_grpo",
                )
                normalized_by_episode_step[
                    (*_downstream_episode_key(sample), step_index)
                ] = normalized_reward
            continue

        sample = group[0]
        metadata = ensure_metadata(sample)
        raw_reward = float(sample.reward) if isinstance(sample.reward, (int, float)) else 0.0
        action_correct = bool(metadata.get("action_correct", False))

        if action_correct or raw_reward > 0.0:
            previous_reward = None
            if step_index > 0:
                previous_reward = normalized_by_episode_step.get(
                    (*_downstream_episode_key(sample), step_index - 1)
                )

            if previous_reward is not None:
                normalized_reward = float(previous_reward)
                strategy = "singleton_success_copy_previous_step"
            else:
                normalized_reward = 1.0
                strategy = "singleton_success_positive_fallback"
        else:
            normalized_reward = -1.0
            strategy = "singleton_failure_negative_one"

        _set_normalized_downstream_reward(
            sample,
            normalized_reward=normalized_reward,
            step_index=step_index,
            group_size=1,
            strategy=strategy,
        )
        normalized_by_episode_step[
            (*_downstream_episode_key(sample), step_index)
        ] = normalized_reward


def finalize_extractor_rewards(
    extractor_samples: list[Sample],
    downstream_results: list[dict],
    *,
    num_experiences: int,
    num_tasks: int,
    format_reward_mode: str = "penalty",
    reward_key: str = "episode_reward",
    reward_metadata_key: str = "raw_mean_downstream_solver_reward",
    reward_metric_name: str = "episode_reward",
) -> None:
    per_exp_solver_reward = [0.0] * num_experiences
    per_exp_win_rate = [0.0] * num_experiences
    per_exp_counts = [0] * num_experiences

    for result in downstream_results:
        n_idx = int(result.get("n_idx", -1))
        if n_idx < 0 or n_idx >= num_experiences:
            continue
        # Each downstream_result is one downstream task/episode, even if that
        # episode produced multiple per-step samples during rollout.
        per_exp_solver_reward[n_idx] += float(result.get(reward_key, 0.0) or 0.0)
        per_exp_win_rate[n_idx] += float(bool(result.get("won", False)))
        per_exp_counts[n_idx] += 1

    for n_idx in range(num_experiences):
        denom = float(per_exp_counts[n_idx] or num_tasks or 1)
        per_exp_solver_reward[n_idx] /= denom
        per_exp_win_rate[n_idx] /= denom

    for n_idx, extractor_sample in enumerate(extractor_samples):
        metadata = ensure_metadata(extractor_sample)
        format_penalty = float(metadata.get("format_penalty", 0.0))
        format_score = float(metadata.get("format_score", 1.0) or 0.0)
        mean_solver_reward = per_exp_solver_reward[n_idx]
        reward_task_count = int(per_exp_counts[n_idx] or num_tasks or 0)
        metadata["format_reward_mode"] = format_reward_mode
        metadata["extractor_reward_metric"] = reward_metric_name
        metadata["downstream_reward_task_count"] = reward_task_count
        metadata["reward_zeroed_by_format"] = 0.0
        metadata[reward_metadata_key] = mean_solver_reward

        if format_reward_mode == "strict_zero":
            extractor_sample.reward = mean_solver_reward if format_score >= 1.0 else 0.0
            metadata["reward_zeroed_by_format"] = 0.0 if format_score >= 1.0 else 1.0
        elif format_reward_mode == "ignore":
            extractor_sample.reward = mean_solver_reward
        else:
            extractor_sample.reward = mean_solver_reward + format_penalty

        metadata["downstream_win_rate"] = per_exp_win_rate[n_idx]
        metadata["raw_downstream_win_reward"] = 10.0 * per_exp_win_rate[n_idx]
        metadata["raw_training_reward"] = float(extractor_sample.reward)


def apply_reward_weights(
    args,
    *,
    extractor_samples: list[Sample],
    downstream_samples: list[Sample],
) -> None:
    extractor_weight = float(require_arg(args, "extractor_reward_weight"))
    solver_weight = float(require_arg(args, "solver_reward_weight"))

    for sample in extractor_samples:
        if isinstance(sample.reward, (int, float)):
            sample.reward *= extractor_weight
            ensure_metadata(sample)["reward_weight_applied"] = extractor_weight
        maybe_print_random_sample(sample, tag="train/extractor")

    for sample in downstream_samples:
        if isinstance(sample.reward, (int, float)):
            sample.reward *= solver_weight
            ensure_metadata(sample)["reward_weight_applied"] = solver_weight
        maybe_print_random_sample(sample, tag="train/downstream")


def apply_language_audit_rewards(extractor_samples: list[Sample]) -> None:
    for extractor_sample in extractor_samples:
        metadata = ensure_metadata(extractor_sample)
        audit_applied = float(metadata.get("skill_language_audit_applied", 0.0) or 0.0) > 0.5
        audit_pass = float(metadata.get("skill_language_audit_pass", 0.0) or 0.0) > 0.5
        audit_reward_zeroed = float(
            metadata.get("skill_language_audit_reward_zeroed", 0.0) or 0.0
        ) > 0.5

        reward_before_audit = (
            float(extractor_sample.reward) if isinstance(extractor_sample.reward, (int, float)) else 0.0
        )
        metadata["raw_extractor_reward_before_language_audit"] = reward_before_audit
        metadata["reward_zeroed_by_language_audit"] = 1.0 if (
            audit_reward_zeroed or (audit_applied and not audit_pass)
        ) else 0.0
        if metadata["reward_zeroed_by_language_audit"] > 0.5:
            extractor_sample.reward = 0.0
        metadata["raw_training_reward"] = (
            float(extractor_sample.reward) if isinstance(extractor_sample.reward, (int, float)) else 0.0
        )


def attach_train_metadata(
    samples: list[Sample],
    *,
    sample_role: str,
) -> None:
    for sample in samples:
        train_metadata = dict(sample.train_metadata or {})
        train_metadata["training_stage"] = "experience"
        train_metadata["sample_role"] = sample_role
        sample.train_metadata = train_metadata
