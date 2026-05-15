"""Web rollout logger for training."""

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
from slime.utils.misc import group_by
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _meta_float(sample: Sample, key: str, default: float | None = None) -> float | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _training_reward(sample: Sample) -> float | None:
    value = _meta_float(sample, "raw_training_reward")
    if value is not None:
        return value
    reward = sample.reward
    return float(reward) if isinstance(reward, (int, float)) else None


def _env_reward(sample: Sample) -> float:
    value = _meta_float(sample, "raw_env_reward")
    if value is not None:
        return value
    won = bool((sample.metadata or {}).get("won", False)) if isinstance(sample.metadata, dict) else False
    return 10.0 * float(won)


def _solver_raw_reward(sample: Sample) -> float | None:
    if sample.group_index is None or int(sample.group_index) < 1:
        return None
    value = _meta_float(sample, "raw_env_reward")
    if value is not None:
        return value
    value = _meta_float(sample, "raw_step_reward")
    if value is not None:
        return value
    reward = sample.reward
    return float(reward) if isinstance(reward, (int, float)) else None


def _won(sample: Sample) -> bool:
    return bool((sample.metadata or {}).get("won", False)) if isinstance(sample.metadata, dict) else False


def _task_key(sample: Sample, fallback_idx: int) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    label = sample.label if isinstance(sample.label, dict) else {}

    episode_id = str(metadata.get("episode_id") or "").strip()
    if episode_id:
        return f"episode:{episode_id}"

    annotation_id = str(label.get("annotation_id") or metadata.get("annotation_id") or "").strip()
    if annotation_id:
        group_index = sample.group_index if sample.group_index is not None else "none"
        index = sample.index if sample.index is not None else "none"
        return f"annotation:{annotation_id}:group:{group_index}:index:{index}"

    source_file = str(label.get("source_file") or metadata.get("source_file") or "").strip()
    task_index = str(label.get("task_index") or metadata.get("task_index") or "").strip()
    if source_file and task_index:
        group_index = sample.group_index if sample.group_index is not None else "none"
        index = sample.index if sample.index is not None else "none"
        return f"file:{source_file}:task:{task_index}:group:{group_index}:index:{index}"

    return f"fallback:{fallback_idx}"


def _task_success_values(samples: list[Sample]) -> list[float]:
    by_task: dict[str, float] = {}
    for idx, sample in enumerate(samples):
        key = _task_key(sample, idx)
        by_task[key] = max(by_task.get(key, 0.0), 1.0 if _won(sample) else 0.0)
    return list(by_task.values())


def _add_success_metrics(log_dict: dict, prefix: str, samples: list[Sample]) -> None:
    sample_success_flags = [1.0 if _won(sample) else 0.0 for sample in samples]
    task_success_flags = _task_success_values(samples)
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
    extractor = [sample for sample in samples if sample.group_index == 0]
    solver = [sample for sample in samples if sample.group_index is not None and sample.group_index >= 1]

    log_dict: dict = {}
    if rollout_extra_metrics:
        log_dict.update(rollout_extra_metrics)

    log_dict["rollout/final_sample_count"] = len(samples)
    log_dict["rollout/final_extractor_count"] = len(extractor)
    log_dict["rollout/final_solver_count"] = len(solver)

    if extractor:
        ext_training_rewards = [
            reward for sample in extractor if (reward := _training_reward(sample)) is not None
        ]
        ext_downstream_solver_rewards = [
            reward
            for sample in extractor
            if (reward := _meta_float(sample, "raw_mean_downstream_solver_reward")) is not None
        ]
        ext_downstream_num_correct_steps = [
            value
            for sample in extractor
            if (value := _meta_float(sample, "raw_mean_downstream_num_correct_steps")) is not None
        ]
        ext_win_rates = [
            win_rate
            for sample in extractor
            if (win_rate := _meta_float(sample, "downstream_win_rate")) is not None
        ]
        ext_format_scores = [
            score for sample in extractor if (score := _meta_float(sample, "format_score")) is not None
        ]
        ext_format_penalties = [
            penalty for sample in extractor if (penalty := _meta_float(sample, "format_penalty")) is not None
        ]
        ext_skill_language_audit_applied = [
            applied
            for sample in extractor
            if (applied := _meta_float(sample, "skill_language_audit_applied")) is not None
        ]
        ext_skill_language_audit_pass = [
            passed
            for sample in extractor
            if (_meta_float(sample, "skill_language_audit_applied", 0.0) or 0.0) > 0.5
            if (passed := _meta_float(sample, "skill_language_audit_pass")) is not None
        ]
        ext_skill_language_audit_penalties = [
            penalty
            for sample in extractor
            if (penalty := _meta_float(sample, "skill_language_audit_penalty")) is not None
            if abs(penalty) > 1e-8
        ]
        ext_skill_language_audit_reward_zeroed = [
            zeroed
            for sample in extractor
            if (zeroed := _meta_float(sample, "skill_language_audit_reward_zeroed")) is not None
        ]
        ext_skill_language_audit_uncommon_char = [
            flag
            for sample in extractor
            if (_meta_float(sample, "skill_language_audit_applied", 0.0) or 0.0) > 0.5
            if (flag := _meta_float(sample, "skill_language_audit_has_uncommon_characters")) is not None
        ]
        ext_skill_language_audit_non_english_char = [
            flag
            for sample in extractor
            if (_meta_float(sample, "skill_language_audit_applied", 0.0) or 0.0) > 0.5
            if (flag := _meta_float(sample, "skill_language_audit_has_non_english_characters")) is not None
        ]
        ext_resp_lens = [sample.effective_response_length for sample in extractor]

        if ext_training_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_training_rewards),
                "rollout/extractor/training_reward/",
            )
        if ext_downstream_solver_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_downstream_solver_rewards),
                "rollout/extractor/mean_downstream_solver_reward/",
            )
        if ext_downstream_num_correct_steps:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_downstream_num_correct_steps),
                "rollout/extractor/mean_downstream_num_correct_steps/",
            )
        if ext_win_rates:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_win_rates),
                "rollout/extractor/downstream_win_rate/",
            )
        if ext_format_scores:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_format_scores),
                "rollout/extractor/format_score/",
            )
        if ext_format_penalties:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_format_penalties),
                "rollout/extractor/format_penalty/",
            )
        if ext_skill_language_audit_applied:
            log_dict["rollout/extractor/skill_language_audit/applied_frac"] = float(
                np.mean(ext_skill_language_audit_applied)
            )
        if ext_skill_language_audit_pass:
            log_dict["rollout/extractor/skill_language_audit/pass_rate"] = float(
                np.mean(ext_skill_language_audit_pass)
            )
        if ext_skill_language_audit_penalties:
            log_dict |= dict_add_prefix(
                compute_statistics(ext_skill_language_audit_penalties),
                "rollout/extractor/skill_language_audit/penalty/",
            )
        if ext_skill_language_audit_reward_zeroed:
            log_dict["rollout/extractor/skill_language_audit/reward_zeroed_frac"] = float(
                np.mean(ext_skill_language_audit_reward_zeroed)
            )
        if ext_skill_language_audit_uncommon_char:
            log_dict["rollout/extractor/skill_language_audit/uncommon_char_rate"] = float(
                np.mean(ext_skill_language_audit_uncommon_char)
            )
        if ext_skill_language_audit_non_english_char:
            log_dict["rollout/extractor/skill_language_audit/non_english_char_rate"] = float(
                np.mean(ext_skill_language_audit_non_english_char)
            )
        log_dict |= dict_add_prefix(
            compute_statistics(ext_resp_lens),
            "rollout/extractor/response_len/",
        )
        log_dict["rollout/extractor/count"] = len(extractor)
        log_dict["rollout/extractor/truncated_ratio"] = float(
            np.mean([int(sample.status == Sample.Status.TRUNCATED) for sample in extractor])
        )
        log_dict["rollout/extractor/repetition_frac"] = float(
            np.mean([int(has_repetition(sample.response)) for sample in extractor])
        )

    if solver:
        solver_training_rewards = [
            reward for sample in solver if (reward := _training_reward(sample)) is not None
        ]
        solver_raw_rewards = [
            reward for sample in solver if (reward := _solver_raw_reward(sample)) is not None
        ]
        solver_env_rewards = [_env_reward(sample) for sample in solver]
        solver_resp_lens = [sample.response_length for sample in solver]
        solver_eff_resp_lens = [sample.effective_response_length for sample in solver]
        solver_num_steps = [_meta_float(sample, "num_steps", 0.0) or 0.0 for sample in solver]
        solver_workflow_compliance = [
            float(_meta_float(sample, "workflow_compliance", 0.0) or 0.0)
            for sample in solver
            if _meta_float(sample, "workflow_compliance") is not None
        ]

        if solver_training_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(solver_training_rewards),
                "rollout/solver/training_reward/",
            )
        if solver_raw_rewards:
            log_dict["rollout/raw_reward"] = float(np.mean(solver_raw_rewards))
        if solver_env_rewards:
            log_dict |= dict_add_prefix(
                compute_statistics(solver_env_rewards),
                "rollout/solver/env_reward/",
            )
        if solver_workflow_compliance:
            log_dict |= dict_add_prefix(
                compute_statistics(solver_workflow_compliance),
                "rollout/solver/workflow_compliance/",
            )
        log_dict |= dict_add_prefix(compute_statistics(solver_resp_lens), "rollout/solver/response_len/")
        log_dict |= dict_add_prefix(
            compute_statistics(solver_eff_resp_lens),
            "rollout/solver/effective_response_len/",
        )
        log_dict |= dict_add_prefix(compute_statistics(solver_num_steps), "rollout/solver/num_steps/")
        log_dict["rollout/solver/count"] = len(solver)
        _add_success_metrics(log_dict, "rollout/solver", solver)
        log_dict["rollout/solver/truncated_ratio"] = float(
            np.mean([int(sample.status == Sample.Status.TRUNCATED) for sample in solver])
        )
        log_dict["rollout/solver/repetition_frac"] = float(
            np.mean([int(has_repetition(sample.response)) for sample in solver])
        )

        solver_groups = group_by(solver, lambda sample: sample.group_index)
        zero_std_groups = 0
        for group in solver_groups.values():
            rewards = [_training_reward(sample) for sample in group]
            rewards = [reward for reward in rewards if reward is not None]
            if len(rewards) > 1 and len(set(rewards)) == 1:
                zero_std_groups += 1
        log_dict["rollout/solver/zero_std_group_count"] = zero_std_groups

        task_type_groups: dict[str, list[Sample]] = {}
        for sample in solver:
            task_type = (
                str(sample.label.get("task_type", "unknown"))
                if isinstance(sample.label, dict)
                else "unknown"
            )
            task_type_groups.setdefault(task_type, []).append(sample)
        for task_type, task_samples in sorted(task_type_groups.items()):
            log_dict[f"rollout/solver/{task_type}/count"] = len(task_samples)
            _add_success_metrics(log_dict, f"rollout/solver/{task_type}", task_samples)
            task_rewards = [
                reward for sample in task_samples if (reward := _training_reward(sample)) is not None
            ]
            if task_rewards:
                log_dict |= dict_add_prefix(
                    compute_statistics(task_rewards),
                    f"rollout/solver/{task_type}/training_reward/",
                )
            task_steps = [_meta_float(sample, "num_steps", 0.0) or 0.0 for sample in task_samples]
            if task_steps:
                log_dict |= dict_add_prefix(
                    compute_statistics(task_steps),
                    f"rollout/solver/{task_type}/num_steps/",
                )

    source_episode_won = [
        float(_meta_float(sample, "source_episode_won", 0.0) or 0.0)
        for sample in extractor
        if _meta_float(sample, "source_episode_won") is not None
    ]
    if source_episode_won:
        log_dict["rollout/source_episode/task_success_rate"] = float(np.mean(source_episode_won))
        log_dict["rollout/source_episode/win_rate"] = log_dict["rollout/source_episode/task_success_rate"]

    log_dict["perf/rollout_time"] = rollout_time
    all_resp_lens = [sample.response_length for sample in samples]
    eff_resp_lens = [sample.effective_response_length for sample in samples]
    non_gen_times = [sample.non_generation_time for sample in samples]

    if non_gen_times and max(non_gen_times) > 0:
        log_dict |= dict_add_prefix(
            compute_statistics(non_gen_times),
            "perf/non_generation_time/",
        )

    if hasattr(args, "rollout_num_gpus") and args.rollout_num_gpus is not None:
        log_dict["perf/tokens_per_gpu_per_sec"] = (
            sum(all_resp_lens) / rollout_time / args.rollout_num_gpus
        )
        log_dict["perf/effective_tokens_per_gpu_per_sec"] = (
            sum(eff_resp_lens) / rollout_time / args.rollout_num_gpus
        )

    if all_resp_lens:
        log_dict["perf/longest_sample_tokens_per_sec"] = max(all_resp_lens) / rollout_time

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step

    logger.info(f"rollout {rollout_id}: {log_dict}")
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True
