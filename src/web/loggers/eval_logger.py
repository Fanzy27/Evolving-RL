"""Eval logger for Mind2Web web experiments.

Records both task-level success and action-level success for baseline eval
and skill-conditioned eval variants.
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _get_task_type(sample: Sample) -> str:
    label = getattr(sample, "label", None)
    if isinstance(label, dict):
        return str(label.get("task_type", "unknown"))
    return "unknown"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _meta_float(sample: Sample, key: str, default: float = 0.0) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _meta_optional_float(sample: Sample, key: str) -> float | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if key not in metadata:
        return None
    try:
        return float(metadata.get(key))
    except Exception:
        return None


def _won(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(metadata.get("won", False))


def _task_key(sample: Sample, fallback_idx: int) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    label = sample.label if isinstance(sample.label, dict) else {}

    annotation_id = str(label.get("annotation_id") or metadata.get("annotation_id") or "").strip()
    if annotation_id:
        return f"annotation:{annotation_id}"

    source_file = str(label.get("source_file") or metadata.get("source_file") or "").strip()
    task_index = str(label.get("task_index") or metadata.get("task_index") or "").strip()
    if source_file and task_index:
        return f"file:{source_file}:task:{task_index}"

    episode_id = str(metadata.get("episode_id") or "").strip()
    if episode_id:
        return f"episode:{episode_id}"

    return f"fallback:{fallback_idx}"


def _task_success_values(samples: list[Sample], value_getter) -> list[float]:
    by_task: dict[str, float] = {}
    for idx, sample in enumerate(samples):
        key = _task_key(sample, idx)
        by_task[key] = max(by_task.get(key, 0.0), float(value_getter(sample)))
    return list(by_task.values())


def _branch_num_steps(sample: Sample, num_steps_key: str) -> float:
    return max(0.0, _meta_float(sample, num_steps_key, 0.0))


def _branch_num_correct_steps(
    sample: Sample,
    *,
    num_correct_key: str,
    num_steps_key: str,
    action_accuracy_key: str,
) -> float:
    explicit = _meta_optional_float(sample, num_correct_key)
    if explicit is not None:
        return max(0.0, explicit)
    num_steps = _branch_num_steps(sample, num_steps_key)
    action_accuracy = max(0.0, min(1.0, _meta_float(sample, action_accuracy_key, 0.0)))
    return action_accuracy * num_steps


def _branch_action_accuracy(
    sample: Sample,
    *,
    num_correct_key: str,
    num_steps_key: str,
    action_accuracy_key: str,
) -> float:
    explicit = _meta_optional_float(sample, action_accuracy_key)
    if explicit is not None:
        return max(0.0, min(1.0, explicit))
    num_steps = _branch_num_steps(sample, num_steps_key)
    if num_steps <= 0:
        return 0.0
    num_correct = _branch_num_correct_steps(
        sample,
        num_correct_key=num_correct_key,
        num_steps_key=num_steps_key,
        action_accuracy_key=action_accuracy_key,
    )
    return max(0.0, min(1.0, num_correct / num_steps))


def _collect_branch_metrics(
    samples: list[Sample],
    *,
    won_key: str,
    num_steps_key: str,
    num_correct_key: str,
    action_accuracy_key: str,
) -> dict[str, Any]:
    task_success_flags = _task_success_values(samples, lambda sample: _meta_float(sample, won_key, 0.0))
    sample_success_flags = [float(_meta_float(sample, won_key, 0.0)) for sample in samples]
    num_steps = [_branch_num_steps(sample, num_steps_key) for sample in samples]
    num_correct_steps = [
        min(
            _branch_num_steps(sample, num_steps_key),
            _branch_num_correct_steps(
                sample,
                num_correct_key=num_correct_key,
                num_steps_key=num_steps_key,
                action_accuracy_key=action_accuracy_key,
            ),
        )
        for sample in samples
    ]
    action_accuracy = [
        _branch_action_accuracy(
            sample,
            num_correct_key=num_correct_key,
            num_steps_key=num_steps_key,
            action_accuracy_key=action_accuracy_key,
        )
        for sample in samples
    ]
    total_steps = sum(num_steps)
    total_correct = sum(num_correct_steps)
    task_count = len(task_success_flags)
    return {
        "count": len(samples),
        "task_count": task_count,
        "total_num_steps": total_steps,
        "total_num_correct_steps": total_correct,
        "sample_success_rate": _mean(sample_success_flags),
        "task_success_rate": _mean(task_success_flags),
        "action_success_rate": (total_correct / total_steps) if total_steps > 0 else 0.0,
        "success_steps_per_task": (total_correct / task_count) if task_count > 0 else 0.0,
        "num_steps": num_steps,
        "num_correct_steps": num_correct_steps,
        "action_accuracy": action_accuracy,
    }


def _metric_key(prefix: str, name: str, suffix: str) -> str:
    return f"{prefix}/{name}_{suffix}" if suffix else f"{prefix}/{name}"


def _stats_prefix(prefix: str, name: str, suffix: str) -> str:
    return f"{prefix}/{name}_{suffix}/" if suffix else f"{prefix}/{name}/"


def _write_branch_metrics(
    log_dict: dict[str, Any],
    prefix: str,
    *,
    suffix: str,
    metrics: dict[str, Any],
) -> None:
    log_dict[_metric_key(prefix, "sample_success_rate", suffix)] = metrics["sample_success_rate"]
    log_dict[_metric_key(prefix, "action_success_rate", suffix)] = metrics["action_success_rate"]
    log_dict[_metric_key(prefix, "success_steps_per_task", suffix)] = metrics[
        "success_steps_per_task"
    ]
    log_dict[_metric_key(prefix, "win_rate", suffix)] = metrics["task_success_rate"]


def _print_branch_summary(label: str, metrics: dict[str, Any], count: int) -> str:
    return (
        f"  {label:<18}: "
        f"sample={metrics['sample_success_rate']:.4f}  "
        f"action={metrics['action_success_rate']:.4f}  "
        f"correct/task={metrics['success_steps_per_task']:.4f}  "
        f"steps={_mean(metrics['num_steps']):.1f}  "
        f"(n={count})"
    )


def _has_skill_meta(samples: list[Sample]) -> bool:
    return any(
        isinstance(sample.metadata, dict) and "won_with_skill" in sample.metadata
        for sample in samples
    )


def log_eval_by_task_type(
    rollout_id: int,
    args,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    return _log_eval_by_task_type_impl(
        rollout_id=rollout_id,
        args=args,
        data=data,
        extra_metrics=extra_metrics,
        logger_name="log_eval_by_task_type",
    )


def log_eval_success_steps_per_task(
    rollout_id: int,
    args,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    return _log_eval_by_task_type_impl(
        rollout_id=rollout_id,
        args=args,
        data=data,
        extra_metrics=extra_metrics,
        logger_name="log_eval_success_steps_per_task",
    )


def _log_eval_by_task_type_impl(
    *,
    rollout_id: int,
    args,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
    logger_name: str,
) -> bool:
    log_dict: dict[str, Any] = extra_metrics.copy() if extra_metrics else {}
    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step

    for dataset_name, dataset_data in data.items():
        samples: list[Sample] = dataset_data.get("samples") or []
        if not samples:
            continue

        prefix = f"eval/{dataset_name}"
        if _has_skill_meta(samples):
            _log_skill_eval(log_dict, prefix, samples, dataset_data, rollout_id)
        else:
            _log_baseline_eval(log_dict, prefix, samples, dataset_data, rollout_id)

    logger.info("%s eval %s: %s", logger_name, rollout_id, log_dict)
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True


def _log_skill_eval(
    log_dict: dict[str, Any],
    prefix: str,
    samples: list[Sample],
    dataset_data: dict[str, Any],
    rollout_id: int,
) -> None:
    branches = {
        "with_skill": _collect_branch_metrics(
            samples,
            won_key="won_with_skill",
            num_steps_key="num_steps_with_skill",
            num_correct_key="num_correct_steps_with_skill",
            action_accuracy_key="action_accuracy_with_skill",
        ),
        "no_skill": _collect_branch_metrics(
            samples,
            won_key="won_no_skill",
            num_steps_key="num_steps_no_skill",
            num_correct_key="num_correct_steps_no_skill",
            action_accuracy_key="action_accuracy_no_skill",
        ),
    }

    log_dict[f"{prefix}/count"] = len(samples)
    for suffix, metrics in branches.items():
        _write_branch_metrics(log_dict, prefix, suffix=suffix, metrics=metrics)

    if "truncated" in dataset_data:
        truncated = dataset_data["truncated"]
        log_dict[f"{prefix}/truncated_ratio"] = sum(truncated) / len(truncated)

    task_type_groups: dict[str, list[Sample]] = {}
    for sample in samples:
        task_type_groups.setdefault(_get_task_type(sample), []).append(sample)

    lines = [
        f"\n{'=' * 80}",
        f"Eval Results [{prefix.split('/', 1)[-1]}]  (rollout {rollout_id})",
        _print_branch_summary("with-skill", branches["with_skill"], len(samples)),
        _print_branch_summary("no-skill", branches["no_skill"], len(samples)),
    ]
    for task_type, task_samples in sorted(task_type_groups.items()):
        group_with = _collect_branch_metrics(
            task_samples,
            won_key="won_with_skill",
            num_steps_key="num_steps_with_skill",
            num_correct_key="num_correct_steps_with_skill",
            action_accuracy_key="action_accuracy_with_skill",
        )
        group_no = _collect_branch_metrics(
            task_samples,
            won_key="won_no_skill",
            num_steps_key="num_steps_no_skill",
            num_correct_key="num_correct_steps_no_skill",
            action_accuracy_key="action_accuracy_no_skill",
        )
        lines.append(
            f"  {task_type:<30}: "
            f"with(sample={group_with['sample_success_rate']:.4f}, action={group_with['action_success_rate']:.4f})  "
            f"no(sample={group_no['sample_success_rate']:.4f}, action={group_no['action_success_rate']:.4f})  "
            f"(n={len(task_samples)})"
        )
    lines.append("=" * 80)
    print("\n".join(lines), flush=True)


def _log_baseline_eval(
    log_dict: dict[str, Any],
    prefix: str,
    samples: list[Sample],
    dataset_data: dict[str, Any],
    rollout_id: int,
) -> None:
    metrics = _collect_branch_metrics(
        samples,
        won_key="won",
        num_steps_key="num_steps",
        num_correct_key="num_correct_steps",
        action_accuracy_key="action_accuracy",
    )

    log_dict[f"{prefix}/count"] = len(samples)
    _write_branch_metrics(log_dict, prefix, suffix="", metrics=metrics)

    if "truncated" in dataset_data:
        truncated = dataset_data["truncated"]
        log_dict[f"{prefix}/truncated_ratio"] = sum(truncated) / len(truncated)

    task_type_groups: dict[str, list[Sample]] = {}
    for sample in samples:
        task_type_groups.setdefault(_get_task_type(sample), []).append(sample)

    lines = [
        f"\n{'=' * 80}",
        f"Eval Results [{prefix.split('/', 1)[-1]}]  (rollout {rollout_id})",
        _print_branch_summary("baseline", metrics, len(samples)),
    ]
    for task_type, task_samples in sorted(task_type_groups.items()):
        group_metrics = _collect_branch_metrics(
            task_samples,
            won_key="won",
            num_steps_key="num_steps",
            num_correct_key="num_correct_steps",
            action_accuracy_key="action_accuracy",
        )
        lines.append(
            f"  {task_type:<30}: "
            f"sample={group_metrics['sample_success_rate']:.4f}  "
            f"action={group_metrics['action_success_rate']:.4f}  "
            f"(n={len(task_samples)})"
        )
    lines.append("=" * 80)
    print("\n".join(lines), flush=True)
