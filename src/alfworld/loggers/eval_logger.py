import logging
from typing import Any

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_task_type(sample: Sample) -> str:
    label = getattr(sample, "label", None)
    if isinstance(label, dict):
        return str(label.get("task_type", "unknown"))
    return "unknown"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _meta_float(sample: Sample, key: str, default: float = 0.0) -> float:
    m = sample.metadata
    if isinstance(m, dict):
        v = m.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return default


def _won(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(metadata.get("won", False))


def _has_skill_meta(samples: list[Sample]) -> bool:
    return any(
        isinstance(s.metadata, dict) and "won_with_skill" in s.metadata
        for s in samples
    )


# ---------------------------------------------------------------------------
# Main log function
# ---------------------------------------------------------------------------


def log_eval_by_task_type(
    rollout_id: int,
    args,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Custom eval log function with per-task_type win rate for ALFWorld."""
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
            _log_baseline_eval(log_dict, prefix, samples, dataset_data)

    logger.info(f"eval {rollout_id}: {log_dict}")
    logging_utils.log(args, log_dict, step_key="eval/step")
    return True


# ---------------------------------------------------------------------------
# Skill eval (with-skill + no-skill)
# ---------------------------------------------------------------------------


def _log_skill_eval(
    log_dict: dict,
    prefix: str,
    samples: list[Sample],
    dataset_data: dict,
    rollout_id: int,
) -> None:
    won_with = [_meta_float(s, "won_with_skill") for s in samples]
    won_no = [_meta_float(s, "won_no_skill") for s in samples]

    steps_with = [_meta_float(s, "num_steps_with_skill") for s in samples]
    steps_no = [_meta_float(s, "num_steps_no_skill") for s in samples]

    overall_with = _mean(won_with)
    overall_no = _mean(won_no)

    log_dict[f"{prefix}/win_rate_with_skill"] = overall_with
    log_dict[f"{prefix}/win_rate_no_skill"] = overall_no
    log_dict[f"{prefix}/count"] = len(samples)

    log_dict |= dict_add_prefix(
        compute_statistics(steps_with),
        f"{prefix}/num_steps_with_skill/"
    )
    log_dict |= dict_add_prefix(
        compute_statistics(steps_no),
        f"{prefix}/num_steps_no_skill/"
    )

    if "truncated" in dataset_data:
        truncated = dataset_data["truncated"]
        log_dict[f"{prefix}/truncated_ratio"] = sum(truncated) / len(truncated)

    task_type_groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        tt = _get_task_type(s)
        task_type_groups.setdefault(tt, []).append(i)

    for tt, indices in sorted(task_type_groups.items()):
        tt_with = [won_with[i] for i in indices]
        tt_no = [won_no[i] for i in indices]

        tt_steps_with = [steps_with[i] for i in indices]
        tt_steps_no = [steps_no[i] for i in indices]

        log_dict[f"{prefix}/{tt}/win_rate_with_skill"] = _mean(tt_with)
        log_dict[f"{prefix}/{tt}/win_rate_no_skill"] = _mean(tt_no)
        log_dict[f"{prefix}/{tt}/count"] = len(indices)

        log_dict |= dict_add_prefix(
            compute_statistics(tt_steps_with),
            f"{prefix}/{tt}/num_steps_with_skill/"
        )
        log_dict |= dict_add_prefix(
            compute_statistics(tt_steps_no),
            f"{prefix}/{tt}/num_steps_no_skill/"
        )

    lines = [
        f"\n{'=' * 80}",
        f"Eval Results [{prefix.split('/', 1)[-1]}]  (rollout {rollout_id})",
        f"  Overall  with-skill  : win={overall_with:.4f}  steps={_mean(steps_with):.1f}  (n={len(samples)})",
        f"  Overall  no-skill    : win={overall_no:.4f}  steps={_mean(steps_no):.1f}",
    ]
    for tt, indices in sorted(task_type_groups.items()):
        tw = _mean([won_with[i] for i in indices])
        tn = _mean([won_no[i] for i in indices])

        sw = _mean([steps_with[i] for i in indices])
        sn = _mean([steps_no[i] for i in indices])

        lines.append(
            f"  {tt:<30}: "
            f"with(win={tw:.4f}, steps={sw:.1f})  "
            f"no(win={tn:.4f}, steps={sn:.1f})  "
            f"(n={len(indices)})"
        )
    lines.append("=" * 80)
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Baseline eval
# ---------------------------------------------------------------------------


def _log_baseline_eval(
    log_dict: dict,
    prefix: str,
    samples: list[Sample],
    dataset_data: dict,
) -> None:
    won_flags = [1.0 if _won(sample) else 0.0 for sample in samples]
    num_steps = [_meta_float(s, "num_steps") for s in samples]

    log_dict[f"{prefix}/win_rate"] = _mean(won_flags)
    log_dict[f"{prefix}/count"] = len(samples)
    log_dict |= dict_add_prefix(compute_statistics(num_steps), f"{prefix}/num_steps/")

    if "truncated" in dataset_data:
        truncated = dataset_data["truncated"]
        log_dict[f"{prefix}/truncated_ratio"] = sum(truncated) / len(truncated)

    task_type_groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        tt = _get_task_type(s)
        task_type_groups.setdefault(tt, []).append(i)

    for tt, indices in sorted(task_type_groups.items()):
        tt_won = [won_flags[i] for i in indices]
        tt_steps = [num_steps[i] for i in indices]
        log_dict[f"{prefix}/{tt}/win_rate"] = _mean(tt_won)
        log_dict[f"{prefix}/{tt}/count"] = len(indices)
        log_dict |= dict_add_prefix(compute_statistics(tt_steps), f"{prefix}/{tt}/num_steps/")
