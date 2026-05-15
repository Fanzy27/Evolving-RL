"""Reward computation and filtering for ALFWorld training."""

from __future__ import annotations

import math

from slime.utils.types import Sample

from src.alfworld.utils.common import ensure_metadata, maybe_print_random_sample, require_arg

_EXTRACTOR_GROUP_MARGIN_Z_THRESHOLD = 1.0
_EXTRACTOR_GROUP_MAX_MEAN_SUCCESS_RATE = 10.7
_FILTER_EPS = 1e-8


def grpo_normalize(samples: list[Sample]) -> None:
    if len(samples) <= 1:
        return

    rewards = [s.reward if s.reward is not None else 0.0 for s in samples]
    mean_r = sum(rewards) / len(rewards)
    variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = variance ** 0.5
    for sample, reward in zip(samples, rewards):
        sample.reward = (reward - mean_r) / (std_r + 1e-6)


def _require_num_repeat(args) -> int:
    num_repeat = int(getattr(args, "num_repeat", 1) or 1)
    if num_repeat <= 0:
        raise ValueError(f"num_repeat must be a positive integer, got {num_repeat}")
    return num_repeat


def _downstream_flat_index(
    *,
    num_experiences: int,
    num_repeat: int,
    k_idx: int,
    n_idx: int,
    repeat_idx: int,
) -> int:
    return ((k_idx * num_experiences + n_idx) * num_repeat) + repeat_idx


def finalize_extractor_rewards(
    extractor_samples: list[Sample],
    downstream_samples: list[Sample],
    *,
    num_experiences: int,
    num_tasks: int,
    num_repeat: int,
) -> None:
    per_exp_solver_reward = [0.0] * num_experiences
    per_exp_win_rate = [0.0] * num_experiences
    denom = float(max(num_tasks * num_repeat, 1))

    for n_idx in range(num_experiences):
        total_reward = 0.0
        total_won = 0.0
        for k_idx in range(num_tasks):
            for repeat_idx in range(num_repeat):
                downstream_sample = downstream_samples[
                    _downstream_flat_index(
                        num_experiences=num_experiences,
                        num_repeat=num_repeat,
                        k_idx=k_idx,
                        n_idx=n_idx,
                        repeat_idx=repeat_idx,
                    )
                ]
                reward = downstream_sample.reward
                total_reward += float(reward) if isinstance(reward, (int, float)) else 0.0
                total_won += float(bool(ensure_metadata(downstream_sample).get("won", False)))
        per_exp_solver_reward[n_idx] = total_reward / denom
        per_exp_win_rate[n_idx] = total_won / denom

    for n_idx, extractor_sample in enumerate(extractor_samples):
        metadata = ensure_metadata(extractor_sample)
        format_score = float(metadata.get("format_score", 1.0) or 0.0)
        audit_applied = float(metadata.get("skill_language_audit_applied", 0.0) or 0.0) > 0.5
        audit_pass = float(metadata.get("skill_language_audit_pass", 0.0) or 0.0) > 0.5
        audit_reward_zeroed = float(
            metadata.get("skill_language_audit_reward_zeroed", 0.0) or 0.0
        ) > 0.5
        base_reward = per_exp_win_rate[n_idx]
        reward_before_audit = base_reward if format_score >= 1.0 else 0.0
        metadata["format_reward_mode"] = "strict_zero"
        metadata["extractor_reward_source"] = "won"
        metadata["reward_zeroed_by_format"] = 0.0 if format_score >= 1.0 else 1.0
        metadata["raw_mean_downstream_solver_reward"] = per_exp_solver_reward[n_idx]
        metadata["raw_extractor_base_reward"] = base_reward
        metadata["downstream_win_rate"] = base_reward
        metadata["raw_downstream_win_reward"] = base_reward
        metadata["downstream_eval_repeats"] = num_repeat
        metadata["raw_extractor_reward_before_language_audit"] = reward_before_audit
        metadata["reward_zeroed_by_language_audit"] = 1.0 if (
            audit_reward_zeroed or (audit_applied and not audit_pass)
        ) else 0.0
        extractor_sample.reward = (
            0.0 if metadata["reward_zeroed_by_language_audit"] > 0.5 else reward_before_audit
        )
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


def _compute_extractor_group_filter_stats(
    downstream_samples: list[Sample],
    *,
    num_experiences: int,
    num_tasks: int,
    num_repeat: int,
) -> dict:
    if num_experiences < 2 or num_tasks <= 0:
        return {
            "keep": True,
            "reason": "insufficient_candidates",
            "top1_index": -1,
            "top2_index": -1,
            "top1_mean_won": 0.0,
            "top2_mean_won": 0.0,
            "margin": 0.0,
            "margin_se": 0.0,
            "margin_z": float("inf"),
            "tau": _EXTRACTOR_GROUP_MARGIN_Z_THRESHOLD,
        }

    per_skill_wins: list[list[float]] = [[] for _ in range(num_experiences)]
    for k_idx in range(num_tasks):
        for n_idx in range(num_experiences):
            for repeat_idx in range(num_repeat):
                downstream_sample = downstream_samples[
                    _downstream_flat_index(
                        num_experiences=num_experiences,
                        num_repeat=num_repeat,
                        k_idx=k_idx,
                        n_idx=n_idx,
                        repeat_idx=repeat_idx,
                    )
                ]
                won = 1.0 if ensure_metadata(downstream_sample).get("won", False) else 0.0
                per_skill_wins[n_idx].append(won)

    mean_wins: list[float] = []
    mean_win_vars: list[float] = []
    for skill_wins in per_skill_wins:
        if not skill_wins:
            mean_wins.append(0.0)
            mean_win_vars.append(0.0)
            continue
        p_hat = sum(skill_wins) / len(skill_wins)
        # Treat repeated downstream evaluations as i.i.d. Bernoulli trials when
        # estimating the standard error of the mean downstream win rate.
        mean_wins.append(p_hat)
        mean_win_vars.append((p_hat * (1.0 - p_hat)) / float(len(skill_wins)))

    order = sorted(
        range(num_experiences),
        key=lambda idx: (mean_wins[idx], -idx),
        reverse=True,
    )
    top1_index = order[0]
    top2_index = order[1]
    top1_mean_won = mean_wins[top1_index]
    top2_mean_won = mean_wins[top2_index]
    margin = top1_mean_won - top2_mean_won
    margin_se = math.sqrt(max(mean_win_vars[top1_index] + mean_win_vars[top2_index], 0.0))
    margin_z = margin / (margin_se + _FILTER_EPS)
    keep = margin_z >= _EXTRACTOR_GROUP_MARGIN_Z_THRESHOLD

    return {
        "keep": keep,
        "reason": "passed" if keep else "margin_below_threshold",
        "top1_index": top1_index,
        "top2_index": top2_index,
        "top1_mean_won": top1_mean_won,
        "top2_mean_won": top2_mean_won,
        "margin": margin,
        "margin_se": margin_se,
        "margin_z": margin_z,
        "tau": _EXTRACTOR_GROUP_MARGIN_Z_THRESHOLD,
    }


def _annotate_extractor_group_filter(
    extractor_samples: list[Sample],
    downstream_samples: list[Sample],
    *,
    filter_stats: dict,
) -> None:
    for extractor_sample in extractor_samples:
        metadata = ensure_metadata(extractor_sample)
        metadata["extractor_group_filter_keep"] = 1.0 if filter_stats["keep"] else 0.0
        metadata["extractor_group_filter_reason"] = str(filter_stats["reason"])
        metadata["extractor_group_filter_tau"] = float(filter_stats["tau"])
        metadata["extractor_group_filter_top1_index"] = int(filter_stats["top1_index"])
        metadata["extractor_group_filter_top2_index"] = int(filter_stats["top2_index"])
        metadata["extractor_group_filter_top1_mean_won"] = float(filter_stats["top1_mean_won"])
        metadata["extractor_group_filter_top2_mean_won"] = float(filter_stats["top2_mean_won"])
        metadata["extractor_group_filter_margin"] = float(filter_stats["margin"])
        metadata["extractor_group_filter_margin_se"] = float(filter_stats["margin_se"])
        metadata["extractor_group_filter_margin_z"] = float(filter_stats["margin_z"])

    for downstream_sample in downstream_samples:
        metadata = ensure_metadata(downstream_sample)
        metadata["extractor_group_filter_keep"] = 1.0 if filter_stats["keep"] else 0.0
        metadata["extractor_group_filter_reason"] = str(filter_stats["reason"])
        metadata["extractor_group_filter_tau"] = float(filter_stats["tau"])
        metadata["extractor_group_filter_margin"] = float(filter_stats["margin"])
        metadata["extractor_group_filter_margin_se"] = float(filter_stats["margin_se"])
        metadata["extractor_group_filter_margin_z"] = float(filter_stats["margin_z"])


def _compute_group_mean_success_filter_stats(
    extractor_samples: list[Sample],
) -> dict:
    if not extractor_samples:
        return {
            "keep": True,
            "reason": "empty_group",
            "group_mean_success_rate": 0.0,
            "threshold": _EXTRACTOR_GROUP_MAX_MEAN_SUCCESS_RATE,
        }

    success_rates = [
        float(ensure_metadata(sample).get("downstream_win_rate", 0.0))
        for sample in extractor_samples
    ]
    group_mean_success_rate = sum(success_rates) / float(len(success_rates))
    keep = group_mean_success_rate <= _EXTRACTOR_GROUP_MAX_MEAN_SUCCESS_RATE

    return {
        "keep": keep,
        "reason": "passed" if keep else "group_mean_success_rate_above_threshold",
        "group_mean_success_rate": group_mean_success_rate,
        "threshold": _EXTRACTOR_GROUP_MAX_MEAN_SUCCESS_RATE,
    }


def _annotate_group_mean_success_filter(
    extractor_samples: list[Sample],
    downstream_samples: list[Sample],
    *,
    filter_stats: dict,
) -> None:
    for extractor_sample in extractor_samples:
        metadata = ensure_metadata(extractor_sample)
        metadata["extractor_group_mean_success_filter_keep"] = (
            1.0 if filter_stats["keep"] else 0.0
        )
        metadata["extractor_group_mean_success_filter_reason"] = str(filter_stats["reason"])
        metadata["extractor_group_mean_success_filter_threshold"] = float(
            filter_stats["threshold"]
        )
        metadata["extractor_group_mean_success_filter_group_mean_success_rate"] = float(
            filter_stats["group_mean_success_rate"]
        )

    for downstream_sample in downstream_samples:
        metadata = ensure_metadata(downstream_sample)
        metadata["extractor_group_mean_success_filter_keep"] = (
            1.0 if filter_stats["keep"] else 0.0
        )
        metadata["extractor_group_mean_success_filter_reason"] = str(filter_stats["reason"])
        metadata["extractor_group_mean_success_filter_threshold"] = float(
            filter_stats["threshold"]
        )
        metadata["extractor_group_mean_success_filter_group_mean_success_rate"] = float(
            filter_stats["group_mean_success_rate"]
        )


def filter_high_mean_success_samples(
    extractor_samples: list[Sample],
    downstream_samples: list[Sample],
    *,
    num_experiences: int,
    num_tasks: int,
    num_repeat: int,
) -> list[Sample]:
    del num_experiences, num_tasks, num_repeat

    filter_stats = _compute_group_mean_success_filter_stats(extractor_samples)
    _annotate_group_mean_success_filter(
        extractor_samples,
        downstream_samples,
        filter_stats=filter_stats,
    )

    if filter_stats["keep"]:
        print(
            "[alfworld/group_mean_success_filter] keep "
            f"mean_success_rate={filter_stats['group_mean_success_rate']:.4f} "
            f"threshold={filter_stats['threshold']:.4f}"
        )
        return extractor_samples

    print(
        "[alfworld/group_mean_success_filter] drop extractor group "
        f"mean_success_rate={filter_stats['group_mean_success_rate']:.4f} "
        f"threshold={filter_stats['threshold']:.4f}"
    )
    return []


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
