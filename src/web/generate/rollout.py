"""Rollout: experience training with web-skill extraction."""

from __future__ import annotations

import asyncio
from copy import deepcopy

from slime.utils.types import Sample

from src.web.generate.episode import run_episode_trace
from src.web.generate.skill_audit import audit_extractor_samples
from src.web.generate.retrieval import retrieve_tasks
from src.web.generate.extractor import (
    build_extractor_sample,
    evaluate_downstream,
    mark_source_episode_won,
    parse_default_skill_output,
    resolve_task_type,
)
from src.web.reward.functions import (
    attach_train_metadata,
    grpo_normalize,
    apply_reward_weights,
    finalize_extractor_rewards,
    normalize_step_rewards,
    apply_language_audit_rewards,
)
from src.web.prompts import format_solver_with_skill_messages
from src.web.utils.common import ensure_metadata, ensure_train_placeholder_sample, require_arg


def _parse_skill_output(args, response_text: str) -> dict:
    return parse_default_skill_output(args, response_text)


async def run_rollout(
    args,
    sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> list:
    num_experiences = int(require_arg(args, "n_experiences"))
    retrieval_topk = int(require_arg(args, "retrieval_topk"))
    retrieve_url_train = require_arg(args, "web_retrieve_url_train")
    label = sample.label if isinstance(sample.label, dict) else {}
    source_task_type = resolve_task_type(label)

    try:
        src_trace = await run_episode_trace(
            args,
            sample=sample,
            skill=None,
            sampling_params=sampling_params,
        )
    except Exception as exc:
        print(f"[web/rollout] source episode error: {exc}")
        failed = Sample(status=Sample.Status.FAILED)
        failed.prompt = sample.prompt
        failed.label = sample.label
        failed.reward = 0.0
        failed.tokens = []
        failed.loss_mask = []
        failed.response = ""
        failed.response_length = 0
        failed.metadata = {
            "training_stage": "experience",
            "task_description": str(label.get("task_description") or ""),
            "won": False,
            "done": False,
            "num_steps": 0,
            "action_accuracy": 0.0,
        }
        return [ensure_train_placeholder_sample(failed, reason="source_episode_error")]

    src_sample = src_trace["episode_sample"]
    src_metadata = ensure_metadata(src_sample)
    trajectory = str(src_trace.get("trajectory") or src_sample.response or "")
    won_score = 1.0 if src_metadata.get("won", False) else 0.0
    task_description = src_metadata.get("task_description", label.get("task_description", ""))

    extractor_samples = list(
        await asyncio.gather(
            *[
                build_extractor_sample(
                    args,
                    task_description=task_description,
                    trajectory=trajectory,
                    won_score=won_score,
                    sampling_params=sampling_params,
                    n_idx=n_idx,
                    parser_fn=_parse_skill_output,
                    base_metadata={"training_stage": "experience"},
                    trajectory_task_label=label,
                )
                for n_idx in range(num_experiences)
            ]
        )
    )
    mark_source_episode_won(
        extractor_samples,
        won_score=won_score,
        training_stage="experience",
        source_task_description=task_description,
        source_task_type=source_task_type,
    )
    skill_language_audit_task = asyncio.create_task(
        audit_extractor_samples(args, extractor_samples)
    )

    downstream_tasks = await retrieve_tasks(
        query=task_description,
        K=retrieval_topk,
        url=retrieve_url_train,
    )

    if not downstream_tasks:
        print(
            "[web/rollout] WARNING: no downstream tasks retrieved. "
            "Extractor samples will receive reward=0."
        )
        await skill_language_audit_task
        for extractor_sample in extractor_samples:
            extractor_sample.reward = 0.0
            metadata = ensure_metadata(extractor_sample)
            metadata["raw_training_reward"] = 0.0
            metadata["raw_extractor_reward_before_language_audit"] = 0.0
            metadata["reward_zeroed_by_language_audit"] = 1.0 if (
                float(metadata.get("skill_language_audit_reward_zeroed", 0.0) or 0.0) > 0.5
            ) else 0.0
            metadata["extractor_reward_metric"] = "num_correct_steps"
            metadata["raw_mean_downstream_num_correct_steps"] = 0.0
            metadata["downstream_win_rate"] = 0.0
            metadata["raw_downstream_win_reward"] = 0.0
        attach_train_metadata(extractor_samples, sample_role="extractor")
        if not evaluation:
            grpo_normalize(extractor_samples)
        return extractor_samples

    solver_sampling_params = deepcopy(sampling_params)
    solver_sampling_params["temperature"] = float(require_arg(args, "solver_temperature"))

    downstream_results = list(
        await asyncio.gather(
            *[
                evaluate_downstream(
                    args,
                    task_info=downstream_tasks[k_idx],
                    skill_sample=extractor_samples[n_idx],
                    source_task_description=task_description,
                    source_task_type=source_task_type,
                    n_idx=n_idx,
                    k_idx=k_idx,
                    sampling_params=solver_sampling_params,
                    training_stage="experience",
                    message_builder=format_solver_with_skill_messages,
                )
                for k_idx in range(len(downstream_tasks))
                for n_idx in range(num_experiences)
            ]
        )
    )
    await skill_language_audit_task
    downstream_samples = [
        step_sample
        for result in downstream_results
        for step_sample in result.get("samples", [])
    ]

    finalize_extractor_rewards(
        extractor_samples,
        downstream_results,
        num_experiences=num_experiences,
        num_tasks=len(downstream_tasks),
        format_reward_mode="strict_zero",
        reward_key="num_correct_steps",
        reward_metadata_key="raw_mean_downstream_num_correct_steps",
        reward_metric_name="num_correct_steps",
    )
    apply_language_audit_rewards(extractor_samples)

    if not evaluation:
        grpo_normalize(extractor_samples)
        normalize_step_rewards(downstream_samples)

    apply_reward_weights(
        args,
        extractor_samples=extractor_samples,
        downstream_samples=downstream_samples,
    )
    attach_train_metadata(extractor_samples, sample_role="extractor")
    attach_train_metadata(downstream_samples, sample_role="downstream_solver")
    return extractor_samples + downstream_samples
