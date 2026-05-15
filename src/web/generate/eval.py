"""Web evaluation pipeline: skill-conditioned episode evaluation.

Retrieves a task semantically similar to the eval sample, extracts a skill,
then evaluates the target with and without that skill.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

from slime.utils.types import Sample

from src.web.generate.episode import run_episode, run_episode_trace
from src.web.generate.retrieval import retrieve_tasks
from src.web.generate.extractor import (
    generate_extractor_sample_from_trajectory,
    resolve_task_description,
    task_info_to_sample,
)
from src.web.utils.common import maybe_print_random_sample, require_arg


def _prompt_to_text(prompt: Any) -> str:
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, dict):
        return str(prompt.get("content") or "").strip()
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                content = str(item.get("content") or "").strip()
                if content:
                    parts.append(content)
            else:
                content = str(item or "").strip()
                if content:
                    parts.append(content)
        return "\n".join(parts).strip()
    return str(prompt).strip()


def _episode_metrics(metadata: dict[str, Any]) -> dict[str, float]:
    num_steps = int(metadata.get("num_steps", 0) or 0)
    num_correct_steps = int(metadata.get("num_correct_steps", 0) or 0)
    action_accuracy = float(metadata.get("action_accuracy", 0.0) or 0.0)
    if num_steps > 0:
        action_accuracy = num_correct_steps / float(num_steps)
    return {
        "won": 1.0 if metadata.get("won", False) else 0.0,
        "num_steps": float(num_steps),
        "num_correct_steps": float(num_correct_steps),
        "action_accuracy": float(action_accuracy),
    }


async def _extract_skill_from_task_info(
    args,
    task_info: dict,
    sampling_params: dict,
    log_tag: str = "web_eval_pipeline",
) -> str:
    retrieved_sample = task_info_to_sample(task_info)
    try:
        src_trace = await run_episode_trace(
            args,
            sample=retrieved_sample,
            skill=None,
            sampling_params=sampling_params,
        )
    except Exception as exc:
        print(f"[{log_tag}] Solver error on retrieved task: {exc}")
        return ""

    src_episode = src_trace["episode_sample"]
    src_metadata = src_episode.metadata if isinstance(src_episode.metadata, dict) else {}
    retrieved_task_desc = src_metadata.get("task_description", "")
    trajectory = str(src_trace.get("trajectory") or src_episode.response or "")
    won_score = 1.0 if src_metadata.get("won", False) else 0.0

    try:
        exp_sample = await generate_extractor_sample_from_trajectory(
            args,
            task_description=retrieved_task_desc,
            trajectory=trajectory,
            won_score=won_score,
            sampling_params=sampling_params,
            n_idx=0,
            base_metadata={"training_stage": "eval"},
            trajectory_task_label=src_episode.label if isinstance(src_episode.label, dict) else None,
        )
        return exp_sample.metadata.get("skill", exp_sample.response or "")
    except Exception as exc:
        print(f"[{log_tag}] Extractor error: {exc}")
        return ""


async def _eval_with_skill_text(
    args,
    sample: Sample,
    sampling_params: dict,
    skill_text: str,
    skill_meta_suffix: str,
) -> Sample:
    with_sample = deepcopy(sample)
    no_sample = deepcopy(sample)

    try:
        if skill_text:
            with_result, no_result = await asyncio.gather(
                run_episode(
                    args,
                    sample=with_sample,
                    skill=skill_text,
                    sampling_params=sampling_params,
                    evaluation=True,
                ),
                run_episode(
                    args,
                    sample=no_sample,
                    skill=None,
                    sampling_params=sampling_params,
                    evaluation=True,
                ),
            )
        else:
            no_result = await run_episode(
                args,
                sample=no_sample,
                skill=None,
                sampling_params=sampling_params,
                evaluation=True,
            )
            with_result = no_result
    except Exception as exc:
        print(f"[web/eval_pipeline] Episode error ({skill_meta_suffix}): {exc}")
        sample.reward = 0.0
        sample.tokens = []
        sample.loss_mask = []
        sample.response = ""
        sample.metadata = {
            f"won_{skill_meta_suffix}": 0.0,
            "won_no_skill": 0.0,
            f"num_steps_{skill_meta_suffix}": 0,
            "num_steps_no_skill": 0,
            f"num_correct_steps_{skill_meta_suffix}": 0,
            "num_correct_steps_no_skill": 0,
            f"action_accuracy_{skill_meta_suffix}": 0.0,
            "action_accuracy_no_skill": 0.0,
        }
        return sample

    with_metadata = with_result.metadata if isinstance(with_result.metadata, dict) else {}
    no_metadata = no_result.metadata if isinstance(no_result.metadata, dict) else {}
    with_metrics = _episode_metrics(with_metadata)
    no_metrics = _episode_metrics(no_metadata)
    won_with = with_metrics["won"]
    won_no = no_metrics["won"]

    result = with_result
    result.reward = 10.0 * won_with
    if not isinstance(result.metadata, dict):
        result.metadata = {}
    result.metadata[f"won_{skill_meta_suffix}"] = won_with
    result.metadata["won_no_skill"] = won_no
    result.metadata[f"num_steps_{skill_meta_suffix}"] = int(with_metrics["num_steps"])
    result.metadata["num_steps_no_skill"] = int(no_metrics["num_steps"])
    result.metadata[f"num_correct_steps_{skill_meta_suffix}"] = int(
        with_metrics["num_correct_steps"]
    )
    result.metadata["num_correct_steps_no_skill"] = int(no_metrics["num_correct_steps"])
    result.metadata[f"action_accuracy_{skill_meta_suffix}"] = float(
        with_metrics["action_accuracy"]
    )
    result.metadata["action_accuracy_no_skill"] = float(no_metrics["action_accuracy"])
    return result


def _resolve_eval_query(sample: Sample) -> str:
    label = sample.label if isinstance(sample.label, dict) else {}
    query = resolve_task_description(label)
    if query:
        return query
    return _prompt_to_text(sample.prompt) or str(label.get("task_type") or "")


async def run_eval(
    args,
    sample: Sample,
    sampling_params: dict,
) -> Sample:
    retrieve_url_test = require_arg(args, "web_retrieve_url_test")
    query = _resolve_eval_query(sample)

    retrieved = await retrieve_tasks(query=query, K=1, url=retrieve_url_test)
    if retrieved:
        skill_text = await _extract_skill_from_task_info(
            args, retrieved[0], sampling_params, log_tag="web_eval/similar"
        )
    else:
        print(
            f"[web/eval_pipeline] WARNING: /search returned no results for '{query[:60]}'. "
            "Skill will be empty; only baseline episode is run."
        )
        skill_text = ""

    result = await _eval_with_skill_text(
        args, sample, sampling_params, skill_text, skill_meta_suffix="with_skill"
    )

    maybe_print_random_sample(result, tag="eval/with_experience")
    return result
