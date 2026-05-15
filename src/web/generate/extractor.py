"""Extractor helpers for the Mind2Web-backed web stage training."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

import numpy as np
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from src.web.generate.episode import run_episode_trace
from src.web.prompts import (
    format_extractor_messages,
    skill_json_to_markdown,
)
from env.web.mind2web_env import (
    build_step_observation,
    expected_refs_for_step,
    normalize_model_action,
    resolve_task,
)
from src.web.utils.common import (
    ensure_metadata,
    ensure_train_placeholder_sample,
    maybe_print_random_sample,
    require_arg,
)


SKILL_TRAJECTORY_TOKEN_LIMIT = 25600
SKILL_SNAPSHOT_CONTEXT_RADIUS = 5
STEP_HEADER_PATTERN = re.compile(r"\n\n\[Step (?P<step>\d+)\]\n")
SNAPSHOT_REF_PATTERN = re.compile(r"\[ref=(e\d+)\]")


# ---------------------------------------------------------------------------
# Trajectory compression helpers
# ---------------------------------------------------------------------------


def _compress_trajectory_observations(
    trajectory: str,
    *,
    task_payload: dict | None,
) -> str:
    if not trajectory or not task_payload:
        return trajectory

    spans = _split_step_blocks(trajectory)
    if not spans:
        return trajectory

    rebuilt_parts: list[str] = []
    cursor = 0
    for step_index, start, end in spans:
        rebuilt_parts.append(trajectory[cursor:start])
        block = trajectory[start:end]
        snapshot_header_match = re.search(
            r"Current page snapshot \(source=.*?\):\n",
            block,
            re.DOTALL,
        )
        if snapshot_header_match is None:
            rebuilt_parts.append(block)
            cursor = end
            continue

        snapshot_start = snapshot_header_match.end()
        snapshot_end = block.find("\n\nPrevious actions:\n", snapshot_start)
        if snapshot_end == -1:
            rebuilt_parts.append(block)
            cursor = end
            continue

        snapshot_text = block[snapshot_start:snapshot_end]
        try:
            obs = build_step_observation(task_payload, step_index)
            gt_refs = set(expected_refs_for_step(task_payload, step_index, obs["refs"]))
        except Exception:
            gt_refs = set()

        agent_refs = set(_extract_action_refs_from_block(block))
        filtered_snapshot = _filter_snapshot_lines(snapshot_text, agent_refs | gt_refs)
        rebuilt_parts.append(block[:snapshot_start])
        rebuilt_parts.append(filtered_snapshot)
        rebuilt_parts.append(block[snapshot_end:])
        cursor = end

    rebuilt_parts.append(trajectory[cursor:])
    return "".join(rebuilt_parts)


def _maybe_compact_trajectory_for_skill_prompt(
    *,
    tokenizer,
    messages_builder,
    trajectory_text: str,
    task_locator: dict | None,
    builder_kwargs: dict,
) -> list[dict]:
    messages = messages_builder(**builder_kwargs)
    if _token_length(tokenizer, messages) <= SKILL_TRAJECTORY_TOKEN_LIMIT:
        return messages

    task_payload = _resolve_skill_trajectory_task(task_locator)
    compressed_trajectory = _compress_trajectory_observations(
        trajectory_text,
        task_payload=task_payload,
    )
    if compressed_trajectory == trajectory_text:
        return messages

    compact_kwargs = dict(builder_kwargs)
    if "trajectory" in compact_kwargs:
        compact_kwargs["trajectory"] = compressed_trajectory
    return messages_builder(**compact_kwargs)


def _filter_snapshot_lines(snapshot_text: str, keep_refs: set[str]) -> str:
    lines = str(snapshot_text or "").splitlines()
    if not lines or not keep_refs:
        return snapshot_text

    ref_line_indices: dict[str, int] = {}
    for idx, line in enumerate(lines):
        match = SNAPSHOT_REF_PATTERN.search(line)
        if match is not None:
            ref_line_indices[match.group(1)] = idx

    target_indices = sorted(
        {
            ref_line_indices[ref]
            for ref in keep_refs
            if ref in ref_line_indices
        }
    )
    if not target_indices:
        return snapshot_text

    keep_indices: set[int] = set()
    for idx in target_indices:
        start = max(0, idx - SKILL_SNAPSHOT_CONTEXT_RADIUS)
        end = min(len(lines), idx + SKILL_SNAPSHOT_CONTEXT_RADIUS + 1)
        keep_indices.update(range(start, end))

    ordered_keep = sorted(keep_indices)
    if len(ordered_keep) == len(lines):
        return snapshot_text

    filtered_lines: list[str] = []
    prev_idx = -1
    for idx in ordered_keep:
        if prev_idx >= 0 and idx - prev_idx > 1:
            omitted = idx - prev_idx - 1
            filtered_lines.append(f"- ... omitted {omitted} unrelated elements ...")
        filtered_lines.append(lines[idx])
        prev_idx = idx
    return "\n".join(filtered_lines)


def _split_step_blocks(trajectory: str) -> list[tuple[int, int, int]]:
    matches = list(STEP_HEADER_PATTERN.finditer(str(trajectory or "")))
    if not matches:
        return []

    spans: list[tuple[int, int, int]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(trajectory)
        spans.append((int(match.group("step")), start, end))
    return spans


def _extract_action_refs_from_block(block_text: str) -> list[str]:
    refs: list[str] = []
    for action_text in re.findall(r"<action>(.*?)</action>", str(block_text or ""), re.IGNORECASE | re.DOTALL):
        parsed = normalize_model_action(action_text)
        ref = str(parsed.get("ref") or "").strip()
        if ref:
            refs.append(ref)
    return refs


def _resolve_skill_trajectory_task(task_locator: dict | None) -> dict | None:
    payload = _coerce_mapping(task_locator)
    if not payload:
        return None
    source_file = payload.get("source_file") or payload.get("task_source_file")
    task_index = payload.get("task_index")
    if source_file in (None, "") or task_index in (None, ""):
        return None
    try:
        return resolve_task(payload)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _coerce_mapping(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _is_nonempty_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    return True


def _first_nonempty(*values):
    for value in values:
        if _is_nonempty_value(value):
            return value
    return None


def _token_length(tokenizer, messages: list[dict]) -> int:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])


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


# ---------------------------------------------------------------------------
# Task info helpers
# ---------------------------------------------------------------------------


def task_info_to_sample(task_info: dict) -> Sample:
    downstream = Sample()
    label = _coerce_mapping(task_info.get("label"))
    label.update(_coerce_mapping(task_info.get("metadata")))

    for key in [
        "annotation_id",
        "task_description",
        "task_type",
        "website",
        "domain",
        "subdomain",
        "source_file",
        "task_index",
    ]:
        value = task_info.get(key)
        if value is not None and value != "":
            label[key] = value

    if "task_description" not in label:
        label["task_description"] = str(
            _first_nonempty(
                task_info.get("question"),
                task_info.get("prompt"),
                task_info.get("confirmed_task"),
            )
            or ""
        )

    downstream.label = label
    downstream.prompt = _first_nonempty(
        task_info.get("prompt"),
        task_info.get("question"),
        label.get("task_description"),
    )
    downstream.id = task_info.get("id") or label.get("annotation_id")
    return downstream


def resolve_task_description(label: dict | None) -> str:
    coerced = _coerce_mapping(label)
    return str(
        coerced.get("task_description")
        or coerced.get("confirmed_task")
        or coerced.get("prompt")
        or ""
    )


def resolve_task_type(label: dict | None) -> str:
    coerced = _coerce_mapping(label)
    return str(
        coerced.get("task_type")
        or coerced.get("domain")
        or coerced.get("website")
        or ""
    )


# ---------------------------------------------------------------------------
# Skill output parsing
# ---------------------------------------------------------------------------


def parse_default_skill_output(args, response_text: str) -> dict:
    skill_markdown, error_msg = skill_json_to_markdown(response_text)
    skill_format_penalty = abs(float(require_arg(args, "skill_format_penalty")))
    if skill_markdown is None:
        return {
            "skill": str(response_text or "").strip(),
            "format_penalty": -skill_format_penalty,
            "format_score": 0.0,
            "skill_parse_error": error_msg,
        }
    return {
        "skill": skill_markdown,
        "format_penalty": 0.0,
        "format_score": 1.0,
    }


# ---------------------------------------------------------------------------
# Extractor sample generation
# ---------------------------------------------------------------------------


def _make_failed_extractor_sample(
    *,
    n_idx: int,
    status,
    requested_mode: str,
    actual_mode: str,
    base_metadata: dict | None = None,
) -> Sample:
    sample = Sample(group_index=0, index=n_idx, status=status)
    sample.reward = 0.0
    sample.tokens = []
    sample.loss_mask = []
    sample.response = ""
    sample.response_length = 0
    metadata = ensure_metadata(sample)
    metadata.update(base_metadata or {})
    metadata.setdefault("skill", "")
    metadata.setdefault("format_penalty", 0.0)
    metadata["requested_skill_source_mode"] = requested_mode
    metadata["skill_source_mode"] = actual_mode
    return ensure_train_placeholder_sample(
        sample,
        reason=f"extractor_{actual_mode}_{getattr(status, 'value', str(status))}",
    )


async def generate_extractor_sample(
    args,
    *,
    messages: list[dict],
    sampling_params: dict,
    n_idx: int,
    requested_mode: str = "generate",
    actual_mode: str = "generate",
    parser_fn=None,
    base_metadata: dict | None = None,
) -> Sample:
    parser_fn = parser_fn or parse_default_skill_output

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    try:
        output = await post(url, {"text": prompt_text, "sampling_params": sampling_params})
    except Exception as exc:
        print(
            f"[web/generate_extractor_sample] POST error "
            f"({actual_mode}, n={n_idx}): {exc}"
        )
        return _make_failed_extractor_sample(
            n_idx=n_idx,
            status=Sample.Status.FAILED,
            requested_mode=requested_mode,
            actual_mode=actual_mode,
            base_metadata=base_metadata,
        )

    finish_type = output["meta_info"]["finish_reason"]["type"]
    if finish_type == "abort":
        return _make_failed_extractor_sample(
            n_idx=n_idx,
            status=Sample.Status.ABORTED,
            requested_mode=requested_mode,
            actual_mode=actual_mode,
            base_metadata=base_metadata,
        )

    response_text = output["text"]
    response_token_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    parsed_metadata = dict(base_metadata or {})
    try:
        parsed_metadata.update(parser_fn(args, response_text))
    except Exception as exc:
        print(
            f"[web/generate_extractor_sample] parse error "
            f"({actual_mode}, n={n_idx}): {exc}"
        )
        parsed_metadata.update(
            {
                "skill": str(response_text or "").strip(),
                "format_penalty": -abs(float(require_arg(args, "skill_format_penalty"))),
                "format_score": 0.0,
                "skill_parse_error": str(exc),
            }
        )

    sample = Sample()
    sample.prompt = prompt_text
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response = response_text
    sample.response_length = len(response_token_ids)
    sample.loss_mask = [1] * len(response_token_ids)
    sample.group_index = 0
    sample.index = n_idx
    sample.status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.COMPLETED

    metadata = ensure_metadata(sample)
    metadata.update(parsed_metadata)
    metadata["requested_skill_source_mode"] = requested_mode
    metadata["skill_source_mode"] = actual_mode
    return ensure_train_placeholder_sample(
        sample,
        reason=f"extractor_{actual_mode}_generated",
    )


async def generate_extractor_sample_from_trajectory(
    args,
    *,
    task_description: str,
    trajectory: str,
    won_score: float,
    sampling_params: dict,
    n_idx: int,
    parser_fn=None,
    base_metadata: dict | None = None,
    trajectory_task_label: dict | None = None,
) -> Sample:
    tokenizer = GenerateState(args).tokenizer
    messages = _maybe_compact_trajectory_for_skill_prompt(
        tokenizer=tokenizer,
        messages_builder=format_extractor_messages,
        trajectory_text=trajectory,
        task_locator=trajectory_task_label,
        builder_kwargs={
            "task_description": task_description,
            "trajectory": trajectory,
            "won_score": won_score,
        },
    )
    return await generate_extractor_sample(
        args,
        messages=messages,
        sampling_params=sampling_params,
        n_idx=n_idx,
        requested_mode="generate",
        actual_mode="generate",
        parser_fn=parser_fn or parse_default_skill_output,
        base_metadata=base_metadata,
    )


async def build_extractor_sample(
    args,
    *,
    task_description: str,
    trajectory: str,
    won_score: float,
    sampling_params: dict,
    n_idx: int,
    parser_fn=None,
    base_metadata: dict | None = None,
    trajectory_task_label: dict | None = None,
) -> Sample:
    return await generate_extractor_sample_from_trajectory(
        args,
        task_description=task_description,
        trajectory=trajectory,
        won_score=won_score,
        sampling_params=sampling_params,
        n_idx=n_idx,
        parser_fn=parser_fn or parse_default_skill_output,
        base_metadata=base_metadata,
        trajectory_task_label=trajectory_task_label,
    )


# ---------------------------------------------------------------------------
# Downstream evaluation
# ---------------------------------------------------------------------------


async def evaluate_downstream(
    args,
    *,
    task_info: dict,
    skill_sample: Sample,
    source_task_description: str,
    source_task_type: str,
    n_idx: int,
    k_idx: int,
    sampling_params: dict,
    training_stage: str,
    message_builder=None,
    solver_reward_fn=None,
) -> dict:
    downstream_sample = task_info_to_sample(task_info)
    downstream_label = downstream_sample.label if isinstance(downstream_sample.label, dict) else {}
    downstream_task_description = resolve_task_description(downstream_label)
    downstream_task_type = resolve_task_type(downstream_label)
    skill_metadata = skill_sample.metadata if isinstance(skill_sample.metadata, dict) else {}
    skill_text = str(skill_metadata.get("skill") or skill_sample.response or "").strip()
    workflow_expected_labels = None

    try:
        trace = await run_episode_trace(
            args,
            sample=downstream_sample,
            skill=skill_text if skill_text else None,
            sampling_params=sampling_params,
            evaluation=False,
            message_builder=message_builder,
            workflow_expected_labels=workflow_expected_labels,
        )
    except Exception as exc:
        print(
            f"[web/evaluate_downstream] error "
            f"({training_stage}, k={k_idx}, n={n_idx}): {exc}"
        )
        failed = Sample(status=Sample.Status.FAILED)
        failed.reward = 0.0
        failed.tokens = []
        failed.loss_mask = []
        failed.response = ""
        failed.label = downstream_label
        failed.metadata = {
            "task_description": downstream_task_description,
            "won": False,
            "done": False,
            "num_steps": 0,
            "episode_won": False,
            "episode_done": False,
            "action_accuracy": 0.0,
            "sample_granularity": "step",
        }
        failed = ensure_train_placeholder_sample(
            failed,
            reason=f"downstream_exception:{training_stage}",
        )
        trace = {
            "episode_sample": failed,
            "step_samples": [failed],
            "trajectory": "",
            "won": False,
            "done": False,
            "task_description": downstream_task_description,
            "website": str(downstream_label.get("website") or ""),
            "domain": str(downstream_label.get("domain") or ""),
        }

    episode_sample = trace["episode_sample"]
    episode_metadata = ensure_metadata(episode_sample)
    step_samples = list(trace.get("step_samples") or [])
    won = bool(episode_metadata.get("won", False))
    episode_reward = (
        sum(float(sample.reward) for sample in step_samples if isinstance(sample.reward, (int, float)))
        / len(step_samples)
        if step_samples
        else 0.0
    )
    episode_aux_reward = (
        solver_reward_fn(episode_sample, skill_sample, args)
        if solver_reward_fn is not None
        else episode_reward
    )

    workflow_keys = [
        "workflow_validation_source",
        "workflow_require_full_coverage",
        "workflow_expected_labels",
        "workflow_step_count",
        "workflow_compliance",
        "workflow_observed_labels",
        "workflow_observed_error_labels",
        "workflow_missing_labels",
        "workflow_covers_all_labels",
        "workflow_reward_reason",
        "reasoning_count",
        "workflow_reasoning_close_count",
        "action_count",
        "workflow_action_close_count",
        "workflow_tagged_reasoning_count",
        "has_error_handling_label",
        "workflow_turn_validation",
        "workflow_follow_reward",
        "workflow_success_bonus",
        "workflow_reward_zeroed_by_format",
        "workflow_missing_error_handling_penalty",
    ]

    for sample in step_samples:
        sample.group_index = k_idx + 1
        sample.index = n_idx

        metadata = ensure_metadata(sample)
        metadata["training_stage"] = training_stage
        metadata["skill_text"] = skill_text
        metadata["source_task_description"] = str(source_task_description or "")
        metadata["source_task_type"] = str(source_task_type or "")
        metadata["downstream_task_description"] = str(
            episode_metadata.get("task_description") or downstream_task_description or ""
        )
        metadata["downstream_task_type"] = str(downstream_task_type or "")
        metadata["requested_skill_source_mode"] = str(
            skill_metadata.get("requested_skill_source_mode") or ""
        )
        metadata["skill_source_mode"] = str(skill_metadata.get("skill_source_mode") or "")
        metadata["episode_reward_mean"] = float(episode_reward)
        metadata["episode_aux_reward"] = float(episode_aux_reward)
        metadata["raw_env_reward"] = float(sample.reward) if isinstance(sample.reward, (int, float)) else 0.0
        metadata["raw_training_reward"] = float(sample.reward) if isinstance(sample.reward, (int, float)) else 0.0
        metadata["episode_won"] = won
        metadata["won"] = won
        metadata["episode_done"] = bool(episode_metadata.get("done", False))
        metadata["done"] = bool(episode_metadata.get("done", False))
        for audit_key in [
            "skill_language_audit_enabled",
            "skill_language_audit_applied",
            "skill_language_audit_pass",
            "skill_language_audit_has_uncommon_characters",
            "skill_language_audit_has_non_english_characters",
            "skill_language_audit_has_repetition",
            "skill_language_audit_penalty",
            "skill_language_audit_status",
            "skill_language_audit_judgment",
            "skill_language_audit_reason",
            "skill_language_audit_reward_zeroed",
            "reward_zeroed_by_language_audit",
            "raw_extractor_reward_before_language_audit",
        ]:
            if audit_key in skill_metadata:
                metadata[audit_key] = skill_metadata[audit_key]

        if "workflow_labels" in skill_metadata:
            metadata["workflow_labels"] = list(skill_metadata.get("workflow_labels") or [])
            metadata["workflow_step_count"] = len(metadata["workflow_labels"])
        for key in workflow_keys:
            if key in episode_metadata:
                metadata[key] = deepcopy(episode_metadata[key])

    return {
        "samples": step_samples,
        "won": won,
        "done": bool(episode_metadata.get("done", False)),
        "episode_reward": float(episode_reward),
        "episode_aux_reward": float(episode_aux_reward),
        "num_correct_steps": int(episode_metadata.get("num_correct_steps", 0) or 0),
        "trajectory": str(trace.get("trajectory") or episode_sample.response or ""),
        "task_description": str(
            episode_metadata.get("task_description") or downstream_task_description or ""
        ),
        "task_type": str(downstream_task_type or ""),
        "skill_text": skill_text,
        "skill_source_mode": str(skill_metadata.get("skill_source_mode") or ""),
        "requested_skill_source_mode": str(
            skill_metadata.get("requested_skill_source_mode") or ""
        ),
        "source_file": str(downstream_label.get("source_file") or ""),
        "task_index": downstream_label.get("task_index"),
        "k_idx": int(k_idx),
        "n_idx": int(n_idx),
    }


def mark_source_episode_won(
    extractor_samples: list[Sample],
    *,
    won_score: float,
    training_stage: str,
    source_task_description: str,
    source_task_type: str,
) -> None:
    for sample in extractor_samples:
        metadata = ensure_metadata(sample)
        metadata["training_stage"] = training_stage
        metadata["source_episode_won"] = float(bool(won_score))
        metadata["source_task_description"] = source_task_description
        metadata["source_task_type"] = source_task_type
