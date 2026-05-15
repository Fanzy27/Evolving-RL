"""Mind2Web episode runner with per-step training samples and task-level eval traces."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from src.web.prompts import (
    format_solver_messages,
    format_solver_with_skill_messages,
)
from src.web.utils.common import (
    ensure_train_placeholder_sample,
    maybe_print_random_env_sample,
    require_arg,
)
from env.web.mind2web_env import normalize_model_action


RETURN_LOGPROB = False
SEMAPHORE: asyncio.Semaphore | None = None
PROMPT_CONTEXT_MODE = "system_plus_task_plus_current_observation_plus_action_history"


def _get_semaphore(args) -> asyncio.Semaphore:
    global SEMAPHORE
    if SEMAPHORE is None:
        SEMAPHORE = asyncio.Semaphore(int(require_arg(args, "web_concurrency")))
    return SEMAPHORE


def _resolve_context_limit(args, evaluation: bool) -> int:
    if evaluation and hasattr(args, "eval_max_context_len") and args.eval_max_context_len is not None:
        return int(args.eval_max_context_len)
    return int(require_arg(args, "rollout_max_context_len"))


def _ensure_stop_string(request_sampling_params: dict, stop_string: str) -> None:
    current_stop = request_sampling_params.get("stop")
    if current_stop is None:
        request_sampling_params["stop"] = [stop_string]
        return
    if isinstance(current_stop, str):
        if current_stop != stop_string:
            request_sampling_params["stop"] = [current_stop, stop_string]
        return
    stop_list = list(current_stop)
    if stop_string not in stop_list:
        stop_list.append(stop_string)
    request_sampling_params["stop"] = stop_list


def _apply_generation_budget(
    args,
    request_sampling_params: dict,
    remaining_context_budget: int,
) -> int:
    solver_max_response_len = int(require_arg(args, "solver_max_response_len"))
    max_new_tokens = min(remaining_context_budget, solver_max_response_len)
    request_sampling_params["max_new_tokens"] = max_new_tokens
    return max_new_tokens


def extract_action(text: str) -> tuple[dict[str, str] | str, bool]:
    text = str(text or "")
    match = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE | re.DOTALL)
    if match is None:
        return text[-200:], False

    action_text = match.group(1).strip()
    parsed = normalize_model_action(action_text)
    is_valid = bool(action_text) and bool(parsed.get("op")) and bool(parsed.get("ref"))
    return (parsed if parsed.get("op") else action_text), is_valid


def format_step_observation(
    task_description: str,
    obs: str,
    previous_actions: list[str],
    action_schema: str,
    html_field: str,
    step_idx: int,
) -> str:
    formatted_previous = "\n".join(previous_actions) if previous_actions else "None"
    return (
        f"\n\n[Step {step_idx}]\n"
        f"Current task:\n{task_description or 'Unknown task'}\n\n"
        f"Current page snapshot (source={html_field or 'unknown'}):\n{obs}\n\n"
        f"Previous actions:\n{formatted_previous}\n\n"
        f"Action format:\n{action_schema}\n"
    )


def _copy_workflow_summary(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in [
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
    ]:
        if key in source:
            target[key] = deepcopy(source[key])


def _build_fallback_step_sample(
    source_sample: Sample,
    *,
    prompt_text: str,
    task_description: str,
    website: str,
    domain: str,
    episode_id: str | None,
    status,
    termination_reason: str,
) -> Sample:
    fallback = Sample(status=status)
    fallback.prompt = prompt_text
    fallback.label = deepcopy(source_sample.label)
    fallback.reward = 0.0
    fallback.tokens = []
    fallback.loss_mask = []
    fallback.response = ""
    fallback.response_length = 0
    fallback.group_index = source_sample.group_index
    fallback.index = source_sample.index
    fallback.metadata = {
        "task_description": task_description,
        "website": website,
        "domain": domain,
        "episode_id": episode_id or "",
        "sample_granularity": "step",
        "prompt_context_mode": PROMPT_CONTEXT_MODE,
        "action_correct": False,
        "raw_step_reward": 0.0,
        "step_won": False,
        "step_done": False,
        "step_index": 0,
        "termination_reason": termination_reason,
    }
    return ensure_train_placeholder_sample(fallback, reason=termination_reason or "fallback_step")


def _build_step_sample(
    source_sample: Sample,
    *,
    prompt_text: str,
    prompt_token_ids: list[int],
    obs_text: str,
    obs_token_ids: list[int],
    gen_text: str,
    gen_token_ids: list[int],
    gen_log_probs: list[float] | None,
    reward: float,
    status,
    step_idx: int,
    episode_id: str | None,
    task_description: str,
    website: str,
    domain: str,
    action: Any,
    action_valid: bool,
    action_correct: bool,
    step_done: bool,
    step_won: bool,
    match_mode: str,
    termination_reason: str,
) -> Sample:
    step_sample = Sample(status=status)
    step_sample.prompt = f"{prompt_text}{obs_text}"
    step_sample.label = deepcopy(source_sample.label)
    step_sample.tokens = list(prompt_token_ids) + list(obs_token_ids) + list(gen_token_ids)
    step_sample.response = gen_text
    step_sample.response_length = len(gen_token_ids)
    step_sample.loss_mask = [1] * len(gen_token_ids)
    step_sample.reward = float(reward)
    step_sample.group_index = source_sample.group_index
    step_sample.index = step_idx
    if RETURN_LOGPROB and gen_log_probs is not None:
        step_sample.rollout_log_probs = list(gen_log_probs)

    step_sample.metadata = {
        "task_description": task_description,
        "website": website,
        "domain": domain,
        "episode_id": episode_id or "",
        "sample_granularity": "step",
        "prompt_context_mode": PROMPT_CONTEXT_MODE,
        "step_index": step_idx,
        "last_action": deepcopy(action),
        "last_action_valid": bool(action_valid),
        "action_correct": bool(action_correct),
        "raw_step_reward": float(reward),
        "raw_env_reward": float(reward),
        "raw_training_reward": float(reward),
        "step_won": bool(step_won),
        "step_done": bool(step_done),
        "match_mode": str(match_mode or "none"),
        "termination_reason": termination_reason,
    }
    return ensure_train_placeholder_sample(
        step_sample,
        reason=termination_reason or "step_sample",
    )


def _finalize_episode_metadata(
    *,
    won: bool,
    env_done: bool,
    task_description: str,
    attempted_steps: int,
    correct_steps: int,
    website: str,
    domain: str,
    last_action: Any,
    last_action_valid: bool,
    workflow_turn_checks: list[dict] | None,
    summarize_workflow_turn_checks,
    workflow_expected_labels: list[str] | None,
    termination_reason: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "won": won,
        "done": env_done,
        "task_description": task_description,
        "num_steps": attempted_steps,
        "num_correct_steps": correct_steps,
        "action_accuracy": (correct_steps / attempted_steps) if attempted_steps > 0 else 0.0,
        "website": website,
        "domain": domain,
        "last_action": deepcopy(last_action),
        "last_action_valid": bool(last_action_valid),
        "prompt_context_mode": PROMPT_CONTEXT_MODE,
        "sample_granularity": "episode",
        "termination_reason": termination_reason,
    }

    if workflow_turn_checks is not None and summarize_workflow_turn_checks is not None:
        workflow_compliance = summarize_workflow_turn_checks(
            workflow_turn_checks,
            workflow_expected_labels or [],
            require_full_coverage=bool(won),
        )
        metadata["workflow_validation_source"] = "env_generate_turn_check"
        metadata["workflow_require_full_coverage"] = bool(won)
        metadata["workflow_expected_labels"] = list(workflow_expected_labels or [])
        metadata["workflow_step_count"] = len(workflow_expected_labels or [])
        metadata["workflow_compliance"] = workflow_compliance["compliant"]
        metadata["workflow_observed_labels"] = list(workflow_compliance["observed_labels"])
        metadata["workflow_observed_error_labels"] = list(
            workflow_compliance["observed_error_labels"]
        )
        metadata["workflow_missing_labels"] = list(workflow_compliance["missing_labels"])
        metadata["workflow_covers_all_labels"] = not workflow_compliance["missing_labels"]
        metadata["workflow_reward_reason"] = str(workflow_compliance["reason"])
        metadata["reasoning_count"] = int(workflow_compliance["reasoning_count"])
        metadata["workflow_reasoning_close_count"] = int(
            workflow_compliance["reasoning_close_count"]
        )
        metadata["action_count"] = int(workflow_compliance["action_count"])
        metadata["workflow_action_close_count"] = int(workflow_compliance["action_close_count"])
        metadata["workflow_tagged_reasoning_count"] = int(
            workflow_compliance["tagged_reasoning_count"]
        )
        metadata["has_error_handling_label"] = bool(
            workflow_compliance["observed_error_labels"]
        )
        metadata["workflow_turn_validation"] = workflow_turn_checks

    return metadata


def _build_episode_sample(
    source_sample: Sample,
    *,
    prompt_text: str,
    prompt_token_ids: list[int],
    trajectory_text: str,
    trajectory_token_ids: list[int],
    trajectory_loss_mask: list[int],
    trajectory_log_probs: list[float] | None,
    status,
    reward: float,
    metadata: dict[str, Any],
) -> Sample:
    episode_sample = Sample(status=status)
    episode_sample.prompt = prompt_text
    episode_sample.label = deepcopy(source_sample.label)
    episode_sample.tokens = list(prompt_token_ids) + list(trajectory_token_ids)
    episode_sample.response = trajectory_text
    episode_sample.response_length = len(trajectory_token_ids)
    episode_sample.loss_mask = list(trajectory_loss_mask)
    episode_sample.reward = float(reward)
    episode_sample.group_index = source_sample.group_index
    episode_sample.index = source_sample.index
    episode_sample.metadata = dict(metadata)
    if RETURN_LOGPROB and trajectory_log_probs is not None:
        episode_sample.rollout_log_probs = list(trajectory_log_probs)
    return episode_sample


def _annotate_step_samples_with_episode_summary(
    step_samples: list[Sample],
    *,
    episode_metadata: dict[str, Any],
) -> None:
    if not step_samples:
        return

    for step_idx, step_sample in enumerate(step_samples):
        metadata = step_sample.metadata if isinstance(step_sample.metadata, dict) else {}
        metadata["episode_won"] = bool(episode_metadata.get("won", False))
        metadata["episode_done"] = bool(episode_metadata.get("done", False))
        metadata["won"] = bool(episode_metadata.get("won", False))
        metadata["done"] = bool(episode_metadata.get("done", False))
        metadata["num_steps"] = int(episode_metadata.get("num_steps", 0) or 0)
        metadata["num_correct_steps"] = int(
            episode_metadata.get("num_correct_steps", 0) or 0
        )
        metadata["action_accuracy"] = float(episode_metadata.get("action_accuracy", 0.0) or 0.0)
        metadata["episode_reward_mean"] = float(
            sum(
                float(sample.reward)
                for sample in step_samples
                if isinstance(sample.reward, (int, float))
            )
            / len(step_samples)
        )
        metadata["is_terminal_step"] = 1.0 if step_idx == len(step_samples) - 1 else 0.0
        metadata["source_episode_won"] = float(bool(episode_metadata.get("won", False)))
        _copy_workflow_summary(metadata, episode_metadata)
        step_sample.metadata = metadata


async def run_episode_trace(
    args,
    sample: Sample,
    sampling_params: dict | None = None,
    skill: str | None = None,
    evaluation: bool = False,
    message_builder=None,
    workflow_expected_labels: list[str] | None = None,
) -> dict[str, Any]:
    if hasattr(args, "partial_rollout") and args.partial_rollout:
        raise ValueError("Partial rollout is not supported for Mind2Web web episodes.")

    state = GenerateState(args)
    tokenizer = state.tokenizer
    gen_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    env_url = require_arg(args, "web_env_url")
    max_turns = int(require_arg(args, "max_episode_steps"))
    max_context_length = _resolve_context_limit(args, evaluation)
    sampling_params = dict(sampling_params or {})

    prompt_text = ""
    prompt_token_ids: list[int] = []
    trajectory_text = ""
    trajectory_token_ids: list[int] = []
    trajectory_loss_mask: list[int] = []
    trajectory_log_probs: list[float] | None = [] if RETURN_LOGPROB else None
    step_samples: list[Sample] = []

    won = False
    env_done = False
    episode_id: str | None = None
    task_description = ""
    step_idx = -1
    correct_steps = 0
    website = ""
    domain = ""
    current_obs = ""
    previous_actions: list[str] = []
    action_schema = ""
    html_field = ""
    last_action: Any = ""
    last_action_valid = False
    episode_status = Sample.Status.PENDING
    termination_reason = ""
    workflow_turn_checks: list[dict] | None = None
    summarize_workflow_turn_checks = None
    evaluate_workflow_turn_output = None

    sem = _get_semaphore(args)
    async with sem:
        try:
            reset_resp = await post(f"{env_url}/reset", {"label": sample.label})
            episode_id = reset_resp["episode_id"]
            current_obs = reset_resp["obs"]
            previous_actions = list(reset_resp.get("previous_actions", []))
            action_schema = reset_resp["action_schema"]
            html_field = reset_resp.get("html_field", "")
            task_description = reset_resp.get("task_description", "")
            website = reset_resp.get("website", "")
            domain = reset_resp.get("domain", "")
        except Exception as exc:
            task_description = (
                str(sample.label.get("task_description", ""))
                if isinstance(sample.label, dict)
                else ""
            )
            episode_status = Sample.Status.FAILED
            termination_reason = f"reset_error:{exc}"
            episode_metadata = _finalize_episode_metadata(
                won=False,
                env_done=False,
                task_description=task_description,
                attempted_steps=0,
                correct_steps=0,
                website="",
                domain="",
                last_action="",
                last_action_valid=False,
                workflow_turn_checks=workflow_turn_checks,
                summarize_workflow_turn_checks=summarize_workflow_turn_checks,
                workflow_expected_labels=workflow_expected_labels,
                termination_reason=termination_reason,
            )
            episode_sample = _build_episode_sample(
                sample,
                prompt_text="",
                prompt_token_ids=[],
                trajectory_text="",
                trajectory_token_ids=[],
                trajectory_loss_mask=[],
                trajectory_log_probs=[] if RETURN_LOGPROB else None,
                status=episode_status,
                reward=0.0,
                metadata=episode_metadata,
            )
            return {
                "episode_sample": episode_sample,
                "step_samples": [
                    _build_fallback_step_sample(
                        sample,
                        prompt_text="",
                        task_description=task_description,
                        website="",
                        domain="",
                        episode_id=None,
                        status=Sample.Status.FAILED,
                        termination_reason=termination_reason,
                    )
                ],
                "trajectory": "",
                "won": False,
                "done": False,
                "task_description": task_description,
                "website": "",
                "domain": "",
            }

        if message_builder is not None:
            messages = message_builder(task_description, skill)
        elif skill:
            messages = format_solver_with_skill_messages(task_description, skill)
        else:
            messages = format_solver_messages(task_description)

        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        for step_idx in range(max_turns):
            obs_text = format_step_observation(
                task_description=task_description,
                obs=current_obs,
                previous_actions=previous_actions,
                action_schema=action_schema,
                html_field=html_field,
                step_idx=step_idx,
            )
            obs_token_ids = tokenizer(obs_text, add_special_tokens=False)["input_ids"]

            trajectory_text += obs_text
            trajectory_token_ids += obs_token_ids
            trajectory_loss_mask += [0] * len(obs_token_ids)
            if RETURN_LOGPROB and trajectory_log_probs is not None:
                trajectory_log_probs += [0.0] * len(obs_token_ids)

            current_input_length = len(prompt_token_ids) + len(obs_token_ids)
            remaining_context_budget = max_context_length - current_input_length
            if remaining_context_budget <= 0:
                episode_status = Sample.Status.TRUNCATED
                termination_reason = "context_budget_exhausted_before_generation"
                break

            request_sampling_params = deepcopy(sampling_params)
            _ensure_stop_string(request_sampling_params, "</action>")
            if _apply_generation_budget(
                args,
                request_sampling_params,
                remaining_context_budget,
            ) <= 0:
                episode_status = Sample.Status.TRUNCATED
                termination_reason = "generation_budget_zero"
                break

            payload: dict[str, Any] = {
                "input_ids": prompt_token_ids + obs_token_ids,
                "sampling_params": request_sampling_params,
            }
            if RETURN_LOGPROB:
                payload["return_logprob"] = True

            try:
                output = await post(gen_url, payload)
            except Exception as exc:
                episode_status = Sample.Status.FAILED
                termination_reason = f"sglang_error:{exc}"
                break

            finish_type = output["meta_info"]["finish_reason"]["type"]
            gen_text = output["text"]
            if RETURN_LOGPROB and "output_token_logprobs" in output["meta_info"]:
                logprob_data = output["meta_info"]["output_token_logprobs"]
                gen_token_ids = [item[1] for item in logprob_data]
                gen_log_probs = [item[0] for item in logprob_data]
            else:
                gen_token_ids = tokenizer(gen_text, add_special_tokens=False)["input_ids"]
                gen_log_probs = None

            trajectory_text += gen_text
            trajectory_token_ids += gen_token_ids
            trajectory_loss_mask += [1] * len(gen_token_ids)
            if RETURN_LOGPROB and trajectory_log_probs is not None:
                if gen_log_probs is not None:
                    trajectory_log_probs += gen_log_probs
                else:
                    trajectory_log_probs += [0.0] * len(gen_token_ids)

            if workflow_turn_checks is not None and evaluate_workflow_turn_output is not None:
                workflow_turn_checks.append(evaluate_workflow_turn_output(gen_text))

            action, is_valid = extract_action(gen_text)
            last_action = action
            last_action_valid = is_valid

            step_status = Sample.Status.COMPLETED
            step_reward = 0.0
            action_correct = False
            step_done = False
            step_won = False
            match_mode = "none"
            step_termination_reason = ""

            if finish_type == "abort":
                step_status = Sample.Status.ABORTED
                episode_status = Sample.Status.ABORTED
                termination_reason = "generation_abort"
                step_termination_reason = termination_reason
            elif finish_type == "length":
                step_status = Sample.Status.TRUNCATED
                episode_status = Sample.Status.TRUNCATED
                termination_reason = "generation_length"
                step_termination_reason = termination_reason
            else:
                try:
                    step_resp = await post(
                        f"{env_url}/step",
                        {"episode_id": episode_id, "action": action},
                    )
                    current_obs = step_resp["obs"]
                    previous_actions = list(step_resp.get("previous_actions", previous_actions))
                    action_schema = step_resp.get("action_schema", action_schema)
                    html_field = step_resp.get("html_field", html_field)
                    env_done = bool(step_resp["done"])
                    won = bool(step_resp["won"])
                    action_correct = bool(step_resp.get("action_correct"))
                    step_done = env_done
                    step_won = won
                    match_mode = str(step_resp.get("match_mode", "none"))
                    step_reward = float(
                        step_resp.get("reward", 1.0 if action_correct else 0.0)
                    )
                    if action_correct:
                        correct_steps += 1
                    if env_done:
                        termination_reason = (
                            "task_completed" if won else "task_terminated_by_env"
                        )
                    step_termination_reason = termination_reason
                except Exception as exc:
                    step_status = Sample.Status.FAILED
                    episode_status = Sample.Status.FAILED
                    termination_reason = f"step_error:{exc}"
                    step_termination_reason = termination_reason

            step_sample = _build_step_sample(
                sample,
                prompt_text=prompt_text,
                prompt_token_ids=prompt_token_ids,
                obs_text=obs_text,
                obs_token_ids=obs_token_ids,
                gen_text=gen_text,
                gen_token_ids=gen_token_ids,
                gen_log_probs=gen_log_probs,
                reward=step_reward,
                status=step_status,
                step_idx=step_idx,
                episode_id=episode_id,
                task_description=task_description,
                website=website,
                domain=domain,
                action=action,
                action_valid=is_valid,
                action_correct=action_correct,
                step_done=step_done,
                step_won=step_won,
                match_mode=match_mode,
                termination_reason=step_termination_reason,
            )
            step_samples.append(step_sample)

            if episode_status in {
                Sample.Status.ABORTED,
                Sample.Status.TRUNCATED,
                Sample.Status.FAILED,
            }:
                break

            if env_done:
                break

        if episode_id and not env_done:
            try:
                await post(f"{env_url}/close", {"episode_id": episode_id})
            except Exception:
                pass

    attempted_steps = len(step_samples)
    if episode_status == Sample.Status.PENDING:
        episode_status = Sample.Status.COMPLETED if env_done else Sample.Status.TRUNCATED
        if not termination_reason:
            termination_reason = "task_completed" if won else "max_episode_steps_reached"

    episode_metadata = _finalize_episode_metadata(
        won=won,
        env_done=env_done,
        task_description=task_description,
        attempted_steps=attempted_steps,
        correct_steps=correct_steps,
        website=website,
        domain=domain,
        last_action=last_action,
        last_action_valid=last_action_valid,
        workflow_turn_checks=workflow_turn_checks,
        summarize_workflow_turn_checks=summarize_workflow_turn_checks,
        workflow_expected_labels=workflow_expected_labels,
        termination_reason=termination_reason,
    )
    episode_sample = _build_episode_sample(
        sample,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
        trajectory_text=trajectory_text,
        trajectory_token_ids=trajectory_token_ids,
        trajectory_loss_mask=trajectory_loss_mask,
        trajectory_log_probs=trajectory_log_probs,
        status=episode_status,
        reward=10.0 * float(won),
        metadata=episode_metadata,
    )

    if not step_samples:
        step_samples = [
            _build_fallback_step_sample(
                sample,
                prompt_text=prompt_text,
                task_description=task_description,
                website=website,
                domain=domain,
                episode_id=episode_id,
                status=episode_status,
                termination_reason=termination_reason,
            )
        ]

    _annotate_step_samples_with_episode_summary(
        step_samples,
        episode_metadata=episode_metadata,
    )
    maybe_print_random_env_sample(
        episode_sample,
        tag="web/env_generate_eval" if evaluation else "web/env_generate_task",
    )

    return {
        "episode_sample": episode_sample,
        "step_samples": step_samples,
        "trajectory": trajectory_text,
        "won": won,
        "done": env_done,
        "task_description": task_description,
        "website": website,
        "domain": domain,
    }


async def run_episode(
    args,
    sample: Sample,
    sampling_params: dict | None = None,
    skill: str | None = None,
    evaluation: bool = False,
    message_builder=None,
    workflow_expected_labels: list[str] | None = None,
) -> Sample | list[Sample]:
    trace = await run_episode_trace(
        args,
        sample=sample,
        sampling_params=sampling_params,
        skill=skill,
        evaluation=evaluation,
        message_builder=message_builder,
        workflow_expected_labels=workflow_expected_labels,
    )
    return trace["episode_sample"] if evaluation else trace["step_samples"]
