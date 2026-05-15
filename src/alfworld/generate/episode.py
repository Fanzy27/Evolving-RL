"""ALFWorld episode runner — async, mirrors generate_with_search.py.

Runs a single ALFWorld episode by interleaving:
  1. LLM generation (SGLang HTTP) → <think>...</think><action>...</action>
  2. Environment step (ALFWorld HTTP server) → next observation

Loss mask design (same convention as generate_with_search.py):
  - Model-generated tokens (think + action): loss_mask = 1
  - Environment observation tokens (obs + admissible actions): loss_mask = 0

The `run_episode` function returns a fully annotated Sample object.
"""

import asyncio
from copy import deepcopy
import re

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from src.alfworld.prompts import (
    format_solver_messages,
    format_solver_with_skill_messages,
)
from src.alfworld.utils.common import require_arg

RETURN_LOGPROB = False

SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore(args) -> asyncio.Semaphore:
    global SEMAPHORE
    if SEMAPHORE is None:
        concurrency = int(require_arg(args, "alfworld_concurrency"))
        SEMAPHORE = asyncio.Semaphore(concurrency)
    return SEMAPHORE


def _resolve_context_limit(args, evaluation: bool) -> int:
    if evaluation and hasattr(args, "eval_max_context_len") and args.eval_max_context_len is not None:
        return int(args.eval_max_context_len)
    return int(require_arg(args, "rollout_max_context_len"))


def _apply_generation_budget(
    args,
    request_sampling_params: dict,
    remaining_context_budget: int,
) -> int:
    solver_max_response_len = int(require_arg(args, "solver_max_response_len"))
    max_new_tokens = min(remaining_context_budget, solver_max_response_len)
    request_sampling_params["max_new_tokens"] = max_new_tokens
    return max_new_tokens


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


# ---------------------------------------------------------------------------
# Action extraction (mirrors alfworld_projection logic from SkillRL)
# ---------------------------------------------------------------------------

def extract_action(text: str) -> tuple[str, bool]:
    """Extract action from <action>...</action> tag.

    Returns:
        (action_str, is_valid) where is_valid requires:
        - a non-empty <action>...</action>
        - a matching <reasoning>...</reasoning> block
        - no Chinese characters
    """
    text_lower = text.lower()

    has_reasoning = "<reasoning>" in text_lower and "</reasoning>" in text_lower
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))

    match = re.search(r"<action>(.*?)</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return text[-30:], False

    action = match.group(1).strip().lower()
    is_valid = bool(action) and has_reasoning and not has_chinese
    return action, is_valid


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def format_step_observation(obs: str, admissible_commands: list[str], step_idx: int) -> str:
    """Format environment observation as text to be appended to the response.

    This text is injected as context (loss_mask=0) between model-generated steps.
    """
    formatted_cmds = "\n ".join(f"'{c}'" for c in admissible_commands)
    return (
        f"\n\n[Step {step_idx}]\n"
        f"Observation: {obs}\n"
        f"Admissible actions: [{formatted_cmds}]\n\n"
    )


def _normalize_action_text(text: str) -> str:
    return str(text or "").strip().lower()


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    args,
    sample: Sample,
    sampling_params: dict | None = None,
    skill: str | None = None,
    evaluation: bool = False,
    message_builder=None,
    workflow_expected_labels: list[str] | None = None,
) -> Sample:
    """Run one complete ALFWorld episode.

    Mirrors generate_with_search.generate():
    - Calls SGLang HTTP for each model turn
    - Calls ALFWorld HTTP server for each environment step
    - Maintains loss_mask: 0 for obs tokens, 1 for model tokens

    Args:
        args:              Training configuration namespace.
        task_description:  Task description string. If empty, fetched from the env
                           /reset response (supports datasets without task_description).
        skill:             Optional Markdown skill card (injected into system prompt).
        sampling_params:   SGLang sampling parameters dict.
        evaluation:        If True, runs in eval mode (greedy, etc.).

    Returns:
        Annotated Sample with tokens, response, loss_mask, reward, and
        metadata["task_description"] with the resolved task description.
    """
    if hasattr(args, "partial_rollout") and args.partial_rollout:
        raise ValueError("Partial rollout is not supported for ALFWorld episodes.")

    state = GenerateState(args)
    tokenizer = state.tokenizer
    gen_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    env_url = require_arg(args, "alfworld_env_url")
    max_turns = int(require_arg(args, "max_episode_steps"))
    return_logprob = RETURN_LOGPROB
    max_context_length = _resolve_context_limit(args, evaluation)
    sampling_params = dict(sampling_params or {})
    enforce_turn_output_check = not evaluation
    # Episode state
    game_file_path = "/mnt/tidal-alsh-share2/usr/fanzhiyuan2/projects/extractor/data/alfworld/official/json_2.1.1/" + sample.label['split'] + "/" + sample.label['rel_dir'] + "/game.tw-pddl"
    response = ""
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] | None = [] if return_logprob else None
    prompt_token_ids: list[int] = []

    won = False
    env_done = False
    episode_id: str | None = None
    step_idx = -1
    task_description = ""
    workflow_turn_checks: list[dict] | None = None
    summarize_workflow_turn_checks = None
    evaluate_workflow_turn_output = None

    sem = _get_semaphore(args)
    async with sem:
        # ── Reset the environment ────────────────────────────────────────────
        try:
            reset_payload = {"game_file_path": game_file_path}
            reset_resp = await post(f"{env_url}/reset", reset_payload)
            episode_id = reset_resp["episode_id"]
            current_obs = reset_resp["obs"]
            admissible = reset_resp["admissible_commands"]
            # If task_description not in dataset, fetch it from the env reset response
            task_description = reset_resp.get("task_description", "")
        except Exception as exc:
            ident = game_file_path
            print(f"[run_episode] /reset error ({ident}): {exc}")
            sample.status = Sample.Status.FAILED
            sample.reward = 0.0
            sample.tokens = []
            sample.response = ""
            sample.loss_mask = []
            return sample

        # Build initial prompt after reset so task_description is always available
        if message_builder is not None:
            messages = message_builder(task_description, skill)
        elif skill:
            messages = format_solver_with_skill_messages(task_description, skill)
        else:
            messages = format_solver_messages(task_description)

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        sample.prompt = prompt_text

        # ── Episode loop ─────────────────────────────────────────────────────
        for step_idx in range(max_turns):
            # 1) Append environment observation (loss_mask = 0)
            obs_text = format_step_observation(current_obs, admissible, step_idx)
            obs_token_ids = tokenizer(obs_text, add_special_tokens=False)["input_ids"]
            current_input_length = len(prompt_token_ids) + len(response_token_ids)
            remaining_context_budget = max_context_length - current_input_length
            if remaining_context_budget <= 0 or len(obs_token_ids) > remaining_context_budget:
                sample.status = Sample.Status.TRUNCATED
                break

            response += obs_text
            response_token_ids += obs_token_ids
            loss_mask += [0] * len(obs_token_ids)
            if return_logprob:
                rollout_log_probs += [0.0] * len(obs_token_ids)

            # 2) LLM generates <think>...</think><action>...</action>
            current_input_ids = prompt_token_ids + response_token_ids
            remaining_context_budget = max_context_length - len(current_input_ids)
            if remaining_context_budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            request_sampling_params = deepcopy(sampling_params)
            _ensure_stop_string(request_sampling_params, "</action>")
            if _apply_generation_budget(
                args,
                request_sampling_params,
                remaining_context_budget,
            ) <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            payload: dict = {
                "input_ids": current_input_ids,
                "sampling_params": request_sampling_params,
            }
            if return_logprob:
                payload["return_logprob"] = True

            try:
                output = await post(gen_url, payload)
            except Exception as exc:
                print(f"[run_episode] SGLang error at step {step_idx}: {exc}")
                break

            finish_type = output["meta_info"]["finish_reason"]["type"]
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break

            gen_text: str = output["text"]

            if return_logprob and "output_token_logprobs" in output["meta_info"]:
                logprob_data = output["meta_info"]["output_token_logprobs"]
                gen_token_ids = [item[1] for item in logprob_data]
                gen_log_probs = [item[0] for item in logprob_data]
            else:
                gen_token_ids = tokenizer(gen_text, add_special_tokens=False)["input_ids"]
                gen_log_probs = None

            response += gen_text
            response_token_ids += gen_token_ids
            loss_mask += [1] * len(gen_token_ids)
            if return_logprob and gen_log_probs is not None:
                rollout_log_probs += gen_log_probs

            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                break

            if workflow_turn_checks is not None and evaluate_workflow_turn_output is not None:
                workflow_turn_checks.append(evaluate_workflow_turn_output(gen_text))

            # 3) Extract action and step the environment
            action, is_valid = extract_action(gen_text)
            # if enforce_turn_output_check and not is_valid:
            #     sample.status = Sample.Status.TRUNCATED
            #     break

            try:
                step_resp = await post(
                    f"{env_url}/step",
                    {"episode_id": episode_id, "action": action},
                )
                current_obs = step_resp["obs"]
                admissible = step_resp["admissible_commands"]
                env_done = step_resp["done"]
                won = step_resp["won"]
            except Exception as exc:
                print(f"[run_episode] /step error at step {step_idx}: {exc}")
                break

            if env_done:
                break

        # ── Close episode (if still open) ────────────────────────────────────
        if episode_id and not env_done:
            try:
                await post(f"{env_url}/close", {"episode_id": episode_id})
            except Exception:
                pass

    # ── Assemble Sample ──────────────────────────────────────────────────────
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response = response
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.reward = 10.0 * float(won)
    sample.metadata = {
        "won": won,
        "done": env_done,
        "task_description": task_description,
        "num_steps": step_idx + 1,
        "last_action": action if "action" in locals() else "",
        "last_action_valid": bool(is_valid) if "is_valid" in locals() else False,
        "turn_output_check_enforced": bool(enforce_turn_output_check),
    }
    if workflow_turn_checks is not None and summarize_workflow_turn_checks is not None:
        workflow_compliance = summarize_workflow_turn_checks(
            workflow_turn_checks,
            workflow_expected_labels or [],
            require_full_coverage=bool(won),
        )
        sample.metadata["workflow_validation_source"] = "env_generate_turn_check"
        sample.metadata["workflow_require_full_coverage"] = bool(won)
        sample.metadata["workflow_expected_labels"] = list(workflow_expected_labels or [])
        sample.metadata["workflow_step_count"] = len(workflow_expected_labels or [])
        sample.metadata["workflow_compliance"] = workflow_compliance["compliant"]
        sample.metadata["workflow_observed_labels"] = list(
            workflow_compliance["observed_labels"]
        )
        sample.metadata["workflow_observed_error_labels"] = list(
            workflow_compliance["observed_error_labels"]
        )
        sample.metadata["workflow_missing_labels"] = list(
            workflow_compliance["missing_labels"]
        )
        sample.metadata["workflow_covers_all_labels"] = not workflow_compliance["missing_labels"]
        sample.metadata["workflow_reward_reason"] = str(workflow_compliance["reason"])
        sample.metadata["reasoning_count"] = int(workflow_compliance["reasoning_count"])
        sample.metadata["workflow_reasoning_close_count"] = int(
            workflow_compliance["reasoning_close_count"]
        )
        sample.metadata["action_count"] = int(workflow_compliance["action_count"])
        sample.metadata["workflow_action_close_count"] = int(
            workflow_compliance["action_close_count"]
        )
        sample.metadata["workflow_tagged_reasoning_count"] = int(
            workflow_compliance["tagged_reasoning_count"]
        )
        sample.metadata["has_error_handling_label"] = bool(
            workflow_compliance["observed_error_labels"]
        )
        sample.metadata["workflow_turn_validation"] = workflow_turn_checks

    if return_logprob and rollout_log_probs is not None:
        sample.rollout_log_probs = rollout_log_probs

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED if env_done else Sample.Status.TRUNCATED

    return sample
