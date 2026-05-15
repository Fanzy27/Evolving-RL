"""ALFWorld evaluation pipeline: skill-conditioned episode evaluation.

Retrieves a task semantically similar to the eval sample, extracts a skill,
then evaluates the target with and without that skill.
Metadata keys: won_with_skill, won_no_skill,
               num_steps_with_skill, num_steps_no_skill.
"""

import asyncio
from copy import deepcopy

from slime.utils.types import Sample

from src.alfworld.generate.episode import run_episode
from src.alfworld.generate.retrieval import retrieve_tasks
from src.alfworld.generate.extractor import (
    generate_extractor_sample_from_trajectory,
    resolve_retrieval_query,
    resolve_task_description,
    task_info_to_sample,
)
from src.alfworld.utils.common import ensure_metadata, maybe_print_random_sample, require_arg


async def _extract_skill_from_task_info(
    args,
    task_info: dict,
    sampling_params: dict,
    log_tag: str = "eval_pipeline",
) -> str:
    """Run Solver (no skill) on a retrieved task, then Extractor → skill text.

    Returns an empty string on any failure.
    """
    retrieved_sample = task_info_to_sample(task_info)

    # Step 2: Solver episode on retrieved task
    try:
        src_episode = await run_episode(
            args,
            sample=retrieved_sample,
            skill=None,
            sampling_params=sampling_params,
        )
    except Exception as exc:
        print(f"[{log_tag}] Solver error on retrieved task: {exc}")
        return ""

    # Step 3: Extractor → skill text
    src_metadata = src_episode.metadata if isinstance(src_episode.metadata, dict) else {}
    retrieved_task_desc = src_metadata.get("task_description", "")
    trajectory = src_episode.response
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
    """Steps 4-6: run target with/without skill and package results.

    Args:
        skill_meta_suffix: metadata key suffix for the skill run (e.g. "with_skill").
                           The no-skill run always uses "no_skill".
    """
    with_sample = deepcopy(sample)
    no_sample = deepcopy(sample)

    try:
        if skill_text:
            with_result, no_result = await asyncio.gather(
                run_episode(args, sample=with_sample, skill=skill_text, sampling_params=sampling_params),
                run_episode(args, sample=no_sample, skill=None, sampling_params=sampling_params),
            )
        else:
            no_result = await run_episode(
                args, sample=no_sample, skill=None, sampling_params=sampling_params
            )
            with_result = no_result
    except Exception as exc:
        print(f"[eval_pipeline] Episode error ({skill_meta_suffix}): {exc}")
        sample.reward = 0.0
        sample.tokens = []
        sample.loss_mask = []
        sample.response = ""
        sample.metadata = {
            f"won_{skill_meta_suffix}": 0.0,
            "won_no_skill": 0.0,
            f"num_steps_{skill_meta_suffix}": 0,
            "num_steps_no_skill": 0,
        }
        return sample

    with_metadata = with_result.metadata if isinstance(with_result.metadata, dict) else {}
    no_metadata = no_result.metadata if isinstance(no_result.metadata, dict) else {}
    won_with = 1.0 if with_metadata.get("won", False) else 0.0
    won_no = 1.0 if no_metadata.get("won", False) else 0.0

    result = with_result
    result.reward = 10.0 * won_with
    if not isinstance(result.metadata, dict):
        result.metadata = {}
    result.metadata[f"won_{skill_meta_suffix}"] = won_with
    result.metadata["won_no_skill"] = won_no
    result.metadata[f"num_steps_{skill_meta_suffix}"] = with_metadata.get("num_steps", 0)
    result.metadata["num_steps_no_skill"] = no_metadata.get("num_steps", 0)
    return result


# ---------------------------------------------------------------------------
# Public eval functions
# ---------------------------------------------------------------------------


def _resolve_eval_query(args, sample: Sample) -> str:
    label = sample.label if isinstance(sample.label, dict) else {}
    query = resolve_retrieval_query(
        sample,
        fallback_task_description=resolve_task_description(label),
    )
    if query:
        return query
    return str(label.get("task_type") or "")


async def run_eval(
    args,
    sample: Sample,
    sampling_params: dict,
) -> Sample:
    """Evaluate one ALFWorld sample with a semantically similar retrieved skill.

    Step 1 queries the retrieve server (/search) with the eval sample's task
    description to find the most relevant task for skill generation.
    Metadata keys written:
      won_with_skill, won_no_skill,
      num_steps_with_skill, num_steps_no_skill.
    """
    retrieve_url_test = require_arg(args, "alfworld_retrieve_url_test")
    query = _resolve_eval_query(args, sample)

    # ------------------------------------------------------------------
    # Similar retrieved skill
    # ------------------------------------------------------------------
    retrieved = await retrieve_tasks(
        query=query, K=1, url=retrieve_url_test
    )

    if retrieved:
        skill_text = await _extract_skill_from_task_info(
            args, retrieved[0], sampling_params, log_tag="eval_pipeline/similar"
        )
    else:
        print(
            f"[eval_pipeline] WARNING: /search returned no results for '{query[:60]}'. "
            "Skill will be empty; only baseline episode is run."
        )
        skill_text = ""

    result = await _eval_with_skill_text(
        args, sample, sampling_params, skill_text, skill_meta_suffix="with_skill"
    )

    maybe_print_random_sample(result, tag="eval/with_experience")

    return result
