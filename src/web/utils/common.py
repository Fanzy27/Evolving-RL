"""Compatibility helpers reused from the ALFWorld refactor."""

from __future__ import annotations

import json
import random

from src.alfworld.utils.common import *  # noqa: F401,F403
from slime.utils.types import Sample


def ensure_metadata(sample: Sample) -> dict:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    return sample.metadata


_WEB_RANDOM_LOG_DENOMINATOR = 500


def ensure_train_placeholder_sample(sample: Sample, *, reason: str = "") -> Sample:
    """Make a sample safe for trainer ingestion even if rollout failed.

    Megatron expects:
    - total_length > response_length
    - response_length >= 1
    - len(loss_mask) == response_length

    For failed/empty samples we create a masked one-token placeholder so the
    sample contributes no gradient but does not crash training.
    """
    tokens = list(sample.tokens or [])
    response_length = int(sample.response_length or 0)
    prompt_length = len(tokens) - response_length

    if response_length >= 1 and prompt_length >= 1:
        return sample

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    metadata["train_placeholder"] = 1.0
    if reason:
        metadata["train_placeholder_reason"] = str(reason)
    sample.metadata = metadata
    sample.remove_sample = True

    response_length = max(1, response_length)
    min_total_length = response_length + 1
    if len(tokens) < min_total_length:
        tokens = ([0] * (min_total_length - len(tokens))) + list(tokens)

    sample.tokens = list(tokens)
    sample.response_length = int(response_length)
    sample.loss_mask = [0] * sample.response_length
    if sample.reward is None:
        sample.reward = 0.0
    return sample


def maybe_print_random_env_sample(sample: Sample, *, tag: str) -> None:
    if random.randint(1, _WEB_RANDOM_LOG_DENOMINATOR) != 1:
        return

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    summary = {
        "tag": tag,
        "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
        "reward": float(sample.reward) if isinstance(sample.reward, (int, float)) else sample.reward,
        "response_length": int(sample.response_length or 0),
        "effective_response_length": int(sample.effective_response_length or 0),
        "metadata": metadata,
    }

    print("--------------------------------")
    print("[web/random_env_sample]")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    prompt_text = sample.prompt if isinstance(sample.prompt, str) else json.dumps(
        sample.prompt,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    response_text = str(sample.response or "")

    print("[prompt]")
    print(prompt_text)

    print("[response]")
    print(response_text)
