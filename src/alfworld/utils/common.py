"""Shared utility helpers for runtime validation and sample debugging."""

from __future__ import annotations

import json
import random
import sys

from slime.utils.types import Sample


_RANDOM_LOG_DENOMINATOR = 64


def require_arg(args, name: str):
    if not hasattr(args, name):
        raise ValueError(
            f"Missing required argument '{name}'. Set it explicitly in the bash script."
        )

    value = getattr(args, name)
    if value is None:
        raise ValueError(
            f"Argument '{name}' is None. Set it explicitly in the bash script."
        )

    return value


def require_cli_flags(flags: list[str], argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    missing = [
        flag
        for flag in flags
        if flag not in argv and not any(arg.startswith(f"{flag}=") for arg in argv)
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            f"Missing required CLI flags: {missing_text}. Set them explicitly in the bash script."
        )


def _metadata(sample: Sample) -> dict:
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _should_log_random_sample() -> bool:
    return random.randint(1, _RANDOM_LOG_DENOMINATOR) == 1


def _is_reward_detail_key(key: str) -> bool:
    key_lower = key.lower()
    return any(
        token in key_lower
        for token in (
            "reward",
            "bonus",
            "penalty",
            "score",
            "won",
            "win_rate",
        )
    )


def ensure_metadata(sample: Sample) -> dict:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    return sample.metadata


def maybe_print_random_sample(sample: Sample, *, tag: str) -> None:
    if not _should_log_random_sample():
        return

    metadata = _metadata(sample)
    reward_payload = {
        "tag": tag,
        "reward": float(sample.reward) if isinstance(sample.reward, (int, float)) else sample.reward,
    }

    for key in sorted(metadata):
        if _is_reward_detail_key(key):
            reward_payload[key] = metadata[key]

    print("--------------------------------")
    print("[alfworld/random_sample]")
    print(json.dumps(reward_payload, ensure_ascii=False, indent=2, default=str))

    sample_text = f"{sample.prompt or ''}{sample.response or ''}".strip()
    if sample_text:
        print("[sample_text]")
        print(sample_text)
