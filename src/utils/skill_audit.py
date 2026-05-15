"""Language-quality audit for extractor skills (domain-parameterized).

Rule-based audit only: detects suspicious Unicode characters, non-English text,
and repetitive patterns in extractor-generated skill markdown.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import unicodedata
from collections import Counter

from slime.utils.types import Sample

_AUDIT_MAX_CONCURRENCY = 32
_PASSED_AUDIT_PRINT_PROB = 0.01
_FAILED_AUDIT_PRINT_PROB = 1.0

_DEFAULT_ENABLE_RULE_BASED_AUDIT = "1"
_SUSPICIOUS_UNICODE_NAME_KEYWORDS = (
    "ZERO WIDTH",
    "FULLWIDTH",
    "HALFWIDTH",
    "MATHEMATICAL",
    "MODIFIER LETTER",
    "COMBINING",
    "PRIVATE USE",
    "SMALL FORM VARIANT",
)
_MAX_SUSPICIOUS_CHAR_REPORTS = 8
_MAX_CONSECUTIVE_CHAR_REPEAT = 8
_MAX_CONSECUTIVE_TOKEN_REPEAT = 8
_MIN_REPEATED_LINE_COUNT = 3
_MIN_REPEATED_LINE_SHARE = 0.5
_MIN_REPEATED_CHUNK_COUNT = 4


def _ensure_metadata(sample: Sample) -> dict:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    return sample.metadata


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "pass", "passed"}


def _get_arg_or_env(args, attr_name: str, env_name: str, default=None):
    value = getattr(args, attr_name, None)
    if value is not None:
        return value

    env_value = os.getenv(env_name)
    if env_value is not None and env_value != "":
        return env_value
    return default


def _resolve_rule_based_audit_enabled(args, domain: str) -> bool:
    return _bool_value(
        _get_arg_or_env(
            args,
            f"{domain}_skill_language_audit_enable_rule_based",
            f"{domain.upper()}_SKILL_LANGUAGE_AUDIT_ENABLE_RULE_BASED",
            _DEFAULT_ENABLE_RULE_BASED_AUDIT,
        )
    )


def _format_char_for_log(char: str) -> str:
    codepoint = f"U+{ord(char):04X}"
    name = unicodedata.name(char, "UNKNOWN")
    return f"{repr(char)}({codepoint}, {name})"


def _collect_suspicious_characters(text: str) -> list[str]:
    suspicious: list[str] = []
    seen: set[str] = set()
    for char in text:
        if char in {"\n", "\r", "\t"}:
            continue

        reason = ""
        category = unicodedata.category(char)
        name = unicodedata.name(char, "")
        if char == "\ufffd":
            reason = "replacement_character"
        elif ord(char) < 32 or ord(char) == 127:
            reason = "control_character"
        elif category in {"Cf", "Cs", "Co", "Cn"}:
            reason = f"unicode_category_{category.lower()}"
        elif any(keyword in name for keyword in _SUSPICIOUS_UNICODE_NAME_KEYWORDS):
            reason = "suspicious_unicode_name"

        if not reason:
            continue

        descriptor = f"{reason}:{_format_char_for_log(char)}"
        if descriptor in seen:
            continue
        seen.add(descriptor)
        suspicious.append(descriptor)
        if len(suspicious) >= _MAX_SUSPICIOUS_CHAR_REPORTS:
            break

    return suspicious


def _collect_non_english_characters(text: str) -> list[str]:
    non_english: list[str] = []
    seen: set[str] = set()
    for char in text:
        if char in {"\n", "\r", "\t"}:
            continue
        if ord(char) <= 127:
            continue
        category = unicodedata.category(char)
        if category[:1] in {"S", "P", "Z"}:
            continue

        descriptor = _format_char_for_log(char)
        if descriptor in seen:
            continue
        seen.add(descriptor)
        non_english.append(descriptor)
        if len(non_english) >= _MAX_SUSPICIOUS_CHAR_REPORTS:
            break

    return non_english


def _detect_repetitive_text_reason(text: str) -> str:
    collapsed = re.sub(r"[ \t]+", " ", str(text or "").strip())
    if not collapsed:
        return ""

    char_match = re.search(r"(.)\1{" + str(_MAX_CONSECUTIVE_CHAR_REPEAT - 1) + r",}", collapsed)
    if char_match:
        return f"long repeated character sequence detected: {char_match.group(0)[:32]!r}"

    tokens = [token.lower() for token in re.findall(r"\S+", collapsed)]
    if tokens:
        run_len = 1
        for idx in range(1, len(tokens)):
            if tokens[idx] == tokens[idx - 1]:
                run_len += 1
                if run_len >= _MAX_CONSECUTIVE_TOKEN_REPEAT:
                    return f"repeated token sequence detected: {tokens[idx]!r}"
            else:
                run_len = 1

        for chunk_size in (2, 3, 4):
            if len(tokens) < chunk_size * _MIN_REPEATED_CHUNK_COUNT:
                continue
            for start_idx in range(0, len(tokens) - chunk_size + 1):
                chunk = tuple(tokens[start_idx : start_idx + chunk_size])
                if len(" ".join(chunk)) < 8:
                    continue
                repeat_count = 1
                next_idx = start_idx + chunk_size
                while next_idx + chunk_size <= len(tokens):
                    if tuple(tokens[next_idx : next_idx + chunk_size]) != chunk:
                        break
                    repeat_count += 1
                    if repeat_count >= _MIN_REPEATED_CHUNK_COUNT:
                        return f"repetitive text pattern detected: {' '.join(chunk)!r}"
                    next_idx += chunk_size

    lines = [line.strip().lower() for line in str(text or "").splitlines() if line.strip()]
    if lines:
        line_counter = Counter(line for line in lines if len(line) >= 12)
        if line_counter:
            most_common_line, count = line_counter.most_common(1)[0]
            if count >= _MIN_REPEATED_LINE_COUNT and (count / float(len(lines))) >= _MIN_REPEATED_LINE_SHARE:
                return f"repeated line pattern detected: {most_common_line[:120]!r}"

    return ""


def _run_rule_based_audit(skill_text: str) -> dict:
    suspicious_characters = _collect_suspicious_characters(skill_text)
    non_english_characters = _collect_non_english_characters(skill_text)
    repetition_reason = _detect_repetitive_text_reason(skill_text)

    reasons: list[str] = []
    if suspicious_characters:
        reasons.append(
            "suspicious characters detected: " + "; ".join(suspicious_characters[:_MAX_SUSPICIOUS_CHAR_REPORTS])
        )
    if non_english_characters:
        reasons.append(
            "non-English textual characters detected: "
            + "; ".join(non_english_characters[:_MAX_SUSPICIOUS_CHAR_REPORTS])
        )
    if repetition_reason:
        reasons.append(repetition_reason)

    passed = not reasons
    return {
        "applied": True,
        "passed": passed,
        "reward_zeroed": not passed,
        "status": "passed" if passed else "failed",
        "judgment": "PASS" if passed else "FAIL",
        "reason": " | ".join(reasons),
        "has_uncommon_characters": bool(suspicious_characters),
        "has_repetition": bool(repetition_reason),
        "has_non_english_characters": bool(non_english_characters),
        "suspicious_characters": suspicious_characters,
        "non_english_characters": non_english_characters,
    }


def _initialize_audit_metadata(metadata: dict, *, enabled: bool) -> None:
    metadata["skill_language_audit_enabled"] = 1.0 if enabled else 0.0
    metadata["skill_language_audit_applied"] = 0.0
    metadata["skill_language_audit_pass"] = 0.0
    metadata["skill_language_audit_has_uncommon_characters"] = 0.0
    metadata["skill_language_audit_has_non_english_characters"] = 0.0
    metadata["skill_language_audit_has_repetition"] = 0.0
    metadata["skill_language_audit_penalty"] = 0.0
    metadata["skill_language_audit_status"] = "disabled" if not enabled else "skipped"
    metadata["skill_language_audit_judgment"] = "SKIP"
    metadata["skill_language_audit_reason"] = ""
    metadata["skill_language_audit_reward_zeroed"] = 0.0
    metadata["reward_zeroed_by_language_audit"] = 0.0


def _maybe_print_audit_sample(
    sample: Sample,
    *,
    tag: str,
    print_prob: float,
) -> None:
    if random.random() >= print_prob:
        return

    metadata = _ensure_metadata(sample)
    payload = {
        "tag": tag,
        "judgment": metadata.get("skill_language_audit_judgment"),
        "status": metadata.get("skill_language_audit_status"),
        "reason": metadata.get("skill_language_audit_reason"),
        "penalty": metadata.get("skill_language_audit_penalty"),
        "skill_language_audit_reward_zeroed": metadata.get("skill_language_audit_reward_zeroed"),
        "reward_zeroed_by_language_audit": metadata.get("reward_zeroed_by_language_audit"),
        "has_uncommon_characters": metadata.get("skill_language_audit_has_uncommon_characters"),
        "has_non_english_characters": metadata.get("skill_language_audit_has_non_english_characters"),
        "has_repetition": metadata.get("skill_language_audit_has_repetition"),
        "source_task_description": metadata.get("source_task_description"),
        "source_task_type": metadata.get("source_task_type"),
        "requested_skill_source_mode": metadata.get("requested_skill_source_mode"),
        "skill_source_mode": metadata.get("skill_source_mode"),
    }

    print("--------------------------------")
    print(f"[{tag}]")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    skill_text = str(metadata.get("skill") or "").strip()
    if skill_text:
        print("[skill_markdown]")
        print(skill_text)


async def _audit_single_extractor_sample(args, sample: Sample, domain: str) -> None:
    metadata = _ensure_metadata(sample)
    skill_text = str(metadata.get("skill") or "").strip()
    format_score = float(metadata.get("format_score", 0.0) or 0.0)

    enabled = _resolve_rule_based_audit_enabled(args, domain)
    _initialize_audit_metadata(metadata, enabled=enabled)

    if format_score < 1.0 or not skill_text:
        metadata["skill_language_audit_status"] = "skipped_non_markdown_skill"
        metadata["skill_language_audit_reason"] = "skill_markdown_unavailable"
        return

    if not enabled:
        metadata["skill_language_audit_status"] = "disabled"
        metadata["skill_language_audit_reason"] = "audit_disabled"
        return

    result = _run_rule_based_audit(skill_text)

    metadata["skill_language_audit_applied"] = 1.0
    metadata["skill_language_audit_pass"] = 1.0 if result["passed"] else 0.0
    metadata["skill_language_audit_has_uncommon_characters"] = (
        1.0 if result["has_uncommon_characters"] else 0.0
    )
    metadata["skill_language_audit_has_non_english_characters"] = (
        1.0 if result["has_non_english_characters"] else 0.0
    )
    metadata["skill_language_audit_has_repetition"] = (
        1.0 if result["has_repetition"] else 0.0
    )
    metadata["skill_language_audit_reward_zeroed"] = 1.0 if result["reward_zeroed"] else 0.0
    metadata["skill_language_audit_status"] = result["status"]
    metadata["skill_language_audit_judgment"] = result["judgment"]
    metadata["skill_language_audit_reason"] = result["reason"]

    if result["reward_zeroed"]:
        _maybe_print_audit_sample(
            sample,
            tag=f"{domain}/skill_language_audit_failed_sample",
            print_prob=_FAILED_AUDIT_PRINT_PROB,
        )
        return

    if result["passed"]:
        _maybe_print_audit_sample(
            sample,
            tag=f"{domain}/skill_language_audit_passed_sample",
            print_prob=_PASSED_AUDIT_PRINT_PROB,
        )


async def audit_extractor_samples(
    args,
    extractor_samples: list[Sample],
    *,
    domain: str,
) -> None:
    if not extractor_samples:
        return

    semaphore = asyncio.Semaphore(_AUDIT_MAX_CONCURRENCY)

    async def _audit_with_limit(sample: Sample) -> None:
        async with semaphore:
            await _audit_single_extractor_sample(args, sample, domain)

    await asyncio.gather(*[_audit_with_limit(sample) for sample in extractor_samples])
