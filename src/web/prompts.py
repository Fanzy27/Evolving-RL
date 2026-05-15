"""Prompt templates for the Mind2Web-backed web experience pipeline.

Three roles share the same underlying model weights:
  1. Solver (no skill)       - solves the source web task
  2. Extractor               - extracts a reusable web skill from trajectory
  3. Solver (with skill)     - solves downstream web tasks using a skill card
"""

from __future__ import annotations

import json
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Solver prompt (no skill)
# ---------------------------------------------------------------------------

SOLVER_SYSTEM_PROMPT = (
    "You are an intelligent agent in a web browsing environment. "
    "Your goal is to complete the user's web task step by step.\n\n"
    "At each step, you will receive a compact interactive page snapshot. "
    "Each actionable element is identified by a ref like [ref=e12]. "
    "You must choose actions only from the refs that appear in the current observation.\n\n"
    "You MUST first reason inside <reasoning> and </reasoning> tags. "
    "Then output exactly one JSON action inside <action> and </action> tags.\n\n"
    "Allowed operations:\n"
    '- CLICK: {"op":"CLICK","ref":"e12"}\n'
    '- TYPE: {"op":"TYPE","ref":"e7","value":"text to type"}\n'
    '- SELECT: {"op":"SELECT","ref":"e9","value":"option text"}\n\n'
    "Do not invent refs. Do not output multiple actions. "
    "If the page contains many options, pick the single best next action."
)


def format_solver_messages(task_description: str) -> list[dict]:
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Your task is to: {task_description}"},
    ]


# ---------------------------------------------------------------------------
# 2. Extractor prompt
# ---------------------------------------------------------------------------

SKILL_JSON_SCHEMA_PROMPT = """You may optionally write a short plain-text analysis before the final skill JSON.

Output rules:
- Any analysis must be plain text, not JSON.
- The final skill must appear as a single JSON object at the end of the response.
- You may wrap the final JSON object in a ```json fenced block if helpful.
- The parser will ignore non-JSON text and extract only the final skill JSON object.

Use exactly this JSON schema:

{
  "name": "string, kebab-case",
  "description": "string, 1-2 sentences stating what the skill does and when to use it",
  "metadata": {
    "source_task_type": "string",
    "success_signal": "success or failure",
    "abstraction_level": "low, medium, or high",
    "skill_category": "string, a short descriptive category label"
  },
  "skill_overview": [
    "string",
    "string"
  ],
  "when_to_use": [
    "string"
  ],
  "when_not_to_use": [
    "string"
  ],
  "inputs_to_identify": [
    "string"
  ],
  "workflow": [
    {
      "step": 1,
      "goal": "string",
      "action": "string",
      "tools_or_strategy": "string",
      "completion_condition": "string"
    }
  ],
  "decision_rules": [
    "string"
  ],
  "output_requirements": [
    "string"
  ]
}

Field requirements:
- "name": must be valid kebab-case
- "description": must include both what the skill does and when to use it
- "skill_overview": prefer 2-5 items when evidence is sufficient; otherwise [] is allowed
- "when_to_use": prefer 2-6 items when evidence is sufficient; otherwise [] is allowed
- "when_not_to_use": prefer 1-4 items when evidence is sufficient; otherwise [] is allowed
- "inputs_to_identify": prefer at least 2 items when supported by evidence; otherwise [] is allowed
- "workflow": prefer 2-8 steps when the evidence supports a stable workflow
- "decision_rules": prefer 1-5 items when supported by evidence; otherwise [] is allowed
- "error_handling": prefer 1-5 items when supported by evidence; otherwise [] is allowed
- "output_requirements": prefer 1-5 items when supported by evidence; otherwise [] is allowed

Additional constraints:
1. Do not copy the trajectory verbatim.
2. Do not include one-off page IDs, exact refs, exact dates, or accidental details unless they represent a stable pattern.
3. If the task is only weakly generalizable, output a narrow but reusable skill.
4. The workflow must be executable and concrete.
5. Decision rules must contain real conditional guidance, not just restate the workflow.
6. Error handling must address realistic web interaction risks, especially those revealed by failure examples.
7. The final JSON must be internally consistent, specific, and free of filler.
8. Every required field must be present.
9. If evidence is insufficient, keep the field present but you may use conservative empty values such as "" or [] instead of inventing unsupported content.
10. For failed examples, a short analysis of the failure cause and the transferable lesson is strongly encouraged before the final JSON.

Before producing the final JSON, identify:
- the task type
- the key success or failure factors
- the reusable patterns vs. one-off details
"""


EXTRACTOR_INSTRUCTION_PROMPT = """Now use the example above to extract a reusable skill.

The goal is not to retell the case, but to capture a reusable skill that could help on similar web tasks.

A good skill should capture:
- what kind of web task this is
- when the skill should be used
- when it should not be used
- the stable workflow
- the decision rules behind the workflow
- common failure modes and how to handle them
- the expected final outcome

Follow these rules:

1. Abstract from the example
Infer the broader task type, user intent, target outcome, and repeatable workflow pattern.
Do not rewrite the full trajectory. Generalize it into a reusable skill.

2. Make trigger conditions explicit
The skill must clearly say:
- what it does
- when it should be used
- what page cues, fields, widgets, or states indicate applicability
- when it should not be used

3. Preserve workflow structure
If the trajectory shows a multi-step process, convert it into an ordered workflow.
If it shows branching choices, convert them into decision rules.

4. Learn from success and failure
- If the example succeeded, extract the key steps and checks that made it work.
- If the example failed, extract where it failed, why it failed, and what rules or checks would reduce future failures.

5. Stay evidence-grounded
Every major step, rule, and error-handling item should be supported by the request, trajectory, or success/failure signal.
Do not invent unsupported tools, procedures, or domain knowledge.
If evidence is limited, produce a narrower skill.

6. Analysis is allowed before the final skill JSON
You may first write a short analysis section, especially for failed examples where failure diagnosis is useful.
If you do, keep it concise, keep it plain text, and put the final JSON object last.
""" + "\n\n" + SKILL_JSON_SCHEMA_PROMPT


def format_extractor_messages(
    task_description: str,
    trajectory: str,
    won_score: float,
) -> list[dict]:
    success = won_score >= 0.9
    extractor_prompt = (
        f"Task: {task_description}\n\n"
        "Agent's Trajectory:\n"
        "<<<BEGIN_TRAJECTORY>>>\n"
        f"{trajectory}\n"
        "<<<END_TRAJECTORY>>>\n\n"
        f"True Environment Success: {success}\n\n"
        "The case details above are the full evidence you should rely on. "
        "Please read them first, then assess the trajectory quality and extract a reusable skill. "
        "If the sample failed, a brief failure analysis is especially helpful. "
        "Any analysis should come before the final skill JSON.\n\n"
        "Use the following extraction guidance and output requirements:\n\n"
        f"{EXTRACTOR_INSTRUCTION_PROMPT}"
    )
    return [{"role": "user", "content": extractor_prompt}]


# ---------------------------------------------------------------------------
# Skill JSON -> Markdown conversion
# ---------------------------------------------------------------------------

REQUIRED_SKILL_FIELDS = ["name", "description"]


def _extract_json_object(text: str) -> tuple[Optional[dict], str]:
    text = str(text or "").strip()
    if not text:
        return None, "Invalid JSON: empty input"

    skill_keys = {
        "name",
        "description",
        "metadata",
        "skill_overview",
        "when_to_use",
        "when_not_to_use",
        "inputs_to_identify",
        "workflow",
        "decision_rules",
        "error_handling",
        "output_requirements",
    }

    def candidate_rank(obj: dict, end_idx: int, raw_len: int) -> tuple[int, int, int]:
        score = sum(1 for key in skill_keys if key in obj)
        return (score, raw_len, end_idx)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, ""
    except json.JSONDecodeError:
        pass

    best_obj: Optional[dict] = None
    best_rank: tuple[int, int, int] | None = None

    for fence in ("```json", "```JSON", "```"):
        search_start = 0
        while True:
            start = text.find(fence, search_start)
            if start == -1:
                break
            content_start = start + len(fence)
            end = text.find("```", content_start)
            if end != -1:
                candidate = text[content_start:end].strip()
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        rank = candidate_rank(obj, end, len(candidate))
                        if best_rank is None or rank > best_rank:
                            best_obj = obj
                            best_rank = rank
                except json.JSONDecodeError:
                    pass
            search_start = start + len(fence)

    in_string = False
    escape = False
    brace_count = 0
    start_idx = None
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if brace_count == 0:
                start_idx = idx
            brace_count += 1
        elif ch == "}":
            if brace_count > 0:
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    candidate = text[start_idx : idx + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            rank = candidate_rank(obj, idx, len(candidate))
                            if best_rank is None or rank > best_rank:
                                best_obj = obj
                                best_rank = rank
                    except json.JSONDecodeError:
                        pass

    if best_obj is not None:
        return best_obj, ""
    return None, "Invalid JSON: no valid JSON object found in input"


def skill_json_to_markdown(json_str: str) -> tuple[str | None, str]:
    skill, err = _extract_json_object(json_str)
    if err:
        return None, err
    if not isinstance(skill, dict):
        return None, "Invalid JSON: top-level value is not an object"

    missing = [field for field in REQUIRED_SKILL_FIELDS if not skill.get(field)]
    if missing:
        return None, f"Missing required fields: {', '.join(missing)}"

    lines: list[str] = [f"# Skill: {skill['name']}", f"\n{skill['description']}"]

    meta = skill.get("metadata", {})
    if isinstance(meta, dict):
        parts = []
        for key, label in [
            ("skill_category", "Category"),
            ("source_task_type", "Source"),
            ("abstraction_level", "Level"),
            ("success_signal", "Signal"),
        ]:
            if meta.get(key):
                parts.append(f"**{label}**: {meta[key]}")
        if parts:
            lines.append("\n" + " | ".join(parts))

    overview = skill.get("skill_overview", [])
    if overview:
        lines.append("\n## Overview")
        for item in overview:
            lines.append(f"- {item}")

    when_to_use = skill.get("when_to_use", [])
    if when_to_use:
        lines.append("\n## When to Use")
        for item in when_to_use:
            lines.append(f"- {item}")

    when_not_to_use = skill.get("when_not_to_use", [])
    if when_not_to_use:
        lines.append("\n## When NOT to Use")
        for item in when_not_to_use:
            lines.append(f"- {item}")

    inputs = skill.get("inputs_to_identify", [])
    if inputs:
        lines.append("\n## Inputs to Identify")
        for item in inputs:
            lines.append(f"- {item}")

    workflow = skill.get("workflow", [])
    if workflow:
        lines.append("\n## Workflow")
        for step in workflow:
            num = step.get("step", "?")
            goal = step.get("goal", "")
            action = step.get("action", "")
            strategy = step.get("tools_or_strategy", "")
            condition = step.get("completion_condition", "")
            lines.append(f"{num}. **{goal}**: {action}")
            if strategy:
                lines.append(f"   - Strategy: {strategy}")
            if condition:
                lines.append(f"   - Done when: {condition}")

    rules = skill.get("decision_rules", [])
    if rules:
        lines.append("\n## Decision Rules")
        for item in rules:
            lines.append(f"- {item}")

    errors = skill.get("error_handling", [])
    if errors:
        lines.append("\n## Error Handling")
        for err_item in errors:
            if not isinstance(err_item, dict):
                continue
            error = err_item.get("error", "")
            cause = err_item.get("cause", "")
            response = err_item.get("response", "")
            lines.append(f"- **{error}**: {cause} -> {response}")

    output_reqs = skill.get("output_requirements", [])
    if output_reqs:
        lines.append("\n## Output Requirements")
        for item in output_reqs:
            lines.append(f"- {item}")

    return "\n".join(lines), ""


# ---------------------------------------------------------------------------
# 3. Solver prompt (with skill)
# ---------------------------------------------------------------------------

SOLVER_WITH_SKILL_SYSTEM_PROMPT_TEMPLATE = (
    "You are an intelligent agent in a web browsing environment. "
    "Your goal is to complete the user's web task step by step.\n\n"
    "At each step, you will receive a compact interactive page snapshot with refs like [ref=e12]. "
    "You MUST first reason step-by-step about the current situation inside <reasoning> and </reasoning> tags. "
    "Then output exactly one JSON action inside <action> and </action> tags.\n\n"
    "Allowed operations:\n"
    '- CLICK: {{"op":"CLICK","ref":"e12"}}\n'
    '- TYPE: {{"op":"TYPE","ref":"e7","value":"text to type"}}\n'
    '- SELECT: {{"op":"SELECT","ref":"e9","value":"option text"}}\n\n'
    "You must choose actions only from refs that appear in the current observation. "
    "Do not invent refs. Do not output multiple actions.\n\n"
    "A relevant skill extracted from a previous similar task is provided below for your reference. "
    "In your reasoning, you MUST explicitly analyze whether the provided skill is applicable.\n\n"
    "Example output format:\n"
    "<reasoning>\n"
    "I should first check whether the reference skill applies to this page and task state. "
    "The next actionable control is the location textbox, so typing there is the best next step.\n"
    "</reasoning>\n"
    '<action>{{"op":"CLICK","ref":"e12"}}</action>\n\n'
    "Reference Skill:\n{skill}"
)


def format_solver_with_skill_messages(task_description: str, skill: str) -> list[dict]:
    system = SOLVER_WITH_SKILL_SYSTEM_PROMPT_TEMPLATE.format(skill=skill)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Your task is to: {task_description}"},
    ]
