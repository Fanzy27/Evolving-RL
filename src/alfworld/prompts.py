"""Prompt templates for the ALFWorld Experience Extractor rollout pipeline.

Three roles share the same underlying model weights:
  1. Solver (no experience)  – plays ALFWorld episode on source task (Step 1)
  2. Extractor               – extracts a generalizable skill (JSON → Markdown) from trajectory (Step 2)
  3. Solver (with skill)     – plays downstream ALFWorld tasks using a retrieved skill (Step 4)
"""

import json

# ---------------------------------------------------------------------------
# 1. Solver prompt (no experience)  ← Step 1
# ---------------------------------------------------------------------------

SOLVER_SYSTEM_PROMPT = (
    "You are an intelligent agent in the household environment. "
    "Your goal is to complete the given household task by taking actions step by step.\n\n"
    "At each step, you will be given the current observation and a list of admissible actions. "
    "You MUST first reason step-by-step about the current situation inside <reasoning> and </reasoning> tags. "
    "Then choose exactly ONE admissible action and output it inside <action> and </action> tags.\n\n"
    "The action MUST be copied verbatim from the admissible actions list — do not paraphrase or invent actions.\n\n"
    "Example output format:\n"
    "<reasoning>\n"
    "I need to find the apple. The task says to put it in the fridge. "
    "I should first look around the room.\n"
    "</reasoning>\n"
    "<action>look</action>"
)



def format_solver_messages(task_description: str) -> list[dict]:
    """Chat messages for the Solver without any experience context (Step 1).

    The initial observation and admissible commands are NOT included here —
    they are prepended as the first observation in the episode response.

    Args:
        task_description: The raw task description (e.g. "put a fork in a drawer").

    Returns:
        A list of chat message dicts compatible with tokenizer.apply_chat_template().
    """
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Your task is to: {task_description}"},
    ]


# ---------------------------------------------------------------------------
# 2. Extractor prompt  ← Step 2
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
  "error_handling": [
    {
      "error": "string",
      "cause": "string",
      "response": "string"
    }
  ],
  "output_requirements": [
    "string"
  ],
  ]
}

Field requirements:
- "name": must be valid kebab-case
- "description": must include both what the skill does and when to use it
- "skill_overview": prefer 2-5 items when evidence is sufficient; otherwise [] is allowed
- "when_to_use": prefer 2-6 items when evidence is sufficient; otherwise [] is allowed
- "inputs_to_identify": prefer at least 2 items when supported by evidence; otherwise [] is allowed
- "workflow": prefer 2-8 steps when the evidence supports a stable workflow
- "decision_rules": prefer 1-5 items when supported by evidence; otherwise [] is allowed
- "error_handling": prefer 1-5 items when supported by evidence; otherwise [] is allowed
- "output_requirements": prefer 1-5 items when supported by evidence; otherwise [] is allowed

Additional constraints:
1. Do not copy the trajectory verbatim.
2. Do not include one-off entities, IDs, paths, or accidental details unless they represent a stable pattern.
3. If the task is only weakly generalizable, output a narrow but reusable skill.
4. The workflow must be executable and concrete.
5. Decision rules must contain real conditional guidance, not just repeat the workflow.
6. Error handling must address realistic risks, especially those revealed by failure examples.
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

The goal is not to retell the case, but to capture a reusable skill that could help on similar tasks.

A good skill should capture:
- what kind of task this is
- when the skill should be used
- when it should not be used
- the stable workflow
- the decision rules behind the workflow
- common failure modes and how to handle them
- the expected final output

Follow these rules:

1. Abstract from the example
Infer the broader task type, user intent, target outcome, and repeatable workflow pattern.
Do not rewrite the full trajectory. Generalize it into a reusable skill.

2. Make trigger conditions explicit
The skill must clearly say:
- what it does
- when it should be used
- how users may express this need
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





def _strip_admissible_actions_block(trajectory: str) -> str:
    text = str(trajectory or "")
    if not text:
        return ""

    result_parts: list[str] = []
    pos = 0
    marker = "Admissible actions:"

    while True:
        marker_idx = text.find(marker, pos)
        if marker_idx == -1:
            result_parts.append(text[pos:])
            break

        line_start = text.rfind("\n", 0, marker_idx) + 1
        result_parts.append(text[pos:line_start])

        bracket_start = text.find("[", marker_idx)
        if bracket_start == -1:
            line_end = text.find("\n", marker_idx)
            if line_end == -1:
                break
            pos = line_end + 1
            result_parts.append("\n")
            continue

        depth = 0
        end_idx = bracket_start
        while end_idx < len(text):
            char = text[end_idx]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    end_idx += 1
                    break
            end_idx += 1

        if depth > 0:
            pos = marker_idx
            result_parts.append(text[pos:])
            break

        while end_idx < len(text) and text[end_idx] in " \t":
            end_idx += 1
        while end_idx < len(text) and text[end_idx] in "\r\n":
            end_idx += 1

        pos = end_idx
        result_parts.append("\n")

    cleaned = "".join(result_parts)
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

    normalized_lines: list[str] = []
    prev_blank = False
    for line in cleaned.split("\n"):
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        normalized_lines.append(line)
        prev_blank = is_blank

    return "\n".join(normalized_lines).strip()


def format_extractor_messages(
    task_description: str, trajectory: str, won_score: float
) -> list[dict]:
    """Chat messages for the Extractor agent (Step 2).

    Args:
        task_description: The household task the Solver attempted.
        trajectory: The full episode transcript (obs + reasoning + action per step).
        won_score: 1.0 if the episode succeeded, 0.0 if failed.

    Returns:
        A list of chat message dicts compatible with tokenizer.apply_chat_template().
    """

    success = won_score >= 0.9
    # cleaned_trajectory = _strip_admissible_actions_block(trajectory)
    cleaned_trajectory = trajectory

    extractor_prompt = (
        f"Task: {task_description}\n\n"
        "Agent's Trajectory:\n"
        "<<<BEGIN_TRAJECTORY>>>\n"
        f"{cleaned_trajectory}\n"
        "<<<END_TRAJECTORY>>>\n\n"
        f"True Environment Success: {success}\n\n"
        "The case details above are the full evidence you should rely on. "
        "Please read them first, then assess the trajectory quality and extract a reusable skill. "
        "If the sample failed, a brief failure analysis is especially helpful. "
        "Any analysis should come before the final skill JSON.\n\n"
        "Use the following extraction guidance and output requirements:\n\n"
        f"{EXTRACTOR_INSTRUCTION_PROMPT}"
    )
    return [
        {"role": "user", "content": extractor_prompt},
    ]


# ---------------------------------------------------------------------------
# Skill JSON → Markdown conversion
# ---------------------------------------------------------------------------

# Fields that must be present and non-empty for a valid skill. Other fields may
# remain empty when the evidence is insufficient.
REQUIRED_SKILL_FIELDS = ["name", "description"]


from typing import Optional

def _extract_json_object(text: str) -> tuple[Optional[dict], str]:
    """Extract and parse the best valid JSON object from arbitrary text.

    Supports:
    - raw JSON object
    - fenced code block like ```json ... ```
    - surrounding extra text before/after the JSON
    - optional analysis before the final JSON

    Returns:
        (obj, "")
        (None, error_message)
    """
    text = text.strip()
    skill_keys = {
        "name",
        "description",
        "metadata",
        "skill_overview",
        "when_to_use",
        "inputs_to_identify",
        "workflow",
        "decision_rules",
        "error_handling",
        "output_requirements",
    }

    def _candidate_rank(obj: dict, end_idx: int, raw_len: int) -> tuple[int, int, int]:
        score = sum(1 for key in skill_keys if key in obj)
        return (score, raw_len, end_idx)

    # 1) Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, ""
    except json.JSONDecodeError:
        pass

    best_obj: Optional[dict] = None
    best_rank: tuple[int, int, int] | None = None

    # 2) Try fenced code block extraction
    fence_starts = ["```json", "```JSON", "```"]
    for fence in fence_starts:
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
                        rank = _candidate_rank(obj, end, len(candidate))
                        if best_rank is None or rank > best_rank:
                            best_obj = obj
                            best_rank = rank
                except json.JSONDecodeError:
                    pass
            search_start = start + len(fence)

    # 3) Scan for balanced {...} and choose the best valid candidate
    in_string = False
    escape = False
    brace_count = 0
    start_idx = None

    for i, ch in enumerate(text):
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
                start_idx = i
            brace_count += 1
        elif ch == "}":
            if brace_count > 0:
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    candidate = text[start_idx:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            rank = _candidate_rank(obj, i, len(candidate))
                            if best_rank is None or rank > best_rank:
                                best_obj = obj
                                best_rank = rank
                    except json.JSONDecodeError:
                        continue

    if best_obj is not None:
        return best_obj, ""

    return None, "Invalid JSON: no valid JSON object found in input"


def skill_json_to_markdown(json_str: str) -> tuple[str | None, str]:
    """Parse extractor JSON output and convert to a Markdown skill card.

    Args:
        json_str: Raw string output from the Extractor (may contain extra text).

    Returns:
        (markdown_text, error_message)
        - On success:              (markdown, "")
        - Missing required fields: (None, "Missing required fields: ...")
        - Invalid JSON:            (None, "Invalid JSON: ...")
    """
    skill, err = _extract_json_object(json_str)
    if err:
        return None, err

    if not isinstance(skill, dict):
        return None, "Invalid JSON: top-level value is not an object"

    missing = [f for f in REQUIRED_SKILL_FIELDS if not skill.get(f)]
    if missing:
        return None, f"Missing required fields: {', '.join(missing)}"

    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    lines.append(f"# Skill: {skill['name']}")
    lines.append(f"\n{skill['description']}")

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

    # ── Overview ─────────────────────────────────────────────────────────────
    overview = skill.get("skill_overview", [])
    if overview:
        lines.append("\n## Overview")
        for item in overview:
            lines.append(f"- {item}")

    # ── When to use / When NOT to use ────────────────────────────────────────
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

    # ── Inputs ───────────────────────────────────────────────────────────────
    inputs = skill.get("inputs_to_identify", [])
    if inputs:
        lines.append("\n## Inputs to Identify")
        for item in inputs:
            lines.append(f"- {item}")

    # ── Workflow ─────────────────────────────────────────────────────────────
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

    # ── Decision rules ───────────────────────────────────────────────────────
    rules = skill.get("decision_rules", [])
    if rules:
        lines.append("\n## Decision Rules")
        for item in rules:
            lines.append(f"- {item}")

    # ── Error handling ───────────────────────────────────────────────────────
    errors = skill.get("error_handling", [])
    if errors:
        lines.append("\n## Error Handling")
        for err in errors:
            error = err.get("error", "")
            cause = err.get("cause", "")
            response = err.get("response", "")
            lines.append(f"- **{error}**: {cause} → {response}")

    # ── Output requirements ─────────────────────────────────────────────────
    output_reqs = skill.get("output_requirements", [])
    if output_reqs:
        lines.append("\n## Output Requirements")
        for item in output_reqs:
            lines.append(f"- {item}")

    return "\n".join(lines), ""



# ---------------------------------------------------------------------------
# 3. Solver prompt (with skill)  ← Step 4
# ---------------------------------------------------------------------------

SOLVER_WITH_SKILL_SYSTEM_PROMPT_TEMPLATE = (
    "You are an intelligent agent in the ALFRED household environment. "
    "Your goal is to complete the given household task by taking actions step by step.\n\n"
    "At each step, you will be given the current observation and a list of admissible actions. "
    "You MUST first reason step-by-step about the current situation inside <reasoning> and </reasoning> tags. "
    "Then choose exactly ONE admissible action and output it inside <action> and </action> tags.\n\n"
    "The action MUST be copied verbatim from the admissible actions list — do not paraphrase or invent actions.\n\n"
    "A relevant skill extracted from a previous similar task is provided below for your reference. "
    "In your reasoning, you MUST explicitly analyze whether the provided skill is applicable."
    "Example output format:\n"
    "<reasoning>\n"
    "I need to find the apple. The task says to put it in the fridge. "
    "I should first look around the room.\n"
    "</reasoning>\n"
    "<action>look</action>\n\n"
    "Reference Skill:\n{skill}"
)




def format_solver_with_skill_messages(
    task_description: str, skill: str
) -> list[dict]:
    """Chat messages for the Solver conditioned on a retrieved skill (Step 4).

    Args:
        task_description: The downstream task the Solver should attempt.
        skill: The Markdown-formatted skill card extracted by the Extractor.

    Returns:
        A list of chat message dicts compatible with tokenizer.apply_chat_template().
    """
    system = SOLVER_WITH_SKILL_SYSTEM_PROMPT_TEMPLATE.format(skill=skill)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Your task is to: {task_description}"},
    ]
