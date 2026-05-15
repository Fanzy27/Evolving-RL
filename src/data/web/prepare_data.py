"""Prepare Mind2Web task registries for the web experiment pipeline.

The output intentionally mirrors the ALFWorld parquet/jsonl training shape:

    id        synthetic sample id
    question  plain-text task description for retrieval
    prompt    chat-message array consumed by `--apply-chat-template`
    metadata  dict label used by the environment / rollout code
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import re

import pandas as pd


def _task_type(task: dict) -> str:
    domain = str(task.get("domain") or "").strip()
    subdomain = str(task.get("subdomain") or "").strip()
    if domain and subdomain:
        return f"{domain}/{subdomain}"
    return domain or subdomain or str(task.get("website") or "")


def _normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _infer_split(source_file: str) -> str:
    path = Path(source_file)
    lower_parts = [part.lower() for part in path.parts]
    lower_name = path.stem.lower()
    for candidate in ("train", "test_task", "test_website", "test_domain", "valid", "eval"):
        if candidate in lower_parts or lower_name.startswith(candidate):
            return candidate
    return lower_name or "unknown"


def _build_prompt(task_description: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": task_description}]


def _build_metadata(task: dict, *, source_file: str, task_index: int, split: str) -> dict:
    annotation_id = _normalize_text(task.get("annotation_id"))
    task_description = _normalize_text(task.get("confirmed_task"))
    metadata = {
        "annotation_id": annotation_id,
        "task_id": annotation_id,
        "task_description": task_description,
        "task_type": _normalize_text(_task_type(task)),
        "website": _normalize_text(task.get("website")),
        "domain": _normalize_text(task.get("domain")),
        "subdomain": _normalize_text(task.get("subdomain")),
        "source_file": source_file,
        "task_index": int(task_index),
        "num_steps": len(task.get("actions") or []),
        "split": split,
    }
    return metadata


def collect_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    split_counts: dict[str, int] = defaultdict(int)
    for raw_path in paths:
        source_file = str(Path(raw_path).expanduser().resolve())
        try:
            tasks = json.loads(Path(source_file).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[prepare_data] warning: skipping unreadable file {source_file}: {exc}")
            continue
        if not isinstance(tasks, list):
            print(f"[prepare_data] warning: skipping non-array file {source_file}")
            continue
        split = _infer_split(source_file)
        for task_index, task in enumerate(tasks):
            metadata = _build_metadata(
                task,
                source_file=source_file,
                task_index=task_index,
                split=split,
            )
            split_index = split_counts[split]
            split_counts[split] += 1
            records.append(
                {
                    "id": f"{split}_{split_index}",
                    "question": metadata["task_description"],
                    "prompt": _build_prompt(metadata["task_description"]),
                    "metadata": metadata,
                }
            )
    return records


def write_jsonl(records: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            ordered = {
                "id": record["id"],
                "question": record["question"],
                "prompt": record["prompt"],
                "metadata": record["metadata"],
            }
            fh.write(json.dumps(ordered, ensure_ascii=False) + "\n")


def write_parquet(records: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_records = [
        {
            "id": record["id"],
            "question": record["question"],
            "prompt": record["prompt"],
            "metadata": record["metadata"],
        }
        for record in records
    ]
    pd.DataFrame(ordered_records, columns=["id", "question", "prompt", "metadata"]).to_parquet(
        path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mind2Web task registry for web experiments.")
    parser.add_argument(
        "--data-glob",
        nargs="+",
        required=True,
        help="One or more glob patterns pointing to Mind2Web split json files.",
    )
    parser.add_argument("--output", required=True, help="Output path (.jsonl or .parquet).")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.data_glob:
        paths.extend(sorted(glob.glob(pattern)))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No Mind2Web files matched --data-glob.")

    records = collect_records(paths)
    output = str(Path(args.output).expanduser())
    if output.endswith(".parquet"):
        write_parquet(records, output)
    else:
        write_jsonl(records, output)

    print(f"[prepare_data] wrote {len(records)} task records to {output}")


if __name__ == "__main__":
    main()
