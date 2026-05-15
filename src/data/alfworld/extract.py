#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from glob import glob
from typing import Any, Dict, List, Optional
from collections import Counter

import pandas as pd


KNOWN_SPLITS = ["train", "valid_seen", "valid_train", "valid_unseen"]

TASK_TYPE_MAP = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}


def safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_list_of_str(x: Any) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, list):
        vals = [str(v) for v in x if v is not None]
        return vals if vals else None
    return [str(x)]


def normalize_task_type(task_type: Any) -> Any:
    if isinstance(task_type, int):
        return TASK_TYPE_MAP.get(task_type, task_type)
    return task_type


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return False


def extract_task_and_trial(rel_dir: str):
    rel_dir = rel_dir.replace("\\", "/").strip("/")
    parts = rel_dir.split("/")

    task_id = None
    trial_id = None
    game_rel_path = None

    if len(parts) >= 2:
        task_id = parts[0]
        trial_id = parts[1]
        game_rel_path = "/".join(parts[:2])
    elif len(parts) == 1:
        task_id = parts[0]

    return task_id, trial_id, game_rel_path


def clean_prompt(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[ \t\r\n\.]+$", "", text)
    return text


def extract_prompt_from_grammar(grammar: Optional[str]) -> Optional[str]:
    if not grammar or not isinstance(grammar, str):
        return None

    m = re.search(r'Your task is to:\s*([^"\n]+)', grammar, flags=re.IGNORECASE)
    if not m:
        return None

    prompt = clean_prompt(m.group(1))
    return prompt if prompt else None


def extract_expert_trajectory(game_data: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    if not game_data:
        return None
    for key in ["walkthrough", "expert_plan", "plan"]:
        vals = normalize_list_of_str(game_data.get(key))
        if vals:
            return vals
    return None


def extract_first_ann(traj_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the first annotation from traj_data['anns'], if present.

    Returns a dict with:
        task_desc  (str)        – natural-language task description
        high_descs (list[str])  – per-step high-level descriptions
    or None if no annotations are available.
    """
    anns = traj_data.get("turk_annotations")
    anns = anns.get("anns")
    # import pdb; pdb.set_trace()
    if not anns or not isinstance(anns, list):
        return None
    
    first = anns[0]
    if not isinstance(first, dict):
        return None

    task_desc = first.get("task_desc")
    high_descs = normalize_list_of_str(first.get("high_descs"))

    if not task_desc and not high_descs:
        return None

    result: Dict[str, Any] = {}
    if task_desc:
        result["task_desc"] = str(task_desc)
    if high_descs:
        result["high_descs"] = high_descs
    return result


def build_record(
    split: str,
    split_dir: str,
    traj_json_path: str,
    sample_index: int,
    stats: Counter,
    keep_without_walkthrough: bool = True,
) -> Optional[Dict[str, Any]]:
    stats["total"] += 1

    game_dir = os.path.dirname(traj_json_path)
    rel_dir = os.path.relpath(game_dir, split_dir).replace("\\", "/")
    task_id, _, _ = extract_task_and_trial(rel_dir)

    game_file_path = os.path.join(game_dir, "game.tw-pddl")

    traj_data = safe_read_json(traj_json_path)
    if traj_data is None:
        stats["missing_or_bad_traj_data"] += 1
        return None

    if not os.path.exists(game_file_path):
        stats["missing_game_file"] += 1
        return None

    game_data = safe_read_json(game_file_path)
    if game_data is None:
        stats["missing_or_bad_game_data"] += 1
        return None

    if "solvable" not in game_data:
        stats["missing_solvable"] += 1
        return None

    if not to_bool(game_data.get("solvable")):
        stats["unsolvable"] += 1
        return None

    grammar = game_data.get("grammar")
    if not grammar:
        stats["missing_grammar"] += 1
        return None

    prompt_text = extract_prompt_from_grammar(grammar)
    if not prompt_text:
        stats["missing_prompt"] += 1
        return None

    expert_trajectory = extract_expert_trajectory(game_data)
    if not expert_trajectory:
        stats["missing_walkthrough"] += 1
        if not keep_without_walkthrough:
            return None

    task_type = normalize_task_type(traj_data.get("task_type"))
    first_ann = extract_first_ann(traj_data)
    if first_ann is None:
        stats["missing_ann"] += 1

    metadata: Dict[str, Any] = {
        "split": split,
        "task_type": task_type,
        "rel_dir": rel_dir,
        "task_id": task_id,
        "expert_trajectory": expert_trajectory,
    }
    if first_ann is not None:
        metadata["ann"] = first_ann

    record = {
        "id": f"{split}_{sample_index}",
        "question": prompt_text,
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "metadata": metadata,
    }

    stats["kept"] += 1
    return record


def convert_one_split(data_dir: str, split: str, output_dir: str, keep_without_walkthrough: bool = True):
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[Skip] split dir not found: {split_dir}")
        return

    traj_files = sorted(glob(os.path.join(split_dir, "**", "traj_data.json"), recursive=True))
    if not traj_files:
        print(f"[Skip] no traj_data.json found under: {split_dir}")
        return

    stats = Counter()
    records = []

    for traj_json_path in traj_files:
        try:
            rec = build_record(
                split=split,
                split_dir=split_dir,
                traj_json_path=traj_json_path,
                sample_index=len(records),
                stats=stats,
                keep_without_walkthrough=keep_without_walkthrough,
            )
            if rec is not None:
                records.append(rec)
        except Exception as e:
            stats["exceptions"] += 1
            print(f"[Warn] failed on {traj_json_path}: {e}")

    df = pd.DataFrame(records)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}.parquet")
    df.to_parquet(output_path, index=False)

    print(f"[extract] {split}: {len(df)} tasks -> {output_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Convert ALFWorld dataset to a unified parquet format with id/prompt/metadata."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="ALFWorld root dir, e.g. /root/.cache/alfworld/json_2.1.1"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save split parquet files"
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=KNOWN_SPLITS,
        help=f"Splits to export. Default: {' '.join(KNOWN_SPLITS)}"
    )
    parser.add_argument(
        "--drop_without_walkthrough",
        action="store_true",
        help="If set, drop samples without walkthrough. Default: keep them."
    )
    args = parser.parse_args()

    keep_without_walkthrough = not args.drop_without_walkthrough

    for split in args.splits:
        convert_one_split(
            data_dir=args.data_dir,
            split=split,
            output_dir=args.output_dir,
            keep_without_walkthrough=keep_without_walkthrough,
        )


if __name__ == "__main__":
    main()
