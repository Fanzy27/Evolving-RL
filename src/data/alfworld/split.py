"""Split ALFWorld train.parquet by task type into two subsets."""

import argparse
import os

import pandas as pd


TASK_TYPES_1 = [
    "pick_and_place_simple",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

TASK_TYPES_2 = [
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
]


def main():
    parser = argparse.ArgumentParser(description="Split ALFWorld training data by task type.")
    parser.add_argument("--input", required=True, help="Path to train.parquet")
    parser.add_argument("--output-dir", required=True, help="Directory to write train_1.parquet and train_2.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    df1 = df[df["metadata"].apply(lambda x: isinstance(x, dict) and x.get("task_type") in TASK_TYPES_1)].copy()
    df2 = df[df["metadata"].apply(lambda x: isinstance(x, dict) and x.get("task_type") in TASK_TYPES_2)].copy()

    df1.to_parquet(os.path.join(args.output_dir, "train_seen.parquet"), index=False)
    df2.to_parquet(os.path.join(args.output_dir, "train_unseen.parquet"), index=False)

    print(f"[split] train_seen: {len(df1)} rows, train_unseen: {len(df2)} rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
