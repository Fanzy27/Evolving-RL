#!/usr/bin/env bash
# Extract ALFWorld raw dataset to parquet and split by task type.
# Requires: data/alfworld/task_files/json_2.1.1/ to exist.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
DATA_DIR="${ROOT_DIR}/data/alfworld"
TASK_FILES="${DATA_DIR}/task_files"

if [ ! -d "${TASK_FILES}/json_2.1.1" ]; then
  echo "[alfworld] ERROR: raw dataset not found at ${TASK_FILES}/json_2.1.1" >&2
  echo "[alfworld] Place the ALFWorld official dataset under ${TASK_FILES}/" >&2
  exit 1
fi

echo "[alfworld] extracting parquets from raw dataset ..."
python3 "${ROOT_DIR}/src/data/alfworld/extract.py" \
  --data_dir "${TASK_FILES}/json_2.1.1" \
  --output_dir "${DATA_DIR}"

echo "[alfworld] splitting train.parquet by task type ..."
python3 "${ROOT_DIR}/src/data/alfworld/split.py" \
  --input "${DATA_DIR}/train.parquet" \
  --output-dir "${DATA_DIR}"

echo "[alfworld] data preparation complete"
