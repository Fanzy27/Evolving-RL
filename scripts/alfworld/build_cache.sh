#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
DATA_DIR="${ROOT_DIR}/data/alfworld"
CACHE_DIR="${DATA_DIR}/cache"
MODEL_NAME="Qwen/Qwen3-Embedding-4B"
QUESTION_COL="question"
BATCH_SIZE=32

mkdir -p "${CACHE_DIR}"

python3 "${ROOT_DIR}/env/alfworld/retrieve_server/build_unified_cache.py" \
  --parquets "${DATA_DIR}/train.parquet" "${DATA_DIR}/valid_train.parquet" "${DATA_DIR}/valid_seen.parquet" "${DATA_DIR}/valid_unseen.parquet" \
  --output "${CACHE_DIR}/unified_embeddings.npz" \
  --model_name "${MODEL_NAME}" \
  --question_col "${QUESTION_COL}" \
  --batch_size "${BATCH_SIZE}"

echo "[alfworld/cache] cache ready at ${CACHE_DIR}/unified_embeddings.npz"
