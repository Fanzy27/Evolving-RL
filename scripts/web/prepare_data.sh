#!/usr/bin/env bash
# Download Mind2Web dataset and convert to parquet/jsonl.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
DATA_DIR="${ROOT_DIR}/data/web"
DATASET_DIR="${DATA_DIR}/Mind2Web_dataset"
AUX_DIR="${DATA_DIR}/Mind2Web_data"
HF_URL="${HF_URL:-https://huggingface.co/datasets/osunlp/Mind2Web}"
TEST_ZIP_PASSWORD="${MIND2WEB_TEST_ZIP_PASSWORD:-mind2web}"

mkdir -p "${DATA_DIR}"

# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

if [ ! -e "${DATASET_DIR}" ]; then
  echo "[web] cloning Mind2Web dataset from HuggingFace ..."
  git lfs install --skip-repo >/dev/null 2>&1 || true
  git clone "${HF_URL}" "${DATASET_DIR}"
  ( cd "${DATASET_DIR}" && git lfs pull || true )
fi

if [ -f "${DATASET_DIR}/test.zip" ]; then
  DATASET_DIR="${DATASET_DIR}" TEST_ZIP_PASSWORD="${TEST_ZIP_PASSWORD}" python3 - <<'PY'
import os, shutil, tempfile, zipfile
from pathlib import Path

dataset_dir = Path(os.environ["DATASET_DIR"]).resolve()
password = os.environ["TEST_ZIP_PASSWORD"].encode("utf-8")
zip_path = dataset_dir / "test.zip"

restored = []
with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        if info.is_dir() or not info.filename.endswith(".json"):
            continue
        target = dataset_dir / info.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        fd, tmp_name = tempfile.mkstemp(prefix=f".tmp_{target.stem}_", suffix=target.suffix, dir=str(target.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with zf.open(info, "r", pwd=password) as src, tmp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            if tmp_path.stat().st_size != info.file_size:
                raise RuntimeError(f"size mismatch: {tmp_path.stat().st_size} != {info.file_size}")
            tmp_path.replace(target)
            restored.append(info.filename)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
if restored:
    print(f"[web] restored {len(restored)} official test split files from test.zip")
else:
    print("[web] official test split files already complete")
PY
fi

# ---------------------------------------------------------------------------
# 2. Convert JSON -> parquet / jsonl
# ---------------------------------------------------------------------------

echo "[web] converting Mind2Web JSON to parquet/jsonl ..."
for split in train test_task test_website test_domain; do
  case "${split}" in
    train)  globs=("${DATASET_DIR}/data/train/*.json") ;;
    *)      globs=("${DATASET_DIR}/data/${split}/*.json" "${DATASET_DIR}/${split}/*.json") ;;
  esac
  python3 "${ROOT_DIR}/src/data/web/prepare_data.py" --data-glob "${globs[@]}" --output "${DATA_DIR}/${split}.parquet"
  python3 "${ROOT_DIR}/src/data/web/prepare_data.py" --data-glob "${globs[@]}" --output "${DATA_DIR}/${split}.jsonl"
done

# ---------------------------------------------------------------------------
# 3. Merge splits
# ---------------------------------------------------------------------------

DATA_DIR="${DATA_DIR}" python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

data_dir = Path(os.environ["DATA_DIR"]).resolve()

def merge_parquet(inputs, output):
    pd.concat([pd.read_parquet(p) for p in inputs], ignore_index=True).to_parquet(output, index=False)

def merge_jsonl(inputs, output):
    with Path(output).open("w", encoding="utf-8") as out_f:
        for path in inputs:
            with Path(path).open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)

all_pq = [data_dir / f"{s}.parquet" for s in ("train", "test_task", "test_website", "test_domain")]
all_jl = [data_dir / f"{s}.jsonl"   for s in ("train", "test_task", "test_website", "test_domain")]

merge_parquet(all_pq, data_dir / "all.parquet")
merge_jsonl(all_jl, data_dir / "all.jsonl")

print("[web] merged all.parquet/jsonl from official splits")
PY

echo "[web] data preparation complete"
