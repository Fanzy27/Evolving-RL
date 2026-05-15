#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
WORK_DIR="${ROOT_DIR}"
PYTHON_BIN="python3"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MEGATRON_DIR="/root/Megatron-LM"
SLIME_TRAIN_SCRIPT="slime/train.py"
MODEL_SOURCE_ARGS="slime/scripts/models/qwen2.5-7B.sh"
HF_CHECKPOINT="$WORK_DIR/models/Qwen2.5-7B-Instruct"
REF_CHECKPOINT="$WORK_DIR/models/Qwen2.5-7B-Instruct_torch_dist"
EXP_NAME="mind2web1"

SAVE_CHECKPOINT="${WORK_DIR}/models/trained/web/$EXP_NAME"

DATA_DIR="${ROOT_DIR}/data/web"
TRAIN_DATA="${DATA_DIR}/train.parquet"
TEST_TASK_DATA="${DATA_DIR}/test_task.parquet"
TEST_WEBSITE_DATA="${DATA_DIR}/test_website.parquet"
TEST_DOMAIN_DATA="${DATA_DIR}/test_domain.parquet"
ALL_DATA="${DATA_DIR}/all.parquet"
CACHE_FILE="${DATA_DIR}/cache/unified_embeddings.npz"

WEB_CONFIG="configs/web_server.yaml"
_cfg() { python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
WEB_ENV_URL=$(_cfg "$WEB_CONFIG" web_env_url)
WEB_RETRIEVE_URL_TRAIN=$(_cfg "$WEB_CONFIG" web_retrieve_url_train)
WEB_RETRIEVE_URL_TEST=$(_cfg "$WEB_CONFIG" web_retrieve_url_test)
WEB_HOST=$(_cfg "$WEB_CONFIG" web_host)
WEB_ENV_PORT=$(_cfg "$WEB_CONFIG" web_env_port)
WEB_RETRIEVE_TRAIN_PORT=$(_cfg "$WEB_CONFIG" web_retrieve_train_port)
WEB_RETRIEVE_TEST_PORT=$(_cfg "$WEB_CONFIG" web_retrieve_test_port)
WEB_MAX_DEPTH=$(_cfg "$WEB_CONFIG" web_max_depth)
WEB_MAX_WORKERS=$(_cfg "$WEB_CONFIG" web_max_workers)
WEB_PID_FILE="${ROOT_DIR}/$(_cfg "$WEB_CONFIG" web_pid_file)"
WEB_STARTUP_TIMEOUT=$(_cfg "$WEB_CONFIG" web_startup_timeout)
WEB_RETRIEVE_STARTUP_TIMEOUT=$(_cfg "$WEB_CONFIG" web_retrieve_startup_timeout)
WEB_RETRIEVE_HOST=$(_cfg "$WEB_CONFIG" web_retrieve_host)
WEB_RETRIEVE_QUESTION_COL=$(_cfg "$WEB_CONFIG" web_retrieve_question_col)
LOG_DIR="${ROOT_DIR}/$(_cfg "$WEB_CONFIG" web_log_dir)"
MAX_EPISODE_STEPS="50"
SOLVER_MAX_RESPONSE_LEN="4096"



MASTER_ADDR="127.0.0.1"
NUM_GPUS="8"
RAY_DASHBOARD_PORT="8265"

mkdir -p "${LOG_DIR}" "${DATA_DIR}/cache"

wait_for_health() {
  local url="$1"
  local timeout="${2:-200}"
  "${PYTHON_BIN}" - "$url" "$timeout" <<'PY'
import sys
import time
from urllib.request import Request, urlopen

url = sys.argv[1]
timeout = float(sys.argv[2])
deadline = time.time() + timeout
last_err = None

while time.time() < deadline:
    try:
        with urlopen(Request(url, method="GET"), timeout=5) as resp:
            if 200 <= resp.status < 300:
                sys.exit(0)
    except Exception as exc:
        last_err = exc
    time.sleep(1.0)

raise SystemExit(f"[web/train] timed out waiting for {url}: {last_err}")
PY
}

if [ ! -f "${MODEL_SOURCE_ARGS}" ]; then
  echo "[web/train] missing model args script: ${MODEL_SOURCE_ARGS}" >&2
  exit 1
fi

if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${TEST_TASK_DATA}" ] || [ ! -f "${TEST_WEBSITE_DATA}" ] || [ ! -f "${TEST_DOMAIN_DATA}" ] || [ ! -f "${ALL_DATA}" ]; then
  bash "${ROOT_DIR}/scripts/web/prepare_data.sh"
fi

if [ ! -f "${CACHE_FILE}" ]; then
  bash "${ROOT_DIR}/scripts/web/build_cache.sh"
fi

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
python3 "${ROOT_DIR}/env/web/stop_servers.py" --pid-file "${WEB_PID_FILE}" || true
pkill -f "env/web/retrieve_server.py" 2>/dev/null || true
sleep 2

source "${MODEL_SOURCE_ARGS}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l | tr -d ' ')
if [ "${NVLINK_COUNT}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_CHECKPOINT}"
  --load "${SAVE_CHECKPOINT}"
  --save "${SAVE_CHECKPOINT}"
  --save-interval  50
)

CUSTOM_ARGS=(
  --custom-generate-function-path src.web.pipeline.web_pipeline
  --custom-reward-post-process-path src.web.pipeline.web_reward_post_process
  --custom-eval-rollout-log-function-path src.web.loggers.eval_logger.log_eval_by_task_type
  --custom-rollout-log-function-path src.web.loggers.rollout_logger.log_rollout_data
  --custom-config-path configs/web_server.yaml
)

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt
  --label-key metadata
  --apply-chat-template
  --rollout-shuffle

  --n-experiences 8
  --retrieval-topk 4
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --solver-max-response-len "${SOLVER_MAX_RESPONSE_LEN}"

  --extractor-reward-weight 0.1
  --solver-reward-weight 1.0
  --solver-temperature 1.0
  --skill-format-penalty 1

  --num-rollout 200
  --rollout-batch-size 16
  --n-samples-per-prompt 1
  --rollout-max-context-len 16384
  --rollout-max-response-len 1536
  --rollout-top-p 0.85
  --solver-top 0.85
  --rollout-temperature 1.0
  --global-batch-size 1024
  --balance-data
)

EVAL_ARGS=(
  --skip-eval-before-train
  --eval-prompt-data web_test_task "${TEST_TASK_DATA}" web_test_website "${TEST_WEBSITE_DATA}" web_test_domain "${TEST_DOMAIN_DATA}"
  --eval-interval 25
  --n-samples-per-eval-prompt 1
  --eval-max-response-len 4096
  --eval-temperature 0.0
  --eval-top-p 1.0
  --eval-concurrency 512
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --use-dynamic-global-batch-size
  --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --no-normalize-advantages
  --use-kl-loss
  --solver-entropy-coef 0.0
  --extractor-entropy-coef 0.0
  --solver-kl-loss-coef 0.01
  --extractor-kl-loss-coef 0.01
  --kl-loss-coef 0.01
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.1
  --eps-clip-high 0.15
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

WANDB_ARGS=(
    # --use-wandb
    --wandb-project evolving-rl
    --wandb-group $EXP_NAME
    # --wandb-key    
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR}"
ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${ROOT_DIR}:${MEGATRON_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK\": \"1\"
  }
}"


nohup "${PYTHON_BIN}" "${ROOT_DIR}/env/web/launch_servers.py" \
  --host "${WEB_HOST}" \
  --port "${WEB_ENV_PORT}" \
  --data-root "${DATA_DIR}/Mind2Web_dataset" \
  --max-depth "${WEB_MAX_DEPTH}" \
  --max-workers "${WEB_MAX_WORKERS}" \
  --log-dir "${LOG_DIR}/env" \
  --pid-file "${WEB_PID_FILE}" \
  --startup-timeout "${WEB_STARTUP_TIMEOUT}" > /dev/null 2>&1 &

nohup "${PYTHON_BIN}" "${ROOT_DIR}/env/web/retrieve_server.py" \
  --parquet "${TRAIN_DATA}" \
  --cache_file "${CACHE_FILE}" \
  --port "${WEB_RETRIEVE_TRAIN_PORT}" \
  --host "${WEB_RETRIEVE_HOST}" \
  --question_col "${WEB_RETRIEVE_QUESTION_COL}" > /dev/null 2>&1 &

nohup "${PYTHON_BIN}" "${ROOT_DIR}/env/web/retrieve_server.py" \
  --parquet "${ALL_DATA}" \
  --cache_file "${CACHE_FILE}" \
  --port "${WEB_RETRIEVE_TEST_PORT}" \
  --host "${WEB_RETRIEVE_HOST}" \
  --question_col "${WEB_RETRIEVE_QUESTION_COL}" > /dev/null 2>&1 &

wait_for_health "${WEB_ENV_URL}/health" "${WEB_STARTUP_TIMEOUT}"
wait_for_health "${WEB_RETRIEVE_URL_TRAIN}/health" "${WEB_RETRIEVE_STARTUP_TIMEOUT}"
wait_for_health "${WEB_RETRIEVE_URL_TEST}/health" "${WEB_RETRIEVE_STARTUP_TIMEOUT}"


ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_TRAIN_SCRIPT}" \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${MISC_ARGS[@]}"
