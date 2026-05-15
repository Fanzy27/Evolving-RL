#!/bin/bash
# =============================================================================
# Train ALFWorld experience extraction pipeline.
#
#   Solver (no skill) -> Extractor -> downstream Solver (with skill)
# =============================================================================

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

set -ex

export PYTHONBUFFERED=1

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
MEGATRON_DIR="/root/Megatron-LM"
SLIME_TRAIN_SCRIPT="slime/train.py"
MODEL_SOURCE_ARGS="/root/slime/scripts/models/qwen2.5-7B.sh"
HF_CHECKPOINT="$ROOT_DIR/models/Qwen2.5-7B-Instruct"
REF_CHECKPOINT="$ROOT_DIR/models/Qwen2.5-7B-Instruct_torch_dist"
EXP_NAME="alfworld1"
SAVE_CHECKPOINT="$ROOT_DIR/models/trained/alfworld_new/$EXP_NAME"
TRAIN_DATA="$ROOT_DIR/data/alfworld/train_seen.parquet"
EVAL_DATA="$ROOT_DIR/data/alfworld/valid_unseen.parquet"
ALFWORLD_CONFIG="configs/alfworld_server.yaml"
_cfg() { python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
ALFWORLD_ROUTER_PORT=$(_cfg "$ALFWORLD_CONFIG" alfworld_router_port)
ALFWORLD_WORKER_HOST=$(_cfg "$ALFWORLD_CONFIG" alfworld_worker_host)
ALFWORLD_WORKER_START_PORT=$(_cfg "$ALFWORLD_CONFIG" alfworld_worker_start_port)
ALFWORLD_ROUTER_HOST=$(_cfg "$ALFWORLD_CONFIG" alfworld_router_host)
ALFWORLD_NUM_WORKERS=$(_cfg "$ALFWORLD_CONFIG" alfworld_num_workers)
ALFWORLD_LOG_DIR=$(_cfg "$ALFWORLD_CONFIG" alfworld_log_dir)
ALFWORLD_PID_FILE=$(_cfg "$ALFWORLD_CONFIG" alfworld_pid_file)
ALFWORLD_STARTUP_TIMEOUT=$(_cfg "$ALFWORLD_CONFIG" alfworld_startup_timeout)
ALFWORLD_RETRIEVE_HOST=$(_cfg "$ALFWORLD_CONFIG" alfworld_retrieve_host)
ALFWORLD_RETRIEVE_TRAIN_PORT=$(_cfg "$ALFWORLD_CONFIG" alfworld_retrieve_train_port)
ALFWORLD_RETRIEVE_TEST_PORT=$(_cfg "$ALFWORLD_CONFIG" alfworld_retrieve_test_port)
ALFWORLD_RETRIEVE_QUESTION_COL=$(_cfg "$ALFWORLD_CONFIG" alfworld_retrieve_question_col)
MAX_EPISODE_STEPS=50

source "${MODEL_SOURCE_ARGS}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${REF_CHECKPOINT}"
    --load "${SAVE_CHECKPOINT}"
    --save "${SAVE_CHECKPOINT}"
    --save-interval 25
)

CUSTOM_ARGS=(
    --custom-generate-function-path src.alfworld.pipeline.alfworld_pipeline
    --custom-reward-post-process-path src.alfworld.pipeline.alfworld_reward_post_process
    --custom-eval-rollout-log-function-path src.alfworld.loggers.eval_logger.log_eval_by_task_type
    --custom-rollout-log-function-path src.alfworld.loggers.rollout_logger.log_rollout_data
    --custom-config-path configs/alfworld_server.yaml
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key metadata
    --apply-chat-template
    --rollout-shuffle

    --num-repeat 1
    --n-experiences 8
    --retrieval-topk 4
    --max-episode-steps "${MAX_EPISODE_STEPS}"

    --solver-max-response-len   512
    --extractor-reward-weight 0.2
    --solver-reward-weight 1.0
    --solver-temperature 1
    --skill-format-penalty 1

    --num-rollout 150
    --rollout-batch-size 16
    --n-samples-per-prompt 1
    --rollout-max-context-len 16384
    --rollout-max-response-len 1536
    --rollout-temperature 1
    --solver-top-p    0.9
    --rollout-top-p     0.9
    --global-batch-size 640
    --use-dynamic-batch-size
    --use-dynamic-global-batch-size
    --balance-data
)

EVAL_ARGS=(
    --skip-eval-before-train
    --eval-prompt-data unseen1 "${EVAL_DATA}" unseen2 "${EVAL_DATA}" unseen3 "${EVAL_DATA}"
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
    --max-tokens-per-gpu 20480
)

GRPO_ARGS=(

    --solver-entropy-coef 0
    --extractor-entropy-coef -0.03
    --solver-kl-loss-coef 0.01
    --extractor-kl-loss-coef 0.01
    --advantage-estimator grpo
    --no-normalize-advantages
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type low_var_kl
    --entropy-coef 0
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

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start \
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${ROOT_DIR}:${MEGATRON_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK\": \"1\"
  }
}"


nohup python env/alfworld/launch_servers.py \
  --alf-config-path configs/alfworld.yaml \
  --num-workers "${ALFWORLD_NUM_WORKERS}" \
  --router-port "${ALFWORLD_ROUTER_PORT}" \
  --worker-host "${ALFWORLD_WORKER_HOST}" \
  --worker-start-port "${ALFWORLD_WORKER_START_PORT}" \
  --router-host "${ALFWORLD_ROUTER_HOST}" \
  --python python \
  --log-dir "${ALFWORLD_LOG_DIR}" \
  --pid-file "${ALFWORLD_PID_FILE}" \
  --startup-timeout "${ALFWORLD_STARTUP_TIMEOUT}" \
  --worker-script env/alfworld/worker_server.py \
  --router-script env/alfworld/router_server.py > /dev/null 2>&1 &

nohup python env/alfworld/retrieve_server/retrieve_server.py \
  --parquet data/alfworld/train_seen.parquet \
  --cache_file data/alfworld/cache/unified_embeddings.npz \
  --port "${ALFWORLD_RETRIEVE_TRAIN_PORT}" \
  --host "${ALFWORLD_RETRIEVE_HOST}" \
  --question_col "${ALFWORLD_RETRIEVE_QUESTION_COL}" > /dev/null 2>&1 &

nohup python env/alfworld/retrieve_server/retrieve_server.py \
  --parquet data/alfworld/train.parquet \
  --cache_file data/alfworld/cache/unified_embeddings.npz \
  --port "${ALFWORLD_RETRIEVE_TEST_PORT}" \
  --host "${ALFWORLD_RETRIEVE_HOST}" \
  --question_col "${ALFWORLD_RETRIEVE_QUESTION_COL}" > /dev/null 2>&1 &

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_TRAIN_SCRIPT}" \
    --actor-num-gpus-per-node 8 \
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


