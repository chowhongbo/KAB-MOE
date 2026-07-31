#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)

# ===== Modify only these settings if needed =====
MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-VL-A3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$ROOT/model/Kimi-VL-A3B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TASKS=(chartqa textvqa_val mmstar)
TARGET_SKIP_RATIO=0.5

# ===============================================
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export DECORD_EOF_RETRY_MAX=200000

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPUS[@]}
SEARCH_LOG="$ROOT/result_search.log"
SOLVE_LOG="$ROOT/result_solve.log"
LOG_DIR="$ROOT/logs/eval_$(date +%Y%m%d_%H%M%S)_kimi"
mkdir -p "$LOG_DIR" "$ROOT/storage/eval"

# 1. Use the local model when available; otherwise download it.
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[1/4] Downloading $MODEL_REPO to $MODEL_PATH"
    mkdir -p "$MODEL_PATH"
    python - <<PY
try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise SystemExit("Please install huggingface_hub first: pip install huggingface_hub")

snapshot_download(
    repo_id="$MODEL_REPO",
    local_dir="$MODEL_PATH",
)
PY
else
    echo "[1/4] Using local model: $MODEL_PATH"
fi

# 2. Search the reward matrix with the original 3000 calibration samples.
echo "[2/4] Searching policy"
torchrun --nproc_per_node="$NUM_GPUS" "$ROOT/search.py" \
    --model-path "$MODEL_PATH" \
    --num-samples 3000 \
    --group-size 16 \
    --target-skip-ratio "$TARGET_SKIP_RATIO" \
    2>&1 | tee "$SEARCH_LOG"

# 3. Run the original solver and select only the monotonic result.
echo "[3/4] Solving monotonic policy"
python "$ROOT/solve_matrix.py" \
    --log-file "$SEARCH_LOG" \
    --skip-ratio "$TARGET_SKIP_RATIO" \
    2>&1 | tee "$SOLVE_LOG"

VISUAL_K=$(awk '/^mono result:/{getline; print; exit}' "$SOLVE_LOG" \
    | sed -n "s/.*'pattern': '\([^']*\)'.*/\1/p")

if [[ -z "$VISUAL_K" ]]; then
    echo "Failed to extract the monotonic pattern from $SOLVE_LOG" >&2
    exit 1
fi

TEXT_K="1-26:6"
echo "Monotonic policy: $VISUAL_K"

# 4. Evaluate the searched monotonic policy with the original evaluation command.
echo "[4/4] Evaluating: ${TASKS[*]}"
for TASK in "${TASKS[@]}"; do
    LOG_FILE="$LOG_DIR/$TASK.log"
    echo "========== $TASK =========="

    PYTHONUNBUFFERED=1 accelerate launch --main_process_port 5678 \
        "$ROOT/eval/kimi_fix.py" \
        --model kimi_vl_fix \
        --model_args "pretrained=$MODEL_PATH,visual_topk_by_layer=$VISUAL_K,text_topk_by_layer=$TEXT_K" \
        --tasks "$TASK" \
        --batch_size 1 \
        --output_path "$ROOT/storage/eval/" \
        2>&1 | tee "$LOG_FILE"
done

echo "Finished. Evaluation logs: $LOG_DIR"
