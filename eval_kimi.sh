#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export prefix=$(cd "$(dirname "$0")" && pwd)
# export model_prefix=${MODEL_PREFIX:-/path/to/models}
export model_prefix="/path/to/models"
export PYTHONPATH=$prefix:$PYTHONPATH
export DECORD_EOF_RETRY_MAX=200000
export LOG_DIR="${LOG_DIR:-$prefix/logs}"
mkdir -p "$LOG_DIR"
LOG_FOLDER="eval_$(date +%Y%m%d_%H%M%S)_kimi"
mkdir -p "$LOG_DIR/$LOG_FOLDER"

# Static pattern: layer-wise K for vision/text obtained by ILP, example
# Full 6 experts: text_topk_by_layer=1-26:6,visual_topk_by_layer=1-26:6
VISUAL_K="${VISUAL_K:-1-26:6}"
TEXT_K="${TEXT_K:-1-26:6}"

# Task name (can be textvqa_val, chartqa, mmstar, mmbench_en, mme, realworldqa,
# coco_cap, videomme, egoschema, mvbench, video_mmmu, longvideobench_val_v, etc.)
TASK_NAME="${1:-chartqa}"
LOG_FILE="$LOG_DIR/$LOG_FOLDER/$TASK_NAME.log"

echo "Task: $TASK_NAME | Visual K: $VISUAL_K | Text K: $TEXT_K | Log: $LOG_FILE"

PYTHONUNBUFFERED=1 accelerate launch --main_process_port 5678 \
    $prefix/eval/kimi_fix.py \
    --model kimi_vl_fix \
    --model_args pretrained=$model_prefix/Kimi-VL-A3B-Instruct,visual_topk_by_layer=$VISUAL_K,text_topk_by_layer=$TEXT_K \
    --tasks $TASK_NAME \
    --batch_size 1 \
    --output_path $prefix/storage/eval/ \
    2>&1 | tee "$LOG_FILE"
