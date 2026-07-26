#!/usr/bin/env bash

export prefix=$(cd "$(dirname "$0")" && pwd)
export model_prefix="/path/to/models"

export PYTHONPATH=$prefix:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

torchrun --nproc_per_node=8 search.py \
  --model-path "$model_prefix/Kimi-VL-A3B-Instruct" \
  --num-samples 3000 \
  --group-size 16 \
  --target-skip-ratio 0.5 \
  > result_search.log