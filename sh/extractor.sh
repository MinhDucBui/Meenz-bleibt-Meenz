#!/bin/bash -x


# Record script start time
start_time=$(date +%s)
echo "=== Script started at $(date) ==="
python src/inference.py \
    --model_name "gpt-oss-120b" \
    --batch_size 4 \
    --max_new_tokens 1024
