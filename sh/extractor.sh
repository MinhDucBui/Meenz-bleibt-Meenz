#!/bin/bash -x


# Record script start time
start_time=$(date +%s)
echo "=== Script started at $(date) ==="
#     --model_name "/p/project/westai0073/Models/gpt-oss-120b" \
#     --model_name "/p/project/westai0073/Models/Llama-3.3-70B-Instruct"
python src/inference.py \
    --model_name "gpt-oss-120b" \
    --batch_size 4 \
    --max_new_tokens 1024
