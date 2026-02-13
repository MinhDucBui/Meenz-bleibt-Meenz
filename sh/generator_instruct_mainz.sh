#!/bin/bash -x


# Record script start time
start_time=$(date +%s)
echo "=== Script started at $(date) ==="

models=(
    # Big
    #"Llama-3.3-70B-Instruct"
    #"leo-hessianai-70b-chat"
    #"Qwen2.5-72B-Instruct"

    # Medium
    #"aya-expanse-32b"
    #"Mistral-Small-3.2-24B-Instruct-2506"
    #"phi-4"
    #"gemma-3-27b-it"

    # Small
    #"Qwen3-4B-Instruct-2507"
    #"gemma-3-4b-it"
    #"Qwen2.5-7B-Instruct"
    "Qwen3-30B-A3B-Instruct-2507"
)


# Loop over each model
for model in "${models[@]}"; do
    echo "Running inference for model: $model"
    python src/inference.py \
        --model_name "$model" \
        --mode "generator" \
        --batch_size 128 \
        --max_new_tokens 512
done
