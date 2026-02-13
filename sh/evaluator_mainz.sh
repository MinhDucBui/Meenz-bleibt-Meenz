#!/bin/bash -x


files=(
    "output/generator/Qwen2.5-7B-Instruct.csv"
    "output/generator/gemma-3-4b-it.csv"
    "output/generator/gemma-3-27b-it.csv"
    "output/generator/leo-hessianai-70b-chat.csv"
    "output/generator/phi-4.csv"
    "output/generator/aya-expanse-32b.csv"
    "output/generator/Qwen2.5-72B-Instruct.csv"
    "output/generator/Llama-3.3-70B-Instruct.csv"
    "output/generator/Qwen3-4B-Instruct-2507.csv"
    "output/generator/Llama-3.1-8B-Instruct.csv"
    "output/generator/gpt-oss-120b.csv"
    "output/generator_rulebased/Llama-3.3-70B-Instruct.csv"
    "output/generator/Qwen3-30B-A3B-Instruct-2507.csv"
    "output/generator/Qwen3-30B-A3B-Instruct-2507.csv"
)

python src/inference.py \
    --input_files "${files[@]}" \
    --model_name "Llama-3.3-70B-Instruct" \
    --mode "evaluator" \
    --batch_size 64 \
    --max_new_tokens 8
