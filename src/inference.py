#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration, Mistral3ForConditionalGeneration
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from typing import List, Any
from math import ceil
import argparse
from tqdm import tqdm
import pandas as pd
import os
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import *

def load_model_and_tokenizer(model_name: str):
    if "gemma" in model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name, padding_side="left")
    elif "Mistral" in model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    if "gpt-oss-120b" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",          # let HF handle device placement
            #torch_dtype=torch.float16,   # use FP16 for efficiency
            trust_remote_code=True
        )
    elif "Mistral" in model_name:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto"
        )
    elif "gemma" in model_name:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",          # let HF handle device placement
            torch_dtype="auto",   # use FP16 for efficiency
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",          # let HF handle device placement
            torch_dtype="auto",   # use FP16 for efficiency
        )

    model.eval()
    return model, tokenizer



def build_chat_text(model_name: str, tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> str:

    supports_system = "system" in getattr(tokenizer, "chat_template", "")

    if supports_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        # Insert system prompt inline if not supported
        messages = [
            {
                "role": "user",
                "content": f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
            }
        ]
    if "gemma" in model_name:
        messages = [
            {"role": "system", 
            "content": [{"type": "text", "text": system_prompt}]
            },
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt}
            ]},
        ]

    return messages




def extract_generated_part(tokenizer, model_inputs, gen):
    outputs = []
    for g, inp in zip(gen, model_inputs["input_ids"]):
        # decode full input prompt
        prompt_text = tokenizer.decode(
            inp,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # decode full generated sequence
        text = tokenizer.decode(
            g,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # keep only continuation
        continuation = text[len(prompt_text):].strip()
        outputs.append(continuation)

    return outputs


@torch.inference_mode()
def generate_batch(
    model_name,
    model,
    tokenizer,
    systen_prompt: str,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    **gen_kwargs: Any
) -> List[str]:

    outputs: List[str] = []
    device = model.device
    chat_texts = [build_chat_text(model_name, tokenizer, systen_prompt, p) for p in prompts]
    num_batches = ceil(len(chat_texts) / batch_size)
    i = 0
    for bi in tqdm(range(num_batches)):
        chunk = chat_texts[bi * batch_size: (bi + 1) * batch_size]

        if "gpt-oss-" in model_name:
            model_inputs = tokenizer.apply_chat_template(
                chunk, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
                reasoning_effort="medium"
            ).to(model.device)
        else:
            model_inputs = tokenizer.apply_chat_template(
                chunk, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True
            ).to(model.device)

        gen = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        generated_texts = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        outputs += extract_generated_part(tokenizer, model_inputs, gen)
        if i%10 == 0:
            df = pd.DataFrame({"Output": outputs})
            save_name = f"outputs_partial_{model_name.split('/')[-1]}.csv"
            print(save_name)
            df.to_csv(save_name, index=False, encoding="utf-8")
        i += 1
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Batch generation using Qwen model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",  # erlaubt mehrere Dateien
        default=["data/final_wb.txt"],
        help="Eine oder mehrere Eingabedateien mit Prompts für die Generierung"
    )
    parser.add_argument("--output_folder", type=str, default="output/",
                        help="File containing prompts for generation")
    parser.add_argument("--mode", type=str, default="extractor",
                        help="File containing prompts for generation")
    parser.add_argument("--language", type=str, default="",
                        help="File containing prompts for generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--shots", type=int, default=-1, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample instead of greedy decoding")

    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    for input_file in args.input_files:
        # Load the Data
        if args.mode == "extractor":
            df, system_prompt = load_data_extract_definition()
            prompts = df["raw_prompts"].tolist()
        elif args.mode == "generator":
            df, system_prompt = load_data_def_generation(args.language)
            prompts = df["raw_prompts"].tolist()
        elif args.mode == "generator_rulebased":
            df, system_prompt = load_data_def_generation_rulebased(args.language)
            prompts = df["raw_prompts"].tolist()
        elif args.mode == "evaluator":
            df, system_prompt = load_data_def_evaluator(input_file, args.mode)
            prompts = df["raw_prompts_" + args.mode].tolist()
        elif args.mode == "generatereversed":
            df, system_prompt = load_data_reversed_generate_def(args.mode, args.language)
            prompts = df["raw_prompts_" + args.mode].tolist()
        elif args.mode == "generatereversed_rulebased":
            df, system_prompt = load_data_reversed_generate_def_rulebased(args.mode, args.language)
            prompts = df["raw_prompts_" + args.mode].tolist()
        elif "generatereversed_fewshot" in args.mode:
            fewshot_mode = args.mode.split("_")[-1]
            df, system_prompt = load_data_reverse_generation_def_fewshot(args.language, args.shots, fewshot_mode)
            prompts = df["raw_prompts"].tolist()
        elif args.mode == "cleaner":
            df, system_prompt = load_data_def_cleaner(input_file, args.mode)
            prompts = df["raw_prompts_" + args.mode].tolist()
        elif "generator_fewshot" in args.mode:
            fewshot_mode = args.mode.split("_")[-1]
            df, system_prompt = load_data_def_generation_fewshot(args.language, args.shots, fewshot_mode)
            prompts = df["raw_prompts"].tolist()
        else:
            asddas

        # Inference Time!
        responses = generate_batch(
            args.model_name,
            model,
            tokenizer,
            system_prompt,
            prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )

        # Save
        df["response_" + args.mode] = responses
        if args.mode == "evaluator" or args.mode == "cleaner":
            df.to_csv(input_file, index=False, encoding="utf-8")
        else:
            if args.language == "english":
                save_dir = os.path.join(args.output_folder, args.mode, "english")
            else:
                save_dir = os.path.join(args.output_folder, args.mode)
            os.makedirs(save_dir, exist_ok=True)
            if args.shots != -1:
                save_path = os.path.join(save_dir, f"{args.model_name.split('/')[-1]}_{args.shots}shots.csv")
            else:
                save_path = os.path.join(save_dir, f"{args.model_name.split('/')[-1]}.csv")
            df.to_csv(save_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
