"""
merge_lora.py — Merge LoRA adapters back into the base model.

After LoRA training, the output directory contains only the small adapter
weights (~200MB). This script merges them into the base model to produce a
full model that works with evaluate.py and serve.sbatch as-is.

Usage:
  python merge_lora.py <adapter_path> <base_model_path> <output_path>

Examples:
  # Merge 32B LoRA adapter
  python scripts/merge_lora.py \
      output/qwen3-32b-sql-lora \
      models/Qwen3-32B \
      output/qwen3-32b-sql

  # Merge 8B LoRA adapter
  python scripts/merge_lora.py \
      output/qwen3-8b-sql-lora \
      models/Qwen3-8B \
      output/qwen3-8b-sql
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("adapter_path", help="Path to LoRA adapter directory")
    parser.add_argument("base_model_path", help="Path to base model directory")
    parser.add_argument("output_path", help="Path to save merged model")
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model_path} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {args.adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging adapter weights into base model ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path} ...")
    model.save_pretrained(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)

    print(f"Done! Merged model saved to {args.output_path}")


if __name__ == "__main__":
    main()
