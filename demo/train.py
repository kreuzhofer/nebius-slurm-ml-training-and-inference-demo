"""
train.py — Supervised Fine-Tuning (SFT) for Qwen3 on SQL generation.

This script works for both Qwen3-8B and Qwen3-32B. The model path and output
directory are passed via environment variables (set in the .sbatch files).

What it does:
  1. Loads a pre-trained Qwen3 model
  2. Loads the SQL dataset (natural language → SQL query pairs)
  3. Fine-tunes the model using PyTorch FSDP (Fully Sharded Data Parallel)
     to distribute training across all GPUs on all nodes
  4. Saves the fine-tuned model to shared storage

This is the same kind of training you'd do in a notebook, but:
  - torchrun handles launching 1 process per GPU across all nodes
  - FSDP splits the model across GPUs (essential for 32B — too big for 1 GPU)
  - The HuggingFace Trainer orchestrates the training loop
"""

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)


def tokenize_example(example, tokenizer, max_length=1024):
    """
    Format a SQL example as a chat conversation and tokenize it.

    Input format (from dataset):
      context:  "CREATE TABLE head (age INTEGER)"
      question: "How many heads are older than 56?"
      answer:   "SELECT COUNT(*) FROM head WHERE age > 56"

    We format it as a chat conversation so the model learns:
      System: You are a SQL expert...
      User:   <schema + question>
      Assistant: <SQL query>
    """
    messages = [
        {
            "role": "system",
            "content": "You are a SQL expert. Given a database schema and a question, write the correct SQL query. Output only the SQL query, nothing else.",
        },
        {
            "role": "user",
            "content": f"Schema:\n{example['context']}\n\nQuestion: {example['question']}",
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]

    # apply_chat_template converts the messages into the format the model expects
    # (e.g., <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>...)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    encoded = tokenizer(
        text, max_length=max_length, truncation=True, padding=False
    )

    # For causal language model training, labels = input_ids
    # The model learns to predict each next token
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded


def main():
    # --- Configuration from environment variables (set in .sbatch files) ---
    demo_dir = os.environ.get("DEMO_DIR", "/mnt/data/demo")
    model_path = os.environ.get("MODEL_PATH", f"{demo_dir}/models/Qwen3-8B")
    output_dir = os.environ.get("OUTPUT_DIR", f"{demo_dir}/output/qwen3-8b-sql")
    dataset_path = os.environ.get("DATASET_PATH", f"{demo_dir}/datasets/sql-create-context")

    # Batch size per GPU — smaller for larger models to fit in memory
    per_device_bs = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "4"))
    grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "2"))
    num_epochs = int(os.environ.get("NUM_EPOCHS", "1"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-5"))

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    if rank == 0:
        print(f"Model:      {model_path}")
        print(f"Output:     {output_dir}")
        print(f"Dataset:    {dataset_path}")
        print(f"Batch size: {per_device_bs} per GPU, {grad_accum} accumulation steps")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model in bfloat16 ---
    # bfloat16 uses half the memory of float32 with minimal quality loss
    # FSDP will shard this across all GPUs automatically
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # --- Load and prepare dataset ---
    dataset = load_from_disk(dataset_path)
    train_data = dataset["train"]

    # Split into train (95%) and eval (5%) sets
    split = train_data.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    if rank == 0:
        print(f"Train examples: {len(train_ds)}, Eval examples: {len(eval_ds)}")

    # Tokenize the datasets
    train_ds = train_ds.map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval",
    )

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,

        # How many passes through the data
        num_train_epochs=num_epochs,

        # Batch size per GPU. With 16 GPUs and grad_accum=2:
        # effective_batch_size = 4 * 16 * 2 = 128
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,

        # Optimizer settings
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,

        # Use bfloat16 for faster training on H100s
        bf16=True,

        # Gradient checkpointing saves memory by recomputing activations
        # during backward pass instead of storing them. Trades compute for memory.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # FSDP (Fully Sharded Data Parallel) — this is the key setting!
        # "full_shard" = shard model params, gradients, and optimizer states across GPUs
        # "auto_wrap"  = automatically decide which layers to wrap
        # This is what lets a 32B model train across 16 GPUs
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "backward_prefetch": "backward_pre",
            "forward_prefetch": "true",
        },

        # Logging and saving
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,

        # Data loading
        dataloader_num_workers=4,

        # Don't send metrics to wandb/tensorboard
        report_to="none",
    )

    # --- Create trainer and start training ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8
        ),
    )

    if rank == 0:
        print("Starting training...")

    trainer.train()

    # --- Save the fine-tuned model ---
    # Only rank 0 saves (Trainer handles gathering FSDP shards automatically)
    if trainer.is_world_process_zero():
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
