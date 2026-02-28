"""
train_lora.py — LoRA Fine-Tuning for Qwen3 on SQL generation.

Like train.py but uses LoRA (Low-Rank Adaptation) instead of full fine-tuning.
Only a small set of adapter parameters are trained (~1-2% of total), which:
  - Drastically reduces GPU memory usage
  - Makes 32B training feasible without CPU offloading
  - Trains faster per step (fewer parameters to update)

The base model weights stay frozen. Only the LoRA adapter weights are saved.
"""

import os
import shutil
import torch
import datasets
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

datasets.disable_caching()


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

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    encoded = tokenizer(
        text, max_length=max_length, truncation=True, padding=False
    )

    encoded["labels"] = encoded["input_ids"].copy()
    return encoded


def main():
    # --- Configuration from environment variables (set in .sbatch files) ---
    demo_dir = os.environ.get("DEMO_DIR", "/mnt/data/demo")
    model_path = os.environ.get("MODEL_PATH", f"{demo_dir}/models/Qwen3-8B")
    output_dir = os.environ.get("OUTPUT_DIR", f"{demo_dir}/output/qwen3-8b-sql-lora")
    dataset_path = os.environ.get("DATASET_PATH", f"{demo_dir}/datasets/sql-create-context")

    per_device_bs = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "4"))
    grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "2"))
    num_epochs = int(os.environ.get("NUM_EPOCHS", "1"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-4"))

    # LoRA hyperparameters
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    local_data = os.environ.get("LOCAL_DATA_DIR", "/mnt/local-data")
    local_output_dir = os.path.join(local_data, os.path.basename(output_dir))
    os.makedirs(local_output_dir, exist_ok=True)

    if rank == 0:
        print(f"Model:      {model_path}")
        print(f"Output:     {output_dir}")
        print(f"Local dir:  {local_output_dir}")
        print(f"Dataset:    {dataset_path}")
        print(f"Batch size: {per_device_bs} per GPU, {grad_accum} accumulation steps")
        print(f"LoRA:       r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model in bfloat16 ---
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # --- Apply LoRA ---
    # Target attention + MLP layers for best quality on structured tasks like SQL
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    if rank == 0:
        model.print_trainable_parameters()

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=local_output_dir,

        num_train_epochs=num_epochs,

        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,

        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,

        bf16=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # FSDP with LoRA — use_orig_params is critical for PEFT compatibility
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "backward_prefetch": "backward_pre",
            "forward_prefetch": "true",
            "use_orig_params": True,
        },

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,

        dataloader_num_workers=4,

        report_to="none",
    )

    # --- Load and prepare dataset ---
    dataset = load_from_disk(dataset_path)
    train_data = dataset["train"]
    split = train_data.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    if rank == 0:
        print(f"Train examples: {len(train_ds)}, Eval examples: {len(eval_ds)}")

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
        print("Starting LoRA training...")

    trainer.train()

    # --- Save the LoRA adapter ---
    # With LoRA, we only save the small adapter weights (not the full model).
    # At inference time, load the base model + adapter with PeftModel.from_pretrained().
    if trainer.is_world_process_zero():
        import glob
        checkpoints = sorted(glob.glob(os.path.join(local_output_dir, "checkpoint-*")))
        if checkpoints:
            last_ckpt = checkpoints[-1]
            print(f"Copying LoRA adapter from {last_ckpt} to {output_dir} ...")
            os.makedirs(output_dir, exist_ok=True)
            for fname in os.listdir(last_ckpt):
                src = os.path.join(last_ckpt, fname)
                if os.path.isfile(src) and not fname.startswith(("rng_state", "optimizer", "training_args")):
                    shutil.copy2(src, output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"\nLoRA training complete! Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
