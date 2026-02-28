"""
evaluate.py — Compare base vs fine-tuned model on SQL generation.

This loads both models one at a time on a GPU, runs them on test examples,
and produces:
  1. Accuracy numbers (printed to terminal)
  2. Side-by-side examples showing where the fine-tuned model improved
  3. A bar chart saved as PNG (for your demo presentation)
  4. Detailed results saved as JSON

Usage (submitted via evaluate.sbatch):
  python evaluate.py --base-model /mnt/data/demo/models/Qwen3-8B \
                     --tuned-model /mnt/data/demo/output/qwen3-8b-sql
"""

import argparse
import json
import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use non-interactive backend for matplotlib (no display needed on cluster)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_sql(model, tokenizer, schema, question, max_new_tokens=256):
    """Ask the model to generate SQL for a given schema + question."""
    messages = [
        {
            "role": "system",
            "content": "You are a SQL expert. Given a database schema and a question, write the correct SQL query. Output only the SQL query, nothing else.",
        },
        {
            "role": "user",
            "content": f"Schema:\n{schema}\n\nQuestion: {question}",
        },
    ]

    # Format as chat and generate
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (skip the prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    # Clean up: remove any thinking tags if present
    if "</think>" in response:
        response = response.split("</think>")[-1]

    return response.strip()


def normalize_sql(sql):
    """Normalize SQL for comparison (lowercase, strip whitespace/semicolons)."""
    return " ".join(sql.lower().strip().rstrip(";").split())


def evaluate_model(model_path, test_data, label, device_map="auto"):
    """Load a model, run it on test data, return predictions."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )

    predictions = []
    for i, example in enumerate(test_data):
        sql = generate_sql(model, tokenizer, example["context"], example["question"])
        predictions.append(sql)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_data)}] Q: {example['question'][:60]}...")
            print(f"           Predicted: {sql[:80]}")

    # Free GPU memory before loading next model
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned model")
    parser.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL", "/mnt/data/demo/models/Qwen3-8B"),
    )
    parser.add_argument(
        "--tuned-model",
        default=os.environ.get("TUNED_MODEL", "/mnt/data/demo/output/qwen3-8b-sql"),
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET_PATH", "/mnt/data/demo/datasets/sql-create-context"),
    )
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("RESULTS_DIR", "/mnt/data/demo/results"),
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix for result filenames (e.g., 'qwen3-32b-lora_')",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # --- Load test data ---
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset)["train"]
    test_split = dataset.train_test_split(test_size=0.05, seed=42)["test"]

    # Use a subset for faster evaluation
    num = min(args.num_examples, len(test_split))
    test_data = test_split.select(range(num))
    ground_truth = [ex["answer"] for ex in test_data]
    print(f"Evaluating on {num} test examples")

    # --- Evaluate both models ---
    base_preds = evaluate_model(args.base_model, test_data, "Base Model")
    tuned_preds = evaluate_model(args.tuned_model, test_data, "Fine-tuned Model")

    # --- Compute accuracy ---
    base_correct = sum(
        normalize_sql(p) == normalize_sql(g)
        for p, g in zip(base_preds, ground_truth)
    )
    tuned_correct = sum(
        normalize_sql(p) == normalize_sql(g)
        for p, g in zip(tuned_preds, ground_truth)
    )

    base_acc = base_correct / num * 100
    tuned_acc = tuned_correct / num * 100

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"  RESULTS ({num} test examples)")
    print(f"{'='*60}")
    print(f"  Base model:      {args.base_model}")
    print(f"  Fine-tuned model: {args.tuned_model}")
    print(f"{'='*60}")
    print(f"  Base model accuracy:       {base_acc:5.1f}%  ({base_correct}/{num})")
    print(f"  Fine-tuned model accuracy: {tuned_acc:5.1f}%  ({tuned_correct}/{num})")
    print(f"  Improvement:               {tuned_acc - base_acc:+5.1f}%")
    print(f"{'='*60}")

    # --- Show examples where fine-tuning helped ---
    print(f"\nExamples where fine-tuning FIXED the output:\n")
    shown = 0
    for i in range(num):
        base_ok = normalize_sql(base_preds[i]) == normalize_sql(ground_truth[i])
        tuned_ok = normalize_sql(tuned_preds[i]) == normalize_sql(ground_truth[i])
        if not base_ok and tuned_ok and shown < 5:
            print(f"  Example {i+1}:")
            print(f"    Question:     {test_data[i]['question']}")
            print(f"    Ground truth: {ground_truth[i]}")
            print(f"    Base model:   {base_preds[i]}")
            print(f"    Fine-tuned:   {tuned_preds[i]}")
            print()
            shown += 1
    if shown == 0:
        print("  (No clear examples found — try with more test examples)")

    # --- Generate comparison chart ---
    model_name = os.path.basename(args.base_model)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [f"Base {model_name}", f"Fine-tuned {model_name}"],
        [base_acc, tuned_acc],
        color=["#2196F3", "#4CAF50"],
        width=0.5,
    )
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title("SQL Generation: Base vs Fine-tuned")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, [base_acc, tuned_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=14,
        )
    plt.tight_layout()
    chart_path = os.path.join(args.results_dir, f"{args.prefix}accuracy_comparison.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to: {chart_path}")

    # --- Save detailed results as JSON ---
    results_path = os.path.join(args.results_dir, f"{args.prefix}results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "tuned_model": args.tuned_model,
                "num_examples": num,
                "base_accuracy_pct": round(base_acc, 2),
                "tuned_accuracy_pct": round(tuned_acc, 2),
                "improvement_pct": round(tuned_acc - base_acc, 2),
                "examples": [
                    {
                        "question": test_data[i]["question"],
                        "schema": test_data[i]["context"],
                        "ground_truth": ground_truth[i],
                        "base_prediction": base_preds[i],
                        "tuned_prediction": tuned_preds[i],
                        "base_correct": normalize_sql(base_preds[i]) == normalize_sql(ground_truth[i]),
                        "tuned_correct": normalize_sql(tuned_preds[i]) == normalize_sql(ground_truth[i]),
                    }
                    for i in range(num)
                ],
            },
            f,
            indent=2,
        )
    print(f"Detailed results saved to: {results_path}")

    # --- Save results as Markdown ---
    md_path = os.path.join(args.results_dir, f"{args.prefix}results.md")
    with open(md_path, "w") as f:
        f.write(f"# Evaluation Results ({num} test examples)\n\n")
        f.write(f"| | Model | Accuracy | Correct |\n")
        f.write(f"|---|---|---|---|\n")
        f.write(f"| Base | `{args.base_model}` | {base_acc:.1f}% | {base_correct}/{num} |\n")
        f.write(f"| Fine-tuned | `{args.tuned_model}` | {tuned_acc:.1f}% | {tuned_correct}/{num} |\n\n")
        f.write(f"**Improvement: {tuned_acc - base_acc:+.1f}%**\n\n")
        f.write(f"## Examples where fine-tuning fixed the output\n\n")
        shown = 0
        for i in range(num):
            base_ok = normalize_sql(base_preds[i]) == normalize_sql(ground_truth[i])
            tuned_ok = normalize_sql(tuned_preds[i]) == normalize_sql(ground_truth[i])
            if not base_ok and tuned_ok and shown < 5:
                f.write(f"### Example {i+1}\n\n")
                f.write(f"**Question:** {test_data[i]['question']}\n\n")
                f.write(f"**Ground truth:** `{ground_truth[i]}`\n\n")
                f.write(f"**Base model:** `{base_preds[i]}`\n\n")
                f.write(f"**Fine-tuned:** `{tuned_preds[i]}`\n\n")
                shown += 1
        if shown == 0:
            f.write("(No clear examples found — try with more test examples)\n")
    print(f"Markdown results saved to: {md_path}")


if __name__ == "__main__":
    main()
