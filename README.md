# Nebius Soperator GPU Cluster Demo

Distributed training and inference on a Nebius Cloud Slurm-on-Kubernetes cluster using [Soperator](https://github.com/nebius/nebius-solutions-library/tree/main/soperator).

## Cluster

- 2 worker nodes, 8x NVIDIA H100 80GB each (16 GPUs total)
- InfiniBand interconnect with GPUDirect RDMA
- Shared NFS storage (`/mnt/data`) + local SSD per node (`/mnt/local-data`)
- Deployed via Terraform using the Soperator template

## What's in this repo

```
soperator/                  # Soperator Terraform template (from nebius-solutions-library)
  installations/
    dkreuzh001/             # Cluster-specific config
      terraform.tfvars      # Cluster settings
      .envrc                # Environment setup (Nebius auth, S3 backend)
      main.tf               # Module orchestration
      variables.tf          # Variable definitions
modules/                    # Shared Terraform modules (from nebius-solutions-library)
demo/scripts/               # Demo scripts
  setup.sh                  # Install conda environment with PyTorch, vLLM, etc.
  download.sh               # Download models and dataset from HuggingFace
  train.py                  # SFT training script (FSDP, distributed)
  train_lora.py             # LoRA training script
  train_8b.sbatch           # Slurm job: Qwen3-8B full fine-tuning (16 GPUs)
  train_32b.sbatch          # Slurm job: Qwen3-32B full fine-tuning (16 GPUs)
  train_32b_lora.sbatch     # Slurm job: Qwen3-32B LoRA fine-tuning (16 GPUs)
  evaluate.py               # Compare base vs fine-tuned models
  evaluate.sbatch           # Slurm job: evaluation
  serve.sbatch              # Slurm job: vLLM inference server (single-node)
  serve_235b.sbatch         # Slurm job: vLLM + Ray multi-node inference (16 GPUs)
  merge_lora.py             # Merge LoRA adapters into base model
  query.sh                  # Query the vLLM server
DEMO_SUMMARY.md             # Full walkthrough and results
EMAIL_FOLLOW_UP.md          # Follow-up email summary
```

## Tasks Completed

1. **Distributed fine-tuning** — Qwen3-8B (full SFT) and Qwen3-32B (LoRA) on the sql-create-context dataset across 16 H100 GPUs
2. **Inference serving** — vLLM with OpenAI-compatible API for 8B (1 GPU), 32B (4 GPUs), and 235B MoE (16 GPUs with Ray)
3. **Base vs fine-tuned comparison** — 2% → 88% accuracy (8B), 3% → 84% accuracy (32B) on exact-match SQL generation
4. **GPU utilization >80%** — all 16 GPUs fully utilized during training and multi-node inference

## Quick Start

See [DEMO_SUMMARY.md](DEMO_SUMMARY.md) for the full deployment guide, training details, and results.

## License

Demo scripts (`demo/`) are licensed under the [MIT License](LICENSE.md).

The Soperator Terraform code (`soperator/`, `modules/`) is from the [Nebius Solutions Library](https://github.com/nebius/nebius-solutions-library) and licensed under [Apache 2.0](LICENSE).
