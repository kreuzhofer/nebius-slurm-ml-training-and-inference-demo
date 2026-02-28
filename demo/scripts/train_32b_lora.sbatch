#!/bin/bash
# =============================================================================
# train_32b_lora.sbatch — LoRA fine-tuning for Qwen3-32B
#
# Uses LoRA (Low-Rank Adaptation) instead of full fine-tuning. Only ~1-2% of
# parameters are trained, which drastically reduces memory usage:
#   - Full fine-tuning 32B: OOMs on 16x H100 80GB
#   - LoRA fine-tuning 32B: fits comfortably with batch_size=2 per GPU
#
# HOW TO USE:
#   sbatch /mnt/data/demo/scripts/train_32b_lora.sbatch
# =============================================================================

#SBATCH --job-name=train-32b-lora
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/data/demo/logs/train_32b_lora_%j.out
#SBATCH --error=/mnt/data/demo/logs/train_32b_lora_%j.err

# --- Activate environment ---
export DEMO_DIR="/mnt/data/demo"
export PATH="$DEMO_DIR/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate demo

# --- Model configuration ---
export MODEL_PATH="$DEMO_DIR/models/Qwen3-32B"
export OUTPUT_DIR="$DEMO_DIR/output/qwen3-32b-sql-lora"
export DATASET_PATH="$DEMO_DIR/datasets/sql-create-context"
export PER_DEVICE_BATCH_SIZE=2         # LoRA uses much less memory than full FT
export GRADIENT_ACCUMULATION_STEPS=4   # Effective batch = 2 * 16 * 4 = 128
export NUM_EPOCHS=1
export LEARNING_RATE=2e-4              # LoRA benefits from higher LR than full FT

# --- LoRA hyperparameters ---
export LORA_R=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.05

# --- NCCL settings ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# --- Launch ---
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "Total GPUs:   $(( SLURM_NNODES * 8 ))"
echo "Model:        $MODEL_PATH (32B LoRA)"
echo "Output:       $OUTPUT_DIR"
echo "LoRA:         r=$LORA_R, alpha=$LORA_ALPHA"
echo "=========================================="

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=8 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$DEMO_DIR/scripts/train_lora.py"
