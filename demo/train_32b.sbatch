#!/bin/bash
# =============================================================================
# train_32b.sbatch — Slurm job script for distributed Qwen3-32B fine-tuning
#
# Same as train_8b.sbatch but for the 32B model. Key differences:
#   - Smaller batch size (32B needs more memory per GPU)
#   - More gradient accumulation to compensate
#   - This is what gives you >80% GPU utilization for task 4!
#
# PREREQUISITE: Make sure Qwen3-32B download is complete:
#   tail -f /mnt/data/demo/logs/download_32b.log
#
# HOW TO USE:
#   sbatch /mnt/data/demo/scripts/train_32b.sbatch
# =============================================================================

#SBATCH --job-name=train-32b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/data/demo/logs/train_32b_%j.out
#SBATCH --error=/mnt/data/demo/logs/train_32b_%j.err

# --- Activate environment ---
export DEMO_DIR="/mnt/data/demo"
export PATH="$DEMO_DIR/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate demo

# --- Model configuration ---
export MODEL_PATH="$DEMO_DIR/models/Qwen3-32B"
export OUTPUT_DIR="$DEMO_DIR/output/qwen3-32b-sql"
export DATASET_PATH="$DEMO_DIR/datasets/sql-create-context"
export PER_DEVICE_BATCH_SIZE=1         # Smaller batch — 32B model uses more GPU memory
export GRADIENT_ACCUMULATION_STEPS=8   # Compensate: effective batch = 1 * 16 * 8 = 128
export NUM_EPOCHS=1
export LEARNING_RATE=1e-5              # Slightly lower LR for larger model

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
echo "Model:        $MODEL_PATH (32B — this will fill your GPUs!)"
echo "Output:       $OUTPUT_DIR"
echo "=========================================="

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=8 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$DEMO_DIR/scripts/train.py"
