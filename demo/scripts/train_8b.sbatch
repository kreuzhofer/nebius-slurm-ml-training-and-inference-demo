#!/bin/bash
# =============================================================================
# train_8b.sbatch — Slurm job script for distributed Qwen3-8B fine-tuning
#
# WHAT IS THIS?
# Think of this like a "job ticket" you submit to the cluster. Instead of
# running training directly (like in a notebook), you hand this script to
# Slurm (the job scheduler), and it:
#   1. Reserves 2 nodes with 8 GPUs each (16 GPUs total)
#   2. Runs your training script on all of them simultaneously
#   3. Logs output to a file you can check later
#
# HOW TO USE:
#   sbatch /mnt/data/demo/scripts/train_8b.sbatch
#
# USEFUL COMMANDS:
#   squeue                  — See your running/pending jobs
#   squeue -u root          — See only your jobs
#   scancel <JOBID>         — Cancel a job
#   tail -f /mnt/data/demo/logs/train_8b_<JOBID>.out  — Watch live output
#   sinfo                   — See available nodes and their status
# =============================================================================

# --- Slurm resource requests (lines starting with #SBATCH) ---
#SBATCH --job-name=train-8b           # Name shown in squeue
#SBATCH --nodes=2                      # Use both nodes (2 nodes x 8 GPUs = 16 GPUs)
#SBATCH --ntasks-per-node=1            # 1 task per node (torchrun handles the 8 GPUs)
#SBATCH --gpus-per-node=8              # Request all 8 GPUs on each node
#SBATCH --exclusive                    # Don't share nodes with other jobs
#SBATCH --output=/mnt/data/demo/logs/train_8b_%j.out   # stdout (%j = job ID)
#SBATCH --error=/mnt/data/demo/logs/train_8b_%j.err    # stderr

# --- Activate the conda environment ---
export DEMO_DIR="/mnt/data/demo"
export PATH="$DEMO_DIR/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate demo

# --- Tell the training script which model and where to save ---
export MODEL_PATH="$DEMO_DIR/models/Qwen3-8B"
export OUTPUT_DIR="$DEMO_DIR/output/qwen3-8b-sql"
export DATASET_PATH="$DEMO_DIR/datasets/sql-create-context"
export PER_DEVICE_BATCH_SIZE=4       # 8B is small, can use larger batches
export GRADIENT_ACCUMULATION_STEPS=2
export NUM_EPOCHS=1
export LEARNING_RATE=2e-5

# --- NCCL settings for optimal multi-node GPU communication ---
# NCCL is the library that GPUs use to talk to each other
export NCCL_DEBUG=WARN                 # Set to INFO for debugging connectivity
export NCCL_IB_DISABLE=0              # Enable InfiniBand (fast GPU-to-GPU across nodes)
export NCCL_NET_GDR_LEVEL=5           # Enable GPUDirect RDMA
export NCCL_TIMEOUT=3600              # 1 hour timeout for FSDP all-gather during model save

# --- Figure out which node is the "master" for coordination ---
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "GPUs/node:    $SLURM_GPUS_ON_NODE"
echo "Total GPUs:   $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Model:        $MODEL_PATH"
echo "Output:       $OUTPUT_DIR"
echo "=========================================="

# --- Launch distributed training ---
# srun:     runs the command on each allocated node (2 nodes)
# torchrun: on each node, launches 8 processes (1 per GPU)
# Result:   16 training processes across 16 GPUs, all working together
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=8 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$DEMO_DIR/scripts/train.py"
