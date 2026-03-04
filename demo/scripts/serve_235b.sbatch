#!/bin/bash
# =============================================================================
# serve_235b.sbatch — Multi-node vLLM inference server for Qwen3-235B-A22B
#
# Uses Ray to coordinate vLLM across 2 nodes (16 GPUs total).
# Tensor parallelism (TP=8) splits layers across GPUs within each node.
# Pipeline parallelism (PP=2) splits layers across nodes.
#
# HOW TO USE:
#   sbatch /mnt/data/demo/scripts/serve_235b.sbatch
#
# QUERYING THE SERVER (from login node):
#   # Find which node is the head:
#   squeue  # look at the NODELIST column, first node is the head
#
#   # Query it:
#   curl http://<HEAD_NODE>:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "Qwen3-235B-A22B",
#       "messages": [
#         {"role": "system", "content": "You are a SQL expert. Given a database schema and a question, write the correct SQL query. Output only the SQL query, nothing else."},
#         {"role": "user", "content": "Schema: CREATE TABLE users (id INT, name TEXT, age INT)\nQuestion: Find all users older than 25"}
#       ]
#     }'
# =============================================================================

#SBATCH --job-name=vllm-235b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/data/demo/logs/serve_235b_%j.out
#SBATCH --error=/mnt/data/demo/logs/serve_235b_%j.err

# --- Activate environment ---
export DEMO_DIR="/mnt/data/demo"
export PATH="$DEMO_DIR/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate demo

# --- Triton cache on local SSD so each node compiles independently (no NFS race) ---
export TRITON_CACHE_DIR="/mnt/local-data/.triton_cache"

# --- Arguments ---
MODEL_PATH=${1:-$DEMO_DIR/models/Qwen3-235B-A22B}
PORT=${2:-8000}

# --- Identify nodes ---
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
WORKER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n+2)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | grep -v '^169\.254\.' | head -n1)

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Head node:    $HEAD_NODE ($HEAD_IP)"
echo "Worker nodes: $WORKER_NODES"
echo "Total GPUs:   $(( SLURM_NNODES * 8 ))"
echo "Model:        $MODEL_PATH"
echo "Port:         $PORT"
echo "TP size:      8 (within node)"
echo "PP size:      2 (across nodes)"
echo ""
echo "  Query with:"
echo "    curl http://$HEAD_NODE:$PORT/v1/models"
echo "=========================================="

# --- Step 1: Start Ray head on the first node ---
# --overlap allows multiple srun steps to share the same task slots within this job
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    bash -c "mkdir -p $TRITON_CACHE_DIR && ray start --head --port=6379 --num-gpus=8 --block" &
sleep 15  # Wait for the head to be ready

# --- Step 2: Join worker nodes to the Ray cluster ---
for NODE in $WORKER_NODES; do
    echo "Joining $NODE to Ray cluster..."
    srun --overlap --nodes=1 --ntasks=1 -w "$NODE" \
        bash -c "mkdir -p $TRITON_CACHE_DIR && ray start --address=$HEAD_IP:6379 --num-gpus=8 --block" &
done
sleep 15  # Wait for workers to register

# --- Step 3: Launch vLLM on the head node ---
# vLLM discovers the Ray cluster automatically and distributes across all GPUs.
# TP=8 (tensor parallel within each node), PP=2 (pipeline parallel across nodes)
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$(basename "$MODEL_PATH")" \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 2 \
        --distributed-executor-backend ray \
        --dtype bfloat16 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$PORT" \
        --max-model-len 32768

# Note: the --block flag on ray start keeps those srun tasks alive.
# When the vLLM server exits (or the job is cancelled), Slurm cleans up everything.
