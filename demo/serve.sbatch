#!/bin/bash
# =============================================================================
# serve.sbatch — Start a vLLM inference server for a model.
#
# vLLM is a fast inference engine. It starts an OpenAI-compatible API server
# that you can query with curl or Python requests.
#
# HOW TO USE:
#   # Serve the fine-tuned 8B model (1 GPU):
#   sbatch /mnt/data/demo/scripts/serve.sbatch
#
#   # Serve the fine-tuned 32B model (needs 4 GPUs for tensor parallelism):
#   sbatch --gpus-per-node=4 /mnt/data/demo/scripts/serve.sbatch \
#       /mnt/data/demo/output/qwen3-32b-sql 8000 4
#
#   # Serve the BASE model for comparison (on a different port):
#   sbatch /mnt/data/demo/scripts/serve.sbatch \
#       /mnt/data/demo/models/Qwen3-8B 8001
#
# QUERYING THE SERVER (from login node):
#   # First find which node the server is running on:
#   squeue  # look at the NODELIST column
#
#   # Then query it:
#   curl http://<WORKER_NODE>:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "/mnt/data/demo/output/qwen3-8b-sql",
#       "messages": [
#         {"role": "system", "content": "You are a SQL expert."},
#         {"role": "user", "content": "Schema: CREATE TABLE users (id INT, name TEXT, age INT)\nQuestion: Find all users older than 25"}
#       ]
#     }'
# =============================================================================

#SBATCH --job-name=vllm-serve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/mnt/data/demo/logs/serve_%j.out
#SBATCH --error=/mnt/data/demo/logs/serve_%j.err

# --- Activate environment ---
export DEMO_DIR="/mnt/data/demo"
export PATH="$DEMO_DIR/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate demo

# --- Arguments ---
MODEL_PATH=${1:-$DEMO_DIR/output/qwen3-8b-sql}
PORT=${2:-8000}
TP_SIZE=${3:-1}  # tensor parallel size (use 4 for 32B model)

echo "=========================================="
echo "Starting vLLM server"
echo "  Model:    $MODEL_PATH"
echo "  Port:     $PORT"
echo "  TP size:  $TP_SIZE"
echo "  Node:     $(hostname)"
echo ""
echo "  Query with:"
echo "    curl http://$(hostname):$PORT/v1/models"
echo "=========================================="

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --max-model-len 2048
