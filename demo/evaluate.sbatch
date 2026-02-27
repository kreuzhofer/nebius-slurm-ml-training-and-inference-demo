#!/bin/bash
# =============================================================================
# evaluate.sbatch — Compare base vs fine-tuned model on SQL test set.
#
# This loads each model one at a time, generates SQL for 100 test questions,
# and compares accuracy. Produces a chart + JSON results.
#
# HOW TO USE:
#   # For 8B model comparison:
#   sbatch /mnt/data/demo/scripts/evaluate.sbatch
#
#   # For 32B model comparison (needs more GPUs for loading):
#   sbatch --gpus-per-node=4 /mnt/data/demo/scripts/evaluate.sbatch \
#       /mnt/data/demo/models/Qwen3-32B /mnt/data/demo/output/qwen3-32b-sql
#
# Results will be in /mnt/data/demo/results/
# =============================================================================

#SBATCH --job-name=evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1              # 1 GPU is enough for 8B inference
#SBATCH --output=/mnt/data/demo/logs/evaluate_%j.out
#SBATCH --error=/mnt/data/demo/logs/evaluate_%j.err

# --- Activate environment ---
export DEMO_DIR="/mnt/data/demo"
export PATH="$DEMO_DIR/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate demo

# --- Accept model paths as arguments (with defaults for 8B) ---
BASE_MODEL=${1:-$DEMO_DIR/models/Qwen3-8B}
TUNED_MODEL=${2:-$DEMO_DIR/output/qwen3-8b-sql}

echo "=========================================="
echo "Evaluating: $BASE_MODEL vs $TUNED_MODEL"
echo "=========================================="

python "$DEMO_DIR/scripts/evaluate.py" \
    --base-model "$BASE_MODEL" \
    --tuned-model "$TUNED_MODEL" \
    --num-examples 100
