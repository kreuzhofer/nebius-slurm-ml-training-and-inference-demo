#!/bin/bash
# =============================================================================
# download.sh — Download models and dataset to the shared filesystem.
#
# Run this on the login node (it has internet access, worker nodes may not).
# Downloads go to /mnt/data/demo/models/ which is visible from all nodes.
#
# Usage:
#   source /mnt/data/demo/activate.sh
#   bash /mnt/data/demo/scripts/download.sh
#
# Tip: Qwen3-8B (~16GB) downloads in a few minutes.
#      Qwen3-32B (~64GB) takes ~15-30 min depending on bandwidth.
#      Start 32B download first, then work on 8B while it finishes.
# =============================================================================
set -euo pipefail

DEMO_DIR="/mnt/data/demo"
MODELS_DIR="$DEMO_DIR/models"
DATASETS_DIR="$DEMO_DIR/datasets"

mkdir -p "$MODELS_DIR" "$DATASETS_DIR"

# --- Download Qwen3-8B (small model, ~16GB) ---
echo "=== Downloading Qwen3-8B (~16GB) ==="
if [ -d "$MODELS_DIR/Qwen3-8B" ] && [ "$(ls -A "$MODELS_DIR/Qwen3-8B"/*.safetensors 2>/dev/null)" ]; then
    echo "Qwen3-8B already downloaded, skipping."
else
    huggingface-cli download Qwen/Qwen3-8B \
        --local-dir "$MODELS_DIR/Qwen3-8B" \
        --local-dir-use-symlinks False
    echo "Qwen3-8B downloaded to $MODELS_DIR/Qwen3-8B"
fi

# --- Download SQL dataset (~50MB) ---
echo ""
echo "=== Downloading SQL dataset ==="
if [ -d "$DATASETS_DIR/sql-create-context" ]; then
    echo "Dataset already downloaded, skipping."
else
    python -c "
from datasets import load_dataset
print('Loading sql-create-context from HuggingFace...')
ds = load_dataset('b-mc2/sql-create-context')
ds.save_to_disk('$DATASETS_DIR/sql-create-context')
print(f'Dataset saved. Train examples: {len(ds[\"train\"])}')
"
    echo "Dataset saved to $DATASETS_DIR/sql-create-context"
fi

# --- Download Qwen3-32B (large model, ~64GB) ---
echo ""
echo "=== Downloading Qwen3-32B (~64GB) — this will take a while ==="
echo "    Starting in background so you can work on 8B training meanwhile."
echo "    Check progress: tail -f $DEMO_DIR/logs/download_32b.log"
echo ""

if [ -d "$MODELS_DIR/Qwen3-32B" ] && [ "$(ls -A "$MODELS_DIR/Qwen3-32B"/*.safetensors 2>/dev/null)" ]; then
    echo "Qwen3-32B already downloaded, skipping."
else
    nohup bash -c "
        export PATH=\"$DEMO_DIR/miniconda3/bin:\$PATH\"
        eval \"\$(conda shell.bash hook)\"
        conda activate demo
        huggingface-cli download Qwen/Qwen3-32B \
            --local-dir \"$MODELS_DIR/Qwen3-32B\" \
            --local-dir-use-symlinks False
        echo 'Qwen3-32B download complete!'
    " > "$DEMO_DIR/logs/download_32b.log" 2>&1 &

    echo "32B download started in background (PID: $!)"
    echo "Check progress: tail -f $DEMO_DIR/logs/download_32b.log"
fi

echo ""
echo "=== Downloads started ==="
echo ""
echo "Next steps:"
echo "  1. Submit 8B training job:  sbatch /mnt/data/demo/scripts/train_8b.sbatch"
echo "  2. Check job status:        squeue"
echo "  3. Watch training logs:     tail -f /mnt/data/demo/logs/train_8b_<JOBID>.out"
