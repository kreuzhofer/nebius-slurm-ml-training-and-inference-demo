#!/bin/bash
# =============================================================================
# setup.sh — Run this ONCE on the login node after SSH-ing into the cluster.
#
# What this does:
#   1. Installs Miniconda on the shared filesystem (so all nodes can use it)
#   2. Creates a conda environment with PyTorch + training/inference libraries
#
# Usage:
#   ssh root@<SLURM_IP>
#   bash /mnt/data/demo/setup.sh
#
# After running, activate the env with:
#   source /mnt/data/demo/activate.sh
# =============================================================================
set -euo pipefail

DEMO_DIR="/mnt/data/demo"
CONDA_DIR="$DEMO_DIR/miniconda3"

echo "=== Setting up demo environment ==="
mkdir -p "$DEMO_DIR"/{models,output,datasets,results,logs,scripts}

# --- Step 1: Install Miniconda ---
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo "Installing Miniconda..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "Miniconda installed at $CONDA_DIR"
else
    echo "Miniconda already installed at $CONDA_DIR"
fi

export PATH="$CONDA_DIR/bin:$PATH"

# --- Step 2: Create conda environment ---
# Check for the actual env directory (not grep, which false-matches the path /mnt/data/demo/...)
if [ ! -d "$CONDA_DIR/envs/demo" ]; then
    echo "Creating conda environment 'demo' with Python 3.11..."
    conda create -y -n demo python=3.11 --override-channels -c conda-forge
else
    echo "Conda environment 'demo' already exists"
fi

eval "$(conda shell.bash hook)"
conda activate demo

# --- Step 3: Install packages ---
echo "Installing PyTorch (CUDA 12.4)..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu124

echo "Installing training libraries..."
pip install -q transformers datasets accelerate peft

echo "Installing vLLM for inference serving..."
pip install -q vllm

echo "Installing evaluation/visualization tools..."
pip install -q matplotlib pandas

# --- Step 4: Create activation helper script ---
cat > "$DEMO_DIR/activate.sh" << 'ACTIVATE'
#!/bin/bash
# Source this to activate the demo environment:
#   source /mnt/data/demo/activate.sh
export DEMO_DIR="/mnt/data/demo"
export PATH="$DEMO_DIR/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate demo
echo "Demo environment activated. DEMO_DIR=$DEMO_DIR"
ACTIVATE
chmod +x "$DEMO_DIR/activate.sh"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To activate the environment, run:"
echo "  source /mnt/data/demo/activate.sh"
echo ""
echo "Next step: download models with:"
echo "  bash /mnt/data/demo/scripts/download.sh"
