#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — GCP VM Bootstrap for DLLM Refactoring Agent
# Run this ONCE on the GCP Compute Engine VM after SSH connection.
# Usage: bash scripts/setup_env.sh
# =============================================================================
set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }
fail() { echo -e "${RED}[FAIL] $*${NC}"; exit 1; }

# ── Guard: must be on Linux (the GCP VM) ──────────────────────────────────────
[[ "$(uname)" != "Linux" ]] && fail "This script must be run on the GCP Linux VM, not your Mac."

# =============================================================================
log "[1/8] System packages"
# =============================================================================
sudo apt-get update -qq
sudo apt-get install -y \
    git curl wget build-essential \
    python3-dev libpython3-dev pkg-config \
    htop tmux unzip

# =============================================================================
log "[2/8] Verify CUDA 12.x (should be pre-installed on deeplearning image)"
# =============================================================================
if ! command -v nvcc &>/dev/null; then
    warn "nvcc not found — checking nvidia-smi..."
    nvidia-smi || fail "No GPU detected. Did you provision with --accelerator flag?"
else
    nvcc --version
fi
nvidia-smi

# =============================================================================
log "[3/8] Miniconda (if not present)"
# =============================================================================
if ! command -v conda &>/dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    log "Miniconda installed — re-sourcing shell"
    source ~/.bashrc
else
    log "Conda already present: $(conda --version)"
fi

# =============================================================================
log "[4/8] Create conda env: dllm-refactor (Python 3.11)"
# =============================================================================
if conda env list | grep -q "dllm-refactor"; then
    warn "Env 'dllm-refactor' already exists — skipping creation"
else
    conda create -n dllm-refactor python=3.11 -y
fi

# Activate for the rest of the script
eval "$(conda shell.bash hook)"
conda activate dllm-refactor

# =============================================================================
log "[5/8] PyTorch 2.3 with CUDA 12.4 (compatible with CUDA 12.9 image)"
# =============================================================================
pip install --quiet \
    torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Sanity check
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after install!'
print(f'  ✅ PyTorch {torch.__version__} | CUDA {torch.version.cuda} | Device: {torch.cuda.get_device_name(0)}')
"

# =============================================================================
log "[6/8] Core ML dependencies"
# =============================================================================
pip install --quiet \
    "transformers>=4.45" \
    "accelerate>=0.30" \
    "bitsandbytes>=0.43" \
    datasets \
    evaluate \
    sentencepiece \
    protobuf \
    huggingface_hub

# Dev / evaluation tooling
pip install --quiet \
    black isort \
    pytest pytest-json-report \
    zss          # Zhang-Shasha tree-edit distance for AST structural distance metric

# =============================================================================
log "[7/8] tree-sitter (AST parsing for coherence metrics)"
# =============================================================================
pip install --quiet \
    "tree-sitter>=0.22" \
    tree-sitter-python

# Verify
python -c "
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
PY_LANG = Language(tspython.language())
parser = Parser(PY_LANG)
tree = parser.parse(b'def foo(): pass')
print(f'  ✅ tree-sitter OK — root: {tree.root_node.type}')
"

# =============================================================================
log "[8/8] Clone research repos"
# =============================================================================
REPO_DIR="$HOME/repos"
mkdir -p "$REPO_DIR"

# LLaDA 1.5 — official implementation (GSAI-ML)
if [ ! -d "$REPO_DIR/LLaDA" ]; then
    git clone --depth=1 https://github.com/ML-GSAI/LLaDA.git "$REPO_DIR/LLaDA"
    log "LLaDA cloned"
else
    warn "LLaDA already present — skipping clone"
fi

# DAWN — plug-and-play dependency-aware decoder
if [ ! -d "$REPO_DIR/DAWN" ]; then
    git clone --depth=1 https://github.com/lizhuo-luo/DAWN.git "$REPO_DIR/DAWN"
    log "DAWN cloned"
else
    warn "DAWN already present — skipping clone"
fi

# RefactorBench — primary evaluation benchmark
if [ ! -d "$REPO_DIR/RefactorBench" ]; then
    git clone --depth=1 https://github.com/microsoft/RefactorBench.git "$REPO_DIR/RefactorBench"
    log "RefactorBench cloned"
else
    warn "RefactorBench already present — skipping clone"
fi

# =============================================================================
log "=== Environment Setup Complete ==="
# =============================================================================
echo ""
echo "  Activate with:  conda activate dllm-refactor"
echo "  Project repos:  $REPO_DIR"
echo ""

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "
import torch, transformers, tree_sitter
print(f'  PyTorch     : {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  tree-sitter : {tree_sitter.__version__}')
print(f'  GPU         : {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)')
"
echo ""
echo "✅ Ready to run: python core/refactor_agent.py"
