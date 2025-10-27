# Megatron-LM Training Setup

## Environment Details
- Python: 3.12.3
- PyTorch: 2.9.0+cu128
- GPUs: 2x RTX 3090 (24GB each)
- CUDA Version: 12.8 (PyTorch), 13.0 (System)

## Setup Complete ✓

### Virtual Environment
Created with `uv` in `.venv/` directory

### Dependencies Installed
- megatron-core with mlm extras
- Core training dependencies (torch, tensorstore, etc.)
- Note: Optional packages (Transformer Engine, Apex, causal-conv1d) not installed due to CUDA version mismatch - not required for basic training

## Running Training

### Quick Start
```bash
./run_training.sh
```

### Manual Execution
```bash
source .venv/bin/activate
export PYTHONPATH=/home/me/Repos/Megatron-LM:$PYTHONPATH
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

## Training Configuration

The example uses:
- **Tensor Parallelism**: 2-way (splits model across 2 GPUs)
- **Model**: Small GPT (2 layers, 12 hidden size, 4 attention heads)
- **Data**: Mock dataset (no real data preparation needed)
- **Iterations**: 5 training steps
- **Features**:
  - Distributed training across 2 GPUs
  - Automatic gradient synchronization
  - Checkpoint saving and loading

## Checkpoints

Checkpoints are saved to `ckpt/` directory with distributed checkpoint format:
- `__0_0.distcp`, `__0_1.distcp` - Rank 0 checkpoints
- `__1_0.distcp`, `__1_1.distcp` - Rank 1 checkpoints
- `common.pt` - Common metadata
- `metadata.json` - Checkpoint metadata

## Code Fixes Applied

Fixed compatibility issues with PyTorch 2.9:
- Moved `Iterator` import from `torch.utils.data` to `typing`
- Added `Any` to typing imports in `examples/run_simple_mcore_train_loop.py`

## Next Steps

To train with real data:
1. Prepare your dataset in JSONL format
2. Use `tools/preprocess_data.py` to tokenize and prepare data
3. Modify training scripts to use real data instead of mock data
4. Adjust model size, parallelism settings, and training hyperparameters

For larger models, see examples in:
- `examples/llama/` - LLaMA model training
- `examples/gpt3/` - GPT-3 style models
- `examples/mixtral/` - Mixture of Experts models
