#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH to include Megatron-LM
export PYTHONPATH=/home/me/Repos/Megatron-LM:$PYTHONPATH

# Run training on 2 GPUs
echo "Running Megatron-LM training on 2 RTX 3090s..."
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

echo "Training complete! Check the 'ckpt' directory for saved checkpoints."
