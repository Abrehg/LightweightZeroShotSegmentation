#!/bin/bash

# Clean up previous test weights to avoid confusion
rm -rf weights_test
mkdir -p weights_test

# Set environment variables for local testing
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the script
# We use standard python execution. The script inside automatically detects 
# that it is running on a Mac and switches to Single-Device mode 
# while maintaining the logic flow of the distributed setup.
python train.py \
  --token "YOUR_HUGGINGFACE_TOKEN" \
  --wandb_key "YOUR_WANDB_KEY"