#!/bin/bash

# # Clean up previous test weights to avoid confusion
# rm -rf weights_test
# mkdir -p weights_test

# Set environment variables for local testing
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the script
python test_pipeline.py \
  --token "_" \
  --wandb_key "_"