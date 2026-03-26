#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=sam_finetune_referring
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=10
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr

# ============================================================
# Fine-tune on Referring Expression Datasets
# ============================================================
# Run this AFTER the main train.py pipeline has completed.
# Downloads / expects:
#   - gRefCOCO: data/grefcoco/grefs(unc).json + instances.json
#   - COCO images: data/images/train2014/
#   - Ref-YouTube-VOS: data/ref-youtube-vos/train/{JPEGImages,Annotations}
#                      data/ref-youtube-vos/meta_expressions/train/meta_expressions.json
# ============================================================

PROJECT_DIR="/gpfs/u/home/ZSIS/ZSISsrtk/barn/research"
VENV_NAME="visEnv"
WANDB_API_KEY="_"
CHECKPOINT_DIR="$PROJECT_DIR/weights"

# --- Environment Setup ---
module purge
module load gcc

export http_proxy=http://proxy:8888
export https_proxy=http://proxy:8888
export HTTP_PROXY=http://proxy:8888
export HTTPS_PROXY=http://proxy:8888

ENV_BIN="/gpfs/u/home/ZSIS/ZSISsrtk/barn/miniconda3/envs/$VENV_NAME/bin"
export PATH="$ENV_BIN:$PATH"

echo "Python: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
cd "$PROJECT_DIR"

# --- DDP Setup ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname -I | awk '{print $1}')
export MASTER_PORT=29500

export NO_PROXY="localhost,127.0.0.1,.ccni.rpi.edu,$MASTER_ADDR,$MASTER_IP"
export no_proxy=$NO_PROXY
export NCCL_SOCKET_IFNAME=^docker,lo
export GLOO_SOCKET_IFNAME=^docker,lo

# --- Run Fine-Tuning ---
echo "Starting referring expression fine-tuning..."

srun --ntasks=1 --ntasks-per-node=1 \
    python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
    tuneModel.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --wandb_key "$WANDB_API_KEY"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Fine-tuning completed successfully."
else
    echo "Fine-tuning exited with error code $EXIT_CODE."
fi
echo "Job finished."