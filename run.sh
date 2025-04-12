#!/bin/bash
#SBATCH --job-name=multistage-training
#SBATCH --nodes=8
#SBATCH --gres=gpu:a100:8
#SBATCH --time=72:00:00
#SBATCH --partition=superpod
#SBATCH --output=training-%j.out
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128G

# Load modules
module load cuda/11.8 nccl python/3.10 wandb

# Set up environment
export DATA_DIR=/scratch/$USER/datasets
mkdir -p $DATA_DIR

# Run training
srun python train.py \
    --data-dir $DATA_DIR \
    --train-clip \
    --train-prior \
    --train-teacher \
    --train-student \
    --batch-size 2048 \
    --clip-epochs 10 \
    --prior-epochs 5 \
    --teacher-epochs 10 \
    --student-epochs 20

# Cleanup temporary files
rm -rf $DATA_DIR/tmp