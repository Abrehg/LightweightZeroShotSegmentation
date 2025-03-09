# Sample sbatch script (run.sh)
#!/bin/bash
#SBATCH --job-name=clip-training
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --partition=superpod

srun python train.py \
  --coco-path /datasets/coco \
  --cc3m-path /datasets/cc3m \
  --custom400m-path /datasets/custom400m \
  --batch-size 4096 \
  --epochs 20