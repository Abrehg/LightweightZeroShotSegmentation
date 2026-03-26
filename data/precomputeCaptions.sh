#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=precompute_captions
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=dcs-48hr

PROJECT_DIR="/gpfs/u/home/ZSIS/ZSISsrtk/barn/research/data"
VENV_NAME="visEnv"

module purge
module load gcc

export http_proxy=http://proxy:8888
export https_proxy=http://proxy:8888
export HTTP_PROXY=http://proxy:8888
export HTTPS_PROXY=http://proxy:8888

ENV_BIN="/gpfs/u/home/ZSIS/ZSISsrtk/barn/miniconda3/envs/$VENV_NAME/bin"
export PATH="$ENV_BIN:$PATH"

cd "$PROJECT_DIR"

echo "Starting caption precomputation..."
python -m precomputeCaptions.py --output cache/caption_cache.json

echo "Done. Cache file:"
ls -lh cache/caption_cache.json 2>/dev/null || echo "No cache file created."