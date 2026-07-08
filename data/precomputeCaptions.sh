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

PROJECT_DIR="/gpfs/u/home/ZSIS/ZSISsrtk/barn/research"
VENV_NAME="visEnv"
SEG_DATA_DIR="${SEG_LOCAL_DIR:-/gpfs/u/home/ZSIS/ZSISsrtk/scratch/segmentation}"

module purge
module load gcc

export http_proxy=http://proxy:8888
export https_proxy=http://proxy:8888
export HTTP_PROXY=http://proxy:8888
export HTTPS_PROXY=http://proxy:8888

ENV_BIN="/gpfs/u/home/ZSIS/ZSISsrtk/barn/miniconda3/envs/$VENV_NAME/bin"
export PATH="$ENV_BIN:$PATH"

# Run from the repo root, not data/, so `data.precomputeCaptions`'s relative
# imports (`from .segmentation import ...`) resolve correctly.
cd "$PROJECT_DIR"

echo "Starting caption pass (generates text for anything new; tokenizes + deletes"
echo "text left over from a previous run) against $SEG_DATA_DIR ..."
python -m data.precomputeCaptions --data-dir "$SEG_DATA_DIR"

echo "Done. Dataset files:"
ls -lh "$SEG_DATA_DIR"/*.json.gz 2>/dev/null || echo "No dataset files found at $SEG_DATA_DIR."