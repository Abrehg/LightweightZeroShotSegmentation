#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=sam_clip_training
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ZSIS/ZSISsrtk/barn/train.py"
PYTHON_SCRIPT_NAME="train.py"
VENV_NAME="visEnv"


# --- Sanity Checks ---
if [ -z "$HF_TOKEN" ]; then
    echo "Error: Hugging Face token not provided."
    echo "Usage: sbatch $0 YOUR_HF_TOKEN"
    exit 1
fi

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory $PROJECT_DIR does not exist."
    exit 1
fi

PYTHON_SCRIPT_PATH="$PROJECT_DIR/$PYTHON_SCRIPT_NAME"
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script $PYTHON_SCRIPT_PATH does not exist."
    exit 1
fi

# --- Environment Setup ---
echo "Loading modules..."
module purge
module load gcc
module load spectrum-mpi
module load cuda/11.2

conda activate $VENV_NAME

# Verify Python and pip are from the venv
echo "Python executable: $(which python)"
echo "Pip executable: $(which pip)"

# Navigate to the project directory
echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"


# --- Running the Training Script ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500 # A free port

# --- Running the Training Script with torchrun ---
echo "Starting distributed training script..."
echo "MASTER_ADDR: $MASTER_ADDR"

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN"\
    --wandb_key "$WANDB_API_KEY"


EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

conda deactivate

echo "Job finished."