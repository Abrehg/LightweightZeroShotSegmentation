#!/bin/bash

#SBATCH --job-name=sam_clip_training    # Job name
#SBATCH --output=sam_clip_training_%j.out # Standard output and error log (%j expands to jobId)
#SBATCH --error=sam_clip_training_%j.err  # Separate error file (optional)
#SBATCH --partition=gpu                 # Partition (queue) name - CHANGE THIS
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (usually 1 for single Python script)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task - ADJUST
#SBATCH --gres=gpu:1                    # Number of GPUs per node - ADJUST (e.g., gpu:rtx3090:1)
#SBATCH --mem=32G                       # Memory per node - ADJUST (e.g., 32G, 64G)
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec - ADJUST

# --- User-configurable ---
PROJECT_DIR="/path/to/your/project_directory" # IMPORTANT: Set this to your project's absolute path
PYTHON_SCRIPT_NAME="train.py"
VENV_NAME="myenv" # Name of your virtual environment directory
HF_TOKEN="$1" # Expects the Hugging Face token as the first argument to this script

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

VENV_PATH="$PROJECT_DIR/$VENV_NAME"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment $VENV_PATH does not exist. Please create it first."
    exit 1
fi


# --- Environment Setup ---
echo "Loading modules..."
module purge # Clear any inherited modules
# Load modules required for your environment -
# These are examples, replace with modules available on your cluster
module load python/3.9.12   # Or your preferred Python version
module load cuda/11.7       # Or your required CUDA version
module load cudnn/8.5.0-cuda11.7 # Or your required cuDNN version (often loaded with CUDA)
# Add any other modules your project might need (e.g., gcc for compilation)

echo "Activating Python virtual environment..."
source "$VENV_PATH/bin/activate"

# Verify Python and pip are from the venv
echo "Python executable: $(which python)"
echo "Pip executable: $(which pip)"

# Navigate to the project directory
echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# --- WandB Setup (Optional, if you need specific API key handling beyond .netrc) ---
# If your train.py script handles wandb login or you have a global .netrc, this might not be needed.
# export WANDB_API_KEY="YOUR_WANDB_API_KEY_IF_NOT_USING_NETRC_OR_LOGIN_IN_SCRIPT"

# --- Running the Training Script ---
echo "Starting Python training script: $PYTHON_SCRIPT_NAME"
echo "Using Hugging Face Token: $HF_TOKEN"

# The `stdbuf -oL -eL` commands help ensure that output is line-buffered,
# which can be useful for seeing logs in real-time.
stdbuf -oL -eL python "$PYTHON_SCRIPT_PATH" --token "$HF_TOKEN"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

# --- Deactivate Virtual Environment ---
echo "Deactivating virtual environment..."
deactivate

echo "Job finished."