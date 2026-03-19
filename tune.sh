#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=sam_clip_tuning
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --qos=dcs-48hr

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ZSIS/ZSISsrtk/barn/research"
PYTHON_SCRIPT_NAME="tune.py"
VENV_NAME="visEnv"
HF_TOKEN="_"
PHASE="clip"
N_TRIALS=30

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
module load cuda/11.2

export http_proxy=http://proxy:8888
export https_proxy=http://proxy:8888
export HTTP_PROXY=http://proxy:8888
export HTTPS_PROXY=http://proxy:8888

ENV_BIN="/gpfs/u/home/ZSIS/ZSISsrtk/barn/miniconda3/envs/$VENV_NAME/bin"
export PATH="$ENV_BIN:$PATH"

# Verify Python and pip are from the venv
echo "Python executable: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Navigate to the project directory
echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# --- Running the Training Script ---
echo "Pre-downloading HuggingFace cache to prevent DDP race conditions"
python -c "
import os, socket, requests
os.environ['HF_HUB_HTTP_TIMEOUT'] = '3600'
socket.setdefaulttimeout(3600)

import datasets
datasets.config.DOWNLOAD_DEFAULT_TIMEOUT = 3600
datasets.config.MAX_RETRIES = 10

_original_request = requests.Session.request
def _patched_request(self, method, url, **kwargs):
    kwargs['timeout'] = 3600
    return _original_request(self, method, url, **kwargs)
requests.Session.request = _patched_request

_original_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs['timeout'] = 3600
    return _original_send(self, request, **kwargs)
requests.Session.send = _patched_send

from models.clip_model import create_text_encoder, create_image_encoder
from models.prior_model import TeacherCLIP
from transformers import InstructBlipProcessor
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

print('1/4: Downloading CLIP Base models...')
create_text_encoder()
create_image_encoder()

print('2/4: Downloading Teacher CLIP (Large)...')
TeacherCLIP()

print('3/4: Downloading InstructBlip Processor...')
try:
    InstructBlipProcessor.from_pretrained('Salesforce/instructblip-flan-t5-xl')
except Exception:
    pass 

print('4/4: Downloading LAION streaming builder...')
try:
    load_dataset('laion/relaion400m', split='train', streaming=True, token='$HF_TOKEN')
    print('LAION pre-download successful!')
except Exception as e:
    print(f'LAION pre-download failed: {e}')
"

# --- Running the Training Script with torchrun ---
echo "Starting Optuna hyperparameter sweep: phase=$PHASE, trials=$N_TRIALS"
 
python "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN" \
    --phase "$PHASE" \
    --trials "$N_TRIALS"


EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

echo "Job finished."