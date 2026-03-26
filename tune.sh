#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=sam_clip_tuning
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=dcs-48hr

# ============================================================
# Sequential Tuning Workflow
# ============================================================
# Each phase depends on trained weights from previous phases.
# Run them in order, updating PHASE and structural args each time:
#
#   1) PHASE="clip"    — no weights needed, tunes from scratch
#      → train CLIP with best hyperparams, save to WEIGHTS_DIR
#
#   2) PHASE="prior"   — needs trained text encoder from step 1
#      → set TXT_ENC_LAYERS to best num_layers from step 1
#      → train Prior with best hyperparams, save to WEIGHTS_DIR
#
#   3) PHASE="decoder"  — needs trained text encoder + prior
#      → set TXT_ENC_LAYERS and PRIOR_LAYERS from steps 1-2
#      → train SAM decoder for 3 epochs, save to WEIGHTS_DIR
#
#   4) PHASE="student"  — needs full trained teacher pipeline
#      → set all structural params from steps 1-3
#      → teacher warms up for TEACHER_WARMUP_EPOCHS, then student tunes
# ============================================================

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ZSIS/ZSISsrtk/barn/research"
PYTHON_SCRIPT_NAME="tune.py"
VENV_NAME="visEnv"
HF_TOKEN="_"
N_TRIALS=40

# Current phase to tune (change per run)
PHASE="decoder"

# Path to trained weights from previous phases
WEIGHTS_DIR="$PROJECT_DIR/models/trained"

# Structural params from previous tuning results (must match saved weights).
# Update these after each phase completes with the best trial values.
TXT_ENC_LAYERS=6
PRIOR_LAYERS=8
SAM_LAYERS=2
SAM_MEMORY=10
TEACHER_WARMUP_EPOCHS=3

# --- Sanity Checks ---
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "_" ]; then
    echo "Error: Set HF_TOKEN to your actual HuggingFace token."
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

# Auto-create weights dir and warn if empty (tuning will use random init as fallback)
if [ "$PHASE" != "clip" ]; then
    mkdir -p "$WEIGHTS_DIR"
    if [ -z "$(ls -A $WEIGHTS_DIR 2>/dev/null)" ]; then
        echo "WARNING: WEIGHTS_DIR '$WEIGHTS_DIR' is empty."
        echo "  Phase '$PHASE' will use random initialization for upstream components."
        echo "  Results are approximate — retune after training for best accuracy."
    fi
fi

# --- Environment Setup ---
echo "Loading modules..."
module purge
module load gcc

export http_proxy=http://proxy:8888
export https_proxy=http://proxy:8888
export HTTP_PROXY=http://proxy:8888
export HTTPS_PROXY=http://proxy:8888

ENV_BIN="/gpfs/u/home/ZSIS/ZSISsrtk/barn/miniconda3/envs/$VENV_NAME/bin"
export PATH="$ENV_BIN:$PATH"

export HF_HOME=/gpfs/u/barn/ZSIS/ZSISsrtk/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export SAM_CACHE_DIR=/gpfs/u/barn/ZSIS/ZSISsrtk/.cache/sam_data
mkdir -p "$HF_HOME/hub" "$SAM_CACHE_DIR"

echo "Python executable: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

cd "$PROJECT_DIR"

# # --- Pre-download HuggingFace assets ---
# echo "Pre-downloading HuggingFace cache..."
# python -c "
# import os, socket, requests
# os.environ['HF_HUB_HTTP_TIMEOUT'] = '3600'
# socket.setdefaulttimeout(3600)

# import datasets
# datasets.config.DOWNLOAD_DEFAULT_TIMEOUT = 3600
# datasets.config.MAX_RETRIES = 10

# _original_request = requests.Session.request
# def _patched_request(self, method, url, **kwargs):
#     kwargs['timeout'] = 3600
#     return _original_request(self, method, url, **kwargs)
# requests.Session.request = _patched_request

# _original_send = requests.Session.send
# def _patched_send(self, request, **kwargs):
#     kwargs['timeout'] = 3600
#     return _original_send(self, request, **kwargs)
# requests.Session.send = _patched_send

# from models.clip_model import create_text_encoder, create_image_encoder
# from models.prior_model import TeacherCLIP
# from transformers import InstructBlipProcessor
# from datasets import load_dataset
# import warnings
# warnings.filterwarnings('ignore')

# print('1/4: Downloading CLIP Base models...')
# create_text_encoder()
# create_image_encoder()

# print('2/4: Downloading Teacher CLIP (Large)...')
# TeacherCLIP()

# print('3/4: Downloading InstructBlip Processor...')
# try:
#     InstructBlipProcessor.from_pretrained('Salesforce/instructblip-flan-t5-xl')
# except Exception:
#     pass

# print('4/4: Downloading LAION streaming builder...')
# try:
#     load_dataset('laion/relaion400m', split='train', streaming=True, token='$HF_TOKEN')
#     print('LAION pre-download successful!')
# except Exception as e:
#     print(f'LAION pre-download failed: {e}')
# "

# --- Build the command with phase-appropriate args ---
echo "Starting Optuna hyperparameter sweep: phase=$PHASE, trials=$N_TRIALS"

CMD="python $PYTHON_SCRIPT_PATH \
    --token $HF_TOKEN \
    --phase $PHASE \
    --trials $N_TRIALS"

# Add weight-loading args for phases that depend on previous phases
if [ "$PHASE" != "clip" ]; then
    CMD="$CMD \
    --weights_dir $WEIGHTS_DIR \
    --txt_enc_layers $TXT_ENC_LAYERS"
fi

if [ "$PHASE" = "decoder" ] || [ "$PHASE" = "student" ]; then
    CMD="$CMD \
    --prior_layers $PRIOR_LAYERS"
fi

if [ "$PHASE" = "student" ]; then
    CMD="$CMD \
    --sam_layers $SAM_LAYERS \
    --sam_memory $SAM_MEMORY \
    --teacher_warmup_epochs $TEACHER_WARMUP_EPOCHS"
fi

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Tuning completed successfully."
else
    echo "Tuning exited with error code $EXIT_CODE."
fi

echo "Job finished."