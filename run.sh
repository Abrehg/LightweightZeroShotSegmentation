#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=sam_clip_training
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --nodes=15
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ZSIS/ZSISsrtk/barn/research"
PYTHON_SCRIPT_NAME="train.py"
VENV_NAME="visEnv"
HF_TOKEN="_"
WANDB_API_KEY="_"

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

export http_proxy=http://proxy:8888
export https_proxy=http://proxy:8888
export HTTP_PROXY=http://proxy:8888
export HTTPS_PROXY=http://proxy:8888
export WANDB_CORE=false

export OMP_NUM_THREADS=4

ENV_BIN="/gpfs/u/home/ZSIS/ZSISsrtk/barn/miniconda3/envs/$VENV_NAME/bin"
export PATH="$ENV_BIN:$PATH"

# Verify Python and pip are from the venv
echo "Python executable: $(which python)"

# Navigate to the project directory
echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# --- Running the Training Script ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export MASTER_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname -I | awk '{print $1}')
export MASTER_PORT=29500 

export NO_PROXY="localhost,127.0.0.1,.ccni.rpi.edu,$MASTER_ADDR,$MASTER_IP"
export no_proxy=$NO_PROXY

export NCCL_SOCKET_IFNAME=^docker,lo
export GLOO_SOCKET_IFNAME=^docker,lo
export NCCL_DEBUG=WARN

export NCCL_TIMEOUT=5400000
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

echo "Pre-downloading HuggingFace cache to prevent DDP race conditions"
python -c "
from models.clip_model import create_text_encoder, create_image_encoder
from models.prior_model import TeacherCLIP
import warnings
warnings.filterwarnings('ignore')
 
print('1/2: Downloading CLIP Base models...')
create_text_encoder()
create_image_encoder()
 
print('2/2: Downloading Teacher CLIP (Large)...')
TeacherCLIP()
 
print('Model cache ready.')
"

# --- Running the Training Script with torchrun ---
echo "Starting distributed training script..."
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_IP: $MASTER_IP"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
    "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN"\
    --wandb_key "$WANDB_API_KEY"


EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

echo "Job finished."