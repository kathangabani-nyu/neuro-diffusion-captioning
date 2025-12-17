#!/bin/bash
#SBATCH --job-name=fmri_caption
#SBATCH --account=pr_194_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/train_caption_%j.out
#SBATCH --error=../logs/train_caption_%j.err

# Load modules
module purge
module load anaconda3/2024.02
module load cuda/11.6.2

# Activate environment
source activate /scratch/kdg7224/envs/dynadiff

# Set paths
export PYTHONPATH="/scratch/kdg7224/fmri_caption_project:$PYTHONPATH"
export HF_HOME=/scratch/kdg7224/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/kdg7224/.cache/huggingface/hub

# PyTorch memory optimization to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Navigate to project
cd /scratch/kdg7224/fmri_caption_project

# Create logs directory
mkdir -p logs

echo "Starting fMRI-to-Caption training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Check for latest checkpoint to resume from (most recent latest.pt file)
LATEST_CHECKPOINT=$(find checkpoints_caption -name "latest.pt" -type f -exec ls -t {} + 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint to resume from: $LATEST_CHECKPOINT"
    RESUME_ARG="--resume $LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting from scratch"
    RESUME_ARG=""
fi

# Run training
python train/train_caption_model.py --config config/train_config.yaml $RESUME_ARG

echo "Training completed!"
echo "End time: $(date)"

