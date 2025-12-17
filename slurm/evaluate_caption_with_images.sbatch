#!/bin/bash
#SBATCH --job-name=eval_img_gen
#SBATCH --account=pr_194_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/eval_img_gen_%j.out
#SBATCH --error=../logs/eval_img_gen_%j.err

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

# Navigate to project
cd /scratch/kdg7224/fmri_caption_project

# Create logs directory
mkdir -p logs

echo "Starting fMRI-to-Caption evaluation with image generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Find best model checkpoint
BEST_CHECKPOINT=$(find checkpoints_caption -name "best.pt" -type f -exec ls -t {} + 2>/dev/null | head -1)

if [ -z "$BEST_CHECKPOINT" ] || [ ! -f "$BEST_CHECKPOINT" ]; then
    echo "Error: No best.pt checkpoint found!"
    exit 1
fi

echo "Evaluating model: $BEST_CHECKPOINT"
echo "This will generate images from captions using Stable Diffusion"
echo "This may take several hours depending on test set size..."

# Check and install dependencies
echo "Checking dependencies..."
python3 << 'PYEOF'
import subprocess
import sys

def check_and_install(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
        print(f"✓ {package} is installed")
        return True
    except ImportError:
        print(f"⚠ {package} not found - attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"], 
                                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            print(f"✓ {package} installed")
            return True
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
            return False

# Check pycocoevalcap (critical for caption metrics)
check_and_install("pycocoevalcap", "pycocoevalcap")

# Verify installation
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    print("✓ All caption metric modules imported successfully")
except ImportError as e:
    print(f"✗ ERROR: Failed to import caption metrics: {e}")
    print("  Caption metrics will not be computed!")
PYEOF

# Run evaluation with image generation
# Use fewer inference steps for faster generation (can increase for better quality)
python test/evaluate_caption_with_images.py \
    --checkpoint "$BEST_CHECKPOINT" \
    --generate-images \
    --save-images \
    --num-inference-steps 20

echo "Evaluation completed!"
echo "End time: $(date)"

