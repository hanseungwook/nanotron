#!/bin/bash
#SBATCH --job-name=flash-attn-test    # Job name
#SBATCH --output=flash-attn-test-%j.log   # Standard output and error log
#SBATCH --nodes=1                     # Run on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=16G                     # Memory limit
#SBATCH --time=00:30:00               # Time limit (30 minutes)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --partition=main

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start: $(date)"

# Load modules if needed (uncomment and modify as necessary for your environment)
# module load cuda/11.7
# module load python/3.9

# Activate virtual environment if needed (uncomment and modify for your environment)
source $HOME/nanotron-env/bin/activate

# Display information about available GPUs
nvidia-smi

# Run the script
echo "Starting example execution..."
python3 sample_packing.py

# Print end time
echo "End: $(date)"
