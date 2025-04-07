#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=multiclass_job_output.txt
#SBATCH --error=multiclass_job_error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

# Run your job commands
uv run ./shapesynthesis/train.py ./configs/vae_multiclass_full.yaml
# poetry run python ./shapesynthesis/scratch_emd.py
