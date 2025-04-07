#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=multiclass_eval_output.txt
#SBATCH --error=multiclass_eval_error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

# # Run your job commands
uv run ./shapesynthesis/test_generation.py \
    --encoder_config ./configs/encoder_multiclass_full.yaml \
    --vae_config ./configs/vae_multiclass_full.yaml

# uv run scratch.py


