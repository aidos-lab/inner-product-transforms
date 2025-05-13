#!/bin/bash
#SBATCH --job-name=train_all_vae_airplane
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/train_all_vae_airplane_output.txt
#SBATCH --error=./logs/train_all_vae_airplane_error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

# Run your job commands
uv run python ./shapesynthesis/train.py ./configs/vae_airplane.yaml 
