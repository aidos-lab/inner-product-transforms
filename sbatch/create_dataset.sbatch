#!/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=./logs/create_dataset_output.txt
#SBATCH --error=./logs/create_dataset_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

# Run your job commands
uv run python ./shapesynthesis/datasets/shapenet.py
