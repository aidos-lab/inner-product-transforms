#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=./logs/mnist.txt
#SBATCH --error=./logs/error_mnist.txt
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal
#SBATCH --partition=GPU




# Run your job commands
#uv run python ./shapesynthesis/datasets/shapenet.py
uv run python ./shapesynthesis/datasets/mnist.py
