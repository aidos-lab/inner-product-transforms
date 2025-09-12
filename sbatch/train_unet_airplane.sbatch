#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=./logs/unet_airplane.txt
#SBATCH --error=./logs/error_unet_airplane.txt
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal
#SBATCH --partition=GPU
#SBATCH --nodelist=diufrd203


#######################################################################
### Encoder
#######################################################################

# Set the dev flag, runs a subset of the data throught the whole pipeline 
# for end to end testing. 
# DEV="--dev"

# Uncomment this line to run the full pipeline. 
DEV=""

nvidia-smi

# Train the encoder model 
uv run ./shapesynthesis/train.py ./configs/unet_airplane.yaml --compile $DEV 
