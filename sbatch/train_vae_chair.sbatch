#!/bin/bash
#SBATCH --job-name=full_chair
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/full_chair_output.txt
#SBATCH --error=./logs/full_chair_error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

#######################################################################
### Encoder
#######################################################################

# Set the dev flag, runs a subset of the data throught the whole pipeline 
# for end to end testing. 
DEV="--dev"

# Uncomment this line to run the full pipeline. 
DEV=""

# Train the encoder model 
uv run ./shapesynthesis/train.py ./configs/encoder_new_chair.yaml $DEV

# Test the reconstruction of the model 
uv run ./shapesynthesis/test.py --encoder_config ./configs/encoder_new_chair.yaml $DEV

#######################################################################
### VAE
#######################################################################

# Train the VAE Model 
uv run ./shapesynthesis/train.py ./configs/vae_chair.yaml $DEV

# Test the VAE reconstruction (joint with the encoder)
uv run ./shapesynthesis/test.py --encoder_config ./configs/encoder_new_chair.yaml --vae_config ./configs/vae_chair.yaml $DEV

# Test the VAE generation (joint with the encoder)
uv run ./shapesynthesis/test_generation.py --encoder_config ./configs/encoder_new_chair.yaml --vae_config ./configs/vae_chair.yaml $DEV


