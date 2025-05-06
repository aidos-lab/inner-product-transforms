#! /bin/bash 
set -e 

#######################################################################
### Encoder
#######################################################################

# Set the dev flag, runs a subset of the data throught the whole pipeline 
# for end to end testing. 
DEV="--dev"

# Create the airplane dataset (only needs to be ran once)
uv run ./shapesynthesis/datasets/shapenet.py $DEV


# Train the encoder model 
uv run ./shapesynthesis/train.py ./configs/encoder_new_airplane.yaml $DEV

# Test the reconstruction of the model 
uv run ./shapesynthesis/test.py --encoder_config ./configs/encoder_new_airplane.yaml $DEV

#######################################################################
### VAE
#######################################################################

# Train the VAE Model 
uv run ./shapesynthesis/train.py ./configs/vae_sigma_airplane.yaml $DEV

# Test the VAE reconstruction (joint with the encoder)
uv run ./shapesynthesis/test.py --encoder_config ./configs/encoder_new_airplane.yaml --vae_config ./configs/vae_sigma_airplane.yaml $DEV

# Test the VAE generation (joint with the encoder)
uv run ./shapesynthesis/test_generation.py --encoder_config ./configs/encoder_new_airplane.yaml --vae_config ./configs/vae_sigma_airplane.yaml $DEV

