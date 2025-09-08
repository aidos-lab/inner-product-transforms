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
#SBATCH --nodelist=diufrd203



#######################################################################
### Encoder
#######################################################################

# Set the dev flag, runs a subset of the data throught the whole pipeline 
# for end to end testing. 
DEV="--dev"


# Train the encoder model 
uv run ./shapesynthesis/train.py ./configs/vqvae_mnist.yaml $DEV

# # Test the reconstruction of the model 
# uv run ./shapesynthesis/test.py --encoder_config ./configs/encoder_new_airplane.yaml $DEV
#
# #######################################################################
# ### VAE
# #######################################################################
#
# # Train the VAE Model 
# uv run ./shapesynthesis/train.py ./configs/vae_airplane.yaml $DEV
#
# # Test the VAE reconstruction (joint with the encoder)
# uv run ./shapesynthesis/test.py --encoder_config ./configs/encoder_new_airplane.yaml --vae_config ./configs/vae_airplane.yaml $DEV
#
# # Test the VAE generation (joint with the encoder)
# uv run ./shapesynthesis/test_generation.py --encoder_config ./configs/encoder_new_airplane.yaml --vae_config ./configs/vae_airplane.yaml $DEV


