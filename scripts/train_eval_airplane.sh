#! /bin/bash

DEV="--dev"

# Train both models
# uv run shapesynthesis/train.py ./configs/vae_airplane.yaml $DEV
# uv run shapesynthesis/train.py ./configs/encoder_airplane.yaml $DEV

uv run shapesynthesis/test.py --encoder_config ./configs/encoder_airplane.yaml --dev
# uv run shapesynthesis/test.py --encoder_config ./configs/encoder_airplane.yaml --vae_config ./configs/encoder_airplane.yaml $DEV
