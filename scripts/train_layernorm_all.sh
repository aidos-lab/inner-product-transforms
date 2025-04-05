#! /bin/bash 

DEV="--prod"

# Check if all configs are correct.
uv run python ./shapesynthesis/validate_configuration.py ./configs/encoder_layernorm_car.yaml 
uv run python ./shapesynthesis/validate_configuration.py ./configs/encoder_layernorm_chair.yaml 
uv run python ./shapesynthesis/validate_configuration.py ./configs/encoder_layernorm_airplane.yaml 

# Train all models
uv run python ./shapesynthesis/train.py ./configs/encoder_layernorm_car.yaml $DEV
uv run python ./shapesynthesis/train.py ./configs/encoder_layernorm_chair.yaml $DEV
uv run python ./shapesynthesis/train.py ./configs/encoder_layernorm_airplane.yaml $DEV
