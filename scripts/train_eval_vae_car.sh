#! /bin/bash 

python3 ./shapesynthesis/train.py ./shapesynthesis/configs/vae_car.yaml 
python3 ./shapesynthesis/test_generation.py --encoder_config ./shapesynthesis/configs/encoder_car.yaml --vae_config ./shapesynthesis/configs/vae_car.yaml 

