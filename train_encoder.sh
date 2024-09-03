#! /bin/bash 

python3 train_encoder.py ./configs/config_encoder_shapenet_airplane.yaml $1

python3 train_encoder.py ./configs/config_encoder_shapenet_chair.yaml $1

python3 train_encoder.py ./configs/config_encoder_shapenet_car.yaml $1

