#! /bin/bash

python3 train_vae.py ./configs/config_vae_shapenet_chair.yaml
python3 train_vae.py ./configs/config_vae_shapenet_car.yaml


