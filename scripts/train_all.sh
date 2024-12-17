#! /bin/bash 

DEV="--prod"

# # Chamfer + ECT
# python ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_car.yaml $DEV
# python ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chair.yaml $DEV
# python ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_airplane.yaml $DEV


# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_ect_car.yaml $DEV
# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_ect_chair.yaml $DEV
# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_ect_airplane.yaml $DEV


python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_downsample_car.yaml $DEV
python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_downsample_chair.yaml $DEV
python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_downsample_airplane.yaml $DEV
