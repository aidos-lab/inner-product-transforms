#! /bin/bash 

DEV=""

# Chamfer + ECT
python ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_car.yaml $DEV
python ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chair.yaml $DEV
python ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_airplane.yaml $DEV


# python ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chamfer_car.yaml $DEV
# python ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chamfer_chair.yaml $DEV
# python ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chamfer_airplane.yaml $DEV
