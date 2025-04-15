#! /bin/bash 

DEV="--dev"

python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chamfer_car.yaml $DEV
python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chamfer_chair.yaml $DEV
python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chamfer_airplane.yaml $DEV
python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chair.yaml $DEV


# # Chamfer + ECT
# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_car.yaml $DEV
# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_chair.yaml $DEV
# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_airplane.yaml $DEV



# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_ect_car.yaml $DEV
# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_ect_chair.yaml $DEV
# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_ect_airplane.yaml $DEV


# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_downsample_car.yaml $DEV
# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_downsample_chair.yaml $DEV
# python3 ./shapesynthesis/train.py ./shapesynthesis/configs/encoder_downsample_airplane.yaml $DEV
