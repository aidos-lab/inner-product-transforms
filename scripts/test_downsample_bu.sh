#! /bin/bash 

python ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_airplane.yaml --encoder_upsample_config ./shapesynthesis/configs/encoder_airplane.yaml --uniform
python ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_car.yaml --encoder_upsample_config ./shapesynthesis/configs/encoder_car.yaml --uniform
python ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_chair.yaml --encoder_upsample_config ./shapesynthesis/configs/encoder_chair.yaml --uniform



python ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_airplane.yaml --encoder_upsample_config ./shapesynthesis/configs/encoder_airplane.yaml 
python ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_car.yaml --encoder_upsample_config ./shapesynthesis/configs/encoder_car.yaml 
python ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_chair.yaml --encoder_upsample_config ./shapesynthesis/configs/encoder_chair.yaml 
