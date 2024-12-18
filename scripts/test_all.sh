#! /bin/bash 


python3 ./shapesynthesis/test_downsample.py \
    --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_airplane.yaml \
    --encoder_upsample_config ./shapesynthesis/configs/encoder_airplane.yaml

python3 ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_car.yaml \
    --encoder_upsample_config ./shapesynthesis/configs/encoder_car.yaml

python3 ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_chair.yaml \
    --encoder_upsample_config ./shapesynthesis/configs/encoder_chair.yaml


# # Chamfer + ECT
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_car.yaml 
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chair.yaml 
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_airplane.yaml 


# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chamfer_car.yaml 
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chamfer_chair.yaml 
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chamfer_airplane.yaml 


# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_ect_car.yaml 
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_ect_chair.yaml 
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_ect_airplane.yaml
#
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_downsample_car.yaml 
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_downsample_chair.yaml 
# python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_downsample_airplane.yaml 
