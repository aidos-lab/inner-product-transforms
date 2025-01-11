#! /bin/bash 


# python3 ./shapesynthesis/test_downsample.py \
#     --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_airplane.yaml \
#     --encoder_upsample_config ./shapesynthesis/configs/encoder_airplane.yaml

# python3 ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_car.yaml \
#     --encoder_upsample_config ./shapesynthesis/configs/encoder_car.yaml

# python3 ./shapesynthesis/test_downsample.py --encoder_downsample_config ./shapesynthesis/configs/encoder_downsample_chair.yaml \
#     --encoder_upsample_config ./shapesynthesis/configs/encoder_chair.yaml

# python3 ./shapesynthesis/test_rnd.py --encoder_config ./shapesynthesis/configs/encoder_airplane.yaml --vae_config ./shapesynthesis/configs/vae_airplane.yaml --num_reruns 5
# python3 ./shapesynthesis/test_rnd.py --encoder_config ./shapesynthesis/configs/encoder_chair.yaml --vae_config ./shapesynthesis/configs/vae_chair.yaml --num_reruns 5
# python3 ./shapesynthesis/test_rnd.py --encoder_config ./shapesynthesis/configs/encoder_car.yaml --vae_config ./shapesynthesis/configs/vae_car.yaml --num_reruns 5



# # Chamfer + ECT
# python3 ./shapesynthesis/test_rnd.py --encoder_config ./shapesynthesis/configs/encoder_car.yaml --num_reruns 5
# python3 ./shapesynthesis/test_rnd.py --encoder_config ./shapesynthesis/configs/encoder_chair.yaml --num_reruns 5
# python3 ./shapesynthesis/test_rnd.py --encoder_config ./shapesynthesis/configs/encoder_airplane.yaml --num_reruns 5


python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chamfer_car.yaml --num_reruns 5
python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chamfer_chair.yaml --num_reruns 5
python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chamfer_airplane.yaml --num_reruns 5


python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_ect_car.yaml --num_reruns 5
python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_ect_chair.yaml --num_reruns 5
python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_ect_airplane.yaml --num_reruns 5
