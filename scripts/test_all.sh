#! /bin/bash 

# Chamfer + ECT
python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_car.yaml 
python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chair.yaml 
python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_airplane.yaml 


python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chamfer_car.yaml 
python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chamfer_chair.yaml 
python3 ./shapesynthesis/test.py --encoder_config ./shapesynthesis/configs/encoder_chamfer_airplane.yaml 
