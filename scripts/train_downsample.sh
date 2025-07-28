#! /bin/bash 

uv run shapesynthesis/train.py configs/encoder_downsample_032_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_064_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_128_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_256_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_512_airplane.yaml



uv run shapesynthesis/train.py configs/encoder_downsample_032_car.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_064_car.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_128_car.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_256_car.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_512_car.yaml


uv run shapesynthesis/train.py configs/encoder_downsample_032_chair.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_064_chair.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_128_chair.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_256_chair.yaml
uv run shapesynthesis/train.py configs/encoder_downsample_512_chair.yaml
