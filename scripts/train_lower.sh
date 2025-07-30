#! /bin/bash 

uv run shapesynthesis/train.py configs/encoder_numdir_004_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_numdir_008_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_numdir_016_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_numdir_032_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_numdir_064_airplane.yaml


uv run shapesynthesis/train.py configs/encoder_numres_004_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_numres_008_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_numres_016_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_numres_032_airplane.yaml
uv run shapesynthesis/train.py configs/encoder_numres_064_airplane.yaml

