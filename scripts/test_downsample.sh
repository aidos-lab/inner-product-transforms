#! /bin/bash 


uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_032_airplane.yaml --upsampler_config configs/encoder_airplane.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_064_airplane.yaml --upsampler_config configs/encoder_airplane.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_128_airplane.yaml --upsampler_config configs/encoder_airplane.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_256_airplane.yaml --upsampler_config configs/encoder_airplane.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_512_airplane.yaml --upsampler_config configs/encoder_airplane.yaml 

uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_032_car.yaml --upsampler_config configs/encoder_car.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_064_car.yaml --upsampler_config configs/encoder_car.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_128_car.yaml --upsampler_config configs/encoder_car.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_256_car.yaml --upsampler_config configs/encoder_car.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_512_car.yaml --upsampler_config configs/encoder_car.yaml 


uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_032_chair.yaml --upsampler_config configs/encoder_chair.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_064_chair.yaml --upsampler_config configs/encoder_chair.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_128_chair.yaml --upsampler_config configs/encoder_chair.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_256_chair.yaml --upsampler_config configs/encoder_chair.yaml 
uv run shapesynthesis/test_downsample.py --downsampler_config configs/encoder_downsample_512_chair.yaml --upsampler_config configs/encoder_chair.yaml 
