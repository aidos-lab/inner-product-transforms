#! /bin/bash

uv run shapesynthesis/evaluate_nna.py --results_folder ./results/encoder_airplane
uv run shapesynthesis/evaluate_nna.py --results_folder ./results/encoder_chair
uv run shapesynthesis/evaluate_nna.py --results_folder ./results/encoder_car

uv run shapesynthesis/evaluate_nna.py --results_folder ./results/vae_airplane
uv run shapesynthesis/evaluate_nna.py --results_folder ./results/vae_chair
uv run shapesynthesis/evaluate_nna.py --results_folder ./results/vae_car

uv run shapesynthesis/evaluate_nna.py --results_folder ./results/vae_latent_airplane
uv run shapesynthesis/evaluate_nna.py --results_folder ./results/vae_latent_chair
uv run shapesynthesis/evaluate_nna.py --results_folder ./results/vae_latent_car

