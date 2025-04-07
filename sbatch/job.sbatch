#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=my_job_output.txt
#SBATCH --error=my_job_error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

# Run your job commands
# poetry run python ./shapesynthesis/train.py ./configs/encoder_airplane.yaml
# poetry run python ./shapesynthesis/train.py ./configs/encoder_multiclass.yaml
poetry run python ./shapesynthesis/scratch_dataset.py
