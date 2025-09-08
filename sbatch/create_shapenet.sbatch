#! /bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=./logs/mnist.txt
#SBATCH --error=./logs/error_mnist.txt
#SBATCH --nice=10000
#SBATCH --qos=cpu_normal
#SBATCH --partition=CPU
#SBATCH --nodelist=diufrd201

# Run your job commands
# uv add gdown
# uv run python -m gdown https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j --folder 
uv run python ./shapesynthesis/datasets/shapenet.py




