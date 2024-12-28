#! /bin/bash 
set -o errexit
set -o nounset
set -o pipefail


files=$(ls ./shapesynthesis/configs/ect-render)

for file in $files
do 
    python3 ./shapesynthesis/point_cloud_render.py --config ./shapesynthesis/configs/ect-render/$file 
done