#! /bin/bash 
set -o errexit
set -o nounset
set -o pipefail


files=$(ls ./shapesynthesis/configs/ect-render)

for file in $files
do 
    python3 ./shapesynthesis/test_render_shapenet.py \
        --config ./shapesynthesis/configs/ect-render/$file \
        --num_reruns 10
done