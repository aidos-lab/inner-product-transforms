#! /bin/bash

# Run this script from the root folder 
# bash ./scripts/download_data.sh
# Download data manually from https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j

mkdir -p data/shapenet/raw
mv ShapeNetCore.v2.PC15k.zip data/shapenet/raw
cd data/shapenet/raw
unzip ShapeNetCore.v2.PC15k.zip



