# Shape Synthesis and reconstruction



# Installation 

## Data

```sh 
#! /bin/bash

# Run this script from the root folder 
# bash ./scripts/download_data.sh
# Download data manually from https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j

mkdir -p data/shapenet/raw
mv ShapeNetCore.v2.PC15k.zip data/shapenet/raw
cd data/shapenet/raw
unzip ShapeNetCore.v2.PC15k.zip

```

## Venv 

There are some iffy dependencies on the cuda version of the EMD.  
Custom wheel available for python 3.10

```shell 
poetry install 
```

# Run code 

Train a test model (in a poetry shell).

```
python train.py ./configs/config.yaml
```






