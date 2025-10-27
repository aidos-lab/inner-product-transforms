# Point Cloud Synthesis Using Inner Product Transforms
[![arXiv](https://img.shields.io/badge/arXiv-2410.18987-b31b1b.svg)](https://arxiv.org/abs/2410.18987) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/inner-product-transforms) ![GitHub](https://img.shields.io/github/license/aidos-lab/inner-product-transforms) [![Maintainability](https://qlty.sh/badges/b7958e48-8382-4fb0-ac0b-63b95b7a5426/maintainability.svg)](https://qlty.sh/gh/aidos-lab/projects/inner-product-transforms)


Welcome to the repository for our work on point cloud generation. 

**Please note: This is a work-in-progress repository. Expect some breaking changes!**

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

## Virtual environment

First install the dependencies and then the full virtual 
environment. The dependencies are known to cause some trouble at times 
and the code is provided on an as is basis. The cuda toolkit for 
cuda 12 is required to compile the code, which can be checked with the 
command `nvcc`. If present you should be good to go. 

```shell
cd dependencies
make venv
```

**Please note: Always import torch before either of the two packages, else you will encounter errors.**

Once the dependencies are installed run `uv sync` in the 
top level directory.

Motivation for the custom installation is the fact that 
the kernel is dependent on the specific GPU architecture 
and therefore has to be compiled with an exact python 
version `3.10.12` and an exact cuda version `cu124`. 
Other versions have not been tested.

Full script. 


```shell
cd dependencies
make venv 
cd .. 
uv sync
```


# Training and evaluation.

For training and testing models there is an 
sbatch script. 

After preparing the data, prepare them for training via the command: 

```shell
uv run shapesynthesis/datasets/shapenet.py
```

To check that model training works we can train a VAE and an encoder in the 
development environment via the commands

```
python train_encoder.py ./configs/encoder_airplane.yaml --dev
python train_vae.py ./configs/vae_airplane.yaml --dev
```

It will store the trained model under `trained_models_dev`. 
To evaluate the encoder model, we run 

```shell
uv run ./shapesynthesis/test.py \
    --encoder_config ./configs/encoder_airplane.yaml --dev
```

That is to say that a specific model is tied to a specific 
configuration in a `1:1` relation.

To evaluate the reconstruction of a generative model, we provide both an 
encoder and a VAE model. 

```shell
uv run ./shapesynthesis/test.py \
    --encoder_config ./configs/encoder_airplane.yaml \
    --vae_config ./configs/vae_airplane.yaml --dev 
```

To evaluate the generative performance we run 

```shell
uv run ./shapesynthesis/test_generation.py \
    --encoder_config ./configs/encoder_airplane.yaml \
    --vae_config ./configs/vae_airplane.yaml --dev 
```

In this last case, the Encoder is _not_ taken from the dev folder. 

This takes a while to run (~15-30 minutes) so some patience is 
required. 


## Developing your own Models. 

If you wish to build your own models the following instructions 
should help. 

In the model directory we can add a new file `my_new_model.py`
and add two base classes to it. 
First is the `ModelConfig` class containing all the model configurations 
and one is a `BaseLightningModel` class which can be copied verbatim from 
one of the existing models. 

**ModelConfig.** The ModelConfig class has one mandatory argument and 
that is the relative module path to the specific model. 
In the example above, if the model is implemented in the file 
`my_new_model.py`, the module_path gets the value `models.my_new_model`. 

The remainder of the arguments are arbitrary and depend on the type 
of model you aim to develop. 
The corresponding section in the configuration yaml will have exactly the 
same keys as this class and will serve for input validation. 

**BaseLightningModule** This class contains all specifics for dealing with the 
torch geometric datasets and logging metrics and will need only need minimal change. 
The actual model lives under the `self.model` and can be implemented separately. 


## Citation

If you believe our work was useful for your research, please consider a citation.

```{bibtex}
@inproceedings{Roell25a,
	archiveprefix = {arXiv},
	author = {Ernst R{\"o}ell and Bastian Rieck},
	booktitle = {Advances in Neural Information Processing Systems},
	eprint = {2410.18987},
	primaryclass = {cs.CV},
	publisher = {Curran Associates, Inc.},
	pubstate = {inpress},
	title = {Point Cloud Synthesis Using Inner Product Transforms},
	volume = {38},
	year = {2025},
}
```


