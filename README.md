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

Custom version runs on `uv`. 
For training models the following suffices. 
```shell 
uv sync 
```

For evaluation of the emd, the `emd_kernel.cu` needs to be 
installed. For that run the following in the environment 
where the code will be ran, to have access to the specific 
NVIDIA architecture in the environment variables. This 
also assumes `nvcc` is installed. Also first install 
the virtual environment for availability of the other 
dependencies. 

Motivation for the custom installation is the fact that 
the kernel is dependent on the specific GPU architecture 
and therefore has to be compiled with an exact python 
version `3.10.12` and an exact cuda version `cu124`. 
Other versions have not been tested.

(On the HPC cluster this has to be ran via an sbatch job.)

```shell
cd shapesynthesis/metrics/PyTorchEMD
uv build --wheel --no-build-isolation
```

Under `shapesynthesis/metrics/PyTorchEMD/dist/emd_ext-0.0.0-cp310-cp310-linux_x86_64.whl` the compiled kernel can be 
found and added to the virtual environment via 

```shell
uv add shapesynthesis/metrics/PyTorchEMD/dist/emd_ext-0.0.0-cp310-cp310-linux_x86_64.whl
```

# Training and evaluation.

Train a model via the following command. 

```
python train.py ./configs/encoder_airplane.yaml
```

It will store the trained model under `trained_models`. 
To evaluate an encoder model, we run 

```shell
uv run ./shapesynthesis/test.py \
    --encoder_config ./configs/encoder_airplane.yaml
```
That is to say that a specific model is tied to a specific 
configuration in a `1:1` relation.

To evaluate the reconstruction of a generative model, we provide both an 
encoder and a VAE model. 

```shell
uv run ./shapesynthesis/test.py \
    --encoder_config ./configs/encoder_airplane.yaml \
    --vae_config ./configs/vae_airplane.yaml
```

To evaluate the generative performance we run 

```shell
uv run ./shapesynthesis/test_generation.py \
    --encoder_config ./configs/encoder_airplane.yaml \
    --vae_config ./configs/vae_airplane.yaml
```

This takes a while to run (~30 minutes) so some patience is 
required. 


## Developing your own Models. 

If you wish to build your own models the following instructions 
should help. 

In the model directory we can add a new file `my_new_model.py`
and add two base classes to it. 
First is the `ModelConfig` class containing all the model configurations 
and one is a `BaseLightningModel` class which can be copied verbatim from 
one of the already implemented models. 

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

