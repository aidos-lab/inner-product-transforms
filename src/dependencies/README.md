# Installation instructions. 

This is a helper `pyproject.toml` for installing the accelerated 
implementations of the Earth Movers Distance and Chamfer Distance. 
We first build the packages locally, since they depend on the GPU 
architecture. 

To build both packages, run the makefile. This will create a `dist` folder with the wheels 
files. The `pyproject.toml` depends on the existence of these wheel files. 

Once the environment is build, one can the run `uv sync` in the parent folder to build the 
full environment. 

The full set of commands are: 

```{shell}
make venv
```
For removing the `dist` folder run 

```{shell}
make clean
```



