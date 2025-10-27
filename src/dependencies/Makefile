.PHONY: clean venv 

clean: 
	rm -rf .venv 
	rm -rf dist  

venv:
	uv sync 
	uv build --wheel --no-build-isolation --project ./chamfer3D 
	uv build --wheel --no-build-isolation --project ./PyTorchEMD

