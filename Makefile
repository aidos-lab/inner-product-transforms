lint:
	python -m pylint *.py

black: 
	python -m black . --line-length 80

mypy: 
	python -m mypy main_vae_modelnet.py --ignore-missing-imports