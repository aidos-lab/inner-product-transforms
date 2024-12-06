.PHONY: venv 

venv: 
	poetry shell

readme: 
	quarto render readme.ipynb --to gfm --output Readme.md