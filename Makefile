.PHONY: venv 

venv: 
	poetry shell

readme: 
	quarto render Readme.ipynb --to gfm --output Readme.md