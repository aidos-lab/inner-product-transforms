.PHONY: archive

archive: 
	zip -r shape-synthesis.zip 'dependencies/' \
		'./configs' \
		'./sbatch' \
		'./shapesynthesis' \
		'./pyproject.toml' \
		'./README.md' \
		-x 'dependencies/.venv/*' \
		-x '**/__pycache__/*' \
		-x '**/dist/*' \
		-x '**/uv.lock'
	# zip -r myarchive.zip '../shape-synthesis/' -x './notebooks/*' './analysis/' './scratch/' './scripts' 'uv.lock' './trained_models' './trained_models_dev' 
