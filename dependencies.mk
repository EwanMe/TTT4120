.PHONY: dependencies
dependencies:
	python -m venv ../venv
	../venv/bin/pip install -r ../requirements.txt
