.PHONY: build-assignment-1
build-assignment-1:
	python -m venv assignment-1/venv
	assignment-1/venv/bin/pip install -r assignment-1/requirements.txt

.PHONY: assignment-1
assignment-1: build-assignment-1
	assignment-1/venv/bin/python assignment-1/problem-2.py