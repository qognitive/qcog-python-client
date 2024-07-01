.PHONY: build
build:
	python -m pip cache purge
	python -m pip install --upgrade pip
	python -m pip install ".[dev]"
	python -m build .

.PHONY: clean

clean:
	rm -rf build dist

lint-check:
	ruff check ./qcog_python_client && \
	mypy ./qcog_python_client

lint-write:
	ruff format ./qcog_python_client