.PHONY: build
build:  schema-build
	python -m pip cache purge
	python -m pip install --upgrade pip
	python -m pip install ".[dev]"
	python -m build .

.PHONY: clean

clean:
	rm -rf build dist

docs-build:
	cd docs && \
	python -m sphinx -T -W --keep-going -b html -d _build/doctrees -D language=en . ./html

lint-check:
	ruff check . && \
	mypy ./qcog_python_client

lint-fix:
	ruff check --fix .

lint-write:
	ruff format .

schema-build:
	python schema.py
