
.PHONY: build
build:  schema-build
	python -m pip cache purge
	python -m pip install --upgrade pip
	python -m pip install ".[dev]"
	python -m build .

.PHONY: clean
clean:
	rm -rf build dist

docs-build: schema-build
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

test-unit:
	export PYTHONPATH=.
	pytest -v --cov=qcog_python_client tests/unit

test-integration:
	export PYTHONPATH=. && \
	export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} && \
	export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} && \
	export AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} && \
	pytest -v --cov=qcog_python_client tests/integration
