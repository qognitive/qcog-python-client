.PHONY: build
build:
	python -m pip cache purge
	python -m pip install --upgrade pip
	python -m pip install ".[dev]"
	python -m build .

.PHONY: clean
clean:
	rm -rf build dist
