.PHONY: all install build docs

all: install build

install:
	sage -pip install --upgrade .

build:
	sage --python setup.py build

docs:
	sage -sh -c "sphinx-build -b html docs/src docs"
