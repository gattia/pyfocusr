.PHONY: build install 

requirements:
	python -m pip install -r requirements.txt
# alternatively... 
# python -m pip install cycpd
# conda install -c conda-forge --file requirements.txt

# requirements-mamba:
# 	mamba install --file requirements.txt

# requirements-conda:
# 	conda install --file requirements.txt

build:
	python -m build -o wheelhouse

install:
	pip install .


install-dev: 
	pip install --editable .

dev:
	pip install --upgrade pytest black isort wheel pdoc3 coverage build

dev-mamba:
	mamba install pytest black isort wheel pdoc3 coverage build

dev-conda:
	conda install pytest black isort wheel pdoc3 coverage build

docs:
	pdoc --output-dir docs/ --html --force pyfocusr
	mv docs/pyfocusr/* docs/
	rm -rf docs/pyfocusr

test:
	set -e
	pytest

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .

autoformat:
	set -e
	isort .
	black --config pyproject.toml .

clean:
	rm -rf build dist *.egg-info  

coverage: 
	coverage run -m pytest
	# Make the html version of the coverage results. 
	coverage html 