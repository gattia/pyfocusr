.PHONY: build install 

requirements:
	python -m pip install scikit-build
	python -m pip install -r requirements.txt

# alternatively
# conda install -c conda-forge --file requirements.txt
# python -m pip install -r requirements-pip.txt
	


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
	python -m pip install --upgrade -r requirements-dev.txt

dev-mamba:
	mamba install --file requirements-dev.txt

dev-conda:
	conda install --file requirements-dev.txt
	
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