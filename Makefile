.PHONY: format lint clean test setup_env interrogate isort black autoflake check freeze
.ONESHELL:
SHELL := /bin/bash

setup_env:
	source setup_env.sh

# --------------------------- Project Hygiene ----------------------------------

clean:
	find . -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete
	rm -rf logs \
		__pycache__ \
		.pytest_cache \
		tests/.pytest_cache \
		.mypy_cache \
		.coverage \
		dist \
		build

# ------------------------------- Run Tests ------------------------------------

test:
	python -m pytest \
		--cov=src \
		--cov-report term-missing \
		--durations=10 \
		tests

# ------------------------------ Code Hygiene -----------------------------------

interrogate:
	interrogate -c pyproject.toml --exclude tests

isort:
	isort .

black:
	black --config pyproject.toml .

autoflake:
	autoflake --config pyproject.toml .

check:
	interrogate
	autoflake
	isort
	black

# ------------------------------ Env Hygiene ------------------------------------

freeze:
	pip freeze > tmp_requirements.txt