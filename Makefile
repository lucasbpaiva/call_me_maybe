PYTHON = uv run python
FLAKE8 = uv run flake8
MYPY = uv run mypy

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	$(FLAKE8) src/
	$(MYPY) --warn-return-any\
			--warn-unused-ignores\
			--ignore-missing-imports\
			--disallow-untyped-defs\
			--check-untyped-defs\
			src/

lint-strict:
	$(FLAKE8) src/
	$(MYPY) --strict src/
