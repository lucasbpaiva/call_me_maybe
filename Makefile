ifneq ($(wildcard /goinfre/.),)
	export HF_HOME := /goinfre/$(USER)/hf_cache
	export UV_CACHE_DIR := /goinfre/$(USER)/uv_cache
else
	export HF_HOME := $(HOME)/.cache/huggingface/hub
	export UV_CACHE_DIR := $(HOME)/.cache/uv
endif

VENV = .venv
PYTHON = uv run python
FLAKE8 = uv run flake8
MYPY = uv run mypy

.PHONY: install run debug clean lint lint-strict

setup-env:
	@mkdir -p $(HF_HOME)
	@mkdir -p $(UV_CACHE_DIR)
ifneq ($(wildcard /goinfre/.),)
	@if [ ! -L ".venv" ]; then \
		rm -rf .venv; \
		mkdir -p /goinfre/$(USER)/call_me_maybe_venv; \
		ln -s /goinfre/$(USER)/call_me_maybe_venv .venv; \
	fi
endif

install: setup-env
	@uv sync

run: setup-env
	@$(PYTHON) -m src

debug:
	@$(PYTHON) -m pdb -m src

clean:
	@rm -rf $(VENV)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	@$(FLAKE8) src/
	@$(MYPY) --warn-return-any\
			--warn-unused-ignores\
			--ignore-missing-imports\
			--disallow-untyped-defs\
			--check-untyped-defs\
			src/

lint-strict:
	@$(FLAKE8) src/
	@$(MYPY) --strict src/

test: setup-env
	uv run pytest tests/
