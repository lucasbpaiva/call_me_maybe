ifneq ($(wildcard /goinfre/.),)
	export HF_HOME := /goinfre/$(USER)/hf_cache
	export UV_CACHE_DIR := /goinfre/$(USER)/uv_cache
else
	export HF_HOME := $(HOME)/.cache/huggingface/hub
	export UV_CACHE_DIR := $(HOME)/.cache/uv
endif

export HF_HUB_VERBOSITY := error

VENV = .venv
PYTHON = uv run python
FLAKE8 = uv run flake8
MYPY = uv run mypy
EV_INPUT = data/input/eval_cases.json
EV_FUNC = data/input/eval_functions.json
EV_OUTPUT = data/output/eval_results.json

.PHONY: install run debug clean fclean re test lint lint-strict

setup-env:
	@mkdir -p $(HF_HOME)
	@mkdir -p $(UV_CACHE_DIR)
ifneq ($(wildcard /goinfre/.),)
	@mkdir -p /goinfre/$(USER)/call_me_maybe_venv
	@if [ ! -L ".venv" ]; then \
		rm -rf .venv; \
		ln -s /goinfre/$(USER)/call_me_maybe_venv .venv; \
	fi
endif

install: setup-env
	@uv sync

run: setup-env
	@$(PYTHON) -m src

debug: setup-env
	@$(PYTHON) -m pdb -m src

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +

fclean: clean
	@rm -rf $(VENV)
ifneq ($(wildcard /goinfre/.),)
	@rm -rf /goinfre/$(USER)/call_me_maybe_venv
endif

re: fclean install

lint: setup-env
	@$(FLAKE8) src/ tests/
	@$(MYPY) --warn-return-any\
			--warn-unused-ignores\
			--ignore-missing-imports\
			--disallow-untyped-defs\
			--check-untyped-defs\
			src/ tests/

lint-strict: setup-env
	@$(FLAKE8) src/ tests/
	@$(MYPY) --strict src/ tests/

test: setup-env
	@$(PYTHON) -m tests.generate_golden
	@uv run pytest tests/

eval: setup-env
	@echo "Generating outputs..."
	@uv run python -m src --input $(EV_INPUT) --functions_definition $(EV_FUNC) --output $(EV_OUTPUT)
	@echo "\nEvaluating accuracy..."
	@uv run python src/evaluate.py --generated $(EV_OUTPUT) --expected $(EV_INPUT)
