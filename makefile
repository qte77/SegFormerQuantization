.PHONY: all setup_dev run_app test_all type_check ruff help

# Default target
all: setup_dev

setup_dev:
	@echo "Setting up tools..."
	@pip install uv
	@uv sync --frozen

run_app:
	@uv run python -m src

test_all: ## Run all tests
	@uv run pytest

type_check: ## Check for static typing errors
	@uv run mypy src

ruff: ## Lint: Format and check with ruff
	@uv run ruff format
	@uv run ruff check --fix

help:
	@echo "Usage: make [recipe]"
	@echo "Recipes:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)
