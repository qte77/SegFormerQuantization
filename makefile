setup_env:
	@echo "Setting up tools..."
	@pip install uv
	@uv sync --frozen

run_app:
	@uv run python -m src

test_all:
	@uv run pytest

ruff:
	@uv run ruff format
	@uv run ruff check --fix