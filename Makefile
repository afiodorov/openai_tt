.PHONY: lint format check install dev clean sort-imports

# Python executable (uv run)
PYTHON := uv run

# Install dependencies
install:
	uv pip install -e .

# Install development dependencies
dev:
	uv pip install -e ".[dev]"

# Run linting checks
check:
	ruff check .

# Run linting fixes
lint:
	ruff check --fix .

# Run formatter
format:
	ruff format .

# Sort imports
sort-imports:
	ruff check --select I --fix .

# Run both linting and formatting
lint-all: lint format

# Clean temporary files and caches
clean:
	rm -rf __pycache__
	rm -rf .ruff_cache
	rm -rf .pytest_cache