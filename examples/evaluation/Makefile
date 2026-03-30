.PHONY: default format lint fix check test test-cov run-all-models-aicamera

default: fix test

# Format code (like black)
format:
	uv run ruff format .

# Lint (no changes)
lint:
	uv run ruff check .

# Auto-fix lint + format (what you run locally)
fix:
	uv run ruff check . --fix
	uv run ruff format .

# CI-style: fail if anything would change
check:
	uv run ruff format . --check
	uv run ruff check .

test:
	uv run pytest -q

# Run tests with coverage report
test-cov:
	uv run pytest --cov=src/evaluator --cov-report=term-missing -q

# Run aicamera evaluation for every zoo model (logs and summary in /tmp/eval_results)
run-all-models-aicamera:
	./run_all_models_aicamera.sh
