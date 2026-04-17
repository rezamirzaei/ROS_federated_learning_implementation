UV ?= uv

.PHONY: install install-all pre-commit lint fmt typecheck test bench docs

install:
	$(UV) sync --extra ml --extra dev

install-all:
	$(UV) sync --extra ml --extra dev --extra viz --extra qp --extra ros

pre-commit:
	$(UV) run pre-commit install

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy src/fl_robots/fl_robots

test:
	$(UV) run pytest tests/ -v

bench:
	$(UV) run python scripts/benchmark.py --rounds 3

docs:
	$(UV) run mkdocs serve
