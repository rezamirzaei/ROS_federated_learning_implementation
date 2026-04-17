UV ?= uv

.PHONY: install install-all pre-commit lint fmt typecheck pyright test coverage security bench docs docker-build ci

install:
	$(UV) sync --extra ml --extra dev

install-all:
	$(UV) sync --extra ml --extra dev --extra viz --extra qp --extra ros --extra otel

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

pyright:
	$(UV) run pyright

test:
	$(UV) run pytest tests/ -v

coverage:
	$(UV) run pytest tests/ -v --cov=fl_robots --cov-report=term-missing --cov-report=html

security:
	$(UV) run bandit -q -c bandit.yaml -r src/fl_robots/fl_robots

bench:
	$(UV) run python scripts/benchmark.py --rounds 3

docs:
	$(UV) run mkdocs serve

docker-build:
	docker build -f docker/Dockerfile --target standalone-runtime -t fl-robots-standalone:latest .
	docker build -f docker/Dockerfile --target standalone-test -t fl-robots-test:latest .

ci: lint typecheck pyright coverage security
	@echo "✅ All CI checks passed"
