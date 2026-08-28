# LiveSearchBench developer entry points.
#
#   make install    install runtime + test dependencies
#   make smoke      full offline pipeline check, no API keys, no network (<60s)
#   make test       unit tests
#   make score      score the bundled sample predictions
#   make clean      remove generated artifacts
#
# `make smoke` is the single command that tells you whether a fresh clone works.

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help install test smoke score lint clean

help:
	@grep -E '^#   \S' Makefile | sed 's/^#   /  /'

install:
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests/ -q

smoke:
	@PYTHON=$(PYTHON) ./scripts/smoke_test.sh

score:
	@if ls outputs/evaluations/*/*_results.json >/dev/null 2>&1; then \
	  $(PYTHON) scripts/analysis/score.py outputs/evaluations/*/*_results.json --format table; \
	else \
	  echo "No evaluation runs found under outputs/evaluations/."; \
	  echo "Produce one first, e.g.:"; \
	  echo "  python scripts/eval/DA.py demo.json --dry-run --limit 10"; \
	fi

clean:
	rm -rf .pytest_cache **/__pycache__ outputs/evaluations outputs/questions outputs/extracted_triples
