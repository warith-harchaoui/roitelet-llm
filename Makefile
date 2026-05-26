# Roitelet LLM — developer entry points.
#
# All targets here are intentionally thin wrappers around tools that
# already work standalone. The Makefile exists so newcomers can find the
# right command without grepping CI configs or reading every README.

.PHONY: help install install-eval install-multimodal test eval lint vendor clean

# Default target prints what's available.
help:
	@echo "Roitelet LLM — make targets"
	@echo ""
	@echo "  install            Editable install of runtime deps"
	@echo "  install-eval       Add the DeepEval-based answer-quality extras"
	@echo "  install-multimodal Add whisper.cpp + NeMo + kreuzberg extras"
	@echo "  test               Run the fast default unit-test suite"
	@echo "  eval               Run answer-quality tests against local Ollama"
	@echo "                     (writes a JSON summary to an ignored working directory)"
	@echo "  lint               Run ruff in check mode"
	@echo "  vendor             Re-download Tailwind + marked into web/vendor/"
	@echo "  clean              Remove .pytest_cache and __pycache__ trees"

install:
	pip install -e .

install-eval:
	pip install -e '.[eval]'

install-multimodal:
	pip install -e '.[multimodal]'

# Fast suite. Same default as `pytest` in pyproject.toml (excludes `-m eval`).
test:
	python -m pytest -q

# Answer-quality eval. Slow, needs Ollama running, network-dependent.
# Output goes to an ignored working directory (see .gitignore) so we
# can diff regressions across commits without polluting tracked state.
# Failure of any case is non-fatal at the make level — the JSON is the
# artefact.
EVAL_DIR := .private/eval_runs
EVAL_FILE := $(EVAL_DIR)/$(shell date -u +%Y%m%dT%H%M%SZ).json
eval:
	@mkdir -p $(EVAL_DIR)
	@echo "Writing eval report to $(EVAL_FILE)"
	-python -m pytest -m eval -q --json-report --json-report-file=$(EVAL_FILE) || \
		python -m pytest -m eval -q | tee $(EVAL_FILE)
	@echo ""
	@echo "Report: $(EVAL_FILE)"

lint:
	python -m ruff check .

vendor:
	bash scripts/vendor_web_assets.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
