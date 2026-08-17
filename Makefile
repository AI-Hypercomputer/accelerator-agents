.PHONY: help test test-nexus clean

PYTHON ?= python3
FRAMEWORK_DIR = nexus/framework

help:
	@echo "Available targets:"
	@echo "  make test          Run all framework unit tests"
	@echo "  make test-nexus    Run Nexus framework unit tests"
	@echo "  make clean         Remove cache files and build artifacts"

test: test-nexus

test-nexus:
	PYTHONPATH=$(FRAMEWORK_DIR) $(PYTHON) -m unittest discover -s $(FRAMEWORK_DIR)/tests -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
