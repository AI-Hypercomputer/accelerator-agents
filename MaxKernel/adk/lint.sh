#!/bin/bash
set -e

echo "Running ruff check on MaxKernel..."
ruff check .

echo "Running ruff format check on MaxKernel..."
ruff format --check .

echo "Linting passed!"
