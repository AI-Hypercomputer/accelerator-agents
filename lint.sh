#!/bin/bash
set -e

echo "Running ruff check on MaxKernel..."
ruff check MaxKernel/

echo "Running ruff format check on MaxKernel..."
ruff format --check MaxKernel/

echo "Linting passed!"
