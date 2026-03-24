#!/bin/bash
set -e

echo "Applying formatting to MaxKernel..."
ruff format .

echo "Applying lint fixes to MaxKernel..."
ruff check --fix .

echo "Formatting and fixes complete!"
