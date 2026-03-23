#!/bin/bash
set -e

echo "Applying formatting to MaxKernel..."
ruff format MaxKernel/

echo "Applying lint fixes to MaxKernel..."
ruff check --fix MaxKernel/

echo "Formatting and fixes complete!"
