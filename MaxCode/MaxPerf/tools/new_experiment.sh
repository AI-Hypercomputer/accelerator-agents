#!/usr/bin/env bash
# new_experiment.sh — Scaffold a new MaxPerf experiment directory.
# Usage: new_experiment.sh <slug>
#   slug: kebab-case identifier, no spaces, max 40 chars
#
# Creates:
#   experiments/<YYYYMMDD>-<slug>/experiment.md
#   experiments/<YYYYMMDD>-<slug>/metadata.json
#   Git branch: qwen3coder-maxperf-<YYYYMMDD>-<slug>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  echo "Usage: $0 <slug>"
  echo "  slug: kebab-case (lowercase, hyphens, digits), max 40 chars"
  exit 1
}

if [ $# -ne 1 ]; then
  usage
fi

SLUG="$1"

# Validate slug: kebab-case, no spaces, <= 40 chars
if [ ${#SLUG} -gt 40 ]; then
  echo "ERROR: Slug must be 40 characters or fewer (got ${#SLUG})." >&2
  exit 1
fi

if ! echo "$SLUG" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
  echo "ERROR: Slug must be kebab-case (lowercase letters, digits, hyphens; no leading/trailing hyphens)." >&2
  exit 1
fi

DATESTAMP=$(date +%Y%m%d)
EXPERIMENT_ID="${DATESTAMP}-${SLUG}"
EXP_DIR="$ROOT/experiments/$EXPERIMENT_ID"
BRANCH="qwen3coder-maxperf-${DATESTAMP}-${SLUG}"

if [ -d "$EXP_DIR" ]; then
  echo "ERROR: Experiment directory already exists: experiments/$EXPERIMENT_ID/" >&2
  exit 1
fi

# Create experiment directory
mkdir -p "$EXP_DIR"

# Create experiment.md stub
cat > "$EXP_DIR/experiment.md" << 'EXPERIMENT_EOF'
---
title: ""
type: experiment
tags: [experiment]
hypothesis: ""
model: qwen3coder-480b
created: DATESTAMP_PLACEHOLDER
updated: DATESTAMP_PLACEHOLDER
commit: ""
verdict: ""
agent: ""
---

<!-- TODO: One-paragraph summary of the experiment and its outcome. -->

## Hypothesis

<!-- TODO: State the hypothesis as filed from the queue, including class, origination path, and evidence pointer. -->

## Method

<!-- TODO: Describe the implementation approach — what code was changed, what graph/kernel modification was made. -->

## Profile

<!-- TODO: Diagnostic vector output from TPUDiagnoseAgent. Include roofline analysis, headroom report, DMA idle %, HBM BW utilization %, VREG spill count, collective latency breakdown. -->

## Results

<!-- TODO: TPS/chip, p50/p99 TPOT, HBM peak, compile time, eval scores. -->

## Numeric Equivalence

<!-- TODO: Pass/fail, max deviation, failed prompts if any. Write "N/A" for hypothesis classes that don't require numeric-equiv. -->

## Verdict

<!-- TODO: Accept/reject/invalid per the class-appropriate template from the hypothesis taxonomy. -->

## Next hypotheses

<!-- TODO: Any new hypotheses filed as a result of this experiment's profiling or analysis. -->

## Lessons

<!-- TODO: Generalizable findings, constraint identifications, or knowledge gained. -->
EXPERIMENT_EOF

# Replace DATESTAMP_PLACEHOLDER with actual date
FORMATTED_DATE=$(date +%Y-%m-%d)
if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i '' "s/DATESTAMP_PLACEHOLDER/$FORMATTED_DATE/g" "$EXP_DIR/experiment.md"
else
  sed -i "s/DATESTAMP_PLACEHOLDER/$FORMATTED_DATE/g" "$EXP_DIR/experiment.md"
fi

# Create metadata.json skeleton
cat > "$EXP_DIR/metadata.json" << METADATA_EOF
{
  "experiment_id": "$EXPERIMENT_ID",
  "hypothesis": "",
  "class": "",
  "branch": "$BRANCH",
  "origination": "",
  "agent": "",
  "created": "$FORMATTED_DATE"
}
METADATA_EOF

# Create git branch (if inside a git repo)
if git -C "$ROOT" rev-parse --git-dir > /dev/null 2>&1; then
  git -C "$ROOT" checkout -b "$BRANCH" 2>/dev/null || \
    echo "WARNING: Could not create branch $BRANCH (may already exist or not in a git repo)." >&2
else
  echo "NOTE: Not inside a git repository; skipping branch creation."
fi

echo "Created experiment:"
echo "  experiments/$EXPERIMENT_ID/experiment.md"
echo "  experiments/$EXPERIMENT_ID/metadata.json"
echo "  Branch: $BRANCH"
