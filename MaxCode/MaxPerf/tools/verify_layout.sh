#!/usr/bin/env bash
# verify_layout.sh — Validate MaxPerf directory structure.
# Exits 0 if valid, non-zero with diagnostic messages if not.

set -euo pipefail

# Resolve the max_perf_g root (parent of tools/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

errors=0

err() {
  echo "ERROR: $1" >&2
  errors=$((errors + 1))
}

# --- Required top-level directories ---
for dir in agents experiments raw wiki runs; do
  if [ ! -d "$ROOT/$dir" ]; then
    err "Required directory missing: $dir/"
  fi
done

# --- Required top-level files ---
for f in program.md RESULTS.tsv; do
  if [ ! -f "$ROOT/$f" ]; then
    err "Required file missing: $f"
  fi
done

# --- All 7 agent prompts ---
AGENT_PROMPTS=(
  maxperf_orchestrator.md
  tpu_diagnose.md
  deep_research.md
  max_shard.md
  max_kernel.md
  max_tile.md
  auto_refactor.md
)
for prompt in "${AGENT_PROMPTS[@]}"; do
  if [ ! -f "$ROOT/agents/$prompt" ]; then
    err "Agent prompt missing: agents/$prompt"
  fi
done

# --- RESULTS.tsv header check (18 columns) ---
if [ -f "$ROOT/RESULTS.tsv" ]; then
  EXPECTED_HEADER=$(printf 'date\tclass\tbranch\texperiment_slug\tagent\tverdict\ttps_per_chip\tp50_tpot_ms\tp99_tpot_ms\thbm_peak_gib\tdiagnostic_vector_delta\tnumeric_equiv_pass\tvreg_spill_delta\tcompile_time_s\teval_humaneval\teval_mbpp\tprofile_gcs_path\tnotes')
  ACTUAL_HEADER=$(head -1 "$ROOT/RESULTS.tsv")
  if [ "$ACTUAL_HEADER" != "$EXPECTED_HEADER" ]; then
    err "RESULTS.tsv header does not match 18-column spec"
  fi
fi

# --- Wiki subdirectories ---
WIKI_SUBDIRS=(sources codebases concepts observations analyses hypotheses)
for sub in "${WIKI_SUBDIRS[@]}"; do
  if [ ! -d "$ROOT/wiki/$sub" ]; then
    err "Wiki subdirectory missing: wiki/$sub/"
  fi
done

# --- Raw subdirectories ---
RAW_SUBDIRS=(profiles hlo xla_logs isa numeric_ref code sources)
for sub in "${RAW_SUBDIRS[@]}"; do
  if [ ! -d "$ROOT/raw/$sub" ]; then
    err "Raw subdirectory missing: raw/$sub/"
  fi
done

# --- Every experiment directory contains experiment.md ---
if [ -d "$ROOT/experiments" ]; then
  for exp_dir in "$ROOT"/experiments/*/; do
    # Skip if no subdirectories exist (glob didn't expand)
    [ -d "$exp_dir" ] || continue
    if [ ! -f "$exp_dir/experiment.md" ]; then
      dirname=$(basename "$exp_dir")
      err "Experiment directory missing experiment.md: experiments/$dirname/"
    fi
  done
fi

# --- No files over 200KB accidentally staged ---
while IFS= read -r -d '' large_file; do
  rel_path="${large_file#$ROOT/}"
  # Skip known-large directories (raw/isa, raw/profiles, raw/hlo, raw/xla_logs)
  case "$rel_path" in
    raw/isa/*|raw/profiles/*|raw/hlo/*|raw/xla_logs/*|raw/numeric_ref/*|raw/sources/*|raw/code/*|.git/*|.obsidian/*) continue ;;
  esac
  err "File over 200KB: $rel_path ($(du -h "$large_file" | cut -f1))"
done < <(find "$ROOT" -type f -size +200k -print0 2>/dev/null)

# --- Summary ---
if [ $errors -gt 0 ]; then
  echo ""
  echo "Layout validation FAILED with $errors error(s)." >&2
  exit 1
else
  echo "Layout validation passed."
  exit 0
fi
