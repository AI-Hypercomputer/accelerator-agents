# pylint: skip-file
#!/usr/bin/env python3
"""append_result.py — Append a validated row to RESULTS.tsv.

Usage:
  python append_result.py --date 20260510 --class kernel-novel --branch qwen3coder-maxperf-20260510-foo \\
      --experiment_slug 20260510-foo --agent MaxKernel --verdict accepted \\
      --tps_per_chip 123.4 ...

  python append_result.py --tsv "20260510\\tkernel-novel\\t..."

All 18 columns are required (use empty string for absent values).
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_PATH = os.path.join(ROOT, "RESULTS.tsv")

COLUMNS = [
    "date",
    "class",
    "branch",
    "experiment_slug",
    "agent",
    "verdict",
    "tps_per_chip",
    "p50_tpot_ms",
    "p99_tpot_ms",
    "hbm_peak_gib",
    "diagnostic_vector_delta",
    "numeric_equiv_pass",
    "vreg_spill_delta",
    "compile_time_s",
    "eval_humaneval",
    "eval_mbpp",
    "profile_gcs_path",
    "notes",
]

NUMERIC_COLUMNS = {
    "tps_per_chip",
    "p50_tpot_ms",
    "p99_tpot_ms",
    "hbm_peak_gib",
    "vreg_spill_delta",
    "compile_time_s",
    "eval_humaneval",
    "eval_mbpp",
}


def parse_args():
  parser = argparse.ArgumentParser(
      description="Append a validated row to RESULTS.tsv.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=f"Columns ({len(COLUMNS)}): {', '.join(COLUMNS)}",
  )
  parser.add_argument(
      "--tsv",
      metavar="LINE",
      help="A single TSV-formatted line (18 tab-separated fields).",
  )
  for col in COLUMNS:
    parser.add_argument(f"--{col}", default=None, help=f"Value for {col}.")
  return parser.parse_args()


def validate_row(values):
  """Validate a list of 18 string values. Returns list of error messages."""
  errors = []
  if len(values) != len(COLUMNS):
    errors.append(f"Expected {len(COLUMNS)} columns, got {len(values)}.")
    return errors

  row = dict(zip(COLUMNS, values))

  # Numeric columns must parse as numbers (if non-empty)
  for col in NUMERIC_COLUMNS:
    val = row.get(col, "")
    if val and val.strip():
      try:
        float(val)
      except ValueError:
        errors.append(f"Column '{col}' is not a valid number: {val!r}")

  # experiment_slug must match an existing experiments/ directory
  slug = row.get("experiment_slug", "")
  if slug:
    exp_dir = os.path.join(ROOT, "experiments", slug)
    if not os.path.isdir(exp_dir):
      errors.append(
          f"experiment_slug '{slug}' has no matching directory: "
          f"experiments/{slug}/"
      )

  return errors


def main():
  args = parse_args()

  if args.tsv:
    values = args.tsv.split("\t")
  else:
    values = []
    for col in COLUMNS:
      val = getattr(args, col, None)
      if val is None:
        val = ""
      values.append(val)

  # Check that at least some values were provided
  if all(v == "" for v in values):
    print("ERROR: No values provided. Use --help for usage.", file=sys.stderr)
    sys.exit(1)

  errors = validate_row(values)
  if errors:
    for e in errors:
      print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

  # Verify RESULTS.tsv header
  if not os.path.isfile(RESULTS_PATH):
    print(f"ERROR: {RESULTS_PATH} not found.", file=sys.stderr)
    sys.exit(1)

  with open(RESULTS_PATH, "r") as f:
    header_line = f.readline().rstrip("\n")

  expected_header = "\t".join(COLUMNS)
  if header_line != expected_header:
    print(
        "ERROR: RESULTS.tsv header does not match expected columns.",
        file=sys.stderr,
    )
    sys.exit(1)

  # Append the row
  row_line = "\t".join(values)
  with open(RESULTS_PATH, "a") as f:
    f.write(row_line + "\n")

  slug = values[COLUMNS.index("experiment_slug")]
  print(f"Appended row for experiment '{slug}' to RESULTS.tsv.")


if __name__ == "__main__":
  main()
