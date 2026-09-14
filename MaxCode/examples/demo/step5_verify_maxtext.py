"""
Step 5 (MaxText): Verify the quality of a PyTorch-to-MaxText conversion.

This script is the MaxText counterpart of step5_verify.py.  It locates the
most recent timestamped output directory that contains MaxText artifacts and
produces a scorecard comparing the PyTorch source against the generated
MaxText layers file.

Metrics are the same as step5_verify.py:

  Completeness (AST-based, no LLM)
    Parses both files and compares classes, methods, and standalone
    functions by name.  Score = matched / total source components.

  Correctness (LLM-based, requires GOOGLE_API_KEY)
    Runs the ValidationAgent to detect deviations between the PyTorch
    source and MaxText output.  Score = 100 minus weighted penalties
    (high=5, medium=3, low=1 per deviation).

Requires:
  - Step 3 completed (merged model file created)
  - Step 4 MaxText completed (timestamped MaxText output directory)
  - Optionally GOOGLE_API_KEY for the correctness check

Usage:
    python step5_verify_maxtext.py
"""

import glob
import json
import os
import re
import sys

from config import MERGED_FILE, OUTPUT_DIR, setup

# Add MaxCode to sys.path so agent imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.migration.verification_agent import VerificationAgent
from step5_verify import print_scorecard


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def _find_latest_maxtext_dir():
    """Return (timestamped_dir, layers_path) or (timestamped_dir, None).

    Scans OUTPUT_DIR for timestamped subdirectories (YYYYMMDD_HHMMSS),
    sorted most-recent first, and returns the first one that contains
    MaxText artifacts.

    Returns:
        (str, str)  — directory path and layers .py path, or
        (str, None) — directory has MaxText/configs but no layers file
                       (known block with built-in implementation), or
        (None, None) — no MaxText output found at all.
    """
    if not os.path.isdir(OUTPUT_DIR):
        return None, None

    candidates = sorted(
        [
            d
            for d in os.listdir(OUTPUT_DIR)
            if _TIMESTAMP_RE.match(d)
            and os.path.isdir(os.path.join(OUTPUT_DIR, d))
        ],
        reverse=True,
    )

    for dirname in candidates:
        ts_dir = os.path.join(OUTPUT_DIR, dirname)
        layers_dir = os.path.join(ts_dir, "MaxText", "layers")
        configs_dir = os.path.join(ts_dir, "MaxText", "configs")

        # Check for layers .py files first
        if os.path.isdir(layers_dir):
            py_files = glob.glob(os.path.join(layers_dir, "*.py"))
            if py_files:
                # Return the first (usually only) layers file
                return ts_dir, py_files[0]

        # If configs exist but no layers, this is a known-block run
        if os.path.isdir(configs_dir):
            return ts_dir, None

    return None, None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    setup()

    if not os.path.isfile(MERGED_FILE):
        print("ERROR: Merged model file not found. Run step3_merge.py first.")
        sys.exit(1)

    ts_dir, layers_path = _find_latest_maxtext_dir()

    if ts_dir is None:
        print("ERROR: No MaxText output found in output/. "
              "Run step4_convert_maxtext.py first.")
        sys.exit(1)

    if layers_path is None:
        print("=" * 50)
        print("  Step 5: Verify MaxText Conversion")
        print("=" * 50)
        print(f"\n  Output dir: {ts_dir}")
        print()
        print("  This run produced a YAML config overlay only (known block")
        print("  with built-in MaxText implementation). No layers file was")
        print("  emitted, so verification is not applicable.")
        print()
        print("=" * 50)
        sys.exit(0)

    print("=" * 50)
    print("  Step 5: Verify MaxText Conversion Quality")
    print("=" * 50)
    print(f"  Source: {MERGED_FILE}")
    print(f"  Output: {layers_path}")

    # Read source and output
    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        source_code = f.read()
    with open(layers_path, "r", encoding="utf-8") as f:
        output_code = f.read()

    # Run verification
    api_key = os.environ.get("GOOGLE_API_KEY")
    verifier = VerificationAgent(target="maxtext")

    if api_key:
        print("\n  Running verification (completeness + correctness)...")
    else:
        print("\n  GOOGLE_API_KEY not set -- running completeness check only.")

    result = verifier.verify(source_code, output_code, api_key=api_key)
    overall = print_scorecard(result.completeness, result.correctness)

    # -- Save JSON --
    json_result = {
        "source_file": MERGED_FILE,
        "output_file": layers_path,
        "completeness": result.completeness,
        "overall": overall,
    }
    if result.correctness is not None:
        json_result["correctness"] = {
            "score": result.correctness["score"],
            "deviation_count": result.correctness["deviation_count"],
            "by_category": result.correctness["by_category"],
            "by_severity": result.correctness["by_severity"],
            "deviations": result.correctness["deviations"],
            "filtered_deviations": result.correctness.get("filtered_deviations", []),
        }

    json_path = os.path.join(ts_dir, "verification_maxtext_scorecard.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2)
    print(f"  Results saved to {json_path}")


if __name__ == "__main__":
    main()
