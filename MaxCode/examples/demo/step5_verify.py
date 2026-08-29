"""
Step 5: Verify the quality of a PyTorch-to-JAX conversion.

This script produces a scorecard with two metrics:

  Completeness (AST-based, no LLM)
    Parses both files and compares classes, methods, and standalone
    functions by name.  Score = matched / total source components.

  Correctness (LLM-based, requires GOOGLE_API_KEY)
    Runs the ValidationAgent to detect deviations between the PyTorch
    source and JAX output.  Score = 100 minus weighted penalties
    (high=5, medium=3, low=1 per deviation).

Requires:
  - Step 3 completed (merged model file created)
  - Step 4 completed (JAX output file created)
  - Optionally GOOGLE_API_KEY for the correctness check

Usage:
    python step5_verify.py
"""

import json
import os
import sys

from config import MERGED_FILE, MERGED_UTILS_FILE, OUTPUT_DIR, REPO_URL, setup

# Add MaxCode to sys.path so agent imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.migration.verification_agent import VerificationAgent


# ------------------------------------------------------------------
# Scorecard display
# ------------------------------------------------------------------

def print_scorecard(completeness, correctness=None):
    """Print a formatted verification scorecard."""
    print()
    print("=" * 50)
    print("  Conversion Verification Scorecard")
    print("=" * 50)

    c = completeness
    print()
    print(f"  Completeness: {c['score']:.1f}%  "
          f"({c['found']}/{c['total']} components)")
    print(f"    Classes:    {c['classes']['found']}/{c['classes']['total']}", end="")
    if c["classes"]["missing"]:
        print(f"  (missing: {', '.join(c['classes']['missing'])})", end="")
    print()

    print(f"    Methods:    {c['methods']['found']}/{c['methods']['total']}", end="")
    if c["methods"]["missing"]:
        shown = c["methods"]["missing"][:5]
        extra = len(c["methods"]["missing"]) - len(shown)
        print(f"  (missing: {', '.join(shown)}", end="")
        if extra > 0:
            print(f" +{extra} more", end="")
        print(")", end="")
    print()

    print(f"    Functions:  {c['functions']['found']}/{c['functions']['total']}", end="")
    if c["functions"]["missing"]:
        print(f"  (missing: {', '.join(c['functions']['missing'])})", end="")
    print()

    if c.get("delegated"):
        d = c["delegated"]
        print(f"    Delegated:  {d['count']} components handled by MaxText built-ins")

    if correctness is not None:
        cr = correctness
        n_dev = cr["deviation_count"]
        n_filt = len(cr.get("filtered_deviations", []))
        print()
        print(f"  Correctness:  {cr['score']:.1f}%  "
              f"({n_dev} deviation{'s' if n_dev != 1 else ''} found"
              f"{f', {n_filt} filtered' if n_filt else ''})")
        for sev in ("high", "medium", "low"):
            count = cr["by_severity"].get(sev, 0)
            if count:
                cats = [
                    d.get("category", "unknown")
                    for d in cr["deviations"]
                    if d.get("severity", "").lower() == sev
                ]
                cat_str = ", ".join(sorted(set(cats)))
                print(f"    {sev:8s} {count}  ({cat_str})")
    else:
        print()
        print("  Correctness:  skipped (GOOGLE_API_KEY not set)")

    if correctness is not None:
        overall = round((completeness["score"] + correctness["score"]) / 2, 1)
    else:
        overall = completeness["score"]
    print()
    print(f"  Overall: {overall:.1f}%")
    print()
    print("=" * 50)

    return overall


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _find_jax_output():
    """Return the path to the JAX output file inside OUTPUT_DIR."""
    if not os.path.isdir(OUTPUT_DIR):
        return None
    repo_name = REPO_URL.rstrip("/").rsplit("/", 1)[-1].replace("-", "_")
    expected = f"{repo_name}_jax.py"
    expected_path = os.path.join(OUTPUT_DIR, expected)
    if os.path.isfile(expected_path):
        return expected_path
    for name in os.listdir(OUTPUT_DIR):
        if name.endswith("_jax.py"):
            return os.path.join(OUTPUT_DIR, name)
    return None


def main():
    setup()

    if not os.path.isfile(MERGED_FILE):
        print("ERROR: Merged model file not found. Run step3_merge.py first.")
        sys.exit(1)

    jax_path = _find_jax_output()
    if jax_path is None:
        print("ERROR: No JAX output file found in output/. Run step4_convert.py first.")
        sys.exit(1)

    print("=" * 50)
    print("  Step 5: Verify Conversion Quality")
    print("=" * 50)
    print(f"  Source: {MERGED_FILE}")
    print(f"  Output: {jax_path}")

    # Read source and output
    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        source_code = f.read()
    with open(jax_path, "r", encoding="utf-8") as f:
        output_code = f.read()

    # Run verification
    api_key = os.environ.get("GOOGLE_API_KEY")
    verifier = VerificationAgent()

    if api_key:
        print("\n  Running verification (completeness + correctness)...")
    else:
        print("\n  GOOGLE_API_KEY not set -- running completeness check only.")

    result = verifier.verify(source_code, output_code, api_key=api_key)
    overall = print_scorecard(result.completeness, result.correctness)

    # -- Utility file verification --
    utils_completeness = None
    repo_name = REPO_URL.rstrip("/").rsplit("/", 1)[-1].replace("-", "_")
    utils_jax_path = os.path.join(OUTPUT_DIR, f"{repo_name}_utils_jax.py")

    if os.path.isfile(MERGED_UTILS_FILE) and os.path.isfile(utils_jax_path):
        print()
        print("-" * 50)
        print("  Utility File Verification")
        print("-" * 50)
        print(f"  Source: {MERGED_UTILS_FILE}")
        print(f"  Output: {utils_jax_path}")

        with open(MERGED_UTILS_FILE, "r", encoding="utf-8") as f:
            utils_source = f.read()
        with open(utils_jax_path, "r", encoding="utf-8") as f:
            utils_output = f.read()

        utils_result = verifier.verify(utils_source, utils_output)
        utils_completeness = utils_result.completeness

        u = utils_completeness
        print(f"\n  Utility Completeness: {u['score']:.1f}%  "
              f"({u['found']}/{u['total']} components)")
        print(f"    Classes:    {u['classes']['found']}/{u['classes']['total']}", end="")
        if u["classes"]["missing"]:
            print(f"  (missing: {', '.join(u['classes']['missing'])})", end="")
        print()
        print(f"    Functions:  {u['functions']['found']}/{u['functions']['total']}", end="")
        if u["functions"]["missing"]:
            shown = u["functions"]["missing"][:5]
            extra = len(u["functions"]["missing"]) - len(shown)
            print(f"  (missing: {', '.join(shown)}", end="")
            if extra > 0:
                print(f" +{extra} more", end="")
            print(")", end="")
        print()
    elif os.path.isfile(MERGED_UTILS_FILE):
        print("\n  Utility JAX output not found -- skipping utility verification.")

    # -- Save JSON --
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_result = {
        "source_file": MERGED_FILE,
        "output_file": jax_path,
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
    if utils_completeness is not None:
        json_result["utils_completeness"] = utils_completeness

    json_path = os.path.join(OUTPUT_DIR, "verification_scorecard.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2)
    print(f"  Results saved to {json_path}")


if __name__ == "__main__":
    main()
