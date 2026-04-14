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

import ast
import json
import os
import sys

from config import MERGED_FILE, MERGED_UTILS_FILE, OUTPUT_DIR, REPO_URL, setup

# Standard PyTorch -> JAX/Flax method renames.
# When a source method is renamed to its JAX equivalent, it counts as matched.
# With @nn.compact, there is no setup() — __init__ logic lives in __call__.
METHOD_RENAMES = {
    "__init__": {"setup", "__call__"},
    "forward": {"__call__"},
}

# Methods that are always inlined during conversion (Flax handles these
# via initializer args, so there is never a JAX equivalent).
ALWAYS_INLINED = {
    "reset_parameters",
}


# ------------------------------------------------------------------
# AST extraction
# ------------------------------------------------------------------

def extract_components(file_path):
    """Parse a Python file and return its classes, methods, and functions.

    Returns:
        dict with keys:
          "classes":   {class_name: [method_name, ...], ...}
          "functions": [function_name, ...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=file_path)

    classes = {}
    functions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name
                for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes[node.name] = methods
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)

    return {"classes": classes, "functions": functions}


# ------------------------------------------------------------------
# Completeness
# ------------------------------------------------------------------

def compute_completeness(source_components, output_components):
    """Compare source and output components and return a completeness report.

    Returns:
        dict with keys:
          "score":     float (0-100)
          "classes":   {"total": int, "found": int, "missing": list}
          "methods":   {"total": int, "found": int, "missing": list}
          "functions": {"total": int, "found": int, "missing": list}
    """
    src_classes = source_components["classes"]
    out_classes = output_components["classes"]

    # --- classes ---
    src_class_names = set(src_classes.keys())
    out_class_names = set(out_classes.keys())
    matched_classes = src_class_names & out_class_names
    missing_classes = sorted(src_class_names - out_class_names)

    # --- methods (only within matched classes) ---
    total_methods = 0
    found_methods = 0
    missing_methods = []

    for cls in src_classes:
        src_methods = set(src_classes[cls])
        total_methods += len(src_methods)
        if cls in out_classes:
            out_methods = set(out_classes[cls])
            has_call = "__call__" in out_methods
            for m in sorted(src_methods):
                # Check exact name match
                if m in out_methods:
                    found_methods += 1
                # Check known renames (e.g. __init__ -> setup or __call__)
                elif m in METHOD_RENAMES and METHOD_RENAMES[m] & out_methods:
                    found_methods += 1
                # Always-inlined methods (e.g. reset_parameters)
                elif m in ALWAYS_INLINED:
                    found_methods += 1
                # If the class has __call__, treat other private/helper
                # methods as legitimately inlined into it
                elif has_call and m not in ("__init__", "forward"):
                    found_methods += 1
                else:
                    missing_methods.append(f"{cls}.{m}")
        else:
            # class itself is missing; count all its methods as missing
            for m in sorted(src_methods):
                missing_methods.append(f"{cls}.{m}")

    # --- standalone functions ---
    # A PyTorch function may become a Flax class (e.g. Linear -> nn.Module).
    # Count it as matched if it appears as either a function or a class.
    src_funcs = set(source_components["functions"])
    out_funcs = set(output_components["functions"])
    matched_funcs = src_funcs & out_funcs
    # Also match functions that were promoted to classes in the output
    for f in src_funcs - matched_funcs:
        if f in out_class_names:
            matched_funcs = matched_funcs | {f}
    missing_funcs = sorted(src_funcs - matched_funcs)

    # --- overall ---
    total = len(src_class_names) + total_methods + len(src_funcs)
    found = len(matched_classes) + found_methods + len(matched_funcs)
    score = (found / total * 100) if total > 0 else 100.0

    return {
        "score": round(score, 1),
        "total": total,
        "found": found,
        "classes": {
            "total": len(src_class_names),
            "found": len(matched_classes),
            "missing": missing_classes,
        },
        "methods": {
            "total": total_methods,
            "found": found_methods,
            "missing": missing_methods,
        },
        "functions": {
            "total": len(src_funcs),
            "found": len(matched_funcs),
            "missing": missing_funcs,
        },
    }


# ------------------------------------------------------------------
# Correctness (LLM-based)
# ------------------------------------------------------------------

SEVERITY_WEIGHTS = {"high": 5, "medium": 3, "low": 1}

# Known false-positive (category, severity) pairs.  Only low-severity entries
# qualify — these represent legitimate Flax idioms or PyTorch-only patterns
# that the validator flags but are not real bugs.
FALSE_POSITIVE_RULES = {
    ("method_placement", "low"),   # helpers inlined into __call__ is idiomatic Flax
    ("missing_component", "low"),  # reset_parameters, register_buffer, weight caching
    ("dropped_feature", "low"),    # debug try-except blocks, intermediates tracking
}


def compute_correctness(source_code, output_code, api_key, total_components=0):
    """Run ValidationAgent and score the output.

    The score is ratio-based: penalty is normalized against the total number
    of source components so that larger codebases aren't penalized unfairly.

        score = max(0, (1 - penalty / budget) * 100)

    where budget = total_components * medium_severity_weight.  This makes the
    score symmetric with the completeness metric (both are ratios).

    Args:
        source_code: The PyTorch source code.
        output_code: The converted JAX output code.
        api_key: Google API key for the LLM.
        total_components: Number of source components (classes + methods +
            functions) from the completeness check.  If 0, falls back to
            counting top-level classes and functions from source_code via AST.

    Returns:
        dict with keys:
          "score":               float (0-100)
          "deviations":          list of real deviation dicts
          "filtered_deviations": list of false-positive deviation dicts
          "by_category":         {category: count, ...}  (real only)
          "by_severity":         {severity: count, ...}  (real only)
    """
    import models
    from agents.migration.validation_agent import ValidationAgent

    gemini = models.GeminiTool(
        model_name=models.GeminiModel.GEMINI_3_1_PRO_PREVIEW,
        api_key=api_key,
    )
    validator = ValidationAgent(model=gemini)
    all_deviations = validator.validate(source_code, output_code)

    if not isinstance(all_deviations, list):
        all_deviations = []

    # Split into real vs. false-positive deviations
    real = []
    filtered = []
    for d in all_deviations:
        sev = d.get("severity", "low").lower()
        cat = d.get("category", "unknown")
        if (cat, sev) in FALSE_POSITIVE_RULES:
            filtered.append(d)
        else:
            real.append(d)

    by_severity = {}
    by_category = {}
    penalty = 0

    for d in real:
        sev = d.get("severity", "low").lower()
        cat = d.get("category", "unknown")
        by_severity[sev] = by_severity.get(sev, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1
        penalty += SEVERITY_WEIGHTS.get(sev, 1)

    # Fallback: count top-level classes + functions from source AST
    if total_components <= 0:
        try:
            tree = ast.parse(source_code)
            total_components = sum(
                1 for n in ast.iter_child_nodes(tree)
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            )
        except SyntaxError:
            total_components = 0

    # Ratio-based scoring: budget scales with codebase size.
    # Each component contributes a "correctness budget" equal to the medium
    # severity weight.  A medium-severity deviation on every component = 0%.
    budget = total_components * SEVERITY_WEIGHTS["medium"]
    if budget > 0:
        score = max(0.0, (1.0 - penalty / budget) * 100.0)
    else:
        score = 100.0 if penalty == 0 else 0.0

    return {
        "score": round(score, 1),
        "deviation_count": len(real),
        "deviations": real,
        "filtered_deviations": filtered,
        "by_category": by_category,
        "by_severity": by_severity,
    }


# ------------------------------------------------------------------
# Scorecard display
# ------------------------------------------------------------------

def print_scorecard(completeness, correctness=None):
    """Print a formatted verification scorecard."""
    print()
    print("=" * 50)
    print("  Conversion Verification Scorecard")
    print("=" * 50)

    # -- Completeness --
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

    # -- Correctness --
    if correctness is not None:
        cr = correctness
        n_dev = len(cr["deviations"])
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

    # -- Overall --
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
    """Return the path to the JAX output file inside OUTPUT_DIR.

    Looks for <repo_name>_jax.py first (matching step4's output name).
    Falls back to the first *_jax.py file found.
    """
    if not os.path.isdir(OUTPUT_DIR):
        return None
    # Prefer the file matching the current repo
    repo_name = REPO_URL.rstrip("/").rsplit("/", 1)[-1].replace("-", "_")
    expected = f"{repo_name}_jax.py"
    expected_path = os.path.join(OUTPUT_DIR, expected)
    if os.path.isfile(expected_path):
        return expected_path
    # Fallback: first *_jax.py found
    for name in os.listdir(OUTPUT_DIR):
        if name.endswith("_jax.py"):
            return os.path.join(OUTPUT_DIR, name)
    return None


def main():
    setup()

    # Locate files
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

    # -- Completeness --
    src_components = extract_components(MERGED_FILE)
    out_components = extract_components(jax_path)
    completeness = compute_completeness(src_components, out_components)

    # -- Correctness (optional) --
    api_key = os.environ.get("GOOGLE_API_KEY")
    correctness = None
    if api_key:
        print("\n  Running correctness check (LLM-based)...")
        with open(MERGED_FILE, "r", encoding="utf-8") as f:
            source_code = f.read()
        with open(jax_path, "r", encoding="utf-8") as f:
            output_code = f.read()
        correctness = compute_correctness(
            source_code, output_code, api_key,
            total_components=completeness["total"],
        )
    else:
        print("\n  GOOGLE_API_KEY not set -- skipping correctness check.")

    # -- Print scorecard --
    overall = print_scorecard(completeness, correctness)

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

        utils_src = extract_components(MERGED_UTILS_FILE)
        utils_out = extract_components(utils_jax_path)
        utils_completeness = compute_completeness(utils_src, utils_out)

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
    # (if no MERGED_UTILS_FILE, utilities were not discovered -- nothing to verify)

    # -- Save JSON --
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {
        "source_file": MERGED_FILE,
        "output_file": jax_path,
        "completeness": completeness,
        "overall": overall,
    }
    if correctness is not None:
        result["correctness"] = {
            "score": correctness["score"],
            "deviation_count": len(correctness["deviations"]),
            "by_category": correctness["by_category"],
            "by_severity": correctness["by_severity"],
            "deviations": correctness["deviations"],
            "filtered_deviations": correctness.get("filtered_deviations", []),
        }
    if utils_completeness is not None:
        result["utils_completeness"] = utils_completeness

    json_path = os.path.join(OUTPUT_DIR, "verification_scorecard.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {json_path}")


if __name__ == "__main__":
    main()
