"""
Step 3: Auto-detect model files and merge them into a single file.

This script scans the cloned repository to find all Python files that
define PyTorch nn.Module subclasses (the model code). It then merges
them into a single file in dependency order, so MaxCode can convert
the entire model with full context in one pass.

Non-model files (datasets, training scripts, utilities, etc.) are
automatically excluded. Relative imports between model files are
removed since all code is combined into one file.

Requires:
  - Step 1 completed (repo cloned)

Usage:
    python step3_merge.py
"""

import os
import sys

from config import (
    REPO_DIR, MERGED_FILE, MERGED_UTILS_FILE,
    MERGE_EXCLUDE_PATHS, MERGE_EXCLUDE_CLASSES, MERGE_EXCLUDE_UTILS,
    MAXCODE_DIR,
)

# Add MaxCode to sys.path so agent imports work
sys.path.insert(0, MAXCODE_DIR)

from agents.migration.merge_agent import MergeAgent, _count_module_classes


def main():
    if not os.path.isdir(REPO_DIR):
        print("ERROR: Repository not found. Run step1_clone_repo.py first.")
        raise SystemExit(1)

    print("=" * 70)
    print("Step 3: Auto-Detect and Merge Model Files")
    print("=" * 70)
    print(f"  Scanning: {REPO_DIR}")
    print()

    # Count total Python files for context
    all_py = []
    for root, _, files in os.walk(REPO_DIR):
        for f in sorted(files):
            if f.endswith(".py"):
                all_py.append(os.path.join(root, f))
    print(f"  Found {len(all_py)} Python files total")
    print()

    # Run the merge agent
    merger = MergeAgent()
    result = merger.run(
        REPO_DIR,
        exclude_paths=MERGE_EXCLUDE_PATHS,
        exclude_classes=MERGE_EXCLUDE_CLASSES,
        exclude_utils=MERGE_EXCLUDE_UTILS,
    )

    # --- Report excluded files ---
    if result.excluded_files:
        print("  Filtering results:")
        for full_path, reason in result.excluded_files:
            rel = os.path.relpath(full_path, REPO_DIR)
            print(f"    SKIP  {rel:<45s} ({reason})")
        print()

    # --- Report model files ---
    print(f"  Including {len(result.model_files)} model file(s) in merge:")
    total_lines = 0
    for f in result.model_files:
        rel = os.path.relpath(f, REPO_DIR)
        lines = sum(1 for _ in open(f, encoding="utf-8-sig"))
        total_lines += lines
        print(f"    {rel} ({lines} lines)")

    # --- Report excluded classes ---
    if result.excluded_classes:
        print(f"\n  Filtered {len(result.excluded_classes)} infrastructure class(es):")
        for cls_name, reason in result.excluded_classes:
            print(f"    SKIP  {cls_name:<40s} ({reason})")

    # --- Write merged model file ---
    print(f"\n  Writing merged model file: {MERGED_FILE}")
    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        f.write(result.model_code)

    merged_lines = result.model_code.count("\n") + 1
    final_modules = _count_module_classes(result.model_code)
    if final_modules >= 0:
        print(f"  Final merged file: {merged_lines} lines, "
              f"{final_modules} nn.Module classes")
    else:
        print(f"  Final merged file: {merged_lines} lines "
              "(nn.Module count unavailable -- syntax error in merged code)")

    # --- Utility files ---
    print()
    print("=" * 70)
    print("Step 3b: Discover and Merge Utility Files")
    print("=" * 70)

    if result.utility_files:
        print(f"\n  Keeping {len(result.utility_files)} utility file(s):")
        for full_path in result.utility_files:
            rel = os.path.relpath(full_path, REPO_DIR)
            cat = result.utility_categories.get(full_path, "unknown")
            print(f"    KEEP  {rel:<45s} [{cat}]")

        print(f"\n  Writing merged utility file: {MERGED_UTILS_FILE}")
        with open(MERGED_UTILS_FILE, "w", encoding="utf-8") as f:
            f.write(result.utility_code)

        utils_lines = result.utility_code.count("\n") + 1
        print(f"  Merged utility file: {utils_lines} lines")
    else:
        print("\n  No utility files found.")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
