"""
Step 1: Clone the PyTorch repository from GitHub.

This script clones the Multimodal-Transformer repository, which implements
a multimodal architecture combining language, audio, and vision using
cross-modal attention in PyTorch. After cloning, it lists all Python source
files that MaxCode will discover and convert in Step 3.

If the repo is already cloned, this step is skipped.

Usage:
    python step1_clone_repo.py
"""

import os
from config import REPO_URL, REPO_DIR

def main():
    print("=" * 70)
    print("Step 1: Clone PyTorch Repository")
    print("=" * 70)
    print(f"  Repo:   {REPO_URL}")
    print(f"  Target: {REPO_DIR}")
    print()

    if not os.path.isdir(REPO_DIR):
        ret = os.system(f"git clone {REPO_URL} {REPO_DIR}")
        if ret != 0:
            print("ERROR: git clone failed.")
            raise SystemExit(1)
        print()
    else:
        print("  Already cloned, skipping.")
        print()

    # List all Python files that MaxCode will discover
    print("Python files in the repository:")
    total_lines = 0
    file_count = 0
    for root, _, files in os.walk(REPO_DIR):
        for f in sorted(files):
            if f.endswith(".py"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, REPO_DIR)
                lines = sum(1 for _ in open(full, encoding="utf-8", errors="replace"))
                total_lines += lines
                file_count += 1
                print(f"  {rel} ({lines} lines)")

    print(f"\n  Total: {file_count} files, {total_lines} lines")
    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()
