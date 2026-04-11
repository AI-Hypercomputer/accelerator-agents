"""
Step 1: Clone the PyTorch repository from GitHub.

This script clones a PyTorch repository so MaxCode can convert it to JAX.
After cloning, it lists all Python source files found in the repo.

If the repo is already cloned, this step is skipped.

Usage:
    python step1_clone_repo.py [REPO_URL]

Examples:
    python step1_clone_repo.py
    python step1_clone_repo.py https://github.com/yaohungt/Multimodal-Transformer
    python step1_clone_repo.py https://github.com/openai/whisper
"""

import os
import sys


def main():
    # Accept optional URL from command line; falls back to config default
    if len(sys.argv) > 1:
        repo_url = sys.argv[1]
        # Set env var so config.py picks it up
        os.environ["MAXCODE_REPO_URL"] = repo_url

    # Import AFTER setting env var so config sees the override
    from config import REPO_URL, REPO_DIR, _REPO_URL_FILE

    # Persist the repo URL so step3/step4/step5 use the same repo
    with open(_REPO_URL_FILE, "w") as f:
        f.write(REPO_URL)

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

    # List all Python files
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
