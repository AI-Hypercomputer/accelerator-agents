"""
Step 1: Clone the PyTorch repository from GitHub.

This script clones a PyTorch repository so MaxCode can convert it to JAX.
After cloning, it lists all Python source files found in the repo.

If the repo is already cloned, this step is skipped.

Usage:
    python step1_clone_repo.py [REPO_URL]
    python step1_clone_repo.py [REPO_URL] --subdir PATH

Examples:
    python step1_clone_repo.py
    python step1_clone_repo.py https://github.com/yaohungt/Multimodal-Transformer
    python step1_clone_repo.py https://github.com/openai/whisper
    python step1_clone_repo.py https://github.com/huggingface/transformers --subdir src/transformers/models/qwen3_next
"""

import os
import shutil
import subprocess
import sys


def _parse_github_tree_url(url):
    """Detect URLs like .../tree/main/src/foo and split into repo + subdir."""
    # https://github.com/user/repo/tree/branch/path/to/dir
    if "/tree/" in url:
        base, _, rest = url.partition("/tree/")
        # rest = "main/src/transformers/models/qwen3_next"
        # split off the branch name (first segment)
        parts = rest.split("/", 1)
        subdir = parts[1] if len(parts) > 1 else ""
        return base, subdir
    return url, ""


def _sparse_clone(repo_url, subdir, target_dir):
    """Clone only a subdirectory using git sparse-checkout."""
    print(f"  Sparse-checkout: cloning only {subdir}")
    print()

    # Step 1: bare-minimum clone (no blobs until needed)
    ret = subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse",
         "--depth=1", repo_url, target_dir],
        capture_output=False,
    )
    if ret.returncode != 0:
        print("ERROR: git clone failed.")
        raise SystemExit(1)

    # Step 2: set sparse-checkout to just the subdir
    ret = subprocess.run(
        ["git", "sparse-checkout", "set", subdir],
        cwd=target_dir,
        capture_output=False,
    )
    if ret.returncode != 0:
        print("ERROR: git sparse-checkout failed.")
        raise SystemExit(1)

    # Step 3: flatten — move subdir contents to top level for the pipeline
    nested = os.path.join(target_dir, subdir.replace("/", os.sep))
    if os.path.isdir(nested) and nested != target_dir:
        # Move files up, then remove the nested skeleton
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(target_dir, item)
            shutil.move(src, dst)
        # Remove the now-empty nested directory tree
        top_segment = subdir.split("/")[0]
        skeleton = os.path.join(target_dir, top_segment)
        if os.path.isdir(skeleton):
            shutil.rmtree(skeleton)
        print(f"  Flattened {subdir}/ to repo root")
    print()


def main():
    # Parse arguments
    repo_url = None
    subdir = ""
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--subdir" and i + 1 < len(args):
            subdir = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            repo_url = args[i]
            i += 1
        else:
            i += 1

    if repo_url:
        # Auto-detect tree URLs (user pasted a GitHub folder link)
        parsed_url, parsed_subdir = _parse_github_tree_url(repo_url)
        if parsed_subdir and not subdir:
            repo_url = parsed_url
            subdir = parsed_subdir
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
    if subdir:
        print(f"  Subdir: {subdir}")
    print(f"  Target: {REPO_DIR}")
    print()

    if not os.path.isdir(REPO_DIR):
        if subdir:
            _sparse_clone(REPO_URL, subdir, REPO_DIR)
        else:
            ret = os.system(f'git clone "{REPO_URL}" "{REPO_DIR}"')
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
