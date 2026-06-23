"""
Shared configuration for the MaxCode demo scripts.

All paths are resolved relative to this file's location so the demo
can be run from any working directory.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAXCODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# ---------------------------------------------------------------------------
# Target repo to convert
# ---------------------------------------------------------------------------
DEFAULT_REPO_URL = "https://github.com/yaohungt/Multimodal-Transformer"
_REPO_URL_FILE = os.path.join(SCRIPT_DIR, ".repo_url")


def _resolve_repo_url():
    """Resolve repo URL: env var > .repo_url file > default."""
    from_env = os.environ.get("MAXCODE_REPO_URL")
    if from_env:
        return from_env
    if os.path.isfile(_REPO_URL_FILE):
        with open(_REPO_URL_FILE, "r") as f:
            saved = f.read().strip()
        if saved:
            return saved
    return DEFAULT_REPO_URL


REPO_URL = _resolve_repo_url()
REPO_DIR = os.path.join(SCRIPT_DIR, REPO_URL.rstrip("/").rsplit("/", 1)[-1])

# ---------------------------------------------------------------------------
# Output and RAG paths
# ---------------------------------------------------------------------------
MERGED_FILE = os.path.join(SCRIPT_DIR, "merged_model.py")
MERGED_UTILS_FILE = os.path.join(SCRIPT_DIR, "merged_utils.py")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
RAG_SOURCE_DIR = os.path.join(MAXCODE_DIR, "rag", "sources")

# ---------------------------------------------------------------------------
# Merge filtering (step3)
# ---------------------------------------------------------------------------

# Glob patterns (relative to repo root) for files to exclude from merge.
# Example: ["megatron/model/fused_*.py", "megatron/model/mamba/*"]
MERGE_EXCLUDE_PATHS = []

# Class name patterns to exclude from merged output.
# Supports '*' wildcard.  Example: ["*Pipe", "ColumnParallelLinear"]
MERGE_EXCLUDE_CLASSES = []

# Glob patterns for files to exclude from utility merge.
MERGE_EXCLUDE_UTILS = [
    "setup.py",
    "**/test_*.py",
    "**/tests/**",
    "**/*_test.py",
]


def setup():
    """Common setup: add MaxCode to sys.path and ensure HOME is set."""
    sys.path.insert(0, MAXCODE_DIR)
    if "HOME" not in os.environ:
        os.environ["HOME"] = os.environ.get("USERPROFILE", os.path.expanduser("~"))


def require_api_key():
    """Return the API key or exit with an error message."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable first.")
        print()
        print("  Linux / macOS / Git Bash:   export GOOGLE_API_KEY=<your-key>")
        print("  Windows CMD:                set GOOGLE_API_KEY=<your-key>")
        sys.exit(1)
    return api_key
