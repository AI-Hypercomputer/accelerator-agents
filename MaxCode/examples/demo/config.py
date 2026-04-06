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
REPO_URL = "https://github.com/yaohungt/Multimodal-Transformer"
REPO_DIR = os.path.join(SCRIPT_DIR, "Multimodal-Transformer")

# ---------------------------------------------------------------------------
# Output and RAG paths
# ---------------------------------------------------------------------------
MERGED_FILE = os.path.join(SCRIPT_DIR, "merged_model.py")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
RAG_SOURCE_DIR = os.path.join(MAXCODE_DIR, "rag", "sources")


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
