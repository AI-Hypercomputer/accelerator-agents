#!/usr/bin/env python3
"""Repository hygiene and secret/leak scanner.

Ensures that:
1. No bytecode caches (.pyc, __pycache__) or build artifacts (.egg-info) are tracked in git.
2. Tracked text files do not contain accidental internal URLs, internal paths, or internal credentials.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_TRACKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\.pyc$"),
    re.compile(r"__pycache__"),
    re.compile(r"\.egg-info"),
]

# Patterns for internal links, credentials, and paths that must remain private
SENSITIVE_TEXT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Internal go/ link", re.compile(r"(?:https?://|(?<=\s))go/[a-zA-Z0-9_.-]+")),
    ("Internal issue link", re.compile(r"(?:https?://|(?<=\s))b/\d{6,}")),
    ("Internal cl link", re.compile(r"(?:https?://|(?<=\s))cl/\d{6,}")),
    ("Internal corp domain", re.compile(r"[a-zA-Z0-9.-]+\.corp\.google\.com")),
    (
        "Internal email address",
        re.compile(r"(?<!opensource)(?<![\w.-])[a-zA-Z0-9._%+-]+@google\.com"),
    ),
    ("Internal cloud path", re.compile(r"/google/src/cloud/")),
    ("Internal workstation path", re.compile(r"/usr/local/google/")),
    (
        "Internal tool codename",
        re.compile(r"\b(?:jetski|tpu-coworker|anti-gravity)\b", re.IGNORECASE),
    ),
]

EXCLUDED_DIR_PREFIXES: tuple[str, ...] = (
    "MaxCode/",
    "JAXBench/",
    "maxkernel/claude-code/wiki/",
)

EXCLUDED_FILES: set[str] = {
    # Sanitization helpers specifically designed to strip internal imports when exporting
    "maxkernel/adk/hitl_agent/isolate_object.py",
    "maxkernel/adk/auto_agent/isolate_object.py",
    # Public contact email in open source code of conduct
    "docs/code-of-conduct.md",
    # Scanner itself defines the match rules
    ".github/scripts/check_repo_hygiene.py",
}

BINARY_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".xlsx",
    ".bin",
    ".tar",
    ".gz",
    ".zip",
    ".woff",
    ".woff2",
)


def get_repo_root() -> Path:
    """Resolve the repository root directory."""
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        return Path(root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parents[2]


def get_tracked_files(repo_root: Path) -> list[str]:
    """List all files tracked by git."""
    try:
        output = subprocess.check_output(["git", "ls-files"], cwd=repo_root, text=True)
        return [line.strip() for line in output.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise SystemExit(
            f"[FAIL] Git command failed. Ensure git is installed and run from within a git repository.\nDetail: {e}"
        )


def check_forbidden_tracked_files(tracked_files: list[str]) -> list[str]:
    """Check for accidentally tracked bytecode or build artifact files."""
    violations: list[str] = []
    for file_path in tracked_files:
        for pattern in FORBIDDEN_TRACKED_PATTERNS:
            if pattern.search(file_path):
                violations.append(file_path)
                break
    return violations


def check_sensitive_content(
    repo_root: Path, tracked_files: list[str]
) -> list[tuple[str, int, str, str]]:
    """Scan tracked text files for sensitive links, emails, and paths."""
    violations: list[tuple[str, int, str, str]] = []

    for rel_path in tracked_files:
        if rel_path.startswith(EXCLUDED_DIR_PREFIXES):
            continue
        if rel_path in EXCLUDED_FILES:
            continue
        if rel_path.endswith(BINARY_EXTENSIONS):
            continue

        full_path = repo_root / rel_path
        if not full_path.is_file():
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    for label, pattern in SENSITIVE_TEXT_PATTERNS:
                        match = pattern.search(line)
                        if match:
                            violations.append((rel_path, lineno, label, match.group(0)))
        except (UnicodeDecodeError, PermissionError):
            # Skip non-UTF-8 or inaccessible binary-like files
            continue

    return violations


def main() -> int:
    """Run repository hygiene checks and exit with appropriate status code."""
    repo_root = get_repo_root()
    tracked_files = get_tracked_files(repo_root)

    print(f"[Hygiene] Checking {len(tracked_files)} tracked files in {repo_root}...")

    # 1. Bytecode and cache check
    tracked_violations = check_forbidden_tracked_files(tracked_files)
    if tracked_violations:
        print(
            f"\n[FAIL] Found {len(tracked_violations)} forbidden tracked files (.pyc, __pycache__, .egg-info):",
            file=sys.stderr,
        )
        for path in tracked_violations:
            print(f"  - {path}", file=sys.stderr)
    else:
        print("  [OK] No tracked bytecode or build artifacts found.")

    # 2. Sensitive text patterns check
    content_violations = check_sensitive_content(repo_root, tracked_files)
    if content_violations:
        print(
            f"\n[FAIL] Found {len(content_violations)} internal references/links:",
            file=sys.stderr,
        )
        for path, lineno, label, matched_text in content_violations:
            print(f"  - {path}:{lineno} [{label}] {matched_text}", file=sys.stderr)
    else:
        print("  [OK] No internal URLs, paths, or email references detected.")

    if tracked_violations or content_violations:
        return 1

    print("\n[SUCCESS] Repository hygiene check clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
