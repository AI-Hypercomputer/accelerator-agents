#!/usr/bin/env python3
"""Repository hygiene and secret/leak scanner.

Ensures that:
1. No bytecode caches (.pyc, __pycache__) or build artifacts (.egg-info) are tracked in git.
2. Tracked text files do not contain accidental internal URLs, internal paths, or internal credentials.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path

# Built-in fallback rule sets if hygiene_rules.json is absent
DEFAULT_FORBIDDEN_TRACKED_PATTERNS: list[str] = [
    r"\.pyc$",
    r"__pycache__",
    r"\.egg-info",
]

DEFAULT_SENSITIVE_TEXT_PATTERNS: list[tuple[str, str]] = [
    ("Internal go/ link", r"(?:https?://|(?<=\s))go/[a-zA-Z0-9_.-]+"),
    ("Internal issue link", r"(?:https?://|(?<=\s))b/\d{6,}"),
    ("Internal cl link", r"(?:https?://|(?<=\s))cl/\d{6,}"),
    ("Internal corp domain", r"[a-zA-Z0-9.-]+\.corp\.google\.com"),
    (
        "Internal email address",
        r"(?<!opensource)(?<![\w.-])[a-zA-Z0-9._%+-]+@google\.com",
    ),
    ("Internal cloud path", r"/google/src/cloud/"),
    ("Internal workstation path", r"/usr/local/google/"),
    (
        "Internal tool codename",
        r"(?i)\b(?:jetski|tpu-coworker|anti-gravity)\b",
    ),
]

DEFAULT_EXCLUDED_DIR_PREFIXES: tuple[str, ...] = (
    "MaxCode/",
    "JAXBench/",
    "maxkernel/claude-code/wiki/",
)

DEFAULT_EXCLUDED_FILES: tuple[str, ...] = (
    "maxkernel/adk/hitl_agent/isolate_object.py",
    "maxkernel/adk/auto_agent/isolate_object.py",
    "docs/code-of-conduct.md",
    ".github/scripts/check_repo_hygiene.py",
    ".github/scripts/hygiene_rules.json",
)

DEFAULT_BINARY_EXTENSIONS: tuple[str, ...] = (
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


def load_hygiene_rules(
    config_path: Path | None = None,
) -> tuple[
    list[re.Pattern[str]],
    list[tuple[str, re.Pattern[str]]],
    tuple[str, ...],
    set[str],
    tuple[str, ...],
]:
    """Load hygiene rules from JSON config file or fall back to built-in defaults."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "hygiene_rules.json"

    forbidden_raw = list(DEFAULT_FORBIDDEN_TRACKED_PATTERNS)
    sensitive_raw = list(DEFAULT_SENSITIVE_TEXT_PATTERNS)
    excluded_dirs = DEFAULT_EXCLUDED_DIR_PREFIXES
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    binary_exts = DEFAULT_BINARY_EXTENSIONS

    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "forbidden_file_patterns" in data:
                forbidden_raw = data["forbidden_file_patterns"]
            if "sensitive_text_patterns" in data:
                sensitive_raw = [
                    (item[0], item[1]) for item in data["sensitive_text_patterns"]
                ]
            if "excluded_dir_prefixes" in data:
                excluded_dirs = tuple(data["excluded_dir_prefixes"])
            if "excluded_files" in data:
                excluded_files = set(data["excluded_files"])
            if "binary_extensions" in data:
                binary_exts = tuple(data["binary_extensions"])
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"[WARN] Failed to load {config_path}: {e}. Falling back to default rules.",
                file=sys.stderr,
            )

    compiled_forbidden = [re.compile(p) for p in forbidden_raw]
    compiled_sensitive = [(label, re.compile(p)) for label, p in sensitive_raw]

    return (
        compiled_forbidden,
        compiled_sensitive,
        excluded_dirs,
        excluded_files,
        binary_exts,
    )


(
    FORBIDDEN_TRACKED_PATTERNS,
    SENSITIVE_TEXT_PATTERNS,
    EXCLUDED_DIR_PREFIXES,
    EXCLUDED_FILES,
    BINARY_EXTENSIONS,
) = load_hygiene_rules()


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


def check_forbidden_tracked_files(
    tracked_files: list[str],
    forbidden_patterns: list[re.Pattern[str]] = FORBIDDEN_TRACKED_PATTERNS,
) -> list[str]:
    """Check for accidentally tracked bytecode or build artifact files."""
    violations: list[str] = []
    for file_path in tracked_files:
        for pattern in forbidden_patterns:
            if pattern.search(file_path):
                violations.append(file_path)
                break
    return violations


def scan_single_file(
    repo_root: Path,
    rel_path: str,
    sensitive_patterns: list[tuple[str, re.Pattern[str]]],
) -> list[tuple[str, int, str, str]]:
    """Scan an individual text file for sensitive pattern matches."""
    full_path = repo_root / rel_path
    if not full_path.is_file():
        return []

    file_violations: list[tuple[str, int, str, str]] = []
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                for label, pattern in sensitive_patterns:
                    match = pattern.search(line)
                    if match:
                        file_violations.append(
                            (rel_path, lineno, label, match.group(0))
                        )
    except (UnicodeDecodeError, PermissionError):
        # Skip non-UTF-8 or inaccessible binary-like files
        pass
    return file_violations


def check_sensitive_content(
    repo_root: Path,
    tracked_files: list[str],
    excluded_dirs: tuple[str, ...] = EXCLUDED_DIR_PREFIXES,
    excluded_files: set[str] = EXCLUDED_FILES,
    binary_extensions: tuple[str, ...] = BINARY_EXTENSIONS,
    sensitive_patterns: list[tuple[str, re.Pattern[str]]] = SENSITIVE_TEXT_PATTERNS,
    max_workers: int | None = None,
) -> list[tuple[str, int, str, str]]:
    """Scan tracked text files in parallel for sensitive links, emails, and paths."""
    eligible_files = [
        rel_path
        for rel_path in tracked_files
        if not rel_path.startswith(excluded_dirs)
        and rel_path not in excluded_files
        and not rel_path.endswith(binary_extensions)
    ]

    violations: list[tuple[str, int, str, str]] = []
    if not eligible_files:
        return violations

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                scan_single_file, repo_root, rel_path, sensitive_patterns
            ): rel_path
            for rel_path in eligible_files
        }
        for future in concurrent.futures.as_completed(futures):
            file_violations = future.result()
            if file_violations:
                violations.extend(file_violations)

    violations.sort(key=lambda v: (v[0], v[1]))
    return violations


def main() -> int:
    """Run repository hygiene checks and exit with appropriate status code."""
    repo_root = get_repo_root()
    tracked_files = get_tracked_files(repo_root)

    print(f"[Hygiene] Checking {len(tracked_files)} tracked files in {repo_root}...")

    # 1. Bytecode and cache check
    tracked_violations = check_forbidden_tracked_files(
        tracked_files, FORBIDDEN_TRACKED_PATTERNS
    )
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
    content_violations = check_sensitive_content(
        repo_root,
        tracked_files,
        excluded_dirs=EXCLUDED_DIR_PREFIXES,
        excluded_files=EXCLUDED_FILES,
        binary_extensions=BINARY_EXTENSIONS,
        sensitive_patterns=SENSITIVE_TEXT_PATTERNS,
    )
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
