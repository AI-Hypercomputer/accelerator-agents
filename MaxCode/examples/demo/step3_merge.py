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

import ast
import os
from collections import deque
from config import REPO_DIR, MERGED_FILE


def is_model_file(file_path):
    """Detect if a Python file defines any nn.Module subclass."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Attribute) and base.attr == "Module":
                    return True
                if isinstance(base, ast.Name) and base.id == "Module":
                    return True
    return False


def find_model_files(repo_dir):
    """Walk the repo and return paths of files containing nn.Module classes."""
    model_files = []
    for root, _, files in os.walk(repo_dir):
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            full = os.path.join(root, f)
            if is_model_file(full):
                model_files.append(full)
    return model_files


def get_local_imports(file_path, repo_dir):
    """Parse a Python file's AST and return resolved paths of local imports.

    Handles both absolute-style imports (from modules.transformer import X)
    and relative imports (from .foo import X).  Only returns paths that
    actually exist under repo_dir.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    resolved = set()
    file_dir = os.path.dirname(file_path)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module
        if module is None:
            continue

        # Convert dotted module path to a file path fragment
        module_path = module.replace(".", os.sep)

        if node.level > 0:
            # Relative import: resolve from the file's own directory
            # level=1 means '.', level=2 means '..' etc.
            base = file_dir
            for _ in range(node.level - 1):
                base = os.path.dirname(base)
            candidates = [
                os.path.join(base, module_path + ".py"),
                os.path.join(base, module_path, "__init__.py"),
            ]
        else:
            # Absolute-style import: resolve from repo root
            candidates = [
                os.path.join(repo_dir, module_path + ".py"),
                os.path.join(repo_dir, module_path, "__init__.py"),
            ]

        for candidate in candidates:
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate):
                resolved.add(candidate)
                break

    return resolved


def build_import_graph(model_files, repo_dir):
    """Build a directed graph of imports between model files.

    Returns a dict mapping each model file path to the set of other model
    file paths it imports.
    """
    model_set = set(os.path.normpath(f) for f in model_files)
    graph = {}
    for f in model_files:
        f_norm = os.path.normpath(f)
        all_imports = get_local_imports(f, repo_dir)
        # Keep only edges to other model files
        graph[f_norm] = {imp for imp in all_imports if imp in model_set}
    return graph


def find_entry_points(model_files, import_graph):
    """Find model files that sit at the top of the dependency tree.

    An entry point is a model file that:
      - is NOT imported by any other model file, AND
      - DOES import at least one other model file (i.e. it has dependents)

    Files that are neither imported nor import anything are isolated
    (dead code) and will be excluded from the merge.  If no file meets
    the criteria above (e.g. a single standalone model file), all files
    are returned as entry points so nothing is lost.
    """
    imported_by_someone = set()
    for deps in import_graph.values():
        imported_by_someone.update(deps)

    entries = []
    for f in model_files:
        f_norm = os.path.normpath(f)
        has_deps = bool(import_graph.get(f_norm))
        is_imported = f_norm in imported_by_someone
        if not is_imported and has_deps:
            entries.append(f_norm)

    # Fallback: if no file qualifies (e.g. all files are isolated),
    # treat every file as an entry point so nothing is dropped.
    if not entries:
        entries = [os.path.normpath(f) for f in model_files]

    return entries


def trace_dependencies(entry_points, import_graph):
    """BFS from entry points through the import graph.

    Returns a topologically-sorted list: dependencies first, entry points
    last, so that classes are defined before they are used.
    """
    visited = set()
    order = []  # will be reversed at the end

    # BFS to find all reachable nodes, then topological sort via DFS
    reachable = set()
    queue = deque(entry_points)
    reachable.update(entry_points)
    while queue:
        node = queue.popleft()
        for dep in import_graph.get(node, set()):
            if dep not in reachable:
                reachable.add(dep)
                queue.append(dep)

    # Topological sort (DFS post-order) over the reachable subgraph
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for dep in import_graph.get(node, set()):
            if dep in reachable:
                dfs(dep)
        order.append(node)

    for ep in sorted(entry_points):
        dfs(ep)

    # order is already leaves-first (post-order): dependencies before dependents
    return order


def _is_local_import(line, repo_dir):
    """Check if an import line resolves to a file within the repo."""
    stripped = line.strip()
    # Already handled: relative imports
    if stripped.startswith("from .") or stripped.startswith("from .."):
        return True
    # Check absolute-style 'from X import Y'
    if stripped.startswith("from "):
        parts = stripped.split()
        if len(parts) >= 2:
            module = parts[1]
            module_path = module.replace(".", os.sep)
            if os.path.isfile(os.path.join(repo_dir, module_path + ".py")):
                return True
            if os.path.isfile(os.path.join(repo_dir, module_path, "__init__.py")):
                return True
    return False


def merge_files(file_paths, repo_dir, output_path):
    """Merge model files into a single file with imports de-duplicated."""
    import_lines = set()
    code_sections = []

    for full_path in file_paths:
        rel = os.path.relpath(full_path, repo_dir)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        section_lines = []
        in_docstring = False
        for line in content.split("\n"):
            stripped = line.strip()
            # Track triple-quoted strings (docstrings / multi-line comments)
            triple_count = stripped.count('"""') + stripped.count("'''")
            if triple_count % 2 == 1:
                in_docstring = not in_docstring
            # Inside a docstring, keep the line as-is
            if in_docstring or triple_count > 0:
                section_lines.append(line)
                continue
            # Skip imports that resolve to local repo files (handled by merging)
            if _is_local_import(line, repo_dir):
                continue
            # Collect standard imports (only at top-level indentation)
            if not line[:1].isspace() and (
                stripped.startswith("import ") or stripped.startswith("from ")
            ):
                import_lines.add(line)
            else:
                section_lines.append(line)

        code_sections.append(
            f"\n# {'=' * 70}\n# From {rel}\n# {'=' * 70}\n"
            + "\n".join(section_lines)
        )

    header = '"""\nMerged model file — auto-generated by step3_merge.py\n'
    header += f"Source: {repo_dir}\n"
    header += f"Files:  {len(file_paths)} model files detected\n"
    header += '"""\n\n'

    merged = header + "\n".join(sorted(import_lines)) + "\n" + "\n".join(code_sections)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged)

    return merged


def main():
    if not os.path.isdir(REPO_DIR):
        print("ERROR: Repository not found. Run step1_clone_repo.py first.")
        raise SystemExit(1)

    print("=" * 70)
    print("Step 3: Auto-Detect and Merge Model Files")
    print("=" * 70)
    print(f"  Scanning: {REPO_DIR}")
    print()

    # Scan all .py files
    all_py = []
    for root, _, files in os.walk(REPO_DIR):
        for f in sorted(files):
            if f.endswith(".py"):
                all_py.append(os.path.join(root, f))

    print(f"  Found {len(all_py)} Python files total")
    print()

    # Detect model files
    model_files = find_model_files(REPO_DIR)

    print("  All model files detected (contain nn.Module):")
    for full_path in model_files:
        rel = os.path.relpath(full_path, REPO_DIR)
        lines = sum(1 for _ in open(full_path, encoding="utf-8"))
        print(f"    {rel} ({lines} lines)")

    skipped = len(all_py) - len(model_files)
    print(f"\n  Skipped {skipped} non-model files (datasets, training, utils, etc.)")

    # Build import graph and filter to transitively-imported files only
    print("\n  Building import graph...")
    graph = build_import_graph(model_files, REPO_DIR)

    for src, deps in sorted(graph.items(), key=lambda x: x[0]):
        rel_src = os.path.relpath(src, REPO_DIR)
        if deps:
            dep_names = ", ".join(
                os.path.relpath(d, REPO_DIR) for d in sorted(deps)
            )
            print(f"    {rel_src} -> {dep_names}")
        else:
            print(f"    {rel_src} -> (no model imports)")

    entries = find_entry_points(model_files, graph)
    print(f"\n  Entry point(s): "
          + ", ".join(os.path.relpath(e, REPO_DIR) for e in entries))

    required = trace_dependencies(entries, graph)
    excluded = set(os.path.normpath(f) for f in model_files) - set(required)

    if excluded:
        print(f"\n  Excluded {len(excluded)} file(s) (not imported by any model file):")
        for f in sorted(excluded):
            print(f"    {os.path.relpath(f, REPO_DIR)}")
    else:
        print("\n  No files excluded (all are transitively imported).")

    print(f"\n  Including {len(required)} file(s) in merge:")
    total_lines = 0
    for f in required:
        rel = os.path.relpath(f, REPO_DIR)
        lines = sum(1 for _ in open(f, encoding="utf-8"))
        total_lines += lines
        print(f"    {rel} ({lines} lines)")

    # Merge
    print(f"\n  Merging into: {MERGED_FILE}")
    merged = merge_files(required, REPO_DIR, MERGED_FILE)
    merged_lines = merged.count("\n") + 1
    print(f"  Merged file: {merged_lines} lines, {len(merged)} chars")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
