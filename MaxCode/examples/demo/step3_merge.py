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
import fnmatch
import os
from collections import deque
from config import (
    REPO_DIR, MERGED_FILE, MERGED_UTILS_FILE,
    MERGE_EXCLUDE_PATHS, MERGE_EXCLUDE_CLASSES, MERGE_EXCLUDE_UTILS,
)


def is_model_file(file_path):
    """Detect if a Python file defines any nn.Module subclass."""
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
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
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
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


def _fix_empty_blocks(code):
    """Insert ``pass`` into blocks left empty after import removal.

    When the only statement in an if/else/elif/try/except/for/while/with/def
    body was a local import that got stripped, the block becomes empty and
    causes a SyntaxError.  This function detects those cases and inserts
    ``pass`` to keep the code valid.
    """
    lines = code.split("\n")
    result = []
    # Patterns that introduce a new block (must end with ':')
    block_starters = (
        "if ", "elif ", "else:", "else :",
        "try:", "try :", "except:", "except ",
        "finally:", "finally :",
        "for ", "while ", "with ", "def ", "class ",
    )
    i = 0
    while i < len(lines):
        result.append(lines[i])
        stripped = lines[i].strip()
        # Check if this line starts a block
        if stripped.endswith(":") and any(stripped.startswith(kw) for kw in block_starters):
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            body_indent = indent + "    "
            # Peek ahead: is the next non-blank line at the same or lesser indent?
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines):
                # End of code — block is empty
                result.append(body_indent + "pass")
            else:
                next_stripped = lines[j].lstrip()
                next_indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
                if len(next_indent) <= len(indent) and next_stripped:
                    # Next meaningful line is NOT indented deeper — empty block
                    result.append(body_indent + "pass")
        i += 1
    return "\n".join(result)


def merge_files(file_paths, repo_dir, output_path):
    """Merge model files into a single file with imports de-duplicated."""
    import_lines = set()
    code_sections = []

    for full_path in file_paths:
        rel = os.path.relpath(full_path, repo_dir)
        with open(full_path, "r", encoding="utf-8-sig") as f:
            content = f.read()

        section_lines = []
        in_docstring = False
        skipping_multiline_import = False
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
            # Continue skipping lines from a multi-line local import
            if skipping_multiline_import:
                if ")" in stripped:
                    skipping_multiline_import = False
                continue
            # Skip imports that resolve to local repo files (handled by merging)
            if _is_local_import(line, repo_dir):
                # Check if this is a multi-line import (has '(' but no ')')
                if "(" in stripped and ")" not in stripped:
                    skipping_multiline_import = True
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

    # Post-process: fix empty blocks left behind by import removal.
    # When an if/else/elif/try/except/for/while/with/def block's only
    # content was a local import, removing it leaves invalid syntax.
    fixed_sections = []
    for section in code_sections:
        fixed_sections.append(_fix_empty_blocks(section))
    code_sections = fixed_sections

    header = '"""\nMerged model file - auto-generated by step3_merge.py\n'
    header += f"Source: {repo_dir}\n"
    header += f"Files:  {len(file_paths)} model files detected\n"
    header += '"""\n\n'

    merged = header + "\n".join(sorted(import_lines)) + "\n" + "\n".join(code_sections)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged)

    return merged


# ---------------------------------------------------------------------------
# Smart filtering helpers
# ---------------------------------------------------------------------------

# Infrastructure packages whose presence signals a file wraps HW-specific libs
_INFRA_PACKAGES = {
    "apex",
    "transformer_engine", "te",
    "deepspeed.pipe", "deepspeed.runtime",
}

# Base classes that are never convertible to JAX
_INFRA_BASES = {
    "torch.autograd.Function",
    "autograd.Function",
    "PipelineModule",
    "enum.Enum",
    "Enum",
}


def _base_to_str(base_node):
    """Convert an AST base-class node to a dotted string."""
    if isinstance(base_node, ast.Name):
        return base_node.id
    if isinstance(base_node, ast.Attribute):
        parts = []
        node = base_node
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def detect_infrastructure_imports(file_path):
    """Return set of known infrastructure package names imported by *file_path*."""
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return set()

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if alias.name in _INFRA_PACKAGES or top in _INFRA_PACKAGES:
                    found.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if node.module in _INFRA_PACKAGES or top in _INFRA_PACKAGES:
                    found.add(top)
    return found


def _is_infra_base(base_str):
    """Return True if *base_str* is a known infrastructure base class."""
    if base_str in _INFRA_BASES:
        return True
    # te.pytorch.* (TransformerEngine wrappers)
    if base_str.startswith("te.pytorch.") or base_str.startswith("transformer_engine.pytorch."):
        return True
    return False


def classify_file_classes(file_path):
    """Return list of class info dicts for every ClassDef in *file_path*.

    Each dict has keys: name, bases (list[str]), is_infra (bool).
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return []

    classes = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [_base_to_str(b) for b in node.bases]
        is_infra = bool(bases) and all(_is_infra_base(b) for b in bases)
        classes.append({"name": node.name, "bases": bases, "is_infra": is_infra})
    return classes


def filter_files(model_files, repo_dir):
    """Apply file-level filters to the raw model file list.

    Returns (kept_files, [(removed_path, reason), ...]).
    """
    kept = []
    removed = []

    for full_path in model_files:
        rel = os.path.relpath(full_path, repo_dir).replace("\\", "/")
        basename = os.path.basename(full_path)

        # 1. Config exclude patterns
        excluded = False
        for pat in MERGE_EXCLUDE_PATHS:
            if fnmatch.fnmatch(rel, pat):
                removed.append((full_path, f"matches exclude pattern '{pat}'"))
                excluded = True
                break
        if excluded:
            continue

        # 2. Fused kernel heuristic
        if fnmatch.fnmatch(basename, "fused_*.py"):
            removed.append((full_path, "fused kernel file"))
            continue

        # 3. All-infrastructure file: every class is infra AND file has infra imports
        classes = classify_file_classes(full_path)
        infra_imports = detect_infrastructure_imports(full_path)
        if classes and all(c["is_infra"] for c in classes) and infra_imports:
            pkg_names = ", ".join(sorted(infra_imports))
            removed.append((full_path, f"all classes are {pkg_names} wrappers"))
            continue

        kept.append(full_path)

    return kept, removed


def should_exclude_class(node, exclude_patterns):
    """Check if a ClassDef *node* should be excluded from the merged output.

    Returns (should_exclude: bool, reason: str).
    """
    bases = [_base_to_str(b) for b in node.bases]

    # 1. Config class-name patterns
    for pat in exclude_patterns:
        if fnmatch.fnmatch(node.name, pat):
            return True, f"matches exclude pattern '{pat}'"

    # 2. autograd.Function subclass
    for b in bases:
        if b in ("torch.autograd.Function", "autograd.Function"):
            return True, "autograd.Function subclass"

    # 3. PipelineModule subclass
    if "PipelineModule" in bases:
        return True, "PipelineModule subclass"

    # 4. TransformerEngine wrapper
    for b in bases:
        if b.startswith("te.pytorch.") or b.startswith("transformer_engine.pytorch."):
            return True, "TransformerEngine wrapper"

    # 5. Pipeline wrapper convention (name ends with Pipe)
    if node.name.endswith("Pipe"):
        return True, "pipeline wrapper -- name ends with Pipe"

    # 6. enum.Enum subclass
    for b in bases:
        if b in ("enum.Enum", "Enum"):
            return True, "enum.Enum subclass"

    return False, ""


def filter_classes_from_code(code, exclude_patterns):
    """Remove infrastructure classes from merged source code.

    Uses line-range deletion to preserve formatting and comments.
    Returns (filtered_code, [(class_name, reason), ...]).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"    WARNING: merged code has syntax error (line {e.lineno}), "
              "skipping class filtering")
        return code, []

    lines = code.split("\n")
    # Collect line ranges to remove (1-indexed, inclusive)
    ranges_to_remove = []
    removed_classes = []

    top_level_nodes = list(ast.iter_child_nodes(tree))
    for i, node in enumerate(top_level_nodes):
        if not isinstance(node, ast.ClassDef):
            continue
        exclude, reason = should_exclude_class(node, exclude_patterns)
        if not exclude:
            continue

        start = node.lineno  # 1-indexed
        end = node.end_lineno  # 1-indexed, inclusive

        # Extend to include decorator lines above the class
        if node.decorator_list:
            start = min(d.lineno for d in node.decorator_list)

        # Extend to include blank lines between this class and the next node
        # (so we don't leave big gaps)
        next_start = None
        for j in range(i + 1, len(top_level_nodes)):
            nxt = top_level_nodes[j]
            if hasattr(nxt, "lineno"):
                next_start = nxt.lineno
                break
        if next_start is not None:
            # Remove trailing blank lines up to the next node
            while end + 1 < next_start and lines[end].strip() == "":
                end += 1

        ranges_to_remove.append((start, end))
        removed_classes.append((node.name, reason))

    if not ranges_to_remove:
        return code, []

    # Build set of lines to remove (convert to 0-indexed)
    remove_set = set()
    for start, end in ranges_to_remove:
        for ln in range(start - 1, end):  # start-1 because lines list is 0-indexed
            remove_set.add(ln)

    filtered_lines = [line for idx, line in enumerate(lines) if idx not in remove_set]
    return "\n".join(filtered_lines), removed_classes


def _count_module_classes(code):
    """Count nn.Module subclasses in source code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1  # signal parse failure
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_str = _base_to_str(base)
                if base_str in ("nn.Module", "Module") or base_str.endswith(".Module"):
                    count += 1
                    break
    return count


# ---------------------------------------------------------------------------
# Utility file discovery and merging
# ---------------------------------------------------------------------------

def find_all_local_dependencies(model_files, repo_dir):
    """BFS from model files through ALL local imports (not just model files).

    Returns the set of utility files (local .py files that are transitively
    imported by model files but are NOT themselves model files).
    """
    model_set = set(os.path.normpath(f) for f in model_files)
    visited = set(model_set)
    queue = deque(model_set)

    while queue:
        current = queue.popleft()
        for dep in get_local_imports(current, repo_dir):
            dep_norm = os.path.normpath(dep)
            if dep_norm not in visited:
                visited.add(dep_norm)
                queue.append(dep_norm)

    # Return only the non-model files
    return visited - model_set


def classify_utility_file(file_path, repo_dir):
    """Classify a utility file into a category.

    Returns one of:
      - "init_reexport": __init__.py that only has imports — skip
      - "cuda_kernel": uses load()/load_inline() with .cu/.cpp refs — skip
      - "torch_autograd": has autograd.Function — keep (Python fallback)
      - "torch_utility": imports torch — keep
      - "pure_python": no torch dependency — keep
    """
    basename = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            code = f.read()
        tree = ast.parse(code)
    except SyntaxError:
        return "pure_python"

    # Check if __init__.py with only imports/assignments (re-export)
    if basename == "__init__.py":
        body_types = set(type(n).__name__ for n in ast.iter_child_nodes(tree))
        # Only imports, assignments, and expressions (docstrings)
        reexport_types = {"Import", "ImportFrom", "Assign", "Expr"}
        if body_types <= reexport_types:
            return "init_reexport"

    # Check for CUDA kernel loader patterns
    has_cu_ref = ".cu" in code or ".cpp" in code
    has_load_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # load() or load_inline() calls
            if isinstance(func, ast.Name) and func.id in ("load", "load_inline"):
                has_load_call = True
            elif isinstance(func, ast.Attribute) and func.attr in ("load", "load_inline"):
                has_load_call = True
    if has_cu_ref and has_load_call:
        return "cuda_kernel"

    # Check for autograd.Function
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_str = _base_to_str(base)
                if base_str in ("torch.autograd.Function", "autograd.Function"):
                    return "torch_autograd"

    # Check for torch imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    return "torch_utility"
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "torch" or node.module.startswith("torch.")):
                return "torch_utility"

    return "pure_python"


def filter_utility_files(utility_files, repo_dir):
    """Apply exclusion patterns and classification to utility files.

    Returns (kept, removed_with_reasons, category_map).
    """
    kept = []
    removed = []
    category_map = {}

    for full_path in utility_files:
        rel = os.path.relpath(full_path, repo_dir).replace("\\", "/")

        # Check exclude patterns
        excluded = False
        for pat in MERGE_EXCLUDE_UTILS:
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(os.path.basename(full_path), pat):
                removed.append((full_path, f"matches exclude pattern '{pat}'"))
                excluded = True
                break
        if excluded:
            continue

        category = classify_utility_file(full_path, repo_dir)
        category_map[full_path] = category

        if category == "init_reexport":
            removed.append((full_path, "re-export __init__.py (inlined by merge)"))
        elif category == "cuda_kernel":
            removed.append((full_path, "CUDA kernel loader (no JAX equivalent)"))
        else:
            kept.append(full_path)

    return kept, removed, category_map


def order_utility_files(utility_files, repo_dir):
    """Topologically sort utility files by their import dependencies.

    Dependencies come first so definitions precede usage.
    """
    file_set = set(os.path.normpath(f) for f in utility_files)
    graph = {}
    for f in utility_files:
        f_norm = os.path.normpath(f)
        all_imports = get_local_imports(f, repo_dir)
        graph[f_norm] = {imp for imp in all_imports if imp in file_set}

    visited = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for dep in graph.get(node, set()):
            dfs(dep)
        order.append(node)

    for f in sorted(file_set):
        dfs(f)

    return order


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
    print(f"  Detected {len(model_files)} files containing nn.Module classes")
    print()

    # --- File-level filtering (BEFORE import graph) ---
    print("  Filtering files...")
    model_files, removed_files = filter_files(model_files, REPO_DIR)

    for full_path, reason in removed_files:
        rel = os.path.relpath(full_path, REPO_DIR)
        print(f"    SKIP  {rel:<45s} ({reason})")
    for full_path in model_files:
        rel = os.path.relpath(full_path, REPO_DIR)
        print(f"    KEEP  {rel}")

    if removed_files:
        print(f"  Filtered: {len(removed_files)} files removed, "
              f"{len(model_files)} files remaining")
    else:
        print("  Filtered: no files removed")
    print()

    # Build import graph and filter to transitively-imported files only
    print("  Building import graph...")
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
        lines = sum(1 for _ in open(f, encoding="utf-8-sig"))
        total_lines += lines
        print(f"    {rel} ({lines} lines)")

    # Merge
    print(f"\n  Merging into: {MERGED_FILE}")
    merged = merge_files(required, REPO_DIR, MERGED_FILE)
    merged_lines = merged.count("\n") + 1
    print(f"  Merged file: {merged_lines} lines")

    # --- Class-level filtering (AFTER merge) ---
    print("\n  Filtering infrastructure classes from merged code...")
    filtered, removed_classes = filter_classes_from_code(merged, MERGE_EXCLUDE_CLASSES)

    if removed_classes:
        for cls_name, reason in removed_classes:
            print(f"    SKIP  {cls_name:<40s} ({reason})")
        print(f"  Filtered: {len(removed_classes)} classes removed")

        # Write filtered output
        with open(MERGED_FILE, "w", encoding="utf-8") as f:
            f.write(filtered)
        merged = filtered
    else:
        print("    (no infrastructure classes found)")

    final_lines = merged.count("\n") + 1
    final_modules = _count_module_classes(merged)
    if final_modules >= 0:
        print(f"\n  Final merged file: {final_lines} lines, "
              f"{final_modules} nn.Module classes")
    else:
        print(f"\n  Final merged file: {final_lines} lines "
              "(nn.Module count unavailable -- syntax error in merged code)")

    # ---------------------------------------------------------------
    # Step 3b: Discover and merge utility files
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("Step 3b: Discover and Merge Utility Files")
    print("=" * 70)

    util_files = find_all_local_dependencies(required, REPO_DIR)
    print(f"\n  Discovered {len(util_files)} utility file(s) transitively imported by model files")

    if util_files:
        kept_utils, removed_utils, cat_map = filter_utility_files(
            sorted(util_files), REPO_DIR
        )

        if removed_utils:
            print(f"\n  Filtered out {len(removed_utils)} utility file(s):")
            for full_path, reason in removed_utils:
                rel = os.path.relpath(full_path, REPO_DIR)
                print(f"    SKIP  {rel:<45s} ({reason})")

        if kept_utils:
            print(f"\n  Keeping {len(kept_utils)} utility file(s):")
            for full_path in kept_utils:
                rel = os.path.relpath(full_path, REPO_DIR)
                cat = cat_map.get(full_path, "unknown")
                print(f"    KEEP  {rel:<45s} [{cat}]")

            ordered_utils = order_utility_files(kept_utils, REPO_DIR)
            print(f"\n  Merging {len(ordered_utils)} utility files into: {MERGED_UTILS_FILE}")
            utils_merged = merge_files(ordered_utils, REPO_DIR, MERGED_UTILS_FILE)
            utils_lines = utils_merged.count("\n") + 1
            print(f"  Merged utility file: {utils_lines} lines")
        else:
            print("\n  No utility files remaining after filtering.")
    else:
        print("  No utility files found.")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
