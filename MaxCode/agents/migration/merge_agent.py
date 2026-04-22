"""Merge agent for combining model and utility files before conversion.

This is a pure-logic agent (no LLM calls). It encapsulates the file
discovery, filtering, import-graph analysis, and merge logic that was
previously in examples/demo/step3_merge.py.
"""

import ast
import fnmatch
import os
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MergeResult:
    """Result of merging a repository's model and utility files."""
    model_code: str                              # merged model code
    model_files: list[str]                       # files included in model merge
    utility_code: str | None                     # merged utility code (None if no utils found)
    utility_files: list[str]                     # files included in utility merge
    excluded_files: list[tuple[str, str]] = field(default_factory=list)   # (path, reason)
    excluded_classes: list[tuple[str, str]] = field(default_factory=list) # (class_name, reason)
    utility_categories: dict[str, str] = field(default_factory=dict)     # file -> category


# ---------------------------------------------------------------------------
# Infrastructure detection constants
# ---------------------------------------------------------------------------

_INFRA_PACKAGES = {
    "apex",
    "transformer_engine", "te",
    "deepspeed.pipe", "deepspeed.runtime",
}

_INFRA_BASES = {
    "torch.autograd.Function",
    "autograd.Function",
    "PipelineModule",
    "enum.Enum",
    "Enum",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

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


def _is_local_import(line, repo_dir):
    """Check if an import line resolves to a file within the repo."""
    stripped = line.strip()
    if stripped.startswith("from .") or stripped.startswith("from .."):
        return True
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
    """Insert ``pass`` into blocks left empty after import removal."""
    lines = code.split("\n")
    result = []
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
        if stripped.endswith(":") and any(stripped.startswith(kw) for kw in block_starters):
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            body_indent = indent + "    "
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines):
                result.append(body_indent + "pass")
            else:
                next_indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
                next_stripped = lines[j].lstrip()
                if len(next_indent) <= len(indent) and next_stripped:
                    result.append(body_indent + "pass")
        i += 1
    return "\n".join(result)


def _count_module_classes(code):
    """Count nn.Module subclasses in source code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1
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
# Infrastructure detection helpers
# ---------------------------------------------------------------------------

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
    if base_str.startswith("te.pytorch.") or base_str.startswith("transformer_engine.pytorch."):
        return True
    return False


def classify_file_classes(file_path):
    """Return list of class info dicts for every ClassDef in *file_path*."""
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


def should_exclude_class(node, exclude_patterns):
    """Check if a ClassDef *node* should be excluded from the merged output."""
    bases = [_base_to_str(b) for b in node.bases]

    for pat in exclude_patterns:
        if fnmatch.fnmatch(node.name, pat):
            return True, f"matches exclude pattern '{pat}'"

    for b in bases:
        if b in ("torch.autograd.Function", "autograd.Function"):
            return True, "autograd.Function subclass"

    if "PipelineModule" in bases:
        return True, "PipelineModule subclass"

    for b in bases:
        if b.startswith("te.pytorch.") or b.startswith("transformer_engine.pytorch."):
            return True, "TransformerEngine wrapper"

    if node.name.endswith("Pipe"):
        return True, "pipeline wrapper -- name ends with Pipe"

    for b in bases:
        if b in ("enum.Enum", "Enum"):
            return True, "enum.Enum subclass"

    return False, ""


# ---------------------------------------------------------------------------
# Utility classification
# ---------------------------------------------------------------------------

def classify_utility_file(file_path, repo_dir):
    """Classify a utility file into a category.

    Returns one of: "init_reexport", "cuda_kernel", "torch_autograd",
    "torch_utility", "pure_python".
    """
    basename = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            code = f.read()
        tree = ast.parse(code)
    except SyntaxError:
        return "pure_python"

    if basename == "__init__.py":
        body_types = set(type(n).__name__ for n in ast.iter_child_nodes(tree))
        reexport_types = {"Import", "ImportFrom", "Assign", "Expr"}
        if body_types <= reexport_types:
            return "init_reexport"

    has_cu_ref = ".cu" in code or ".cpp" in code
    has_load_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("load", "load_inline"):
                has_load_call = True
            elif isinstance(func, ast.Attribute) and func.attr in ("load", "load_inline"):
                has_load_call = True
    if has_cu_ref and has_load_call:
        return "cuda_kernel"

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_str = _base_to_str(base)
                if base_str in ("torch.autograd.Function", "autograd.Function"):
                    return "torch_autograd"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    return "torch_utility"
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "torch" or node.module.startswith("torch.")):
                return "torch_utility"

    return "pure_python"


# ---------------------------------------------------------------------------
# MergeAgent
# ---------------------------------------------------------------------------

class MergeAgent:
    """Merges a repository's model and utility files for conversion.

    This is a pure-logic agent (no LLM calls). It handles:
    - Model file discovery (nn.Module detection)
    - File-level and class-level filtering
    - Import graph construction and topological sorting
    - File merging with import deduplication
    - Utility file discovery and classification
    """

    @staticmethod
    def find_model_files(repo_dir):
        """Walk the repo and return paths of files containing nn.Module classes."""
        model_files = []
        for root, _, files in os.walk(repo_dir):
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                full = os.path.join(root, f)
                if MergeAgent._is_model_file(full):
                    model_files.append(full)
        return model_files

    @staticmethod
    def _is_model_file(file_path):
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

    @staticmethod
    def get_local_imports(file_path, repo_dir):
        """Parse a file's AST and return resolved paths of local imports."""
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

            module_path = module.replace(".", os.sep)

            if node.level > 0:
                base = file_dir
                for _ in range(node.level - 1):
                    base = os.path.dirname(base)
                candidates = [
                    os.path.join(base, module_path + ".py"),
                    os.path.join(base, module_path, "__init__.py"),
                ]
            else:
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

    @staticmethod
    def build_model_import_graph(model_files, repo_dir):
        """Build a directed graph of imports between model files."""
        model_set = set(os.path.normpath(f) for f in model_files)
        graph = {}
        for f in model_files:
            f_norm = os.path.normpath(f)
            all_imports = MergeAgent.get_local_imports(f, repo_dir)
            graph[f_norm] = {imp for imp in all_imports if imp in model_set}
        return graph

    @staticmethod
    def find_entry_points(model_files, import_graph):
        """Find model files at the top of the dependency tree."""
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

        if not entries:
            entries = [os.path.normpath(f) for f in model_files]

        return entries

    @staticmethod
    def trace_dependencies(entry_points, import_graph):
        """BFS from entry points, then topological sort (DFS post-order)."""
        visited = set()
        order = []

        reachable = set()
        queue = deque(entry_points)
        reachable.update(entry_points)
        while queue:
            node = queue.popleft()
            for dep in import_graph.get(node, set()):
                if dep not in reachable:
                    reachable.add(dep)
                    queue.append(dep)

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

        return order

    @staticmethod
    def merge_files(file_paths, repo_dir):
        """Merge files into a single string with imports de-duplicated.

        Returns the merged code string (no file I/O for output).
        """
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
                triple_count = stripped.count('"""') + stripped.count("'''")
                if triple_count % 2 == 1:
                    in_docstring = not in_docstring
                if in_docstring or triple_count > 0:
                    section_lines.append(line)
                    continue
                if skipping_multiline_import:
                    if ")" in stripped:
                        skipping_multiline_import = False
                    continue
                if _is_local_import(line, repo_dir):
                    if "(" in stripped and ")" not in stripped:
                        skipping_multiline_import = True
                    continue
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

        fixed_sections = []
        for section in code_sections:
            fixed_sections.append(_fix_empty_blocks(section))
        code_sections = fixed_sections

        header = '"""\nMerged model file - auto-generated by MergeAgent\n'
        header += f"Source: {repo_dir}\n"
        header += f"Files:  {len(file_paths)} files detected\n"
        header += '"""\n\n'

        merged = header + "\n".join(sorted(import_lines)) + "\n" + "\n".join(code_sections)
        return merged

    @staticmethod
    def filter_files(model_files, repo_dir, exclude_paths=None):
        """Apply file-level filters to the raw model file list.

        Returns (kept_files, [(removed_path, reason), ...]).
        """
        if exclude_paths is None:
            exclude_paths = []

        kept = []
        removed = []

        for full_path in model_files:
            rel = os.path.relpath(full_path, repo_dir).replace("\\", "/")
            basename = os.path.basename(full_path)

            excluded = False
            for pat in exclude_paths:
                if fnmatch.fnmatch(rel, pat):
                    removed.append((full_path, f"matches exclude pattern '{pat}'"))
                    excluded = True
                    break
            if excluded:
                continue

            if fnmatch.fnmatch(basename, "fused_*.py"):
                removed.append((full_path, "fused kernel file"))
                continue

            classes = classify_file_classes(full_path)
            infra_imports = detect_infrastructure_imports(full_path)
            if classes and all(c["is_infra"] for c in classes) and infra_imports:
                pkg_names = ", ".join(sorted(infra_imports))
                removed.append((full_path, f"all classes are {pkg_names} wrappers"))
                continue

            kept.append(full_path)

        return kept, removed

    @staticmethod
    def filter_classes_from_code(code, exclude_patterns=None):
        """Remove infrastructure classes from merged source code.

        Returns (filtered_code, [(class_name, reason), ...]).
        """
        if exclude_patterns is None:
            exclude_patterns = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, []

        lines = code.split("\n")
        ranges_to_remove = []
        removed_classes = []

        top_level_nodes = list(ast.iter_child_nodes(tree))
        for i, node in enumerate(top_level_nodes):
            if not isinstance(node, ast.ClassDef):
                continue
            exclude, reason = should_exclude_class(node, exclude_patterns)
            if not exclude:
                continue

            start = node.lineno
            end = node.end_lineno

            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)

            next_start = None
            for j in range(i + 1, len(top_level_nodes)):
                nxt = top_level_nodes[j]
                if hasattr(nxt, "lineno"):
                    next_start = nxt.lineno
                    break
            if next_start is not None:
                while end + 1 < next_start and lines[end].strip() == "":
                    end += 1

            ranges_to_remove.append((start, end))
            removed_classes.append((node.name, reason))

        if not ranges_to_remove:
            return code, []

        remove_set = set()
        for start, end in ranges_to_remove:
            for ln in range(start - 1, end):
                remove_set.add(ln)

        filtered_lines = [line for idx, line in enumerate(lines) if idx not in remove_set]
        return "\n".join(filtered_lines), removed_classes

    @staticmethod
    def find_all_local_dependencies(model_files, repo_dir):
        """BFS from model files through ALL local imports.

        Returns the set of utility files (non-model files that are
        transitively imported by model files).
        """
        model_set = set(os.path.normpath(f) for f in model_files)
        visited = set(model_set)
        queue = deque(model_set)

        while queue:
            current = queue.popleft()
            for dep in MergeAgent.get_local_imports(current, repo_dir):
                dep_norm = os.path.normpath(dep)
                if dep_norm not in visited:
                    visited.add(dep_norm)
                    queue.append(dep_norm)

        return visited - model_set

    @staticmethod
    def filter_utility_files(utility_files, repo_dir, exclude_patterns=None):
        """Apply exclusion patterns and classification to utility files.

        Returns (kept, removed_with_reasons, category_map).
        """
        if exclude_patterns is None:
            exclude_patterns = []

        kept = []
        removed = []
        category_map = {}

        for full_path in utility_files:
            rel = os.path.relpath(full_path, repo_dir).replace("\\", "/")

            excluded = False
            for pat in exclude_patterns:
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

    @staticmethod
    def order_utility_files(utility_files, repo_dir):
        """Topologically sort utility files by their import dependencies."""
        file_set = set(os.path.normpath(f) for f in utility_files)
        graph = {}
        for f in utility_files:
            f_norm = os.path.normpath(f)
            all_imports = MergeAgent.get_local_imports(f, repo_dir)
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

    def run(self, repo_dir, exclude_paths=None, exclude_classes=None,
            exclude_utils=None):
        """Run the full merge pipeline on a repository directory.

        Args:
            repo_dir: Path to the repository root.
            exclude_paths: Glob patterns for files to exclude from merge.
            exclude_classes: Class name patterns to exclude from merged output.
            exclude_utils: Glob patterns for utility files to exclude.

        Returns:
            MergeResult with merged model code, utility code, and metadata.
        """
        if exclude_paths is None:
            exclude_paths = []
        if exclude_classes is None:
            exclude_classes = []
        if exclude_utils is None:
            exclude_utils = []

        all_excluded_files = []
        all_excluded_classes = []

        # 1. Find model files
        model_files = self.find_model_files(repo_dir)

        # 2. File-level filtering
        model_files, removed_files = self.filter_files(
            model_files, repo_dir, exclude_paths
        )
        all_excluded_files.extend(removed_files)

        # 3. Build import graph and trace dependencies
        graph = self.build_model_import_graph(model_files, repo_dir)
        entries = self.find_entry_points(model_files, graph)
        required = self.trace_dependencies(entries, graph)

        # Track files excluded by graph analysis
        required_set = set(required)
        for f in model_files:
            f_norm = os.path.normpath(f)
            if f_norm not in required_set:
                all_excluded_files.append(
                    (f, "not imported by any entry-point model file")
                )

        # 4. Merge model files
        model_code = self.merge_files(required, repo_dir)

        # 5. Class-level filtering
        model_code, removed_classes = self.filter_classes_from_code(
            model_code, exclude_classes
        )
        all_excluded_classes.extend(removed_classes)

        # 6. Discover and merge utility files
        utility_code = None
        utility_files_kept = []
        utility_categories = {}

        util_files = self.find_all_local_dependencies(required, repo_dir)
        if util_files:
            kept_utils, removed_utils, cat_map = self.filter_utility_files(
                sorted(util_files), repo_dir, exclude_utils
            )
            all_excluded_files.extend(removed_utils)
            utility_categories = cat_map

            if kept_utils:
                ordered_utils = self.order_utility_files(kept_utils, repo_dir)
                utility_code = self.merge_files(ordered_utils, repo_dir)
                utility_files_kept = ordered_utils

        return MergeResult(
            model_code=model_code,
            model_files=required,
            utility_code=utility_code,
            utility_files=utility_files_kept,
            excluded_files=all_excluded_files,
            excluded_classes=all_excluded_classes,
            utility_categories=utility_categories,
        )
