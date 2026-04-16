#!/usr/bin/env python3
"""
Script to isolate an object (class or function) from a Python file along with all required imports.

WHAT THIS SCRIPT DOES:
----------------------
Extracts a specified Python object (function, class, or variable) from a source file and creates
a standalone, self-contained Python file with all its dependencies. The output includes:
- All required external library imports (e.g., jax, flax, numpy)
- Recursively extracted code from local workspace packages
- Helper functions, constants, and classes that the target object depends on

RECURSIVE SEARCH vs SIMPLE IMPORTS:
-----------------------------------
The script distinguishes between two types of dependencies:

1. EXTERNAL LIBRARIES (Simple Import - NOT recursively searched):
   - Standard Python libraries (os, sys, typing, etc.)
   - Third-party packages installed via pip (jax, flax, numpy, torch, etc.)
   - Any module NOT found in your workspace

   Example: "import jax.numpy as jnp" → Kept as-is in output

   How it's detected:
   - Module name is NOT in the local_package_names list
   - AND module files don't exist in workspace directories
   - Default assumption: If not local, it's external

2. LOCAL WORKSPACE PACKAGES (Recursively Searched and Extracted):
   - Packages in your workspace (MaxText, tpu_commons, vllm, etc.)
   - Files in third_party/ directories
   - Custom local modules

   Example: "from MaxText.layers import Attention" → Extracts Attention class source code

   How it's detected:
   - Module name starts with a known local package (see local_package_names list)
   - OR module files exist in workspace paths:
     * workspace_root/module_name/
     * workspace_root/third_party/module_name/

   What gets extracted:
   - The imported object's source code
   - All functions/classes/constants it depends on
   - Recursively follows dependencies up to max_depth levels (default: 5)

KNOWN LOCAL PACKAGES:
---------------------
The script maintains a hardcoded list of known local packages for quick detection:
- google3 (Google's internal codebase - auto-detected via 'g4 g4d')
- tpu_commons, MaxText, maxtext
- JetStream, jetstream
- vllm, whisper_jax, torchprime
- maxdiffusion, RecML

This list is an optimization hint, NOT a limitation. The script will still find and recursively
process any package that exists in your workspace, even if it's not in this list.

For google3 imports, the script automatically runs 'g4 g4d' to find the google3 root directory
and resolves imports like 'from google3.foo.bar import X' to the correct file paths.

SPECIAL FEATURES:
-----------------
- Mock implementations: Can substitute external dependencies with local mocks
  (e.g., vllm.logger → tpu_commons.mock.vllm_logger)
- Topological sorting: Dependencies are ordered so definitions come before usage
- Deduplication: Avoids extracting the same object multiple times
- Reference fixing: Updates module.attribute references when extracting definitions

USAGE:
------
Basic usage:
    python isolate_object.py <file_path> <object_name> [options]

Arguments:
    file_path       Path to the Python file containing the object
    object_name     Name of the class or function to extract

Options:
    -o, --output FILE       Write output to FILE instead of stdout
    -d, --max-depth N       Maximum recursion depth for imports (default: 5)
    --debug                 Show debug information including file paths being opened

EXAMPLES:
---------
Extract a class to stdout:
    python isolate_object.py path/to/file.py MyClass

Extract a class to a file:
    python isolate_object.py path/to/file.py MyClass -o isolated_class.py

Extract with custom recursion depth and debug output:
    python isolate_object.py third_party/vllm/model.py Attention -d 3 --debug

Extract from google3:
    python isolate_object.py third_party/google3/path/to/module.py MyFunction -o output.py
"""

import argparse
import ast
import os
import re
import sys
import textwrap
import warnings
from typing import List, Optional, Set, Tuple

# Suppress all deprecation warnings from ast module
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ImportCollector(ast.NodeVisitor):
  """Collects all imports and used names in the AST."""

  def __init__(self):
    self.imports = []
    self.used_names = set()
    self.defined_names = set()
    self.module_attributes = set()  # Track module.attribute patterns
    self.import_aliases = {}  # Track alias -> (module, original_name) mappings

  def visit_Import(self, node):
    for alias in node.names:
      name = alias.asname if alias.asname else alias.name
      self.imports.append(
        f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
      )
      self.defined_names.add(name)
      # Track alias mapping: alias -> (None, original_module_name)
      if alias.asname:
        self.import_aliases[alias.asname] = (None, alias.name)
    self.generic_visit(node)

  def visit_ImportFrom(self, node):
    module = node.module or ""
    if node.names[0].name == "*":
      self.imports.append(f"from {module} import *")
    else:
      names = []
      for alias in node.names:
        name = alias.asname if alias.asname else alias.name
        names.append(
          f"{alias.name}" + (f" as {alias.asname}" if alias.asname else "")
        )
        self.defined_names.add(name)
        # Track alias mapping: alias -> (module, original_name)
        if alias.asname:
          self.import_aliases[alias.asname] = (module, alias.name)
        else:
          # Even without alias, track for potential attribute resolution
          self.import_aliases[alias.name] = (module, alias.name)
      self.imports.append(f"from {module} import {', '.join(names)}")
    self.generic_visit(node)

  def visit_Name(self, node):
    if isinstance(node.ctx, ast.Load):
      self.used_names.add(node.id)
    elif isinstance(node.ctx, ast.Store):
      self.defined_names.add(node.id)
    self.generic_visit(node)

  def visit_Attribute(self, node):
    # For attributes like nn.Module, jnp.array, test_helper.helper_function, etc.
    if isinstance(node.value, ast.Name):
      self.used_names.add(node.value.id)
      # Track module.attribute patterns for local imports
      pattern = f"{node.value.id}.{node.attr}"
      self.module_attributes.add(pattern)

      # If this is an aliased import (e.g., e.f where e is from google3.a.b.c import d as e)
      # Also track the resolved full path
      if node.value.id in self.import_aliases:
        module, original_name = self.import_aliases[node.value.id]
        if module:
          # from module import name as alias -> module.name.attr
          full_path = f"{module}.{original_name}.{node.attr}"
          self.module_attributes.add(full_path)
    elif isinstance(node.value, ast.Attribute) and isinstance(
      node.value.value, ast.Name
    ):
      # Handle nested attributes like module.submodule.function
      self.used_names.add(node.value.value.id)
      self.module_attributes.add(
        f"{node.value.value.id}.{node.value.attr}.{node.attr}"
      )
    self.generic_visit(node)

  def visit_FunctionDef(self, node):
    self.defined_names.add(node.name)
    # Explicitly visit decorators to ensure decorator names are collected
    for decorator in node.decorator_list:
      self.visit(decorator)
    self.generic_visit(node)

  def visit_ClassDef(self, node):
    self.defined_names.add(node.name)
    # Explicitly visit decorators to ensure decorator names are collected
    for decorator in node.decorator_list:
      self.visit(decorator)
    self.generic_visit(node)

  def visit_AnnAssign(self, node):
    """Visit annotated assignments like: exp2: TypeAlias = fa_util.exp2"""
    # Add the target to defined names
    if isinstance(node.target, ast.Name):
      self.defined_names.add(node.target.id)
    # Make sure we visit the annotation and value
    if node.annotation:
      self.visit(node.annotation)
    if node.value:
      self.visit(node.value)
    # Don't call generic_visit to avoid double-visiting


class ImportOnlyCollector(ast.NodeVisitor):
  """Collects only imports, not local definitions."""

  def __init__(self):
    self.imports = []
    self.defined_names = set()

  def visit_Import(self, node):
    for alias in node.names:
      name = alias.asname if alias.asname else alias.name
      self.imports.append(
        f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
      )
      self.defined_names.add(name)
    self.generic_visit(node)

  def visit_ImportFrom(self, node):
    module = node.module or ""
    if node.names[0].name == "*":
      self.imports.append(f"from {module} import *")
    else:
      names = []
      for alias in node.names:
        name = alias.asname if alias.asname else alias.name
        names.append(
          f"{alias.name}" + (f" as {alias.asname}" if alias.asname else "")
        )
        self.defined_names.add(name)
      self.imports.append(f"from {module} import {', '.join(names)}")
    self.generic_visit(node)


class ObjectExtractor:
  """Extracts an object and its dependencies from a Python file."""

  def __init__(self, filename: str, debug: bool = False):
    self.filename = filename
    self.debug = debug
    self._google3_root_cache = None  # Cache for google3 root
    self._google3_root_checked = False  # Track if we've tried to get it

    abs_path = os.path.abspath(filename)
    if self.debug:
      print(f"[DEBUG] Opening file: {abs_path}", file=sys.stderr)

    with open(filename, "r", encoding="utf-8") as f:
      self.source = f.read()
    self.tree = ast.parse(self.source)

  def find_object(self, object_name: str) -> Optional[ast.AST]:
    """Find the AST node for the given object name."""
    for node in ast.walk(self.tree):
      if (
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name == object_name
      ):
        return node
      # Also search for assignments
      elif isinstance(node, ast.Assign):
        for target in node.targets:
          if isinstance(target, ast.Name) and target.id == object_name:
            return node
      # And annotated assignments
      elif (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == object_name
      ):
        return node
    return None

  def get_object_source(self, object_name: str) -> Optional[str]:
    """Extract the source code for the given object."""
    target_node = self.find_object(object_name)
    if not target_node:
      return None

    # Get the source lines
    lines = self.source.splitlines()

    # For functions and classes with decorators, we need to start from the first decorator
    start_line = target_node.lineno - 1  # Convert to 0-indexed
    if (
      isinstance(target_node, (ast.FunctionDef, ast.ClassDef))
      and hasattr(target_node, "decorator_list")
      and target_node.decorator_list
    ):
      # Find the line number of the first decorator
      first_decorator = target_node.decorator_list[0]
      start_line = first_decorator.lineno - 1  # Convert to 0-indexed

    # Use end_lineno if available (Python 3.8+) for all node types
    if hasattr(target_node, "end_lineno") and target_node.end_lineno:
      end_line = target_node.end_lineno
    elif isinstance(target_node, (ast.FunctionDef, ast.ClassDef)):
      # Fallback for functions and classes: find the end by looking for the next top-level definition
      end_line = len(lines)
      for node in ast.walk(self.tree):
        if (
          isinstance(
            node,
            (
              ast.FunctionDef,
              ast.ClassDef,
              ast.Import,
              ast.ImportFrom,
              ast.Assign,
            ),
          )
          and node.lineno > target_node.lineno
          and node.col_offset == 0
        ):  # Top-level only
          end_line = node.lineno - 1
          break
    else:
      # For other nodes without end_lineno, just take the single line
      end_line = target_node.lineno

    # Extract the object source
    object_lines = lines[start_line:end_line]

    # Remove trailing empty lines
    while object_lines and not object_lines[-1].strip():
      object_lines.pop()

    # Join the lines
    source_code = "\n".join(object_lines)

    # Dedent the source code to remove extra indentation (e.g., from if TYPE_CHECKING blocks)
    # Only dedent if all lines have consistent indentation
    if source_code:
      source_code = textwrap.dedent(source_code)

    return source_code

  def get_required_imports(self, object_name: str) -> List[str]:
    """Get all imports required by the specified object."""
    target_node = self.find_object(object_name)
    if not target_node:
      return []

    # Collect imports and used names from the entire file
    file_collector = ImportCollector()
    file_collector.visit(self.tree)

    # Collect used names from just the target object
    object_collector = ImportCollector()
    object_collector.visit(target_node)

    # Find which imports are needed
    needed_imports = []
    needed_names = object_collector.used_names - object_collector.defined_names

    # Check each import to see if it provides any needed names
    # Sort to ensure deterministic order
    for import_stmt in sorted(file_collector.imports):
      # Parse the import to see what names it provides
      try:
        import_tree = ast.parse(import_stmt)
        import_collector = ImportCollector()
        import_collector.visit(import_tree)

        # If this import provides any needed names, include it
        if import_collector.defined_names & needed_names:
          needed_imports.append(import_stmt)
      except:
        # If we can't parse it, include it to be safe
        needed_imports.append(import_stmt)

    return needed_imports

  def get_additional_objects(self, object_name: str) -> List[Tuple[str, str]]:
    """Get additional objects (constants, functions, classes) that the target object depends on."""
    target_node = self.find_object(object_name)
    if not target_node:
      return []

    # Collect used names from the target object
    object_collector = ImportCollector()
    object_collector.visit(target_node)

    # Collect all top-level definitions from the file
    top_level_definitions = {}
    for node in self.tree.body:
      if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        top_level_definitions[node.name] = node
      elif isinstance(node, ast.Assign):
        # Include ALL assignments, not just constants
        for target in node.targets:
          if isinstance(target, ast.Name):
            top_level_definitions[target.id] = node
      elif isinstance(node, ast.AnnAssign):
        # Include annotated assignments like: MY_CONSTANT: TypeAlias = value
        if isinstance(node.target, ast.Name):
          top_level_definitions[node.target.id] = node

    # Find names that are used but not defined in the object itself
    external_names = (
      object_collector.used_names - object_collector.defined_names
    )

    # Filter to only include names that are actually defined at module level
    # and exclude common built-in names and imported names
    builtin_names = {
      "self",
      "True",
      "False",
      "None",
      "int",
      "float",
      "str",
      "list",
      "dict",
      "tuple",
      "set",
    }

    # Get all imported names to exclude them from dependencies
    import_collector = ImportOnlyCollector()
    import_collector.visit(self.tree)
    directly_imported_names = import_collector.defined_names

    # Also identify names that are defined via assignment (like param_with_axes = nn_partitioning.param_with_axes)
    locally_defined_names = set()
    for node in self.tree.body:
      if isinstance(node, ast.Assign):
        for target in node.targets:
          if isinstance(target, ast.Name):
            locally_defined_names.add(target.id)

    additional_objects = []
    # Sort external_names to ensure deterministic order
    for name in sorted(external_names):
      # Skip if it's a builtin or common name
      if name in builtin_names:
        continue

      # Skip if it's a directly imported name (but not locally defined aliases)
      if name in directly_imported_names and name not in locally_defined_names:
        continue

      # Only include if it's defined at the top level of this file
      if name in top_level_definitions:
        node = top_level_definitions[name]
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
          # Only include if it's not the same object we're extracting
          if node.name != object_name:
            obj_source = self.get_object_source(name)
            if obj_source:
              additional_objects.append((name, obj_source))
        elif isinstance(node, ast.Assign):
          # Check if this is an assignment from an external module (like nn_partitioning.something)
          # If so, skip this assignment to avoid circular dependencies
          is_external_assignment = False
          if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if isinstance(node.value, ast.Attribute) and isinstance(
              node.value.value, ast.Name
            ):
              # This looks like "name = module.something"
              module_ref = node.value.value.id
              attr_name = node.value.attr

              # Check if the module name is likely an imported module
              external_module_patterns = ["_partitioning", "_utils", "_layers"]
              if any(
                pattern in module_ref.lower()
                for pattern in external_module_patterns
              ):
                # Skip this assignment since it should be handled by direct imports
                is_external_assignment = True

          if not is_external_assignment:
            # Include regular assignments - use get_object_source for multi-line support
            obj_source = self.get_object_source(name)
            if obj_source:
              additional_objects.append((name, obj_source))
        elif isinstance(node, ast.AnnAssign):
          # Handle annotated assignments like: MY_CONSTANT: TypeAlias = fa_util.exp2
          obj_source = self.get_object_source(name)
          if obj_source:
            additional_objects.append((name, obj_source))

    return additional_objects

  def get_module_usage_patterns(self, object_name: str) -> Set[str]:
    """Get module usage patterns from the target object (e.g., attention_mla.mla_as_linen)."""
    target_node = self.find_object(object_name)
    if not target_node:
      return set()

    collector = ImportCollector()
    collector.visit(target_node)

    module_patterns = set()
    for attr_pattern in collector.module_attributes:
      # Extract the module name (first part before the dot)
      module_name = attr_pattern.split(".")[0]
      module_patterns.add(module_name)

    return module_patterns

  def get_local_import_files(
    self,
    imports: List[str],
    module_usage_patterns: Set[str] = None,
    module_attributes: Set[str] = None,
  ) -> List[str]:
    """Get list of local Python files that are imported."""
    local_files = []
    base_dir = os.path.dirname(os.path.abspath(self.filename))

    # Also search the entire workspace for packages
    workspace_root = self._find_workspace_root(base_dir)

    # Don't get google3 root here - will be fetched lazily only if we see google3 imports

    # Process module_attributes for full module paths (e.g., google3.a.b.c.d.f)
    # This handles cases like: from google3.a.b.c import d as e; f = e.f
    if module_attributes:
      for attr_pattern in sorted(module_attributes):
        parts = attr_pattern.split(".")
        # For patterns with multiple parts, try to resolve as module paths
        # We need at least 2 parts (module.attribute), but we'll try any pattern that might resolve
        if len(parts) >= 2:  # e.g., fa_util.exp2, google3.a.b, etc.
          # Try progressively shorter module paths (e.g., a.b.c.d.f -> try a.b.c.d, a.b.c, etc.)
          for i in range(
            len(parts) - 1, 0, -1
          ):  # Changed from 1 to 0 to try even single-part modules
            module_path = ".".join(parts[:i])

            # Check if this looks like a local package import
            is_local = self._is_local_package(module_path, workspace_root)
            if is_local:
              possible_paths = []

              # Special handling for google3
              if module_path.startswith("google3."):
                google3_root = self._get_google3_root()
                if google3_root:
                  module_path_without_google3 = module_path[
                    8:
                  ]  # Remove 'google3.'
                  possible_paths.append(
                    os.path.join(
                      google3_root,
                      "google3",
                      module_path_without_google3.replace(".", os.sep) + ".py",
                    )
                  )
                  possible_paths.append(
                    os.path.join(
                      google3_root,
                      "google3",
                      module_path_without_google3.replace(".", os.sep),
                      "__init__.py",
                    )
                  )
              else:
                # Try standard paths
                module_as_path = module_path.replace(".", os.sep)
                possible_paths.extend(
                  [
                    os.path.join(workspace_root, module_as_path + ".py"),
                    os.path.join(workspace_root, module_as_path, "__init__.py"),
                    os.path.join(
                      workspace_root, "third_party", module_as_path + ".py"
                    ),
                    os.path.join(
                      workspace_root,
                      "third_party",
                      module_as_path,
                      "__init__.py",
                    ),
                  ]
                )

              for path in possible_paths:
                if os.path.exists(path) and path not in local_files:
                  local_files.append(path)
                  break  # Found the file, stop trying shorter paths

              # If we found a file, don't try shorter module paths
              if any(os.path.exists(p) for p in possible_paths):
                break

    # Add files for module usage patterns (like attention_mla, quantizations)
    # Sort module_usage_patterns to ensure deterministic order
    if module_usage_patterns:
      for module_name in sorted(module_usage_patterns):
        # Try direct paths instead of walking the entire tree
        possible_paths = [
          os.path.join(base_dir, f"{module_name}.py"),
          os.path.join(base_dir, f"{module_name.lower()}.py"),
          os.path.join(workspace_root, f"{module_name}.py"),
          os.path.join(workspace_root, f"{module_name.lower()}.py"),
          os.path.join(workspace_root, "third_party", f"{module_name}.py"),
          os.path.join(
            workspace_root, "third_party", f"{module_name.lower()}.py"
          ),
        ]

        # Only check google3 if we might actually need it (don't call g4 g4d unnecessarily)
        # We'll add google3 paths lazily if needed in the main import loop

        # Only check packages that might actually contain this module
        # based on the current file's location or known patterns
        relevant_packages = []
        file_path_lower = self.filename.lower()

        # If current file is in a package, check that package
        for pkg in [
          "google3",
          "tpu_commons",
          "MaxText",
          "maxtext",
          "JetStream",
          "jetstream",
          "vllm",
          "whisper_jax",
          "torchprime",
          "maxdiffusion",
          "RecML",
        ]:
          if pkg.lower() in file_path_lower:
            relevant_packages.append(pkg)

        # If no relevant packages found, don't search package-specific locations
        for pkg in relevant_packages:
          possible_paths.extend(
            [
              os.path.join(workspace_root, pkg, f"{module_name}.py"),
              os.path.join(workspace_root, pkg, f"{module_name.lower()}.py"),
              os.path.join(
                workspace_root, "third_party", pkg, f"{module_name}.py"
              ),
              os.path.join(
                workspace_root, "third_party", pkg, f"{module_name.lower()}.py"
              ),
            ]
          )

        for path in possible_paths:
          if os.path.exists(path) and path not in local_files:
            local_files.append(path)

    for import_stmt in imports:
      # Parse the import statement to extract module names
      try:
        import_tree = ast.parse(import_stmt)
        for node in ast.walk(import_tree):
          if isinstance(node, ast.ImportFrom):
            if node.module:
              module_path = node.module

              # Handle relative imports (from .module import ...)
              if module_path.startswith("."):
                module_path = module_path[1:]  # Remove leading dot
                search_dir = base_dir
              else:
                search_dir = workspace_root

              # Check if this is a local package (like tpu_commons, MaxText, etc.)
              is_local_package = self._is_local_package(
                module_path, workspace_root
              )

              if is_local_package:
                # Try multiple search strategies for local files
                possible_paths = []

                # Handle google3 imports specially - ONLY call g4 g4d if we see google3
                if module_path.startswith("google3."):
                  google3_root = (
                    self._get_google3_root()
                  )  # Lazy: only called for google3 imports
                  if google3_root:
                    # For google3.foo.bar, look for /path/to/google3/foo/bar.py
                    module_path_without_google3 = module_path[
                      8:
                    ]  # Remove 'google3.'
                    possible_paths.append(
                      os.path.join(
                        google3_root,
                        "google3",
                        module_path_without_google3.replace(".", os.sep)
                        + ".py",
                      )
                    )
                    possible_paths.append(
                      os.path.join(
                        google3_root,
                        "google3",
                        module_path_without_google3.replace(".", os.sep),
                        "__init__.py",
                      )
                    )

                # Direct module path conversion
                possible_paths.append(
                  os.path.join(
                    search_dir, module_path.replace(".", os.sep) + ".py"
                  )
                )
                possible_paths.append(
                  os.path.join(
                    search_dir, module_path.replace(".", os.sep), "__init__.py"
                  )
                )

                # Search specifically in third_party directories
                for third_party_root in ["third_party"]:
                  third_party_path = os.path.join(
                    workspace_root,
                    third_party_root,
                    module_path.replace(".", os.sep) + ".py",
                  )
                  possible_paths.append(third_party_path)

                  third_party_pkg_path = os.path.join(
                    workspace_root,
                    third_party_root,
                    module_path.replace(".", os.sep),
                    "__init__.py",
                  )
                  possible_paths.append(third_party_pkg_path)

                  # Handle packages with duplicated directory structure (e.g., third_party/pkg/pkg/...)
                  # Extract the first component of the module path
                  first_component = module_path.split(".")[0]
                  duplicated_path = os.path.join(
                    workspace_root,
                    third_party_root,
                    first_component,
                    module_path.replace(".", os.sep) + ".py",
                  )
                  possible_paths.append(duplicated_path)

                  duplicated_pkg_path = os.path.join(
                    workspace_root,
                    third_party_root,
                    first_component,
                    module_path.replace(".", os.sep),
                    "__init__.py",
                  )
                  possible_paths.append(duplicated_pkg_path)

                # For packages with different structures, try common patterns
                package_parts = module_path.split(".")

                # Special handling for google3 - DON'T search in other packages!
                if module_path.startswith("google3."):
                  # For google3, we already handled it above, no need to check other packages
                  known_packages = []
                else:
                  # Determine which known packages are relevant for this module
                  # Only check packages that match the module prefix or current file location
                  all_known_packages = [
                    "tpu_commons",
                    "MaxText",
                    "maxtext",
                    "JetStream",
                    "jetstream",
                    "vllm",
                    "whisper_jax",
                    "torchprime",
                    "maxdiffusion",
                    "RecML",
                  ]

                  relevant_packages = []
                  # Check if module starts with a known package name
                  for pkg in all_known_packages:
                    if module_path.startswith(pkg + ".") or module_path == pkg:
                      relevant_packages.append(pkg)

                  # If no match, check if current file is within a known package
                  if not relevant_packages:
                    file_path_lower = self.filename.lower()
                    for pkg in all_known_packages:
                      if pkg.lower() in file_path_lower:
                        relevant_packages.append(pkg)

                  # Use relevant packages, or fall back to first component as hint
                  known_packages = (
                    relevant_packages
                    if relevant_packages
                    else [package_parts[0]]
                  )

                for i in range(len(package_parts)):
                  partial_path = os.path.join(*package_parts[: i + 1])
                  remaining_path = (
                    os.path.join(*package_parts[i + 1 :])
                    if i + 1 < len(package_parts)
                    else ""
                  )

                  # Try in workspace root
                  if remaining_path:
                    full_file_path = os.path.join(
                      workspace_root, partial_path, remaining_path + ".py"
                    )
                    if os.path.exists(full_file_path):
                      possible_paths.append(full_file_path)
                    full_pkg_path = os.path.join(
                      workspace_root,
                      partial_path,
                      remaining_path,
                      "__init__.py",
                    )
                    if os.path.exists(full_pkg_path):
                      possible_paths.append(full_pkg_path)
                  else:
                    target_file_path = os.path.join(
                      workspace_root, partial_path + ".py"
                    )
                    if os.path.exists(target_file_path):
                      possible_paths.append(target_file_path)

                  # Try in third_party
                  if remaining_path:
                    full_file_path = os.path.join(
                      workspace_root,
                      "third_party",
                      partial_path,
                      remaining_path + ".py",
                    )
                    if os.path.exists(full_file_path):
                      possible_paths.append(full_file_path)
                    full_pkg_path = os.path.join(
                      workspace_root,
                      "third_party",
                      partial_path,
                      remaining_path,
                      "__init__.py",
                    )
                    if os.path.exists(full_pkg_path):
                      possible_paths.append(full_pkg_path)
                  else:
                    target_file_path = os.path.join(
                      workspace_root, "third_party", partial_path + ".py"
                    )
                    if os.path.exists(target_file_path):
                      possible_paths.append(target_file_path)

                  # Try within known packages
                  for pkg in known_packages:
                    if remaining_path:
                      full_file_path = os.path.join(
                        workspace_root,
                        pkg,
                        partial_path,
                        remaining_path + ".py",
                      )
                      if os.path.exists(full_file_path):
                        possible_paths.append(full_file_path)
                      full_pkg_path = os.path.join(
                        workspace_root,
                        pkg,
                        partial_path,
                        remaining_path,
                        "__init__.py",
                      )
                      if os.path.exists(full_pkg_path):
                        possible_paths.append(full_pkg_path)

                      # Also try in third_party/pkg
                      full_file_path = os.path.join(
                        workspace_root,
                        "third_party",
                        pkg,
                        partial_path,
                        remaining_path + ".py",
                      )
                      if os.path.exists(full_file_path):
                        possible_paths.append(full_file_path)
                      full_pkg_path = os.path.join(
                        workspace_root,
                        "third_party",
                        pkg,
                        partial_path,
                        remaining_path,
                        "__init__.py",
                      )
                      if os.path.exists(full_pkg_path):
                        possible_paths.append(full_pkg_path)

                # Add any existing paths
                for path in possible_paths:
                  if os.path.exists(path):
                    local_files.append(path)
              else:
                # Not recognized as local package, check for mock implementations
                mock_file = self._find_mock_implementation(
                  module_path, workspace_root
                )
                if mock_file:
                  local_files.append(mock_file)

          elif isinstance(node, ast.Import):
            for alias in node.names:
              module_name = alias.name

              # Check if this is a local package
              is_local_package = self._is_local_package(
                module_name, workspace_root
              )

              if is_local_package:
                possible_paths = []

                # Special handling for google3 - resolve and skip other package checks
                if module_name.startswith("google3."):
                  google3_root = self._get_google3_root()
                  if google3_root:
                    module_path_without_google3 = module_name[
                      8:
                    ]  # Remove 'google3.'
                    possible_paths.append(
                      os.path.join(
                        google3_root,
                        "google3",
                        module_path_without_google3.replace(".", os.sep)
                        + ".py",
                      )
                    )
                    possible_paths.append(
                      os.path.join(
                        google3_root,
                        "google3",
                        module_path_without_google3.replace(".", os.sep),
                        "__init__.py",
                      )
                    )
                  # Don't search in other packages for google3 imports
                  known_packages = []
                else:
                  # Try to find local file for this import
                  possible_paths.append(
                    os.path.join(
                      base_dir, module_name.replace(".", os.sep) + ".py"
                    )
                  )
                  possible_paths.append(
                    os.path.join(
                      workspace_root, module_name.replace(".", os.sep) + ".py"
                    )
                  )
                  possible_paths.append(
                    os.path.join(
                      workspace_root,
                      module_name.replace(".", os.sep),
                      "__init__.py",
                    )
                  )

                  # Search in third_party
                  possible_paths.append(
                    os.path.join(
                      workspace_root,
                      "third_party",
                      module_name.replace(".", os.sep) + ".py",
                    )
                  )
                  possible_paths.append(
                    os.path.join(
                      workspace_root,
                      "third_party",
                      module_name.replace(".", os.sep),
                      "__init__.py",
                    )
                  )

                  # Only search in relevant package directories (not all packages)
                  # Determine relevant packages based on module name or current file location
                  all_known_packages = [
                    "tpu_commons",
                    "MaxText",
                    "maxtext",
                    "JetStream",
                    "jetstream",
                    "vllm",
                    "whisper_jax",
                    "torchprime",
                    "maxdiffusion",
                    "RecML",
                  ]

                  relevant_packages = []
                  for pkg in all_known_packages:
                    if module_name.startswith(pkg + ".") or module_name == pkg:
                      relevant_packages.append(pkg)

                  # If no match, check current file location
                  if not relevant_packages:
                    file_path_lower = self.filename.lower()
                    for pkg in all_known_packages:
                      if pkg.lower() in file_path_lower:
                        relevant_packages.append(pkg)
                        break  # Only add the one we're in

                  # Use relevant packages or just check the module's first component
                  known_packages = (
                    relevant_packages if relevant_packages else []
                  )

                module_as_path = module_name.replace(".", os.sep)
                for pkg in known_packages:
                  possible_paths.append(
                    os.path.join(workspace_root, pkg, module_as_path + ".py")
                  )
                  possible_paths.append(
                    os.path.join(
                      workspace_root, pkg, module_as_path, "__init__.py"
                    )
                  )
                  possible_paths.append(
                    os.path.join(
                      workspace_root, "third_party", pkg, module_as_path + ".py"
                    )
                  )
                  possible_paths.append(
                    os.path.join(
                      workspace_root,
                      "third_party",
                      pkg,
                      module_as_path,
                      "__init__.py",
                    )
                  )

                # Add any existing paths
                for path in possible_paths:
                  if os.path.exists(path):
                    local_files.append(path)
      except:
        # Skip imports we can't parse
        continue

    # Remove duplicates while preserving order, then sort for determinism
    seen = set()
    unique_local_files = []
    for f in local_files:
      if f not in seen:
        seen.add(f)
        unique_local_files.append(f)
    return sorted(unique_local_files)  # Sort for deterministic order

  def _find_workspace_root(self, start_dir: str) -> str:
    """Find the workspace root directory by looking for common markers."""
    current_dir = os.path.abspath(start_dir)

    # Look for common workspace markers
    markers = [
      ".git",
      ".vscode",
      "pyproject.toml",
      "setup.py",
      "requirements.txt",
    ]

    while current_dir != os.path.dirname(current_dir):  # Not at filesystem root
      for marker in markers:
        if os.path.exists(os.path.join(current_dir, marker)):
          return current_dir
      current_dir = os.path.dirname(current_dir)

    # If no markers found, return the start directory
    return start_dir

  def _get_google3_root(self) -> Optional[str]:
    """Get the google3 root directory by running 'g4 g4d'. Only called when needed and cached."""
    # Return cached value if already fetched
    if self._google3_root_checked:
      return self._google3_root_cache

    # Mark as checked so we don't try again
    self._google3_root_checked = True

    try:
      import subprocess

      result = subprocess.run(
        ["g4", "g4d"], capture_output=True, text=True, timeout=5
      )
      if result.returncode == 0:
        g4d_output = result.stdout.strip()
        # Remove the 'google3' suffix to get the parent directory
        if g4d_output.endswith("google3"):
          self._google3_root_cache = g4d_output[: -len("google3")].rstrip("/")
        elif g4d_output.endswith("/google3"):
          self._google3_root_cache = g4d_output[: -len("/google3")]
        else:
          self._google3_root_cache = None
        return self._google3_root_cache
      return None
    except Exception as e:
      self._google3_root_cache = None
      return None

  def _is_local_package(self, module_path: str, workspace_root: str) -> bool:
    """Check if a module path refers to a local package in the workspace."""

    # Check if this is a google3 import
    if module_path.startswith("google3."):
      return True

    # List of known local packages in common ML workspaces
    local_package_names = [
      "google3",
      "tpu_commons",
      "MaxText",
      "maxtext",
      "JetStream",
      "jetstream",
      "vllm",
      "whisper_jax",
      "torchprime",
      "maxdiffusion",
      "RecML",
    ]

    # Check if the module starts with any known local package
    for pkg_name in local_package_names:
      if module_path.startswith(pkg_name):
        return True

    # Check if there's actually a directory structure for this module in the workspace
    module_parts = module_path.split(".")

    # Try various combinations to see if this could be a local module
    for i in range(len(module_parts)):
      partial_path = os.path.join(*module_parts[: i + 1])

      # Check in common locations
      potential_locations = [
        os.path.join(workspace_root, partial_path),
        os.path.join(workspace_root, "third_party", partial_path),
      ]

      # Only check google3 root if the module could plausibly be in google3
      # This avoids calling g4 g4d for every single import
      if (
        module_path.startswith("google3") or "google3" in self.filename.lower()
      ):
        google3_root = self._get_google3_root()
        if google3_root:
          potential_locations.append(
            os.path.join(google3_root, "google3", partial_path)
          )

      for location in potential_locations:
        if os.path.exists(location):
          return True
        # Also check with .py extension
        if os.path.exists(location + ".py"):
          return True

    # If we can't find any evidence this is local, assume it's external
    return False

  def _find_mock_implementation(
    self, module_path: str, workspace_root: str
  ) -> Optional[str]:
    """Find mock implementations of external modules (e.g., vllm.logger -> tpu_commons.mock.vllm_logger)."""
    # Common mappings for mock implementations
    mock_mappings = {
      "vllm.logger": ["tpu_commons.mock.vllm_logger", "tpu_commons.logger"],
      "vllm.config": ["tpu_commons.mock.vllm_config_utils"],
      "vllm.envs": ["tpu_commons.mock.vllm_envs"],
      "vllm.logging": ["tpu_commons.mock.vllm_logging_utils"],
    }

    # Check if we have a known mock mapping
    if module_path in mock_mappings:
      for mock_module in mock_mappings[module_path]:
        # Try to find this mock module
        mock_file = self._find_module_file(mock_module, workspace_root)
        if mock_file:
          return mock_file

    # Try generic pattern: vllm.X -> tpu_commons.mock.vllm_X
    if module_path.startswith("vllm."):
      module_suffix = module_path.split("vllm.", 1)[1]
      mock_module = f"tpu_commons.mock.vllm_{module_suffix.replace('.', '_')}"
      mock_file = self._find_module_file(mock_module, workspace_root)
      if mock_file:
        return mock_file

    return None

  def _find_module_file(
    self, module_path: str, workspace_root: str
  ) -> Optional[str]:
    """Find the file path for a given module path."""
    module_as_path = module_path.replace(".", os.sep)

    # Try various common locations
    possible_paths = [
      os.path.join(workspace_root, module_as_path + ".py"),
      os.path.join(workspace_root, module_as_path, "__init__.py"),
      os.path.join(workspace_root, "third_party", module_as_path + ".py"),
      os.path.join(
        workspace_root, "third_party", module_as_path, "__init__.py"
      ),
    ]

    # For nested packages, also try with duplicated directory structure
    # e.g., tpu_commons.mock.vllm_logger -> third_party/tpu_commons/tpu_commons/mock/vllm_logger.py
    first_component = module_path.split(".")[0]
    possible_paths.extend(
      [
        os.path.join(
          workspace_root, "third_party", first_component, module_as_path + ".py"
        ),
        os.path.join(
          workspace_root,
          "third_party",
          first_component,
          module_as_path,
          "__init__.py",
        ),
      ]
    )

    for path in possible_paths:
      if os.path.exists(path):
        return path

    return None

  def _is_local_package_file(self, file_path: str) -> bool:
    """Check if a file belongs to a local package that should be aggressively extracted."""

    # List of known local package patterns
    local_package_patterns = [
      "google3",
      "tpu_commons",
      "MaxText",
      "maxtext",
      "JetStream",
      "jetstream",
      "vllm",
      "whisper_jax",
      "torchprime",
      "maxdiffusion",
      "RecML",
      "third_party",
    ]

    # Check if the file path contains any local package patterns
    for pattern in local_package_patterns:
      if pattern in file_path:
        return True

    return False

  def extract_from_local_file(
    self,
    file_path: str,
    needed_names: Set[str],
    module_attributes: Set[str] = None,
    external_imports: Set[str] = None,
  ) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Extract needed objects from a local file."""
    if module_attributes is None:
      module_attributes = set()
    if external_imports is None:
      external_imports = set()

    try:
      extractor = ObjectExtractor(file_path, debug=self.debug)

      # Collect all top-level definitions, including those in if TYPE_CHECKING blocks
      definitions = {}

      def collect_definitions_recursive(nodes, depth=0):
        """Recursively collect definitions, including from if blocks."""
        for node in nodes:
          if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
          elif isinstance(node, ast.Assign):
            for target in node.targets:
              if isinstance(target, ast.Name):
                definitions[target.id] = node
          elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
          ):
            definitions[node.target.id] = node
          # Recursively check inside if blocks (for typing.TYPE_CHECKING, etc.)
          elif isinstance(node, ast.If):
            # Check the if branch
            collect_definitions_recursive(node.body, depth + 1)
            # Check the else branch
            collect_definitions_recursive(node.orelse, depth + 1)

      # Start collecting from top-level nodes
      collect_definitions_recursive(extractor.tree.body)

      extracted_objects = []
      extracted_imports = []

      # Get the module name from file path
      module_name = os.path.splitext(os.path.basename(file_path))[0]

      # Extract attributes that are referenced via module.attribute patterns
      for attr_pattern in module_attributes:
        parts = attr_pattern.split(".")
        if len(parts) >= 2:
          # Check both exact module name match and partial matches
          first_part = parts[0]
          attr_name = parts[1]

          # Direct module name match or file path contains the module reference
          if (
            first_part == module_name or first_part in file_path
          ) and attr_name in definitions:
            node = definitions[attr_name]
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
              obj_source = extractor.get_object_source(attr_name)
              if obj_source:
                extracted_objects.append((attr_name, obj_source))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
              # Use get_object_source to handle multi-line assignments
              obj_source = extractor.get_object_source(attr_name)
              if obj_source:
                extracted_objects.append((attr_name, obj_source))

      # For known local packages, be very aggressive in extracting ALL dependencies
      is_local_package_file = self._is_local_package_file(file_path)

      if is_local_package_file:
        # Extract from the definitions dict (which includes TYPE_CHECKING blocks)
        for name, node in definitions.items():
          if isinstance(node, ast.Assign):
            # Extract all assignments that look like constants or are needed
            if (
              name in needed_names
              or isinstance(node.value, ast.Constant)
              or (
                hasattr(ast, "Str") and isinstance(node.value, ast.Str)
              )  # Python < 3.8 compatibility
              or (
                hasattr(ast, "Num") and isinstance(node.value, ast.Num)
              )  # Python < 3.8 compatibility
              or (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
              )
              or (name.isupper() or "_" in name)
            ):
              # Use get_object_source to properly handle multi-line assignments
              obj_source = extractor.get_object_source(name)
              if obj_source and not any(
                obj_name == name for obj_name, _ in extracted_objects
              ):
                extracted_objects.append((name, obj_source))
          elif isinstance(node, ast.AnnAssign):
            # Extract annotated assignments (type aliases) - use get_object_source for multi-line
            if name in needed_names or name.isupper():
              obj_source = extractor.get_object_source(name)
              if obj_source and not any(
                obj_name == name for obj_name, _ in extracted_objects
              ):
                extracted_objects.append((name, obj_source))
          elif isinstance(node, ast.ClassDef):
            # Extract all class definitions, especially enums, if needed
            if name in needed_names:
              obj_source = extractor.get_object_source(name)
              if obj_source and not any(
                obj_name == name for obj_name, _ in extracted_objects
              ):
                extracted_objects.append((name, obj_source))
          elif isinstance(node, ast.FunctionDef):
            # Extract function definitions if needed
            if name in needed_names:
              obj_source = extractor.get_object_source(name)
              if obj_source and not any(
                obj_name == name for obj_name, _ in extracted_objects
              ):
                extracted_objects.append((name, obj_source))

        # For local package files, extract imports that are still needed
        file_collector = ImportCollector()
        file_collector.visit(extractor.tree)

        # Only keep imports that are not from other local packages
        # But treat imports with mocks as external so they get processed
        for import_stmt in file_collector.imports:
          is_external_import = True
          has_mock_impl = False
          try:
            import_tree = ast.parse(import_stmt)
            for node in ast.walk(import_tree):
              if isinstance(node, ast.ImportFrom) and node.module:
                # First check if there's a mock implementation
                mock_file = extractor._find_mock_implementation(
                  node.module,
                  extractor._find_workspace_root(os.path.dirname(file_path)),
                )
                if mock_file:
                  has_mock_impl = True
                  is_external_import = (
                    True  # Treat as external so it gets checked for mocks
                  )
                elif extractor._is_local_package(
                  node.module,
                  extractor._find_workspace_root(os.path.dirname(file_path)),
                ):
                  is_external_import = False
                  break
              elif isinstance(node, ast.Import):
                for alias in node.names:
                  if extractor._is_local_package(
                    alias.name,
                    extractor._find_workspace_root(os.path.dirname(file_path)),
                  ):
                    is_external_import = False
                    break
          except:
            pass

          if is_external_import:
            extracted_imports.append(import_stmt)

      else:
        # Original logic for regular files
        # Extract needed definitions, but skip those already imported from external sources
        for name in needed_names:
          if name in external_imports:
            # Skip this name as it's already imported from an external source
            continue

          if name in definitions:
            node = definitions[name]
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
              obj_source = extractor.get_object_source(name)
              if obj_source and not any(
                obj_name == name for obj_name, _ in extracted_objects
              ):
                extracted_objects.append((name, obj_source))
                # Get imports needed by this object
                obj_imports = extractor.get_required_imports(name)
                extracted_imports.extend(obj_imports)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
              # Include all assignments and annotated assignments (constants, etc.)
              # Use get_object_source to properly handle multi-line assignments
              obj_source = extractor.get_object_source(name)
              if obj_source and not any(
                obj_name == name for obj_name, _ in extracted_objects
              ):
                extracted_objects.append((name, obj_source))

        # Also check for module.attribute patterns, but skip externally imported names
        for attr_pattern in module_attributes:
          if "." in attr_pattern:
            module_part, attr_part = attr_pattern.split(".", 1)
            if (
              module_part == module_name
              and attr_part in definitions
              and attr_part not in external_imports
            ):
              node = definitions[attr_part]
              if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                obj_source = extractor.get_object_source(attr_part)
                if obj_source:
                  # Check if we haven't already added this object
                  if not any(
                    name == attr_part for name, _ in extracted_objects
                  ):
                    extracted_objects.append((attr_part, obj_source))
                    # Get imports needed by this object
                    obj_imports = extractor.get_required_imports(attr_part)
                    extracted_imports.extend(obj_imports)
              elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                # Include all assignments - use get_object_source to handle multi-line
                obj_source = extractor.get_object_source(attr_part)
                if obj_source and not any(
                  name == attr_part for name, _ in extracted_objects
                ):
                  extracted_objects.append((attr_part, obj_source))

      return extracted_imports, extracted_objects
    except Exception as e:
      print(
        f"Warning: Could not extract from {file_path}: {e}", file=sys.stderr
      )
      return [], []

  def _import_provides_needed_names(
    self, import_stmt: str, needed_names: Set[str]
  ) -> bool:
    """Check if an import statement provides any of the needed names."""
    try:
      import_tree = ast.parse(import_stmt)
      import_collector = ImportCollector()
      import_collector.visit(import_tree)
      return bool(import_collector.defined_names & needed_names)
    except:
      return False

  def _is_complete_assignment(self, line: str) -> bool:
    """Check if a line contains a complete assignment (no unclosed parentheses, brackets, etc.)."""
    # Count opening and closing parentheses, brackets, braces
    paren_count = line.count("(") - line.count(")")
    bracket_count = line.count("[") - line.count("]")
    brace_count = line.count("{") - line.count("}")

    # If any are not balanced, it's an incomplete assignment
    if paren_count != 0 or bracket_count != 0 or brace_count != 0:
      return False

    # Also check for common incomplete patterns
    if line.endswith("(") or line.endswith("[") or line.endswith("{"):
      return False

    return True


def isolate_object(
  filename: str, object_name: str, max_depth: int = 5, debug: bool = False
) -> str:
  """
  Extract an object and all its dependencies from a Python file, including recursive imports.

  Args:
      filename: Path to the Python file
      object_name: Name of the class or function to extract
      max_depth: Maximum recursion depth for following imports
      debug: If True, print debug information including file paths being opened

  Returns:
      Complete standalone Python code as a string
  """
  extractor = ObjectExtractor(filename, debug=debug)

  # Get the main object source
  object_source = extractor.get_object_source(object_name)
  if not object_source:
    raise ValueError(f"Object '{object_name}' not found in {filename}")

  # Track processed files to avoid circular dependencies and duplicates
  processed_files = set()
  processed_file_needed_names = {}  # Track what names we've extracted from each file
  processed_objects = (
    set()
  )  # Track (object_name, source_content) to avoid duplicates
  processed_files.add(os.path.abspath(filename))

  # Collect all imports and objects recursively
  all_imports = []
  all_objects = []

  def add_unique_object(name: str, source: str, file_origin: str = ""):
    """Add an object only if it hasn't been added before."""
    # Normalize the source content for comparison
    source_lines = [line.strip() for line in source.split("\n") if line.strip()]
    source_key = tuple(source_lines)

    object_key = (name, source_key)
    if object_key not in processed_objects:
      processed_objects.add(object_key)
      display_name = f"{name} (from {file_origin})" if file_origin else name
      all_objects.append((display_name, source))
      return True
    return False

  def process_file_recursive(
    file_extractor: ObjectExtractor, target_obj: str, depth: int = 0
  ):
    if depth >= max_depth:
      return

    # Get workspace root for this file
    workspace_root = file_extractor._find_workspace_root(
      os.path.dirname(file_extractor.filename)
    )

    # Get required imports for this object
    imports = file_extractor.get_required_imports(target_obj)

    # Get additional objects this depends on
    additional_objects = file_extractor.get_additional_objects(target_obj)

    # Collect used names and module attributes from the target object
    object_collector = ImportCollector()
    target_node = file_extractor.find_object(target_obj)
    if target_node:
      object_collector.visit(target_node)

    # Also collect import aliases from the whole file to resolve aliased imports
    file_collector = ImportCollector()
    file_collector.visit(file_extractor.tree)

    # Merge import aliases from file-level into object collector
    object_collector.import_aliases.update(file_collector.import_aliases)

    # Now revisit the target node with the complete alias mapping to resolve attributes
    if target_node:
      # Re-visit to resolve aliased attributes with the complete import_aliases
      object_collector.visit(target_node)

    # Get module usage patterns to find additional local files
    module_usage_patterns = file_extractor.get_module_usage_patterns(target_obj)

    # Separate local and external imports
    local_files = file_extractor.get_local_import_files(
      imports, module_usage_patterns, object_collector.module_attributes
    )
    external_imports = []
    external_imported_names = set()

    # Process imports to separate local vs external and collect external names
    # Also check for mock implementations
    # Sort to ensure deterministic order
    for import_stmt in sorted(imports):
      is_local = False
      has_mock = False
      for local_file in local_files:
        # Check if this import statement refers to the local file
        try:
          import_tree = ast.parse(import_stmt)
          for node in ast.walk(import_tree):
            if isinstance(node, ast.ImportFrom) and node.module:
              # More comprehensive matching for local modules
              if self._is_local_package(
                node.module,
                file_extractor._find_workspace_root(
                  os.path.dirname(file_extractor.filename)
                ),
              ):
                is_local = True
                break
            elif isinstance(node, ast.Import):
              for alias in node.names:
                if self._is_local_package(
                  alias.name,
                  file_extractor._find_workspace_root(
                    os.path.dirname(file_extractor.filename)
                  ),
                ):
                  is_local = True
                  break
        except:
          continue
        if is_local:
          break

      if not is_local:
        # Check for mock implementations before adding as external import
        try:
          import_tree = ast.parse(import_stmt)
          for node in ast.walk(import_tree):
            if isinstance(node, ast.ImportFrom) and node.module:
              mock_file = file_extractor._find_mock_implementation(
                node.module, workspace_root
              )
              if mock_file:
                has_mock = True
                # Extract from mock file
                for imported_name in node.names:
                  if imported_name.name != "*":
                    try:
                      mock_extractor = ObjectExtractor(
                        mock_file, debug=file_extractor.debug
                      )
                      original_name = imported_name.name
                      alias_name = (
                        imported_name.asname
                        if imported_name.asname
                        else original_name
                      )
                      obj_source = mock_extractor.get_object_source(
                        original_name
                      )
                      if obj_source:
                        # Rename if aliased
                        if alias_name != original_name:
                          obj_source = re.sub(
                            r"\bdef\s+" + re.escape(original_name) + r"\b",
                            f"def {alias_name}",
                            obj_source,
                          )
                          obj_source = re.sub(
                            r"\bclass\s+" + re.escape(original_name) + r"\b",
                            f"class {alias_name}",
                            obj_source,
                          )
                        add_unique_object(
                          alias_name,
                          obj_source,
                          f"{os.path.basename(mock_file)} (mock)",
                        )

                        # Get imports needed by this mock object
                        try:
                          mock_imports = mock_extractor.get_required_imports(
                            original_name
                          )
                          for mock_import in mock_imports:
                            # Check if import is external or has its own mock
                            is_mock_import_external = True
                            try:
                              import_tree = ast.parse(mock_import)
                              for node in ast.walk(import_tree):
                                if (
                                  isinstance(node, ast.ImportFrom)
                                  and node.module
                                ):
                                  if mock_extractor._is_local_package(
                                    node.module, workspace_root
                                  ):
                                    is_mock_import_external = False
                                    break
                            except:
                              pass

                            if (
                              is_mock_import_external
                              and mock_import not in all_imports
                            ):
                              all_imports.append(mock_import)
                        except:
                          pass

                        # Get dependencies
                        try:
                          mock_additional = (
                            mock_extractor.get_additional_objects(original_name)
                          )
                          for add_name, add_source in mock_additional:
                            add_unique_object(
                              add_name,
                              add_source,
                              f"{os.path.basename(mock_file)} (mock)",
                            )

                            # Also get imports for each additional dependency
                            try:
                              add_imports = mock_extractor.get_required_imports(
                                add_name
                              )
                              for add_import in add_imports:
                                is_add_import_external = True
                                try:
                                  import_tree = ast.parse(add_import)
                                  for node in ast.walk(import_tree):
                                    if (
                                      isinstance(node, ast.ImportFrom)
                                      and node.module
                                    ):
                                      if mock_extractor._is_local_package(
                                        node.module, workspace_root
                                      ):
                                        is_add_import_external = False
                                        break
                                except:
                                  pass

                                if (
                                  is_add_import_external
                                  and add_import not in all_imports
                                ):
                                  all_imports.append(add_import)
                            except:
                              pass
                        except:
                          pass
                    except Exception as e:
                      print(
                        f"Warning: Could not extract {imported_name.name} from mock {mock_file}: {e}",
                        file=sys.stderr,
                      )
                break
        except:
          pass

        if not has_mock:
          external_imports.append(import_stmt)
          # Collect names from external imports
          try:
            import_tree = ast.parse(import_stmt)
            import_collector = ImportCollector()
            import_collector.visit(import_tree)
            external_imported_names.update(import_collector.defined_names)
          except:
            pass

    # Add external imports
    all_imports.extend(external_imports)

    # Add additional objects from same file and recursively get their dependencies
    objects_to_process = additional_objects.copy()
    processed_additional = set()

    while objects_to_process:
      obj_name, obj_source = objects_to_process.pop(0)
      if obj_name in processed_additional:
        continue
      processed_additional.add(obj_name)

      add_unique_object(obj_name, obj_source)

      # Get and add imports required by this additional object
      try:
        obj_imports = file_extractor.get_required_imports(obj_name)

        # Also collect module usage patterns for this object
        obj_node = file_extractor.find_object(obj_name)
        obj_module_patterns = set()
        obj_module_attributes = set()
        if obj_node:
          obj_collector = ImportCollector()
          obj_collector.visit(obj_node)

          # Merge file-level import aliases to resolve aliased imports in additional objects
          obj_collector.import_aliases.update(file_collector.import_aliases)

          # Re-visit to resolve aliases
          obj_collector.visit(obj_node)

          obj_module_patterns = file_extractor.get_module_usage_patterns(
            obj_name
          )
          obj_module_attributes = obj_collector.module_attributes

        # Get local files for this object's imports
        obj_local_files = file_extractor.get_local_import_files(
          obj_imports, obj_module_patterns, obj_module_attributes
        )

        for obj_import in obj_imports:
          # Check if it's a local import
          is_local_import = False
          try:
            import_tree = ast.parse(obj_import)
            for node in ast.walk(import_tree):
              if isinstance(node, ast.ImportFrom) and node.module:
                if file_extractor._is_local_package(
                  node.module,
                  file_extractor._find_workspace_root(
                    os.path.dirname(file_extractor.filename)
                  ),
                ):
                  is_local_import = True
                  break
              elif isinstance(node, ast.Import):
                for alias in node.names:
                  if file_extractor._is_local_package(
                    alias.name,
                    file_extractor._find_workspace_root(
                      os.path.dirname(file_extractor.filename)
                    ),
                  ):
                    is_local_import = True
                    break
          except:
            continue

          # If it's not a local import, add it to external imports
          if not is_local_import:
            all_imports.append(obj_import)
          else:
            # It's a local import - add to imports list so it gets processed
            if obj_import not in imports:
              imports.append(obj_import)
      except:
        pass  # Continue if we can't analyze imports

      # Recursively get dependencies of this additional object
      try:
        nested_additional = file_extractor.get_additional_objects(obj_name)
        for nested_name, nested_source in nested_additional:
          if nested_name not in processed_additional:
            objects_to_process.append((nested_name, nested_source))
      except:
        pass  # Continue if we can't analyze dependencies

      # Process local files for this additional object's imports
      for obj_local_file in obj_local_files:
        abs_obj_local_file = os.path.abspath(obj_local_file)
        if abs_obj_local_file not in processed_files:
          # Add to local_files so it gets processed in the main loop below
          if obj_local_file not in local_files:
            local_files.append(obj_local_file)

    # Process local imports recursively
    for local_file in local_files:
      abs_local_file = os.path.abspath(local_file)

      # Determine if we should process this file
      # Process if: (1) never processed before, OR (2) we have new needed names
      should_process = False
      if abs_local_file not in processed_files:
        should_process = True
      else:
        # File was processed before - check if we have new needed names
        # First collect the needed names for this file
        temp_needed_names = set()
        for import_stmt in imports:
          try:
            import_tree = ast.parse(import_stmt)
            for node in ast.walk(import_tree):
              if isinstance(node, ast.ImportFrom):
                if node.module and file_extractor._is_local_package(
                  node.module, workspace_root
                ):
                  module_as_path = node.module.replace(".", os.sep)
                  local_file_normalized = os.path.normpath(local_file)
                  if (
                    local_file_normalized.endswith(module_as_path + ".py")
                    or local_file_normalized.endswith(
                      os.path.join(module_as_path, "__init__.py")
                    )
                    or module_as_path in local_file_normalized
                  ):
                    for alias in node.names:
                      name = alias.asname if alias.asname else alias.name
                      if name != "*":
                        temp_needed_names.add(name)
          except:
            continue

        # Check if there are new names
        previously_needed = processed_file_needed_names.get(
          abs_local_file, set()
        )
        new_names = temp_needed_names - previously_needed
        if new_names:
          should_process = True

      if should_process:
        processed_files.add(abs_local_file)

        # Parse the import to find what names we need from THIS specific file
        needed_names = set()
        for import_stmt in imports:
          try:
            import_tree = ast.parse(import_stmt)
            for node in ast.walk(import_tree):
              if isinstance(node, ast.ImportFrom):
                # Check if this import statement refers to the current local file
                if node.module and file_extractor._is_local_package(
                  node.module, workspace_root
                ):
                  # Convert module path to file path and check if it matches current local_file
                  module_as_path = node.module.replace(".", os.sep)
                  local_file_normalized = os.path.normpath(local_file)

                  # Check if this import matches the local file
                  is_match = False

                  # Standard checks
                  if local_file_normalized.endswith(
                    module_as_path + ".py"
                  ) or local_file_normalized.endswith(
                    os.path.join(module_as_path, "__init__.py")
                  ):
                    is_match = True

                  # For google3 imports, also check without the google3 prefix
                  if not is_match and node.module.startswith("google3."):
                    # Remove google3. prefix and check again
                    module_without_google3 = node.module[
                      8:
                    ]  # Remove 'google3.'
                    module_as_path_no_g3 = module_without_google3.replace(
                      ".", os.sep
                    )
                    if local_file_normalized.endswith(
                      module_as_path_no_g3 + ".py"
                    ) or local_file_normalized.endswith(
                      os.path.join(module_as_path_no_g3, "__init__.py")
                    ):
                      is_match = True
                    # Also check if the file contains google3/ in the path
                    if (
                      not is_match
                      and "google3" + os.sep in local_file_normalized
                    ):
                      # Extract the part after google3/
                      google3_idx = local_file_normalized.find(
                        "google3" + os.sep
                      )
                      if google3_idx != -1:
                        path_after_google3 = local_file_normalized[
                          google3_idx + 8 :
                        ]  # Skip 'google3/'
                        if (
                          path_after_google3 == module_as_path_no_g3 + ".py"
                          or path_after_google3
                          == os.path.join(module_as_path_no_g3, "__init__.py")
                          or path_after_google3.startswith(
                            module_as_path_no_g3 + os.sep
                          )
                        ):
                          is_match = True

                  # Fallback: check if module_as_path is contained in the file path
                  if not is_match and module_as_path in local_file_normalized:
                    is_match = True

                  if is_match:
                    # This import is from the current local file
                    imported_names = [alias.name for alias in node.names]
                    for alias in node.names:
                      name = alias.asname if alias.asname else alias.name
                      if name != "*":
                        needed_names.add(name)
              elif isinstance(node, ast.Import):
                for alias in node.names:
                  if file_extractor._is_local_package(
                    alias.name, workspace_root
                  ):
                    # For direct imports, check if the module name relates to this file
                    module_as_path = alias.name.replace(".", os.sep)
                    local_file_normalized = os.path.normpath(local_file)

                    is_match = False

                    # Standard checks
                    if local_file_normalized.endswith(
                      module_as_path + ".py"
                    ) or local_file_normalized.endswith(
                      os.path.join(module_as_path, "__init__.py")
                    ):
                      is_match = True

                    # For google3 imports, also check without the google3 prefix
                    if not is_match and alias.name.startswith("google3."):
                      module_without_google3 = alias.name[
                        8:
                      ]  # Remove 'google3.'
                      module_as_path_no_g3 = module_without_google3.replace(
                        ".", os.sep
                      )
                      if local_file_normalized.endswith(
                        module_as_path_no_g3 + ".py"
                      ) or local_file_normalized.endswith(
                        os.path.join(module_as_path_no_g3, "__init__.py")
                      ):
                        is_match = True
                      # Also check if the file contains google3/ in the path
                      if (
                        not is_match
                        and "google3" + os.sep in local_file_normalized
                      ):
                        google3_idx = local_file_normalized.find(
                          "google3" + os.sep
                        )
                        if google3_idx != -1:
                          path_after_google3 = local_file_normalized[
                            google3_idx + 8 :
                          ]  # Skip 'google3/'
                          if (
                            path_after_google3 == module_as_path_no_g3 + ".py"
                            or path_after_google3
                            == os.path.join(module_as_path_no_g3, "__init__.py")
                            or path_after_google3.startswith(
                              module_as_path_no_g3 + os.sep
                            )
                          ):
                            is_match = True

                    # Fallback: check if module_as_path is contained in the file path
                    if not is_match and module_as_path in local_file_normalized:
                      is_match = True

                    if is_match:
                      name = alias.asname if alias.asname else alias.name
                      needed_names.add(name)
          except Exception as e:
            continue

        # For local packages, also include names used in object_collector.used_names
        # This helps capture constants and types that are referenced but not explicitly imported
        # Sort to ensure deterministic order
        needed_names.update(sorted(object_collector.used_names))

        # Also add module attributes that might be needed
        # Sort to ensure deterministic order
        for attr_pattern in sorted(object_collector.module_attributes):
          if "." in attr_pattern:
            module_part, attr_part = attr_pattern.split(".", 1)
            needed_names.add(attr_part)

        # Extract needed objects from the local file
        try:
          local_extractor = ObjectExtractor(
            local_file, debug=file_extractor.debug
          )
          local_imports, local_objects = (
            local_extractor.extract_from_local_file(
              local_file,
              needed_names,
              object_collector.module_attributes,
              external_imported_names,
            )
          )

          # Track what names we extracted from this file
          extracted_names_from_file = set()
          for obj_name, _ in local_objects:
            # Extract just the base name (without file annotation)
            base_name = (
              obj_name.split(" (from ")[0]
              if " (from " in obj_name
              else obj_name
            )
            extracted_names_from_file.add(base_name)

          # Update the tracking dictionary
          if abs_local_file in processed_file_needed_names:
            processed_file_needed_names[abs_local_file].update(
              extracted_names_from_file
            )
          else:
            processed_file_needed_names[abs_local_file] = (
              extracted_names_from_file
            )

          # Add imports from local file (filter external vs local)
          for local_import in local_imports:
            # Only add if it's truly external (not another local package import)
            is_external = True
            has_mock = False
            mock_extractions = []  # Track (original_name, alias_name) tuples
            try:
              import_tree = ast.parse(local_import)
              for node in ast.walk(import_tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                  # Check for mock implementations FIRST
                  mock_file = local_extractor._find_mock_implementation(
                    node.module, workspace_root
                  )
                  if mock_file:
                    has_mock = True
                    # Extract the needed names from this mock file
                    for imported_name in node.names:
                      if imported_name.name != "*":
                        original_name = imported_name.name
                        alias_name = (
                          imported_name.asname
                          if imported_name.asname
                          else original_name
                        )
                        mock_extractions.append(
                          (mock_file, original_name, alias_name)
                        )
                  elif local_extractor._is_local_package(
                    node.module, workspace_root
                  ):
                    is_external = False
                    break
                elif isinstance(node, ast.Import):
                  for alias in node.names:
                    if local_extractor._is_local_package(
                      alias.name, workspace_root
                    ):
                      is_external = False
                      break
            except Exception as e:
              pass

            # Process mock extractions
            if mock_extractions:
              for mock_file, original_name, alias_name in mock_extractions:
                try:
                  mock_extractor = ObjectExtractor(
                    mock_file, debug=file_extractor.debug
                  )
                  obj_source = mock_extractor.get_object_source(original_name)
                  if obj_source:
                    # If there's an alias, we need to rename the function/class in the source
                    if alias_name != original_name:
                      # Simple replacement for function/class names
                      obj_source = re.sub(
                        r"\bdef\s+" + re.escape(original_name) + r"\b",
                        f"def {alias_name}",
                        obj_source,
                      )
                      obj_source = re.sub(
                        r"\bclass\s+" + re.escape(original_name) + r"\b",
                        f"class {alias_name}",
                        obj_source,
                      )
                    add_unique_object(
                      alias_name,
                      obj_source,
                      f"{os.path.basename(mock_file)} (mock)",
                    )

                    # Get imports needed by this mock object
                    try:
                      mock_imports = mock_extractor.get_required_imports(
                        original_name
                      )
                      for mock_import in mock_imports:
                        # Check if import is external or has its own mock
                        is_mock_import_external = True
                        try:
                          import_tree = ast.parse(mock_import)
                          for node in ast.walk(import_tree):
                            if isinstance(node, ast.ImportFrom) and node.module:
                              if mock_extractor._is_local_package(
                                node.module, workspace_root
                              ):
                                is_mock_import_external = False
                                break
                        except:
                          pass

                        if (
                          is_mock_import_external
                          and mock_import not in all_imports
                        ):
                          all_imports.append(mock_import)
                    except:
                      pass

                    # Also get dependencies of this mock object
                    try:
                      mock_additional = mock_extractor.get_additional_objects(
                        original_name
                      )
                      for add_name, add_source in mock_additional:
                        add_unique_object(
                          add_name,
                          add_source,
                          f"{os.path.basename(mock_file)} (mock)",
                        )

                        # Also get imports for each additional dependency
                        try:
                          add_imports = mock_extractor.get_required_imports(
                            add_name
                          )
                          for add_import in add_imports:
                            is_add_import_external = True
                            try:
                              import_tree = ast.parse(add_import)
                              for node in ast.walk(import_tree):
                                if (
                                  isinstance(node, ast.ImportFrom)
                                  and node.module
                                ):
                                  if mock_extractor._is_local_package(
                                    node.module, workspace_root
                                  ):
                                    is_add_import_external = False
                                    break
                            except:
                              pass

                            if (
                              is_add_import_external
                              and add_import not in all_imports
                            ):
                              all_imports.append(add_import)
                        except:
                          pass
                    except:
                      pass
                except Exception as e:
                  print(
                    f"Warning: Could not extract {original_name} from mock {mock_file}: {e}",
                    file=sys.stderr,
                  )

            if is_external and not has_mock:
              all_imports.append(local_import)

          # Add objects from local file
          for obj_name, obj_source in local_objects:
            add_unique_object(
              obj_name, obj_source, os.path.basename(local_file)
            )

            # Recursively process dependencies of this extracted object within the same file
            try:
              obj_additional = local_extractor.get_additional_objects(obj_name)
              for add_name, add_source in obj_additional:
                add_unique_object(
                  add_name, add_source, os.path.basename(local_file)
                )
            except:
              pass  # Continue if we can't analyze dependencies

            # Recursively process this object to find more dependencies
            try:
              process_file_recursive(local_extractor, obj_name, depth + 1)
            except Exception as e:
              pass  # Continue if we can't process recursively

        except Exception as e:
          print(
            f"Warning: Could not process {local_file}: {e}", file=sys.stderr
          )

  # Start the recursive processing
  process_file_recursive(extractor, object_name)

  # Remove duplicate imports
  unique_imports = []
  seen_imports = set()
  # Sort to ensure deterministic order
  for imp in sorted(all_imports):
    if imp not in seen_imports:
      unique_imports.append(imp)
      seen_imports.add(imp)

  # Filter out imports that we're already providing as extracted objects
  extracted_object_names = set()
  extracted_from_modules = {}  # Track which module each extracted object came from

  # Include the main object being isolated
  extracted_object_names.add(object_name)
  extracted_from_modules[object_name] = os.path.basename(filename)

  # Include all additional extracted objects
  for name, _ in all_objects:
    # Extract the base name (without the "(from file.py)" part)
    base_name = name.split(" (from ")[0]
    extracted_object_names.add(base_name)

    # Track which file this object came from
    if " (from " in name:
      file_part = name.split(" (from ")[1].rstrip(")")
      extracted_from_modules[base_name] = file_part

  filtered_imports = []
  # Sort to ensure deterministic order
  for import_stmt in sorted(unique_imports):
    # Parse the import to see what names it imports and from which module
    try:
      import_tree = ast.parse(import_stmt)
      import_collector = ImportCollector()
      import_collector.visit(import_tree)

      # Check both alias names and original names against extracted objects
      imported_alias_names = (
        import_collector.defined_names
      )  # The names available after import (aliases)

      # Also collect the original imported names (before aliasing)
      original_imported_names = set()
      for node in ast.walk(import_tree):
        if isinstance(node, ast.ImportFrom):
          for alias in node.names:
            original_imported_names.add(alias.name)  # Original name, not alias
        elif isinstance(node, ast.Import):
          for alias in node.names:
            original_imported_names.add(alias.name)  # Original name, not alias

      # Check for conflicts with either alias names or original names
      alias_conflicts = imported_alias_names & extracted_object_names
      original_conflicts = original_imported_names & extracted_object_names

      if not alias_conflicts and not original_conflicts:
        # No conflicts, keep the entire import
        filtered_imports.append(import_stmt)
      else:
        # There are conflicts - filter out conflicting imports
        for node in ast.walk(import_tree):
          if isinstance(node, ast.ImportFrom):
            if node.module:
              # Filter this import, but reconstruct with remaining names
              remaining_aliases = []
              for alias in node.names:
                alias_name = alias.asname if alias.asname else alias.name
                original_name = alias.name

                # Keep this import if neither the alias nor original name conflicts
                if (
                  alias_name not in extracted_object_names
                  and original_name not in extracted_object_names
                ):
                  remaining_aliases.append(
                    f"{alias.name}"
                    + (f" as {alias.asname}" if alias.asname else "")
                  )

              if remaining_aliases:
                filtered_imports.append(
                  f"from {node.module} import {', '.join(remaining_aliases)}"
                )
              # If no remaining names, the entire import is filtered out
            else:
              # Keep imports without a module (shouldn't happen, but be safe)
              filtered_imports.append(import_stmt)

          elif isinstance(node, ast.Import):
            # For regular imports, filter out conflicting names
            remaining_aliases = []
            for alias in node.names:
              name = alias.asname if alias.asname else alias.name
              if name not in extracted_object_names:
                remaining_aliases.append(
                  f"import {alias.name}"
                  + (f" as {alias.asname}" if alias.asname else "")
                )
            if remaining_aliases:
              filtered_imports.extend(remaining_aliases)

    except:
      # If we can't parse the import, keep it to be safe
      filtered_imports.append(import_stmt)

  # Build the output
  output_lines = []

  # Fix module attribute references in all object sources
  fixed_object_source = fix_module_attribute_references(
    object_source, extracted_object_names
  )

  # Also fix references in all extracted objects
  fixed_all_objects = []
  for name, source in all_objects:
    fixed_source = fix_module_attribute_references(
      source, extracted_object_names
    )
    fixed_all_objects.append((name, fixed_source))

  # After fixing references, check if any module imports are still needed
  final_imports = filter_unused_module_imports(
    filtered_imports, fixed_object_source, fixed_all_objects
  )

  # Sort objects by dependencies (topological sort)
  sorted_objects = topologically_sort_objects(fixed_all_objects)

  # Rebuild output with final imports
  output_lines = []

  # Add header comment
  output_lines.append(
    f"# Isolated {object_name} from {filename} (with recursive imports)"
  )
  output_lines.append("")

  # Add final imports, filtering out any remaining google3 imports and consolidating duplicates
  if final_imports:
    # First pass: remove google3 imports
    non_google3_imports = []
    for import_stmt in final_imports:
      # Skip any imports from google3
      if (
        "from google3." in import_stmt
        or "import google3." in import_stmt
        or import_stmt.strip().startswith("import google3")
      ):
        continue
      non_google3_imports.append(import_stmt)

    # Second pass: consolidate imports from the same module and remove duplicate names
    from_imports = {}  # module -> set of (name, alias) tuples
    direct_imports = []  # List of direct "import X" statements

    for import_stmt in non_google3_imports:
      try:
        import_tree = ast.parse(import_stmt)
        for node in ast.walk(import_tree):
          if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in from_imports:
              from_imports[module] = set()

            for alias in node.names:
              # Store as (original_name, alias_name or None)
              alias_name = alias.asname if alias.asname else None
              from_imports[module].add((alias.name, alias_name))

          elif isinstance(node, ast.Import):
            # Keep direct imports as-is (they're typically unique)
            if import_stmt not in direct_imports:
              direct_imports.append(import_stmt)
      except:
        # If we can't parse it, keep it as-is
        if import_stmt not in direct_imports:
          direct_imports.append(import_stmt)

    # Reconstruct consolidated imports
    cleaned_imports = []

    # Add consolidated from imports
    for module in sorted(from_imports.keys()):
      names = sorted(from_imports[module])
      name_parts = []
      for original_name, alias_name in names:
        if alias_name:
          name_parts.append(f"{original_name} as {alias_name}")
        else:
          name_parts.append(original_name)

      import_stmt = f"from {module} import {', '.join(name_parts)}"
      cleaned_imports.append(import_stmt)

    # Add direct imports
    cleaned_imports.extend(direct_imports)

    if cleaned_imports:
      output_lines.extend(cleaned_imports)
      output_lines.append("")

  # Add additional objects
  for name, source in sorted_objects:
    output_lines.append(f"# Additional dependency: {name}")
    output_lines.append(source)
    output_lines.append("")

  # Add the main object
  output_lines.append(f"# Main object: {object_name}")
  output_lines.append(fixed_object_source)

  return "\n".join(output_lines)


def topologically_sort_objects(
  objects: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
  """Sort objects so that dependencies come before the objects that use them."""

  # Build a dependency graph
  # Extract just the base names for analysis
  object_names = {}
  for name, source in objects:
    base_name = name.split(" (from ")[0]
    object_names[base_name] = (name, source)

  # Find dependencies for each object
  dependencies = {}
  for base_name, (full_name, source) in object_names.items():
    deps = set()
    try:
      tree = ast.parse(source)

      class DependencyFinder(ast.NodeVisitor):
        def __init__(self):
          self.local_names = set()  # Track locally defined names

        def visit_FunctionDef(self, node):
          # Add parameters as local names
          for arg in node.args.args:
            self.local_names.add(arg.arg)
          if node.args.vararg:
            self.local_names.add(node.args.vararg.arg)
          if node.args.kwarg:
            self.local_names.add(node.args.kwarg.arg)
          for arg in node.args.kwonlyargs:
            self.local_names.add(arg.arg)
          # Visit the body
          self.generic_visit(node)

        def visit_Name(self, node):
          if isinstance(node.ctx, ast.Load):
            # This name is being used, check if it's one of our extracted objects
            # and not a local variable/parameter
            if (
              node.id in object_names
              and node.id != base_name
              and node.id not in self.local_names
            ):
              deps.add(node.id)
          elif isinstance(node.ctx, ast.Store):
            # This name is being defined locally
            self.local_names.add(node.id)
          self.generic_visit(node)

        def visit_Attribute(self, node):
          # For type annotations like model_config: ModelConfig
          if (
            isinstance(node.value, ast.Name)
            and node.value.id in object_names
            and node.value.id not in self.local_names
          ):
            deps.add(node.value.id)
          self.generic_visit(node)

      finder = DependencyFinder()
      finder.visit(tree)
    except:
      pass  # If we can't parse, assume no dependencies

    dependencies[base_name] = deps

  # Topological sort using Kahn's algorithm
  sorted_names = []
  in_degree = {name: 0 for name in object_names}

  # Calculate in-degrees
  for name, deps in dependencies.items():
    for dep in deps:
      if dep in in_degree:  # Only count dependencies that are in our object set
        in_degree[name] += 1

  # Queue of nodes with no dependencies
  queue = [name for name, degree in in_degree.items() if degree == 0]

  while queue:
    # Sort to ensure deterministic order when there are multiple options
    queue.sort()
    current = queue.pop(0)
    sorted_names.append(current)

    # Reduce in-degree for dependents
    for name, deps in dependencies.items():
      if current in deps and name not in sorted_names:
        in_degree[name] -= 1
        if in_degree[name] == 0:
          queue.append(name)

  # If there are cycles or remaining nodes, add them at the end
  remaining = [name for name in object_names if name not in sorted_names]
  sorted_names.extend(sorted(remaining))

  # Return objects in sorted order
  result = []
  for name in sorted_names:
    if name in object_names:
      result.append(object_names[name])

  return result


def filter_unused_module_imports(
  imports: List[str], main_source: str, all_objects: List[Tuple[str, str]]
) -> List[str]:
  """Filter out module imports that are no longer used after fixing attribute references."""
  used_names = set()

  # Collect all source code to check
  all_source_code = main_source + "\n"
  for _, obj_source in all_objects:
    all_source_code += obj_source + "\n"

  # Check which names are still referenced
  try:
    tree = ast.parse(all_source_code)

    class NameUsageFinder(ast.NodeVisitor):
      def visit_Attribute(self, node):
        # Track module.attribute usage (e.g., os.path, jnp.array)
        if isinstance(node.value, ast.Name):
          used_names.add(node.value.id)
        self.generic_visit(node)

      def visit_Name(self, node):
        # Track direct name usage
        if isinstance(node.ctx, ast.Load):
          used_names.add(node.id)
        self.generic_visit(node)

    finder = NameUsageFinder()
    finder.visit(tree)

  except Exception as e:
    print(f"Warning: Could not analyze name usage: {e}", file=sys.stderr)
    return imports  # Return all imports if we can't analyze

  # Filter imports to only keep those that are still used
  final_imports = []
  # Sort to ensure deterministic order
  for import_stmt in sorted(imports):
    try:
      import_tree = ast.parse(import_stmt)

      # Collect what names this import provides
      provided_names = set()
      for node in ast.walk(import_tree):
        if isinstance(node, ast.ImportFrom):
          for alias in node.names:
            # The name available after import is the alias (if present) or original name
            name = alias.asname if alias.asname else alias.name
            provided_names.add(name)
        elif isinstance(node, ast.Import):
          for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            provided_names.add(name)

      # Check if any provided names are used
      if provided_names & used_names:
        # For "from X import Y, Z" statements, filter individual names
        for node in ast.walk(import_tree):
          if isinstance(node, ast.ImportFrom):
            # Filter to only used names
            used_aliases = []
            for alias in node.names:
              name = alias.asname if alias.asname else alias.name
              if name in used_names:
                if alias.asname:
                  used_aliases.append(f"{alias.name} as {alias.asname}")
                else:
                  used_aliases.append(alias.name)

            if used_aliases:
              if node.module:
                final_imports.append(
                  f"from {node.module} import {', '.join(sorted(used_aliases))}"
                )
              else:
                final_imports.append(
                  f"from . import {', '.join(sorted(used_aliases))}"
                )
          elif isinstance(node, ast.Import):
            # For regular imports, keep if used
            for alias in node.names:
              name = alias.asname if alias.asname else alias.name
              if name in used_names:
                if alias.asname:
                  final_imports.append(f"import {alias.name} as {alias.asname}")
                else:
                  final_imports.append(f"import {alias.name}")

    except Exception:
      # If we can't parse, keep the import to be safe
      final_imports.append(import_stmt)

  return final_imports


def fix_module_attribute_references(
  source_code: str, extracted_names: Set[str]
) -> str:
  """
  Fix module.attribute references when the attribute has been extracted to the same scope.
  For example, change test_helper.helper_function() to helper_function() if helper_function is extracted.
  But avoid replacing module.attribute in assignment statements where we're assigning from module.attribute to the same attribute name.
  """
  try:
    tree = ast.parse(source_code)

    class AttributeReferenceFixer(ast.NodeTransformer):
      def visit_Assign(self, node):
        # For assignment nodes, we need to be careful not to replace the right-hand side
        # if it's assigning from module.attribute to the same attribute name
        # e.g., with_sharding_constraint = nn_partitioning.with_sharding_constraint

        # Check if this is a simple assignment with one target
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
          target_name = node.targets[0].id

          # Check if the value is a module.attribute where attribute matches the target
          if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.attr == target_name
            and target_name in extracted_names
          ):
            # Don't transform this assignment - keep the original module.attribute reference
            return node

        # For other assignments, apply normal transformation
        return self.generic_visit(node)

      def visit_Attribute(self, node):
        # Check if this is a module.attribute pattern where attribute is extracted
        if isinstance(node.value, ast.Name) and node.attr in extracted_names:
          # Replace module.attribute with just attribute (as a Name node)
          return ast.Name(id=node.attr, ctx=node.ctx)
        return self.generic_visit(node)

    fixer = AttributeReferenceFixer()
    fixed_tree = fixer.visit(tree)

    # Convert back to source code
    try:
      import astor

      return astor.to_source(fixed_tree).strip()
    except ImportError:
      # astor not available, fall back to simple string replacement
      print(
        "Warning: astor not available, using simple string replacement",
        file=sys.stderr,
      )
      return simple_fix_module_references(source_code, extracted_names)

  except Exception as e:
    print(
      f"Warning: Could not parse source for fixing references: {e}",
      file=sys.stderr,
    )
    return simple_fix_module_references(source_code, extracted_names)


def simple_fix_module_references(
  source_code: str, extracted_names: Set[str]
) -> str:
  """Simple string-based replacement for module.attribute references."""
  import re

  fixed_code = source_code
  # Sort to ensure deterministic order
  for name in sorted(extracted_names):
    # More sophisticated pattern to avoid replacing assignment statements
    # where we're assigning from module.name to name
    # Pattern: Don't replace if it's "name = something.name"
    assignment_pattern = (
      r"^(\s*" + re.escape(name) + r"\s*=\s*)\w+\." + re.escape(name) + r"\b"
    )

    # Replace patterns like "module.name" with just "name", but not in assignment statements
    # Split by lines to handle assignments carefully
    lines = fixed_code.split("\n")
    for i, line in enumerate(lines):
      # Check if this line is an assignment from module.name to name
      if re.match(assignment_pattern, line.strip()):
        # Don't modify this line
        continue
      else:
        # Apply normal replacement
        pattern = r"\b\w+\." + re.escape(name) + r"\b"
        lines[i] = re.sub(pattern, name, line)

    fixed_code = "\n".join(lines)

  return fixed_code


def main():
  """Command line interface for the isolate_object function."""
  parser = argparse.ArgumentParser(
    description="Isolate an object from a Python file with all dependencies (including recursive imports)"
  )
  parser.add_argument("filename", help="Path to the Python file")
  parser.add_argument(
    "object_name", help="Name of the class or function to extract"
  )
  parser.add_argument("-o", "--output", help="Output file (default: stdout)")
  parser.add_argument(
    "-d",
    "--max-depth",
    type=int,
    default=5,
    help="Maximum recursion depth for following imports (default: 5)",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Show debug information including exact paths of files being opened",
  )

  args = parser.parse_args()

  try:
    result = isolate_object(
      args.filename, args.object_name, args.max_depth, debug=args.debug
    )

    if args.output:
      with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)
      print(f"Isolated {args.object_name} written to {args.output}")
    else:
      print(result)

  except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
