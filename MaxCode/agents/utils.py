"""Utility functions and classes for agents."""

import ast
import collections
from collections.abc import Mapping, Set
import enum
import logging
import os
import pathlib
from typing import Dict


class AgentDomain(enum.Enum):
  """The high-level domain of an agent's expertise."""

  MIGRATION = "migration"
  KERNEL = "kernel"
  EVALUATION = "evaluation"


class AgentType(enum.Enum):
  """The type of agent."""

  PRIMARY = "primary"
  PYTORCH_TO_JAX_SINGLE_FILE = "pytorch_to_jax_single_file"
  PYTORCH_TO_JAX_REPO = "pytorch_to_jax_repo"
  HF_TO_JAX_SINGLE_FILE = "hf_to_jax_single_file"
  MODEL_CONVERSION = "model_conversion"


def get_module_imports(code: str) -> set[str]:
  """Parses python code and returns a set of modules that are imported.

  E.g., 'import a.b' adds 'a.b', 'from x.y import z' adds 'x.y',
  'from . import m' adds '.m', 'from .n import o' adds '.n'.

  Args:
    code: The Python source code to parse.

  Returns:
    A set of imported module strings.
  """
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return set()
  imports: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for n in node.names:
        imports.add(n.name)
    elif isinstance(node, ast.ImportFrom):
      if node.module:
        imports.add("." * node.level + node.module)
      else:
        # Case: from . import foo, bar
        for n in node.names:
          imports.add("." * node.level + n.name)
  return imports


def build_dependency_graph(repo_path: str) -> Dict[str, set[str]]:
  """Builds a dependency graph of python files in a repo based on imports.

  The graph is an adjacency list where keys are file paths and values are
  sets of file paths of dependencies. File paths are relative to repo_path.
  Dependencies are identified by parsing import statements in each file.
  It resolves:
  - relative imports (.mod, ..mod)
  - absolute imports assuming repo_path is the root for resolution.

  Args:
      repo_path: Path to the repository root directory.

  Returns:
      A dictionary representing the dependency graph.
  """
  repo_root = pathlib.Path(repo_path).resolve()
  py_files_abs_set = set()
  for root, _, files in os.walk(repo_root):
    for filename in files:
      if filename.endswith(".py"):
        py_files_abs_set.add(pathlib.Path(root, filename).resolve())

  file_imports: Dict[pathlib.Path, set[str]] = {}
  for abs_path in py_files_abs_set:
    try:
      with open(abs_path, "r") as f:
        code = f.read()
      file_imports[abs_path] = get_module_imports(code)
    except OSError:
      # Catch file-related errors like FileNotFoundError, PermissionError, etc.
      file_imports[abs_path] = set()

  def get_module_potential_paths(
      base_path: pathlib.Path, module_parts: list[str]
  ) -> list[pathlib.Path]:
    mod_path = base_path
    for part in module_parts:
      mod_path = mod_path / part

    potential = []
    p = mod_path.with_suffix(".py")
    if p in py_files_abs_set:
      potential.append(p)
    p = mod_path / "__init__.py"
    if p in py_files_abs_set:
      potential.append(p)
    return potential

  graph_abs: Dict[pathlib.Path, set[pathlib.Path]] = {
      p: set() for p in py_files_abs_set
  }
  for p_abs, imports in file_imports.items():
    for imp in imports:
      if imp.startswith("."):
        level = 0
        while imp[level] == ".":
          level += 1
        imp_module = imp[level:]

        base_dir = p_abs.parent
        for _ in range(level - 1):
          base_dir = base_dir.parent

        resolved_deps = get_module_potential_paths(
            base_dir, imp_module.split(".")
        )
      else:  # absolute import like 'pkg.mod' or 'os'
        resolved_deps = get_module_potential_paths(repo_root, imp.split("."))

      for dep_path in resolved_deps:
        if dep_path in graph_abs:
          graph_abs[p_abs].add(dep_path)

  graph_rel: Dict[str, set[str]] = {}
  for p_abs, deps in graph_abs.items():
    try:
      p_rel = p_abs.relative_to(repo_root).as_posix()
      graph_rel[p_rel] = {d.relative_to(repo_root).as_posix() for d in deps}
    except ValueError:
      # handle files outside repo_root if symlinks are involved.
      continue

  return graph_rel


def is_model_file(code: str, file_path: str) -> bool:
  """Detects whether code contains a torch.nn.Module subclass definition."""
  try:
    tree = ast.parse(code)
  except SyntaxError as e:
    logging.warning("SyntaxError in file %s: %s", file_path, e)
    return False
  for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
      for base_node in node.bases:
        # Match nn.Module, torch.nn.Module, Module
        if isinstance(base_node, ast.Attribute):
          if base_node.attr == "Module":
            return True
        elif isinstance(base_node, ast.Name):
          if base_node.id == "Module":
            return True
  return False


def topological_sort(graph: Mapping[str, Set[str]]) -> list[str]:
  """Returns files in dependency order (dependencies first) using Kahn's algorithm."""
  in_degree = {node: 0 for node in graph}
  for node, deps in graph.items():
    for dep in deps:
      if dep in in_degree:
        in_degree[node] += 1

  queue = collections.deque(node for node, deg in in_degree.items() if deg == 0)
  result = []

  while queue:
    node = queue.popleft()
    result.append(node)
    # Find nodes that depend on this one and decrement their in-degree
    for other, deps in graph.items():
      if node in deps:
        in_degree[other] -= 1
        if in_degree[other] == 0:
          queue.append(other)

  # Append any remaining nodes (cycles) to avoid dropping files
  for node in graph:
    if node not in result:
      result.append(node)

  return result
