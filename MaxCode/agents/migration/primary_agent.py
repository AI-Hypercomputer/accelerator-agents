"""Primary orchestration agent for repository migration."""

import ast
import os
from collections import deque
from typing import Any, Dict, List, Set

import models
from agents import base
from agents import utils
from agents.migration import model_conversion_agent
from agents.migration import single_file_agent
from rag import rag_agent


def _is_model_file(code: str) -> bool:
  """Detects whether code contains a torch.nn.Module subclass definition."""
  try:
    tree = ast.parse(code)
  except SyntaxError:
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


def _topological_sort(graph: Dict[str, Set[str]]) -> List[str]:
  """Returns files in dependency order (dependencies first) using Kahn's algorithm."""
  in_degree = {node: 0 for node in graph}
  for node, deps in graph.items():
    for dep in deps:
      if dep in in_degree:
        in_degree[node] += 1

  queue = deque(node for node, deg in in_degree.items() if deg == 0)
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


class PrimaryAgent(base.Agent):
  """Primary orchestration agent for repository migration."""

  def __init__(self, model: Any, api_key: str | None = None):
    """Initializes the agent."""
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.PRIMARY,
    )
    self._rag_agent = rag_agent.RAGAgent(
        model,
        embedding_model_name=models.EmbeddingModel.GEMINI_EMBEDDING_001,
        api_key=api_key,
    )
    self._single_file_agent = single_file_agent.PytorchToJaxSingleFileAgent(
        model, self._rag_agent
    )
    self._model_conversion_agent = model_conversion_agent.ModelConversionAgent(
        model, self._rag_agent
    )

  def _convert_file(self, pytorch_code: str) -> str:
    """Routes a file to the appropriate conversion agent."""
    if _is_model_file(pytorch_code):
      return self._model_conversion_agent.run(pytorch_code)
    return self._single_file_agent.run(pytorch_code)

  def run(self, repo_path: str) -> Dict[str, str]:
    """Orchestrates the migration of a repository from PyTorch to JAX.

    Args:
      repo_path: The path to the repository file or directory.

    Returns:
      A dictionary mapping original file paths to converted JAX code.
    """
    if os.path.isfile(repo_path):
      with open(repo_path, "r", encoding="utf-8", errors="replace") as f:
        pytorch_code = f.read()
      converted_code = self._convert_file(pytorch_code)
      return {repo_path: converted_code}
    elif not os.path.isdir(repo_path):
      return {
          repo_path: f"# Error: path {repo_path} is not a file or directory."
      }

    graph = utils.build_dependency_graph(repo_path)
    ordered_files = _topological_sort(graph)
    converted_files: Dict[str, str] = {}

    for file_rel_path in ordered_files:
      file_path = os.path.join(repo_path, file_rel_path)
      with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        pytorch_code = f.read()
      converted_code = self._convert_file(pytorch_code)
      converted_files[file_path] = converted_code

    return converted_files
