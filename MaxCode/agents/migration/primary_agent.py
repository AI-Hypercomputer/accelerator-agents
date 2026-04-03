"""Primary orchestration agent for repository migration."""
import os
from typing import Any

import models
from agents import base
from agents import utils
from agents.migration import model_conversion_agent
from agents.migration import single_file_agent
from rag import rag_agent


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

  def _convert_file(self, pytorch_code: str, file_path: str) -> str:
    """Routes a file to the appropriate conversion agent."""
    if utils.is_model_file(pytorch_code, file_path):
      return self._model_conversion_agent.run(pytorch_code)
    return self._single_file_agent.run(pytorch_code)

  def run(self, repo_path: str) -> dict[str, str]:
    """Orchestrates the migration of a repository from PyTorch to JAX.

    Args:
      repo_path: The path to the repository file or directory.

    Returns:
      A dictionary mapping original file paths to converted JAX code.
    """
    try:
      with open(repo_path, "r", encoding="utf-8", errors="replace") as f:
        pytorch_code = f.read()
      converted_code = self._convert_file(pytorch_code, repo_path)
      return {repo_path: converted_code}
    except OSError:
      # If opening as a file fails, check if it's a directory.
      if not os.path.isdir(repo_path):
        return {
            repo_path: f"# Error: path {repo_path} is not a file or directory."
        }

    if not os.path.isdir(repo_path):
      return {
          repo_path: f"# Error: path {repo_path} is not a file or directory."
      }

    graph = utils.build_dependency_graph(repo_path)
    ordered_files = utils.topological_sort(graph)
    converted_files: dict[str, str] = {}

    for file_rel_path in ordered_files:
      file_path = os.path.join(repo_path, file_rel_path)
      with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        pytorch_code = f.read()
      converted_code = self._convert_file(pytorch_code, file_path)
      converted_files[file_path] = converted_code

    return converted_files
