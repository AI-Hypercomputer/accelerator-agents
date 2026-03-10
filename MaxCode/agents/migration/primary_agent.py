"""Primary orchestration agent for repository migration."""

import os
from typing import Any, Dict

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
        embedding_model_name=models.EmbeddingModel.TEXT_EMBEDDING_004,
        api_key=api_key,
    )
    self._single_file_agent = single_file_agent.PytorchToJaxSingleFileAgent(
        model, self._rag_agent
    )
    self._model_conversion_agent = model_conversion_agent.ModelConversionAgent(
        model, self._rag_agent
    )

  def run(self, repo_path: str) -> Dict[str, str]:
    """Orchestrates the migration of a repository from PyTorch to JAX.

    Args:
      repo_path: The path to the repository file or directory.

    Returns:
      A dictionary mapping original file paths to converted JAX code.
    """
    if os.path.isfile(repo_path):
      with open(repo_path, "r") as f:
        pytorch_code = f.read()
      converted_code = self._single_file_agent.run(pytorch_code)
      return {repo_path: converted_code}
    elif not os.path.isdir(repo_path):
      return {
          repo_path: f"# Error: path {repo_path} is not a file or directory."
      }

    graph = utils.build_dependency_graph(repo_path)
    converted_files: Dict[str, str] = {}
    # conversion order.
    # model_conversion_agent for model files, single_file_agent for others).
    # For now, convert files individually using single_file_agent.

    for file_rel_path in graph:
      file_path = os.path.join(repo_path, file_rel_path)
      with open(file_path, "r") as f:
        pytorch_code = f.read()
      converted_code = self._single_file_agent.run(pytorch_code)
      converted_files[file_path] = converted_code

    return converted_files
