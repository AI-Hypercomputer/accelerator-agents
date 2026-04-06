"""Primary orchestration agent for repository migration."""
import logging
import os
from typing import Any

import models
from agents import base
from agents import utils
from agents.migration import model_conversion_agent
from agents.migration import single_file_agent
from agents.migration import validation_agent
from rag import rag_agent

logger = logging.getLogger(__name__)


class PrimaryAgent(base.Agent):
  """Primary orchestration agent for repository migration."""

  def __init__(self, model: Any, api_key: str | None = None,
               validate: bool = True):
    """Initializes the agent."""
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.PRIMARY,
    )
    self._model_ref = model
    self._validate = validate
    self._validation_results: dict[str, dict] = {}
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

  def _validate_and_repair(self, pytorch_code: str, converted_code: str,
                           file_path: str) -> str:
    """Validates converted code and repairs deviations if found.

    Args:
      pytorch_code: The original PyTorch source code.
      converted_code: The converted JAX/Flax code.
      file_path: The file path (used as key for storing results).

    Returns:
      The final code (repaired if deviations were found, original otherwise).
    """
    validator = validation_agent.ValidationAgent(self._model_ref)
    deviations = validator.validate(pytorch_code, converted_code)
    logger.info("Validation of %s: found %d deviations",
                file_path, len(deviations))

    result = {
        "deviations_found": len(deviations),
        "deviations": deviations,
        "remaining_deviations_count": 0,
        "remaining_deviations": [],
    }

    if deviations:
      repaired_code = validator.repair(
          converted_code, deviations, pytorch_code=pytorch_code
      )
      remaining = validator.validate(pytorch_code, repaired_code)
      logger.info("Re-validation of %s: %d remaining deviations",
                  file_path, len(remaining))
      result["remaining_deviations_count"] = len(remaining)
      result["remaining_deviations"] = remaining
      self._validation_results[file_path] = result
      return repaired_code

    self._validation_results[file_path] = result
    return converted_code

  def get_validation_results(self) -> dict[str, dict]:
    """Returns validation results for all processed files.

    Returns:
      A dictionary mapping file paths to their validation results, each
      containing deviations_found, deviations, remaining_deviations_count,
      and remaining_deviations.
    """
    return self._validation_results

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
      if self._validate:
        converted_code = self._validate_and_repair(
            pytorch_code, converted_code, repo_path
        )
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
      if self._validate:
        converted_code = self._validate_and_repair(
            pytorch_code, converted_code, file_path
        )
      converted_files[file_path] = converted_code

    return converted_files
