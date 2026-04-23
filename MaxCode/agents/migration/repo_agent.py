"""Agent for converting a repository from PyTorch to JAX."""

import os
from typing import Any, Dict

from agents import base
from agents import utils
from agents.migration.prompts import prompts
from rag import rag_agent


class PytorchToJaxRepoAgent(base.Agent):
  """Agent for converting a repository from PyTorch to JAX.

  This agent walks through a repository directory, finds all Python files,
  and converts each file from PyTorch to JAX using a repo-aware prompt.
  Currently, it converts files individually without dependency analysis.
  """

  def __init__(self, model: Any, rag_agent_instance: rag_agent.RAGAgent):
    """Initializes the agent."""
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.PYTORCH_TO_JAX_REPO,
    )
    self._rag_agent = rag_agent_instance

  def run(self, repo_path: str) -> Dict[str, str]:
    """Converts all Python files in a repository from PyTorch to JAX.

    Args:
      repo_path: The path to the repository directory.

    Returns:
      A dictionary mapping original file paths to converted JAX code.
    """
    converted_files: Dict[str, str] = {}
    for root, _, files in os.walk(repo_path):
      for filename in files:
        if filename.endswith(".py"):
          file_path = os.path.join(root, filename)
          try:
            with open(file_path, "r") as f:
              pytorch_code = f.read()
            rag_context_list = self._rag_agent.retrieve_context(pytorch_code)
            rag_context = "\\n\\n".join([
                f"File: {c['file']}\\n```python\\n{c['text']}\\n```"
                for c in rag_context_list
            ])
            converted_code = self.generate(
                prompts.PYTORCH_TO_JAX_REPO_PROMPT,
                {
                    "file_path": file_path,
                    "pytorch_code": pytorch_code,
                    "rag_context": rag_context,
                },
            )
            converted_files[file_path] = converted_code
          except (OSError, ValueError) as e:
            # Skip files that can't be read or processed due to common errors.
            converted_files[file_path] = (
                f"# Error processing file {file_path}: {e}"
            )
            continue
    return converted_files
