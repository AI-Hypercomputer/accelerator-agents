"""Agent for converting a single file from PyTorch to JAX."""

import re
from typing import Any

from agents import base
from agents import utils
from agents.migration.prompts import prompts
from rag import rag_agent


class PytorchToJaxSingleFileAgent(base.Agent):
  """Agent for converting a single file from PyTorch to JAX.

  This agent performs general-purpose conversion of PyTorch API calls to JAX
  API calls within a given file. It is best suited for converting utility
  functions, data loading pipelines, and training/evaluation loops. For
  converting torch.nn.Module definitions to idiomatic Flax equivalents,
  consider using the ModelConversionAgent.
  """

  def __init__(self, model: Any, rag_agent_instance: rag_agent.RAGAgent):
    """Initializes the agent."""
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.PYTORCH_TO_JAX_SINGLE_FILE,
    )
    self._rag_agent = rag_agent_instance

  def _strip_markdown_formatting(self, text: str) -> str:
    """Strips markdown and returns only the first python code block."""
    code_block_match = re.search(
        r"```(?:python)?\n?(.*?)\n?```", text, re.DOTALL
    )
    if code_block_match:
      return code_block_match.group(1).strip()
    return text

  def run(self, pytorch_code: str) -> str:
    """Converts a single file from PyTorch to JAX.

    Args:
      pytorch_code: The PyTorch code to convert.

    Returns:
      The converted JAX code.
    """
    rag_context_list = self._rag_agent.retrieve_context(pytorch_code, top_k=7)
    rag_context = "\n\n".join([
        f"File: {c['file']}\n```python\n{c['text']}\n```"
        for c in rag_context_list
    ])
    generated_code = self.generate(
        prompts.PYTORCH_TO_JAX_SINGLE_FILE_PROMPT,
        {"pytorch_code": pytorch_code, "rag_context": rag_context},
    )
    return self._strip_markdown_formatting(generated_code)
