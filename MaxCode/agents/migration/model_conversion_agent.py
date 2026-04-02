"""Agent for converting a model from PyTorch to JAX."""

import re
from typing import Any

from agents import base
from agents import utils
from agents.migration.prompts import prompts
from rag import rag_agent


class ModelConversionAgent(base.Agent):
  """Agent for converting a model from PyTorch to JAX.

  This agent specializes in converting PyTorch torch.nn.Module class
  definitions into idiomatic JAX/Flax equivalents (flax.linen.Module).
  It uses a prompt optimized for architectural model conversions, which is
  distinct from general API syntax conversion.
  """

  def __init__(self, model: Any, rag_agent_instance: rag_agent.RAGAgent):
    """Initializes the agent."""
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.MODEL_CONVERSION,
    )
    self._rag_agent = rag_agent_instance

  def run(self, pytorch_model_code: str) -> str:
    """Converts a model from PyTorch to JAX.

    Args:
      pytorch_model_code: The PyTorch model code to convert.

    Returns:
      The converted JAX code.
    """
    rag_context_list = self._rag_agent.retrieve_context(pytorch_model_code, top_k=7)
    rag_context = "\n\n".join([
        f"File: {c['file']}\n```python\n{c['text']}\n```"
        for c in rag_context_list
    ])
    generated_code = self.generate(
        prompts.MODEL_CONVERSION_PROMPT,
        {"pytorch_model_code": pytorch_model_code, "rag_context": rag_context},
    )
    return self._strip_markdown_formatting(generated_code)

  def _strip_markdown_formatting(self, text: str) -> str:
    """Strips markdown and returns only the first python code block."""
    code_block_match = re.search(
        r"```(?:python)?\n?(.*?)\n?```", text, re.DOTALL
    )
    if code_block_match:
      return code_block_match.group(1).strip()
    return text
