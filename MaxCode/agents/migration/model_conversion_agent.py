"""Agent for converting a model from PyTorch to a JAX-family target."""

import re
from typing import Any

from agents import base
from agents import utils
from agents.migration.prompts import prompts
from rag import rag_agent

_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\n?(.*?)\n?```", re.DOTALL)


def _strip_markdown_formatting(text: str) -> str:
  """Strips markdown and returns only the first Python code block."""
  code_block_match = _CODE_BLOCK_PATTERN.search(text)
  if code_block_match:
    return code_block_match.group(1).strip()
  # Handle truncated responses: opening ``` present but closing ``` missing
  stripped = text.strip()
  if stripped.startswith("```"):
    first_nl = stripped.find("\n")
    if first_nl != -1:
      stripped = stripped[first_nl + 1:]
    if stripped.endswith("```"):
      stripped = stripped[:-3]
    return stripped.strip()
  # Strip triple-quote wrappers the LLM may use instead of backticks.
  if stripped.startswith('"""') and stripped.endswith('"""'):
    return stripped[3:-3].strip()
  return text


class ModelConversionAgent(base.Agent):
  """Agent for converting a model from PyTorch to JAX/Flax.

  This agent specializes in converting PyTorch torch.nn.Module class
  definitions into idiomatic JAX/Flax equivalents (flax.linen.Module).
  It uses a prompt optimized for architectural model conversions, which is
  distinct from general API syntax conversion.
  """

  def __init__(
      self,
      model: Any,
      rag_agent_instance: rag_agent.RAGAgent,
      target: str = "jax",
  ):
    """Initializes the agent.

    Args:
      model: The LLM model to use for generation.
      rag_agent_instance: RAGAgent for retrieving reference snippets.
      target: Conversion target ("jax" by default). MaxText conversions are
        handled by `MaxTextConversionAgent` rather than this agent, but the
        target is plumbed through for prompt selection symmetry.
    """
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.MODEL_CONVERSION,
    )
    self._rag_agent = rag_agent_instance
    self._target = target

  def run(self, pytorch_model_code: str) -> str:
    """Converts a model from PyTorch to JAX/Flax.

    Args:
      pytorch_model_code: The PyTorch model code to convert.

    Returns:
      The converted JAX code.
    """
    rag_context_list = self._rag_agent.retrieve_per_component_context(
        pytorch_model_code
    )
    rag_context = "\n\n".join([
        f"File: {c['file']}\n```python\n{c['text']}\n```"
        for c in rag_context_list
    ])
    prompt_template = prompts.get_prompt(
        "MODEL_CONVERSION_PROMPT", self._target
    )
    if prompt_template is None:
      prompt_template = prompts.MODEL_CONVERSION_PROMPT
    return _strip_markdown_formatting(
        self.generate(
            prompt_template,
            {
                "pytorch_model_code": pytorch_model_code,
                "rag_context": rag_context,
            },
        )
    )
