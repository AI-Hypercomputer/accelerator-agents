"""Agent for converting a single file from PyTorch to a JAX-family target."""

import re
from typing import Any

from agents import base
from agents import utils
from agents.migration.prompts import prompts
from rag import rag_agent


class PytorchSingleFileAgent(base.Agent):
  """Agent for converting a single file from PyTorch to a JAX-family target.

  This agent performs general-purpose conversion of PyTorch API calls to the
  selected target's API calls within a given file. It is best suited for
  converting utility functions, data loading pipelines, and training/eval
  loops. For converting torch.nn.Module definitions to idiomatic Flax /
  MaxText equivalents, consider using ModelConversionAgent or the
  MaxTextConversionAgent.
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
      target: Conversion target ("jax" or "maxtext"). Selects the prompt.
    """
    super().__init__(
        model=model,
        agent_domain=utils.AgentDomain.MIGRATION,
        agent_type=utils.AgentType.PYTORCH_TO_JAX_SINGLE_FILE,
    )
    self._rag_agent = rag_agent_instance
    self._target = target

  def _strip_markdown_formatting(self, text: str) -> str:
    """Strips markdown and returns only the first python code block."""
    code_block_match = re.search(
        r"```(?:python)?\n?(.*?)\n?```", text, re.DOTALL
    )
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

  def run(self, pytorch_code: str) -> str:
    """Converts a single file from PyTorch to the selected target.

    Args:
      pytorch_code: The PyTorch code to convert.

    Returns:
      The converted code in the target framework.
    """
    rag_context_list = self._rag_agent.retrieve_per_component_context(pytorch_code)
    rag_context = "\n\n".join([
        f"File: {c['file']}\n```python\n{c['text']}\n```"
        for c in rag_context_list
    ])
    prompt_template = prompts.get_prompt(
        "MIGRATE_MODULE_TO_JAX_PROMPT", self._target
    )
    if prompt_template is None:
      prompt_template = prompts.MIGRATE_MODULE_TO_JAX_PROMPT
    generated_code = self.generate(
        prompt_template,
        {"pytorch_code": pytorch_code, "rag_context": rag_context},
    )
    return self._strip_markdown_formatting(generated_code)


# Backwards-compatibility alias for one release.
PytorchToJaxSingleFileAgent = PytorchSingleFileAgent
