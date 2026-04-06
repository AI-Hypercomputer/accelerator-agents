"""Base class for all agents."""

import abc
import logging
from typing import Any, Dict, Optional

from agents import utils


class Agent(abc.ABC):
  """Base class for all agents."""

  def __init__(
      self,
      model: Any,
      agent_domain: utils.AgentDomain = utils.AgentDomain.MIGRATION,
      agent_type: utils.AgentType = utils.AgentType.PRIMARY,
  ):
    """Initializes the agent.

    Args:
      model: The language model instance (e.g., Gemini, Goose) used for
        inference.
      agent_domain: The high-level domain of the agent.
      agent_type: The specific type of the agent.
    """
    self._model = model
    self.agent_domain = agent_domain
    self.agent_type = agent_type

  def generate(
      self, prompt_template: str, prompt_vars: Optional[Dict[str, str]] = None
  ) -> str:
    """Formats a prompt template and calls the model to generate a response."""
    if prompt_vars:
      prompt = prompt_template.format(**prompt_vars)
    else:
      prompt = prompt_template
    logging.info(
        "--- %s PROMPT ---\n%s\n--- END PROMPT ---",
        self.agent_type.name,
        prompt,
    )
    return self._model.generate(prompt)

  @abc.abstractmethod
  def run(self, *args, **kwargs):
    """Runs the agent."""
    raise NotImplementedError
