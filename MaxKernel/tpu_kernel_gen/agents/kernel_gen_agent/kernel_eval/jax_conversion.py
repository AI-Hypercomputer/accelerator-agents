import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types


class JaxConversionChecker(BaseAgent):
  """Checks logic of converted jax code and stops the loop if the jax code is identical to the original code."""

  def __init__(self, name: str, before_agent_callback):
    super().__init__(name=name, before_agent_callback=before_agent_callback)

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    assessment_result = ctx.session.state.get("jax_base_code_correctness_result")
    logging.info(f"[{self.name}] Received assessment result: {assessment_result}")
    if assessment_result.get("result") == "pass":
      logging.info(f"[{self.name}] JAX conversion passed the assessment. Escalating to stop the loop.")
      yield Event(author=self.name, actions=EventActions(escalate=True))
    else:
      logging.info(f"[{self.name}] JAX conversion did not pass the assessment. Continuing the loop.")
      # Yielding an event without content or actions just lets the flow continue.
      yield Event(author=self.name)


def check_whether_to_test(callback_context: CallbackContext):
  if callback_context.state.get("jax_base_code_compilation_result", None) != "Success":
    return types.Content(
      role="model",
      parts=[types.Part(text="The code has compilation errors, so correctness testing is not applicable.")],
    )


jax_conversion_checker = JaxConversionChecker(
  name="JaxConversionChecker",
  before_agent_callback=check_whether_to_test,
)
