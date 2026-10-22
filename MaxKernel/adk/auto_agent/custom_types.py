import logging
from functools import cached_property
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models.google_llm import Gemini
from google.genai import Client, types


class TimeoutGemini(Gemini):
  @cached_property
  def api_client(self) -> Client:
    base_url, api_version = self._base_url_and_api_version
    kwargs_for_http_options = {
      "headers": self._tracking_headers(),
      "retry_options": self.retry_options,
      "base_url": base_url,
      "timeout": 600000,  # 600 seconds in milliseconds
    }
    if api_version:
      kwargs_for_http_options["api_version"] = api_version

    kwargs = {
      "http_options": types.HttpOptions(**kwargs_for_http_options),
    }
    if self.model.startswith("projects/"):
      kwargs["vertexai"] = True

    return Client(**kwargs)


class CustomLlmAgent(LlmAgent):
  """Agent that allows early exit from the loop if a condition is met.

  Automatically uses gemini_model (with retry support) when a string model name is provided.
  """

  def __init__(self, *args, **kwargs):
    """Initialize CustomLlmAgent with automatic Gemini model (with retry) wrapping."""
    # If model is a string, use the pre-configured gemini_model with retry support
    if "model" in kwargs and isinstance(kwargs["model"], str):
      gemini_model = TimeoutGemini(
        model=kwargs["model"],
        retry_options=types.HttpRetryOptions(
          initial_delay=1,
          attempts=10,
        ),
      )
      kwargs["model"] = gemini_model
    super().__init__(*args, **kwargs)

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    # Reset go_to_end flag when new user input is detected
    if (
      hasattr(ctx.session, "contents")
      and ctx.session.contents
      and len(ctx.session.contents) > 0
    ):
      last_message = ctx.session.contents[-1]
      # Check if the last message is from the user (role='user')
      if hasattr(last_message, "role") and last_message.role == "user":
        if ctx.session.state.get("go_to_end", False):
          logging.info(
            f"[{self.name}] New user input detected. Resetting go_to_end flag."
          )
        ctx.session.state["go_to_end"] = False

    if ctx.session.state.get("go_to_end", False):
      logging.info(f"[{self.name}] Early exit condition met. Skipping loop.")
      yield Event(
        author=self.name,
        actions=EventActions(escalate=True),
      )
    else:
      # Delegate to parent implementation (with native retry support at API level)
      async for event in super()._run_async_impl(ctx):
        yield event
