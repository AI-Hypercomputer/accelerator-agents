import logging
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models.google_llm import Gemini
from google.genai import types

from auto_agent import config


class CustomLlmAgent(LlmAgent):
  """Agent that allows early exit from the loop if a condition is met.

  Automatically uses gemini_model (with retry support) when a string model name is provided.
  """

  def __init__(self, *args, **kwargs):
    """Initialize CustomLlmAgent with model-specific configuration and wrapping."""
    # If model is a string, use the pre-configured model or config
    if "model" in kwargs and isinstance(kwargs["model"], str):
      model_str = kwargs["model"]
      is_claude = model_str.startswith("claude")

      if "generate_content_config" not in kwargs:
        kwargs["generate_content_config"] = (
          config.claude_config if is_claude else config.gemini_config
        )

      if "planner" not in kwargs:
        kwargs["planner"] = (
          config.claude_planner if is_claude else config.gemini_planner
        )

      if is_claude:
        from google.adk.models.anthropic_llm import Claude
        cfg = kwargs["generate_content_config"]
        max_tokens = cfg.max_output_tokens if cfg.max_output_tokens else 8192
        claude_model = Claude(
          model=model_str,
          max_tokens=max_tokens,
        )
        kwargs["model"] = claude_model
      else:
        # Wrap in Gemini model for retry support
        gemini_model = Gemini(
          model=model_str,
          retry_options=types.HttpRetryOptions(
            initial_delay=1,
            attempts=5,
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
