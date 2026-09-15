"""Shared configuration for HITL kernel generation agents."""

import os

from google.adk.planners import BuiltInPlanner
from google.genai import types

from hitl_agent.constants import TOP_K, TOP_P

# Environment variables
WORKDIR = os.environ.get("WORKDIR", os.path.dirname(os.path.abspath(__file__)))
TPU_VERSION = os.environ.get("TPU_VERSION", "")
RAG_CORPUS = os.environ.get("RAG_CORPUS", "")
INCLUDE_THOUGHTS = os.environ.get("INCLUDE_THOUGHTS", "true").lower() == "true"
MAX_COMPILATION_RETRIES = int(os.environ.get("MAX_COMPILATION_RETRIES", "6"))

# Model configuration
model_config = types.GenerateContentConfig(
  temperature=0.5,
  top_p=TOP_P,
  top_k=TOP_K,
)

# Planner configuration with thinking/reasoning traces
thinking_planner = BuiltInPlanner(
  thinking_config=types.ThinkingConfig(
    include_thoughts=INCLUDE_THOUGHTS,
    thinking_level="high",
  )
)


# MONKEY PATCH GENERATE_CONTENT to handle rate limits
try:
  import logging

  import tenacity
  from google import genai

  def get_retry_decorator():
    return tenacity.retry(
      wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
      stop=tenacity.stop_after_attempt(10),
      retry=tenacity.retry_if_exception_type(Exception),
      before_sleep=tenacity.before_sleep_log(
        logging.getLogger(__name__), logging.WARNING
      ),
    )

  if not hasattr(genai.models.Models, "_original_generate_content"):
    orig_sync = genai.models.Models.generate_content
    genai.models.Models._original_generate_content = orig_sync

    @get_retry_decorator()
    def wrapped_sync(self, *args, **kwargs):
      return orig_sync(self, *args, **kwargs)

    genai.models.Models.generate_content = wrapped_sync

  if not hasattr(genai.models.AsyncModels, "_original_generate_content"):
    orig_async = genai.models.AsyncModels.generate_content
    genai.models.AsyncModels._original_generate_content = orig_async

    @get_retry_decorator()
    async def wrapped_async(self, *args, **kwargs):
      import asyncio

      # Gemini API occasionally hangs indefinitely on concurrent quotas.
      # Force a 90 second hard timeout so it triggers a tenacity retry
      # instead of infinitely blocking the orchestrator.
      return await asyncio.wait_for(
        orig_async(self, *args, **kwargs), timeout=90
      )

    genai.models.AsyncModels.generate_content = wrapped_async
except ImportError:
  pass
# END MONKEY PATCH
