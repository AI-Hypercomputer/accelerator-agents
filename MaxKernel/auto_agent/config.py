"""Shared configuration for HITL kernel generation agents."""

import os

from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.planners import BuiltInPlanner
from google.genai import types

from auto_agent.constants import MODEL_NAME, TEMPERATURE, TOP_K, TOP_P
from auto_agent.custom_types import TimeoutGemini

# Environment variables
WORKDIR = os.environ.get("WORKDIR", os.path.dirname(os.path.abspath(__file__)))
TPU_VERSION = os.environ.get("TPU_VERSION", "")
RAG_CORPUS = os.environ.get("RAG_CORPUS", "")
INCLUDE_THOUGHTS = os.environ.get("INCLUDE_THOUGHTS", "true").lower() == "true"
MAX_COMPILATION_RETRIES = int(os.environ.get("MAX_COMPILATION_RETRIES", "6"))


compaction_model = TimeoutGemini(model=MODEL_NAME)


# Set events compaction policy to avoid memory overflow
def get_compaction_config():
  return EventsCompactionConfig(
    token_threshold=300000,
    event_retention_size=5,
    compaction_interval=0,
    overlap_size=0,
    summarizer=LlmEventSummarizer(llm=compaction_model),
  )


# Model configuration
model_config = types.GenerateContentConfig(
  temperature=TEMPERATURE,
  top_p=TOP_P,
  top_k=TOP_K,
)


def get_thinking_planner(level: str = "high") -> BuiltInPlanner:
  """Returns a BuiltInPlanner configured with the specified thinking level.

  Args:
    level: The thinking level to use. Can be 'high', 'medium', or 'low'.
  """
  return BuiltInPlanner(
    thinking_config=types.ThinkingConfig(
      include_thoughts=INCLUDE_THOUGHTS,
      thinking_level=level,
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
      return await orig_async(self, *args, **kwargs)

    genai.models.AsyncModels.generate_content = wrapped_async
except ImportError:
  pass
# END MONKEY PATCH
