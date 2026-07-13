"""Shared configuration for HITL kernel generation agents."""

import os

from google.adk.apps.app import EventsCompactionConfig
from google.adk.planners import BuiltInPlanner
from google.genai import types

from auto_agent.constants import TEMPERATURE, TOP_K, TOP_P

# Environment variables
WORKDIR = os.environ.get("WORKDIR", os.path.dirname(os.path.abspath(__file__)))
TPU_VERSION = os.environ.get("TPU_VERSION", "")
RAG_CORPUS = os.environ.get("RAG_CORPUS", "")
INCLUDE_THOUGHTS = os.environ.get("INCLUDE_THOUGHTS", "true").lower() == "true"
MAX_COMPILATION_RETRIES = int(os.environ.get("MAX_COMPILATION_RETRIES", "6"))


# Set events compaction policy to avoid memory overflow
def get_compaction_config():
  return EventsCompactionConfig(
    token_threshold=300000,
    event_retention_size=5,
    compaction_interval=0,
    overlap_size=0,
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
