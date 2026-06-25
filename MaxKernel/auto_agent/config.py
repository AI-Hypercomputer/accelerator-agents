"""Shared configuration for HITL kernel generation agents."""

import os

from google.adk.planners import BuiltInPlanner
from google.genai import types

from auto_agent.constants import TOP_K, TOP_P

# Environment variables
WORKDIR = os.environ.get("WORKDIR", os.path.dirname(os.path.abspath(__file__)))
TPU_VERSION = os.environ.get("TPU_VERSION", "")
RAG_CORPUS = os.environ.get("RAG_CORPUS", "")
INCLUDE_THOUGHTS = os.environ.get("INCLUDE_THOUGHTS", "true").lower() == "true"

# Model configuration
model_config = types.GenerateContentConfig(
  temperature=0.5,
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
