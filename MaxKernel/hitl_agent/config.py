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
