"""Main orchestration agent for HITL kernel generation.

This module contains the root orchestrator that coordinates all subagents
for the human-in-the-loop kernel generation process.
"""

from google.adk.models.anthropic_llm import Claude
from google.adk.models.registry import LLMRegistry

from auto_agent.subagents.kernel_writing import (
  implement_kernel_agent,
  plan_kernel_agent,
  validate_kernel_compilation_agent,
)
from auto_agent.subagents.pipeline_agent import AutonomousPipelineAgent
from auto_agent.subagents.profiling import profile_agent
from auto_agent.subagents.testing import (
  unified_test_agent,
  validated_test_generation_agent,
)

LLMRegistry.register(Claude)

root_agent = AutonomousPipelineAgent(
  name="AutonomousPipelineAgent",
  plan_agent=plan_kernel_agent,
  implement_agent=implement_kernel_agent,
  validate_agent=validate_kernel_compilation_agent,
  test_gen_agent=validated_test_generation_agent,
  test_run_agent=unified_test_agent,
  profile_agent=profile_agent,
  max_iterations=5,
)

__all__ = ["root_agent"]
