"""Main orchestration agent for HITL kernel generation.

This module contains the root orchestrator that coordinates all subagents
for the human-in-the-loop kernel generation process.
"""

from google.adk.apps.app import App, EventsCompactionConfig

from auto_agent.subagents.autotuning.agent import autotune_agent
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

root_agent = AutonomousPipelineAgent(
  name="AutonomousPipelineAgent",
  plan_agent=plan_kernel_agent,
  implement_agent=implement_kernel_agent,
  validate_agent=validate_kernel_compilation_agent,
  test_gen_agent=validated_test_generation_agent,
  test_run_agent=unified_test_agent,
  autotune_agent=autotune_agent,
  profile_agent=profile_agent,
  max_iterations=5,
)

if EVENTS_COMPACTION:
  compaction_config = EventsCompactionConfig(
    token_threshold=200000,
    event_retention_size=100,
    compaction_interval=1,
    overlap_size=0,
  )
else:
  compaction_config = None


app = App(
  name="auto_agent",
  root_agent=root_agent,
  events_compaction_config=compaction_config,
)

__all__ = ["root_agent", "app"]
