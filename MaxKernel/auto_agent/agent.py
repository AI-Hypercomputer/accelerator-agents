"""Main orchestration agent for HITL kernel generation.

This module contains the root orchestrator that coordinates all subagents
for the human-in-the-loop kernel generation process.
"""

from typing import Optional

from google.adk.apps.app import App

from auto_agent.config import get_compaction_config
from auto_agent.constants import EVENTS_COMPACTION
from auto_agent.subagents.autotuning.agent import autotune_agent
from auto_agent.subagents.kernel_writing import (
  implement_kernel_agent,
  plan_kernel_agent,
  prepare_base_kernel_agent,
  validate_kernel_compilation_agent,
)
from auto_agent.subagents.pipeline_agent import AutonomousPipelineAgent
from auto_agent.subagents.profiling import profile_agent
from auto_agent.subagents.testing import (
  unified_test_agent,
  validated_test_generation_agent,
)


def create_root_agent(
  max_iterations: int = 5,
  session_dir: Optional[str] = None,
  end_agent: Optional[str] = None,
  atol: Optional[float] = None,
  rtol: Optional[float] = None,
) -> AutonomousPipelineAgent:
  agent = AutonomousPipelineAgent(
    name="AutonomousPipelineAgent",
    prepare_base_kernel_agent=prepare_base_kernel_agent,
    plan_agent=plan_kernel_agent,
    implement_agent=implement_kernel_agent,
    validate_agent=validate_kernel_compilation_agent,
    test_gen_agent=validated_test_generation_agent,
    test_run_agent=unified_test_agent,
    autotune_agent=autotune_agent,
    profile_agent=profile_agent,
    max_iterations=max_iterations,
    session_dir=session_dir,
    end_agent=end_agent,
    atol=atol,
    rtol=rtol,
  )

  from auto_agent.timing_callbacks import (
    after_agent_callback,
    after_model_callback,
    after_tool_callback,
    before_agent_callback,
    before_model_callback,
    before_tool_callback,
  )

  def _hook(a, name, cb):
    try:
      existing = getattr(a, name, None)
      if existing:
        if isinstance(existing, list):
          existing.append(cb)
        else:
          setattr(a, name, [existing, cb])
      else:
        setattr(a, name, [cb])
    except ValueError:
      pass

  visited = set()

  def _inject(a):
    if id(a) in visited:
      return
    visited.add(id(a))
    _hook(a, "before_agent_callback", before_agent_callback)
    _hook(a, "after_agent_callback", after_agent_callback)
    _hook(a, "before_model_callback", before_model_callback)
    _hook(a, "after_model_callback", after_model_callback)
    _hook(a, "before_tool_callback", before_tool_callback)
    _hook(a, "after_tool_callback", after_tool_callback)

    from google.adk.agents.base_agent import BaseAgent

    for attr_name in dir(a):
      if attr_name.startswith("_") or attr_name == "parent_agent":
        continue
      try:
        attr_val = getattr(a, attr_name, None)
        if isinstance(attr_val, BaseAgent):
          _inject(attr_val)
        elif isinstance(attr_val, list):
          for item in attr_val:
            if isinstance(item, BaseAgent):
              _inject(item)
      except Exception:
        pass

  _inject(agent)
  return agent


root_agent = create_root_agent()

if EVENTS_COMPACTION:
  compaction_config = get_compaction_config()
else:
  compaction_config = None


app = App(
  name="auto_agent",
  root_agent=root_agent,
  events_compaction_config=compaction_config,
)

__all__ = ["create_root_agent", "root_agent", "app"]
