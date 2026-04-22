"""Autotuning agent."""

from hitl_agent.config import model_config, thinking_planner
from hitl_agent.constants import MODEL_NAME
from hitl_agent.custom_types import CustomLlmAgent
from hitl_agent.subagents.autotuning.prompts import autotune_prompt
from hitl_agent.tools.tools import autotune_tool, filesystem_tool_rw

autotune_agent = CustomLlmAgent(
    name="AutotuneAgent",
    model=MODEL_NAME,
    generate_content_config=model_config,
    planner=thinking_planner,
    instruction=autotune_prompt.PROMPT,
    description="Auto-tunes Pallas kernels by searching over parameter spaces.",
    tools=[autotune_tool, filesystem_tool_rw],
)

__all__ = ["autotune_agent"]
