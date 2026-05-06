"""Explanation subagent - provides explanations for the kernel generation process."""

from google.adk.agents import SequentialAgent

from auto_agent.config import model_config, thinking_planner
from auto_agent.constants import MODEL_NAME
from auto_agent.custom_types import CustomLlmAgent
from auto_agent.subagents.explanation.prompts import explanation_prompt
from auto_agent.tools.file_tools import filesystem_tool_r
from auto_agent.tools.tools import vertex_ai_rag_tool

# Explanation LLM agent
explanation_llm_agent = CustomLlmAgent(
  name="ExplanationLlmAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=explanation_prompt.PROMPT,
  description="Provides explanations for the kernel generation process.",
  tools=[filesystem_tool_r, vertex_ai_rag_tool]
  if vertex_ai_rag_tool
  else [filesystem_tool_r],
)

# Explanation orchestrator agent
explanation_agent = SequentialAgent(
  name="ExplanationAgent",
  sub_agents=[explanation_llm_agent],
  description="Provides explanations for the kernel generation process and returns control to orchestration.",
)

__all__ = [
  "explanation_agent",
  "explanation_llm_agent",
]
