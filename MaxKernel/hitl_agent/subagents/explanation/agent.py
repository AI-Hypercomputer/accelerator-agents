"""Explanation subagent - provides explanations for the kernel generation process."""

from google.adk.agents import SequentialAgent

from hitl_agent.custom_types import CustomLlmAgent
from hitl_agent.constants import MODEL_NAME
from hitl_agent.config import model_config, thinking_planner
from hitl_agent.tools.tools import (
    filesystem_tool_rw,
    vertex_ai_rag_tool,
)
from hitl_agent.subagents.explanation.prompts import explanation_prompt

# Explanation LLM agent
explanation_llm_agent = CustomLlmAgent(
    name="ExplanationLlmAgent",
    model=MODEL_NAME,
    generate_content_config=model_config,
    planner=thinking_planner,
    instruction=explanation_prompt.PROMPT,
    description="Provides explanations for the kernel generation process.",
    tools=[filesystem_tool_rw, vertex_ai_rag_tool]
    if vertex_ai_rag_tool else [filesystem_tool_rw],
)

# Explanation orchestrator agent
explanation_agent = SequentialAgent(
    name="ExplanationAgent",
    sub_agents=[explanation_llm_agent],
    description=
    "Provides explanations for the kernel generation process and returns control to orchestration.",
)

__all__ = [
    'explanation_agent',
    'explanation_llm_agent',
]
