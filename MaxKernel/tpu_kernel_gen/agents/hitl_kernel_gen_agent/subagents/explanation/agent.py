"""Explanation subagent - provides explanations for the kernel generation process."""

from google.adk.agents import SequentialAgent

from tpu_kernel_gen.agents.hitl_kernel_gen_agent.config import (
  model_config,
  thinking_planner,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.explanation.prompts import (
  explanation_prompt,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.tools import (
  filesystem_tool_rw,
  vertex_ai_rag_tool,
)
from tpu_kernel_gen.agents.kernel_gen_agent.agent import CustomLlmAgent
from tpu_kernel_gen.agents.kernel_gen_agent.constants import MODEL_NAME

# Explanation LLM agent
explanation_llm_agent = CustomLlmAgent(
  name="ExplanationLlmAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=explanation_prompt.PROMPT,
  description="Provides explanations for the kernel generation process.",
  tools=[filesystem_tool_rw, vertex_ai_rag_tool] if vertex_ai_rag_tool else [filesystem_tool_rw],
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
