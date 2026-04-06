"""Profiling subagent - performance profiling and analysis."""

from google.adk.agents import SequentialAgent

from hitl_agent.callbacks import (
  create_path_saver,
  load_profiling_script_to_state,
  load_single_kernel_to_state,
)
from hitl_agent.config import model_config, thinking_planner
from hitl_agent.constants import MODEL_NAME
from hitl_agent.custom_types import CustomLlmAgent
from hitl_agent.subagents.profiling import offline_tools
from hitl_agent.subagents.profiling.kernel_profile import KernelProfiler
from hitl_agent.subagents.profiling.prompts import (
  analyze_profile_prompt,
  gen_profiling_script,
  read_file_prompt,
  read_profiling_script_prompt,
)
from hitl_agent.tools.tools import filesystem_tool_rw, vertex_ai_rag_tool

# Read file agent for profiling
read_file_for_profiling_agent = CustomLlmAgent(
  name="ReadFileForProfilingAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=read_file_prompt.PROMPT,
  description="Reads the kernel file mentioned by the user for profiling analysis.",
  tools=[filesystem_tool_rw],
  after_tool_callback=create_path_saver("kernel_file_path"),
)

# Profiling script generation agent - writes profiling script to file
generate_profiling_script_agent = CustomLlmAgent(
  name="GenerateProfilingScriptAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=gen_profiling_script.PROMPT,
  description="Generates a profiling script to identify performance bottlenecks in the kernel code and writes it to a file.",
  tools=[filesystem_tool_rw],
  before_agent_callback=load_single_kernel_to_state,
  after_tool_callback=create_path_saver("profiling_script_path"),
)

# Read profiling script agent - loads the generated profiling script file contents into state
read_profiling_script_agent = CustomLlmAgent(
  name="ReadProfilingScriptAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=read_profiling_script_prompt.PROMPT,
  description="Loads the generated profiling script file contents from disk into memory for execution.",
  before_agent_callback=load_profiling_script_to_state,
  include_contents="none",
)

# Profiling execution agent
eval_profile_agent = KernelProfiler(
  name="ProfileEvalAgent",
  input_key="profiling_script",
  output_key="profiling_results",
  auto_manage_servers=True,
)

# Profiling summary agent
summarize_profile_agent = CustomLlmAgent(
  name="SummarizeProfileAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=analyze_profile_prompt.PROMPT,
  description=(
    "Summarizes the profiling results of the kernel and performs deep analysis using offline XProf tools."
  ),
  output_key="profiling_summary",
  include_contents="none",
  tools=[
    offline_tools.load_xplane_and_query,
    offline_tools.get_hlo_dump,
    offline_tools.create_chart_from_xplane,
    offline_tools.get_overview_page_metrics,
    vertex_ai_rag_tool,
  ],
)

# Main profiling orchestrator agent
profile_agent = SequentialAgent(
  name="ProfileAgentOrchestrator",
  sub_agents=[
    read_file_for_profiling_agent,
    generate_profiling_script_agent,
    read_profiling_script_agent,
    eval_profile_agent,
    summarize_profile_agent,
  ],
  description="Profiles the Pallas kernel to identify performance bottlenecks.",
)

__all__ = [
  "profile_agent",
  "read_file_for_profiling_agent",
  "generate_profiling_script_agent",
  "read_profiling_script_agent",
  "eval_profile_agent",
  "summarize_profile_agent",
]
