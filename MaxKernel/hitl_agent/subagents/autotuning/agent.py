"""Autotuning agent following the split pattern (Planner + Runner)."""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from hitl_agent.callbacks import create_path_saver
from hitl_agent.config import model_config, thinking_planner
from hitl_agent.constants import MODEL_NAME
from hitl_agent.custom_types import CustomLlmAgent
from hitl_agent.server_utils.server_manager_mixin import ServerManagerMixin
from hitl_agent.subagents.autotuning.prompts import autotune_prompt, summary_prompt
from .autotune_tool import autotune_kernel
from hitl_agent.tools.tools import filesystem_tool_rw

# 1. Planner Agent (LLM)
# This agent identifies parameters, creates the template, and defines the search space.
# It saves them to session state instead of calling the tool directly.
autotune_planner_agent = CustomLlmAgent(
    name="AutotunePlannerAgent",
    model=MODEL_NAME,
    generate_content_config=model_config,
    planner=thinking_planner,
    instruction=autotune_prompt.PROMPT,
    description="Prepares code template and search space for auto-tuning Pallas kernels.",
    tools=[filesystem_tool_rw],
    after_tool_callback=create_path_saver("autotune_specs_path"),
)

# 2. Runner Agent (Python)
# This agent ensures the server is running locally and calls the autotune tool.
class AutotuneRunner(ServerManagerMixin, BaseAgent):
  """Manages server lifecycle and executes autotuning via HTTP endpoint.
  
  Uses ServerManagerMixin to manage servers.
  """
  
  auto_manage_servers: bool = True

  def __init__(self, name: str):
    BaseAgent.__init__(self, name=name)
    self._servers_started = []

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    try:
      # Read inputs from file saved by Planner Agent
      autotune_specs_path = ctx.session.state.get("autotune_specs_path", "")
      
      if not autotune_specs_path:
        error_msg = "Missing autotune_specs_path in session state."
        logging.error(error_msg)
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={"autotune_results": {"status": "error", "message": error_msg}}
          ),
        )
        return

      if not os.path.exists(autotune_specs_path):
        error_msg = f"Autotune specs file not found at {autotune_specs_path}"
        logging.error(error_msg)
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={"autotune_results": {"status": "error", "message": error_msg}}
          ),
        )
        return

      try:
        with open(autotune_specs_path, "r") as f:
          specs = json.load(f)
        kernel_name = specs.get("kernel_name", "")
        code_template = specs.get("code_template", "")
        search_space = specs.get("search_space", {})
      except Exception as e:
        error_msg = f"Failed to parse autotune specs JSON: {e}"
        logging.error(error_msg)
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={"autotune_results": {"status": "error", "message": error_msg}}
          ),
        )
        return

      if not kernel_name or not code_template or not search_space:
        error_msg = "Missing required inputs in autotune specs file."
        logging.error(error_msg)
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={"autotune_results": {"status": "error", "message": error_msg}}
          ),
        )
        return

      servers_ok, error_msg = await self._ensure_servers_running()
      if not servers_ok:
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={"autotune_results": {"status": "error", "message": f"Server startup failed: {error_msg}"}}
          ),
        )
        return

      logging.info(f"[{self.name}] Running autotune for {kernel_name}")
      
      try:
        results = autotune_kernel(
            kernel_name=kernel_name,
            code_template=code_template,
            search_space=search_space,
        )
        
        results_path = ""
        try:
          results_dir = os.path.dirname(autotune_specs_path)
          results_path = os.path.join(results_dir, "autotune_results.json")
          with open(results_path, "w") as f:
              json.dump(results, f)
          logging.info(f"[{self.name}] Saved results to {results_path}")
        except Exception as e:
          logging.error(f"[{self.name}] Failed to save results to file: {e}")
          
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              "autotune_results": results,
              "autotune_results_path": results_path
            }
          ),
        )
        
      except Exception as e:
        logging.error(f"Exception during autotuning: {e}")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={"autotune_results": {"status": "error", "message": str(e)}}
          ),
        )
    finally:
      await self._cleanup_servers()

autotune_runner = AutotuneRunner(name="AutotuneRunner")

# 3. Summarizer Agent (LLM)
# This agent reads results from state and talks to the user.
autotune_summary_agent = CustomLlmAgent(
    name="AutotuneSummaryAgent",
    model=MODEL_NAME,
    generate_content_config=model_config,
    planner=thinking_planner,
    instruction=summary_prompt.PROMPT,
    description="Summarizes autotuning results for the user.",
    tools=[filesystem_tool_rw],
)

# 4. Combined Sequential Agent
# This maintains the original interface name.
autotune_agent = SequentialAgent(
    name="AutotuneAgent",
    sub_agents=[autotune_planner_agent, autotune_runner, autotune_summary_agent],
)

__all__ = ["autotune_agent"]

