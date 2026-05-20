"""Autotuning agent following the split pattern (Planner + Runner)."""

import json
import logging
import os
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from auto_agent.config import model_config, thinking_planner
from auto_agent.constants import MODEL_NAME
from auto_agent.custom_types import CustomLlmAgent
from auto_agent.subagents.autotuning.autotune_tool import autotune_kernel
from auto_agent.subagents.autotuning.prompts import (
  apply_best_config_prompt,
  autotune_prompt,
  summary_prompt,
)
from auto_agent.tools.file_tools import write_optimized_kernel_tool
from auto_agent.tools.search_api_tool import search_api_tool
from auto_agent.tools.tools import filesystem_tool_r, write_autotune_specs_tool

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
  tools=[filesystem_tool_r, write_autotune_specs_tool, search_api_tool],
)


# 2. Runner Agent
class AutotuneRunner(BaseAgent):
  """Executes autotuning via HTTP endpoint."""

  def __init__(
    self,
    name: str,
    output_key: str,
  ):
    BaseAgent.__init__(self, name=name)
    self.output_key = output_key

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:

    autotune_specs_path = ctx.session.state.get("autotune_specs_path", "")
    autotune_results_path = ctx.session.state.get("autotune_results_path", "")

    if not os.path.exists(autotune_specs_path):
      error_msg = f"Autotune specs file not found at {autotune_specs_path}"
      logging.error(error_msg)
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {"status": "error", "message": error_msg}
          }
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
          state_delta={
            self.output_key: {"status": "error", "message": error_msg}
          }
        ),
      )
      return

    if not kernel_name or not code_template or not search_space:
      error_msg = "Missing required inputs in autotune specs file."
      logging.error(error_msg)
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {"status": "error", "message": error_msg}
          }
        ),
      )
      return

    logging.info(f"[{self.name}] Running autotune for {kernel_name}")

    try:
      results = await autotune_kernel(
        kernel_name=kernel_name,
        code_template=code_template,
        search_space=search_space,
        backend="tpu",
      )

      try:
        with open(autotune_results_path, "w") as f:
          json.dump(results, f)
        logging.info(f"[{self.name}] Saved results to {autotune_results_path}")
      except Exception as e:
        logging.error(f"[{self.name}] Failed to save results to file: {e}")

      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: results,
          }
        ),
      )

    except Exception as e:
      logging.error(f"Exception during autotuning: {e}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={self.output_key: {"status": "error", "message": str(e)}}
        ),
      )


autotune_runner = AutotuneRunner(
  name="AutotuneRunner",
  output_key="autotune_results",
)

# 3. Apply Best Config Agent
apply_best_config_agent = CustomLlmAgent(
  name="ApplyBestConfigAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=apply_best_config_prompt.PROMPT,
  description="Applies autotuning results to the optimized kernel file.",
  tools=[filesystem_tool_r, write_optimized_kernel_tool],
)

# 4. Summarizer Agent
# This agent reads results from state and talks to the user.
autotune_summary_agent = CustomLlmAgent(
  name="AutotuneSummaryAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=summary_prompt.PROMPT,
  description="Summarizes autotuning results for the user.",
  tools=[filesystem_tool_r],
)


class CombinedAutotuneAgent(BaseAgent):
  """Chains autotuning steps and conditionally applies best config."""

  def __init__(self, name: str):
    super().__init__(name=name)

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    logging.info(f"[{self.name}] Running AutotunePlannerAgent...")
    async for event in autotune_planner_agent.run_async(ctx):
      yield event

    logging.info(f"[{self.name}] Running AutotuneRunner...")
    async for event in autotune_runner.run_async(ctx):
      yield event

    autotune_results = ctx.session.state.get("autotune_results", {})
    if (
      autotune_results.get("status") == "success"
      and autotune_results.get("best_cfg") is not None
    ):
      logging.info(f"[{self.name}] Running ApplyBestConfigAgent...")
      async for event in apply_best_config_agent.run_async(ctx):
        yield event
    else:
      logging.warning(
        f"[{self.name}] Autotune was not successful or no best configuration"
        " found. Skipping ApplyBestConfigAgent."
      )

    logging.info(f"[{self.name}] Running AutotuneSummaryAgent...")
    async for event in autotune_summary_agent.run_async(ctx):
      yield event


autotune_agent = CombinedAutotuneAgent(name="AutotuneAgent")

__all__ = ["autotune_agent"]
