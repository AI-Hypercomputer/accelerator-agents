"""Autotuning agent following the split pattern (Planner + Runner)."""

import json
import logging
import os
from typing import AsyncGenerator, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from auto_agent.config import get_thinking_planner, model_config
from auto_agent.constants import MODEL_NAME
from auto_agent.custom_types import CustomLlmAgent
from auto_agent.subagents.autotuning.autotune_tool import autotune_kernel
from auto_agent.subagents.autotuning.prompts import (
  autotune_prompt,
  summary_prompt,
)
from auto_agent.tools.file_tools import (
  filesystem_tool_r,
  write_autotune_specs_tool,
)
from auto_agent.tools.search_api_tool import search_api_tool


# 1. Planner Agent
# This agent identifies parameters, creates the template, and defines the search space.
# It saves them to session state instead of calling the tool directly.
def create_autotune_planner_agent(
  model_name: str = MODEL_NAME,
) -> CustomLlmAgent:
  return CustomLlmAgent(
    name="AutotunePlannerAgent",
    model=model_name,
    generate_content_config=model_config,
    planner=get_thinking_planner("high"),
    instruction=autotune_prompt.PROMPT,
    description="Prepares code template and search space for auto-tuning Pallas kernels.",
    tools=[filesystem_tool_r, write_autotune_specs_tool, search_api_tool],
  )


autotune_planner_agent = create_autotune_planner_agent()


# 2. Runner Agent
class AutotuneRunner(BaseAgent):
  """Executes autotuning via HTTP endpoint."""

  name: Optional[str] = None
  output_key: Optional[str] = None

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

    dependencies = {}
    base_kernel_path = ctx.session.state.get("base_kernel_path", "")
    if base_kernel_path and os.path.exists(base_kernel_path):
      try:
        with open(base_kernel_path, "r") as f:
          dependencies["base_kernel.py"] = f.read()
      except Exception as e:
        logging.warning(
          f"[{self.name}] Failed to read base kernel file {base_kernel_path}: {e}"
        )

    # Read the pre-generated test file (which is the rigorous harness)
    test_file_path = ctx.session.state.get("test_file_path", "")
    harness_code = ""
    if test_file_path and os.path.exists(test_file_path):
      with open(test_file_path, "r") as f:
        harness_code = f.read()

    # Construct a full script for eval_server by placing the kernel and then the harness.
    # Override optimized_mod by defining a dummy class in the script.
    # Since optimized_mod always expects a 'computation' function, alias it here.
    full_code_template = (
      f"{code_template}\n\nclass _optimized_mod:\n"
      f"     computation = computation\n"
      "import sys\n"
      "sys.modules['optimized_kernel'] = _optimized_mod\n\n" + harness_code
    )

    logging.info(f"[{self.name}] Running autotune for {kernel_name}")

    try:
      results = await autotune_kernel(
        kernel_name=kernel_name,
        code_template=full_code_template,
        search_space=search_space,
        backend="tpu",
        dependencies=dependencies,
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


def create_autotune_runner(model_name: str = MODEL_NAME) -> AutotuneRunner:
  return AutotuneRunner(
    name="AutotuneRunner",
    output_key="autotune_results",
  )


autotune_runner = create_autotune_runner()


class ApplyBestConfigAgent(BaseAgent):
  """Programmatic agent that applies autotuning best_config to the optimized kernel file."""

  def __init__(self, name: str):
    super().__init__(name=name)

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:

    # Get the best config directly from state (already verified by CombinedAutotuneAgent)
    best_config = ctx.session.state.get("autotune_results", {}).get(
      "best_config", {}
    )

    # Read the original code template
    autotune_specs_path = ctx.session.state.get("autotune_specs_path", "")
    try:
      with open(autotune_specs_path, "r") as f:
        specs = json.load(f)
      code_template = specs.get("code_template", "")
    except Exception as e:
      error_msg = (
        f"Failed to read code_template from {autotune_specs_path}: {e}"
      )
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            "apply_config_status": {"status": "error", "message": error_msg}
          }
        ),
      )
      return

    # Replace placeholders with best values
    final_code = code_template
    for k, v in best_config.items():
      final_code = final_code.replace(f"{{{k}}}", str(v))

    # Write to optimized kernel
    optimized_kernel_path = ctx.session.state.get("optimized_kernel_path", "")
    try:
      with open(optimized_kernel_path, "w") as f:
        f.write(final_code)
      logging.info(
        f"[{self.name}] Applied best config to {optimized_kernel_path}: {best_config}"
      )
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            "apply_config_status": {
              "status": "success",
              "best_config": best_config,
            }
          }
        ),
      )
    except Exception as e:
      error_msg = (
        f"Failed to write optimized kernel to {optimized_kernel_path}: {e}"
      )
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            "apply_config_status": {"status": "error", "message": error_msg}
          }
        ),
      )


def create_apply_best_config_agent() -> ApplyBestConfigAgent:
  return ApplyBestConfigAgent(name="ApplyBestConfigAgent")


apply_best_config_agent = create_apply_best_config_agent()


# 4. Summarizer Agent
# This agent reads results from state and talks to the user.
def create_autotune_summary_agent(
  model_name: str = MODEL_NAME,
) -> CustomLlmAgent:
  return CustomLlmAgent(
    name="AutotuneSummaryAgent",
    model=model_name,
    generate_content_config=model_config,
    instruction=summary_prompt.PROMPT,
    description="Summarizes autotuning results.",
    tools=[filesystem_tool_r],
    output_key="autotuning_summary",
  )


autotune_summary_agent = create_autotune_summary_agent()


class CombinedAutotuneAgent(BaseAgent):
  """Chains autotuning steps and conditionally applies best config."""

  planner_agent: Optional[BaseAgent] = None
  runner_agent: Optional[BaseAgent] = None
  apply_config_agent: Optional[BaseAgent] = None
  summary_agent: Optional[BaseAgent] = None

  def __init__(
    self,
    name: str,
    planner_agent: BaseAgent,
    runner_agent: BaseAgent,
    apply_config_agent: BaseAgent,
    summary_agent: BaseAgent,
  ):
    super().__init__(
      name=name,
      planner_agent=planner_agent,
      runner_agent=runner_agent,
      apply_config_agent=apply_config_agent,
      summary_agent=summary_agent,
    )

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    logging.info(f"[{self.name}] Running AutotunePlannerAgent...")
    async for event in self.planner_agent.run_async(ctx):
      yield event

    logging.info(f"[{self.name}] Running AutotuneRunner...")
    async for event in self.runner_agent.run_async(ctx):
      yield event

    autotune_results = ctx.session.state.get("autotune_results", {})
    if (
      autotune_results.get("status") == "success"
      and autotune_results.get("best_config") is not None
      and autotune_results.get("best_time_ms") is not None
    ):
      logging.info(f"[{self.name}] Running ApplyBestConfigAgent...")
      async for event in self.apply_config_agent.run_async(ctx):
        yield event
    else:
      logging.warning(
        f"[{self.name}] Autotune was not successful or no best configuration"
        " found. Skipping ApplyBestConfigAgent."
      )

    logging.info(f"[{self.name}] Running AutotuneSummaryAgent...")
    async for event in self.summary_agent.run_async(ctx):
      yield event


def create_autotune_agent(
  model_name: str = MODEL_NAME,
) -> CombinedAutotuneAgent:
  return CombinedAutotuneAgent(
    name="AutotuneAgent",
    planner_agent=create_autotune_planner_agent(model_name),
    runner_agent=create_autotune_runner(model_name),
    apply_config_agent=create_apply_best_config_agent(),
    summary_agent=create_autotune_summary_agent(model_name),
  )


autotune_agent = create_autotune_agent()

__all__ = ["autotune_agent", "create_autotune_agent"]
