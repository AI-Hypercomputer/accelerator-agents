"""Kernel writing and compilation validation agents."""

import logging
import os
from typing import AsyncGenerator, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from auto_agent.callbacks import (
  add_pallas_docs,
  add_workdir_callback,
  extract_fix_summary,
  get_tpu_version_callback,
  load_kernel_and_plan_to_state,
  load_single_kernel_to_state,
)
from auto_agent.config import model_config, thinking_planner
from auto_agent.constants import MODEL_NAME
from auto_agent.custom_types import CustomLlmAgent
from auto_agent.subagents.kernel_writing.kernel_compilation import (
  KernelCompilationChecker,
)
from auto_agent.subagents.kernel_writing.prompts import (
  add_debug_statements,
  cleanup_debug_statements,
  fix_kernel_compilation,
  kernel_compilation_summary,
  kernel_implementation_prompt,
  kernel_planning_prompt,
  read_file_prompt,
)
from auto_agent.tools.search_api_tool import search_api_tool
from auto_agent.tools.tools import filesystem_tool_rw, vertex_ai_rag_tool


class KernelCompilationValidationLoop(BaseAgent):
  """Custom loop agent that validates kernel compilation and fixes errors until valid or max retries reached."""

  compilation_checker: Optional[BaseAgent] = None
  fix_agent: Optional[BaseAgent] = None
  debug_agent: Optional[BaseAgent] = None
  max_retries: int = 4

  def __init__(
    self,
    name: str,
    compilation_checker: BaseAgent,
    fix_agent: BaseAgent,
    debug_agent: Optional[BaseAgent] = None,
    max_retries: int = 4,
  ):
    super().__init__(
      name=name,
      compilation_checker=compilation_checker,
      fix_agent=fix_agent,
      debug_agent=debug_agent,
      max_retries=max_retries,
    )

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Validation loop: compile -> fix -> repeat until valid or max retries."""

    # Check if a kernel file was actually generated
    kernel_path = ctx.session.state.get("optimized_kernel_path", "")

    # Ensure optimized_kernel_path exists in state for template injection
    if "optimized_kernel_path" not in ctx.session.state:
      ctx.session.state["optimized_kernel_path"] = ""

    if not kernel_path:
      logging.error(
        f"[{self.name}] No kernel file path found in state. Kernel implementation may have failed."
      )
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            "kernel_compilation_status": {
              "success": False,
              "retries": 0,
              "message": "No kernel file was generated. Cannot validate compilation.",
              "valid": False,
            }
          }
        ),
      )
      return

    if not os.path.exists(kernel_path):
      error_msg = f"Kernel file not found at {kernel_path}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            "kernel_compilation_status": {
              "success": False,
              "retries": 0,
              "message": error_msg,
              "valid": False,
            }
          }
        ),
      )
      return

    retry_count = 0

    # Initialize compilation history to track all attempts
    if "compilation_history" not in ctx.session.state:
      ctx.session.state["compilation_history"] = []

    while retry_count < self.max_retries:
      logging.info(
        f"[{self.name}] Compilation validation attempt {retry_count + 1}/{self.max_retries}"
      )

      # Run compilation check
      async for event in self.compilation_checker.run_async(ctx):
        yield event

      # Check if compilation passed
      # KernelCompilationChecker returns "Success" string on success, or error message on failure
      compilation_result = ctx.session.state.get("compilation_results", "")
      compilation_valid = compilation_result == "Success"

      # Record this attempt in history
      attempt_record = {
        "attempt": retry_count + 1,
        "result": compilation_result,
        "success": compilation_valid,
        "fix_summary": ctx.session.state.get("fix_summary", None),
      }
      ctx.session.state["compilation_history"].append(attempt_record)
      logging.info(
        f"[{self.name}] Recorded attempt {retry_count + 1} in compilation history"
      )

      # Clear fix_summary for next iteration
      ctx.session.state["fix_summary"] = None

      if compilation_valid:
        logging.info(f"[{self.name}] ✓ Kernel compilation succeeded!")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              "kernel_compilation_status": {
                "success": True,
                "retries": retry_count,
                "message": "Kernel compiled successfully",
                "valid": True,
              }
            }
          ),
        )
        return

      # If compilation failed and we have retries left, try to fix
      if retry_count < self.max_retries - 1:
        logging.info(f"[{self.name}] Compilation failed. Attempting fix...")
        async for event in self.fix_agent.run_async(ctx):
          yield event

        # Add debugging statements after 2nd retry if debug agent is available
        if self.debug_agent and retry_count >= 1:
          logging.info(
            f"[{self.name}] Adding debugging statements to diagnose persistent issues..."
          )
          async for event in self.debug_agent.run_async(ctx):
            yield event

        retry_count += 1
      else:
        # Max retries reached
        logging.error(
          f"[{self.name}] ✗ Max retries reached. Kernel still has compilation errors."
        )
        compilation_error_msg = ctx.session.state.get(
          "compilation_results", "Unknown error"
        )
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              "kernel_compilation_status": {
                "success": False,
                "retries": retry_count,
                "message": f"Kernel compilation failed after {self.max_retries} attempts",
                "valid": False,
                "final_errors": compilation_error_msg,
              }
            }
          ),
        )
        return


class ValidateKernelCompilationAgent(BaseAgent):
  """Orchestrator for kernel validation that handles missing file paths gracefully."""

  read_file_agent: BaseAgent
  validation_loop_agent: BaseAgent
  cleanup_agent: BaseAgent
  summary_agent: BaseAgent

  def __init__(
    self,
    name: str,
    read_file_agent: BaseAgent,
    validation_loop_agent: BaseAgent,
    cleanup_agent: BaseAgent,
    summary_agent: BaseAgent,
    description: str = "",
  ):
    super().__init__(
      name=name,
      description=description,
      read_file_agent=read_file_agent,
      validation_loop_agent=validation_loop_agent,
      cleanup_agent=cleanup_agent,
      summary_agent=summary_agent,
    )

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    # Step 1: Ensure we have a kernel file
    path = ctx.session.state.get("optimized_kernel_path")

    # If path is missing or invalid, try to find it using the read_file_agent
    if not path or not os.path.exists(path):
      async for event in self.read_file_agent.run_async(ctx):
        yield event

      # Check if the agent successfully found the file
      path = ctx.session.state.get("optimized_kernel_path")
      if not path or not os.path.exists(path):
        # If still not found, we assume the agent asked the user for clarification.
        # We stop execution here to wait for user input.
        return

    # Step 2: Run validation loop
    async for event in self.validation_loop_agent.run_async(ctx):
      yield event

    # Step 3: Cleanup debug statements
    async for event in self.cleanup_agent.run_async(ctx):
      yield event

    # Step 4: Summary
    async for event in self.summary_agent.run_async(ctx):
      yield event


# Plan-based kernel writing agents (separate, not sequential)
# These are called independently by the root orchestrator to allow user interaction between steps

plan_kernel_agent = CustomLlmAgent(
  name="PlanKernelAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=kernel_planning_prompt.PROMPT,
  description="Creates or revises a detailed optimization plan for a Pallas kernel.",
  tools=(
    [search_api_tool, filesystem_tool_rw, vertex_ai_rag_tool]
    if vertex_ai_rag_tool
    else [search_api_tool, filesystem_tool_rw]
  ),
  before_agent_callback=[
    add_pallas_docs,
    get_tpu_version_callback,
    add_workdir_callback,
  ],
)

# Kernel compilation validation agents
# Read file agent for validation - extracts kernel path from user message or state
read_file_for_validation_agent = CustomLlmAgent(
  name="ReadFileForValidationAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=read_file_prompt.PROMPT,
  description="Reads the kernel file mentioned by the user or from state for validation.",
  tools=[filesystem_tool_rw],
)

fix_kernel_compilation_agent = CustomLlmAgent(
  name="FixKernelCompilationAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=fix_kernel_compilation.PROMPT,
  description="Fixes compilation errors in the generated kernel while preserving optimization strategy.",
  tools=(
    [search_api_tool, filesystem_tool_rw, vertex_ai_rag_tool]
    if vertex_ai_rag_tool
    else [search_api_tool, filesystem_tool_rw]
  ),
  before_agent_callback=load_kernel_and_plan_to_state,
  after_model_callback=extract_fix_summary,
  include_contents="none",
)

add_debug_statements_agent = CustomLlmAgent(
  name="AddDebugStatementsAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=add_debug_statements.PROMPT,
  description="Adds strategic debugging statements to diagnose persistent compilation issues.",
  tools=[filesystem_tool_rw],
  before_agent_callback=load_kernel_and_plan_to_state,
  include_contents="none",
)

cleanup_debug_statements_agent = CustomLlmAgent(
  name="CleanupDebugStatementsAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=cleanup_debug_statements.PROMPT,
  description="Removes debugging statements from successfully compiled kernel.",
  tools=[filesystem_tool_rw],
  before_agent_callback=load_single_kernel_to_state,
  include_contents="none",
)

kernel_compilation_checker_for_validation = KernelCompilationChecker(
  name="KernelCompilationCheckerForValidation",
  input_key="kernel_code",
  output_key="compilation_results",
  before_agent_callback=load_single_kernel_to_state,
)

kernel_compilation_validation_loop = KernelCompilationValidationLoop(
  name="KernelCompilationValidationLoop",
  compilation_checker=kernel_compilation_checker_for_validation,
  fix_agent=fix_kernel_compilation_agent,
  debug_agent=add_debug_statements_agent,
  max_retries=4,
)

kernel_compilation_summary_agent = CustomLlmAgent(
  name="KernelCompilationSummaryAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=kernel_compilation_summary.PROMPT,
  description="Summarizes kernel compilation validation results with full trace on failure.",
  include_contents="none",
)

# Standalone validation orchestration agent (invoked by root when user requests validation)
validate_kernel_compilation_agent = ValidateKernelCompilationAgent(
  name="ValidateKernelCompilationAgent",
  read_file_agent=read_file_for_validation_agent,
  validation_loop_agent=kernel_compilation_validation_loop,
  cleanup_agent=cleanup_debug_statements_agent,
  summary_agent=kernel_compilation_summary_agent,
  description="Validates kernel compilation with automatic error fixing, debugging, and provides summary. Invoked when user requests validation.",
)

# Implementation agent - implements kernel
implement_kernel_agent = CustomLlmAgent(
  name="ImplementKernelAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=kernel_implementation_prompt.PROMPT,
  description="Implements the optimized Pallas kernel following the plan.",
  tools=(
    [search_api_tool, filesystem_tool_rw, vertex_ai_rag_tool]
    if vertex_ai_rag_tool
    else [search_api_tool, filesystem_tool_rw]
  ),
)

__all__ = [
  "KernelCompilationValidationLoop",
  "ValidateKernelCompilationAgent",
  "plan_kernel_agent",
  "implement_kernel_agent",
  "validate_kernel_compilation_agent",
  "read_file_for_validation_agent",
]
