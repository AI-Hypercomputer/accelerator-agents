"""Specialized correctness-only pipeline for Beam Search workers."""

import logging
import os
from typing import AsyncGenerator

from auto_agent.subagents.pipeline_agent import AutonomousPipelineAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext

class BeamWorkerPipeline(AutonomousPipelineAgent):
  """Subclass of AutonomousPipelineAgent that exits early on correctness success."""

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    iteration = 0
    yield self._initialize_state(ctx)

    while iteration < self.max_iterations:
      logging.info(
        f"[{self.name}] Starting pipeline iteration {iteration + 1}/{self.max_iterations}"
      )

      # Step 1: Plan
      logging.info(f"[{self.name}] Running PlanKernelAgent...")
      async for event in self.plan_agent.run_async(ctx):
        yield event

      # Step 2: Implement
      logging.info(f"[{self.name}] Running ImplementKernelAgent...")
      async for event in self.implement_agent.run_async(ctx):
        yield event

      # Step 3: Validate (Compilation)
      logging.info(f"[{self.name}] Running ValidateKernelCompilationAgent...")
      async for event in self.validate_agent.run_async(ctx):
        yield event

      # Check if compilation succeeded
      compilation_status = ctx.session.state.get("kernel_compilation_status", {})
      if not compilation_status.get("success", False):
        logging.error(
          f"[{self.name}] Compilation failed. Looping back to planning."
        )
        self._save_iteration_files(
          ctx, iteration, keys_to_save=["optimized_kernel_path"]
        )
        iteration += 1
        continue

      # Step 4: Test Gen
      logging.info(f"[{self.name}] Running ValidatedTestGenerationAgent...")
      async for event in self.test_gen_agent.run_async(ctx):
        yield event

      # Check if test generation succeeded
      validation_status = ctx.session.state.get("validation_loop_status", {})
      if not validation_status.get("success", False):
        logging.error(
          f"[{self.name}] Test generation/validation failed. Looping back to planning."
        )
        self._save_iteration_files(
          ctx,
          iteration,
          keys_to_save=["optimized_kernel_path", "test_file_path"],
        )
        iteration += 1
        continue

      # Step 5: Test Run
      logging.info(f"[{self.name}] Running UnifiedTestAgent...")
      async for event in self.test_run_agent.run_async(ctx):
        yield event

      # Check if correctness tests passed (Early Exit check)
      test_results = ctx.session.state.get("test_results", {})
      if not test_results.get("success", False):
        logging.error(f"[{self.name}] Tests failed. Looping back to planning.")
        self._save_iteration_files(
          ctx,
          iteration,
          keys_to_save=["optimized_kernel_path", "test_file_path"]
        )
        iteration += 1
        continue

      kernel_path = ctx.session.state.get("optimized_kernel_path")
      # Snapshot the successful implementation
      kernel_code = ""
      if kernel_path and os.path.exists(kernel_path):
        try:
          with open(kernel_path, "r") as f:
            kernel_code = f.read()
        except Exception as e:
          logging.error(
            f"[{self.name}] Failed to read kernel file for snapshot: {e}"
          )

      # Extract latency
      latency = self._extract_latency(ctx)
      
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            "worker_status": "Success",
            "kernel_code": kernel_code,
            "compilation_status": ctx.session.state.get(
              "kernel_compilation_status", {}
            ),
            "test_status": ctx.session.state.get("test_results", {}),
            "latency_ms": latency,
            "pipeline_status": "Completed",
            "pipeline_iteration": iteration,
            "best_iteration": iteration,
          }
        ),
      )
      self._save_iteration_files(ctx, iteration)
      return  # Terminate worker early since code is correct

    # If we exit the loop, it means we failed to reach correctness
    logging.warning(
      f"[{self.name}] Failed to generate correct code within iteration limit."
    )
    yield Event(
      author=self.name,
      actions=EventActions(
        state_delta={
          "worker_status": "Failed",
          "pipeline_status": "Failed",
        }
      ),
    )
