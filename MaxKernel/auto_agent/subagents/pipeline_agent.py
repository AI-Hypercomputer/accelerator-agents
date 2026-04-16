"""Autonomous pipeline agent for chaining kernel generation steps."""

import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions


class AutonomousPipelineAgent(BaseAgent):
  """Chains kernel generation sub-agents automatically with an improvement loop."""

  plan_agent: BaseAgent
  implement_agent: BaseAgent
  validate_agent: BaseAgent
  test_gen_agent: BaseAgent
  test_run_agent: BaseAgent
  profile_agent: BaseAgent
  max_iterations: int = 2

  def __init__(
    self,
    name: str,
    plan_agent: BaseAgent,
    implement_agent: BaseAgent,
    validate_agent: BaseAgent,
    test_gen_agent: BaseAgent,
    test_run_agent: BaseAgent,
    profile_agent: BaseAgent,
    max_iterations: int = 2,
  ):
    super().__init__(
      name=name,
      plan_agent=plan_agent,
      implement_agent=implement_agent,
      validate_agent=validate_agent,
      test_gen_agent=test_gen_agent,
      test_run_agent=test_run_agent,
      profile_agent=profile_agent,
      max_iterations=max_iterations,
    )

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    iteration = 0
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

      # Step 3: Validate
      logging.info(f"[{self.name}] Running ValidateKernelCompilationAgent...")
      async for event in self.validate_agent.run_async(ctx):
        yield event

      # Check if compilation succeeded
      compilation_status = ctx.session.state.get(
        "kernel_compilation_status", {}
      )
      if not compilation_status.get("success", False):
        logging.error(
          f"[{self.name}] Compilation failed. Looping back to planning."
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
        iteration += 1
        continue

      # Step 5: Test Run
      logging.info(f"[{self.name}] Running UnifiedTestAgent...")
      async for event in self.test_run_agent.run_async(ctx):
        yield event

      # Check if tests passed
      test_results = ctx.session.state.get("test_results", {})
      if not test_results.get("success", False):
        logging.error(f"[{self.name}] Tests failed. Looping back to planning.")
        iteration += 1
        continue

      # Step 6: Profile
      logging.info(f"[{self.name}] Running ProfileAgentOrchestrator...")
      async for event in self.profile_agent.run_async(ctx):
        yield event

      # Step 7: Check if improvement is needed
      needs_improvement = ctx.session.state.get("needs_improvement", False)

      if not needs_improvement:
        logging.info(
          f"[{self.name}] No further improvement needed or agent decided to stop. Stopping pipeline."
        )
        break

      logging.info(
        f"[{self.name}] Improvement needed. Looping back to planning..."
      )
      iteration += 1

    if iteration >= self.max_iterations:
      logging.warning(
        f"[{self.name}] Maximum iterations reached ({self.max_iterations}). Stopping pipeline."
      )

    yield Event(
      author=self.name,
      actions=EventActions(
        state_delta={
          "pipeline_status": "Completed",
          "pipeline_iteration": iteration,
        }
      ),
    )
