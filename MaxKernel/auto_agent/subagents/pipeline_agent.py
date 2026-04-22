"""Autonomous pipeline agent for chaining kernel generation steps."""

import logging
import os
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from auto_agent.config import WORKDIR


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

    # Initialize history if not present
    if "history" not in ctx.session.state:
      ctx.session.state["history"] = []

    # Explicitly dictate standard paths in state to avoid relying on heuristic tool callbacks
    session_dir = os.path.join(WORKDIR, ctx.session.id)
    os.makedirs(session_dir, exist_ok=True)

    if "workdir" not in ctx.session.state:
      ctx.session.state["workdir"] = session_dir
      logging.info(f"[{self.name}] Set workdir: {session_dir}")

    if "base_kernel_path" not in ctx.session.state:
      ctx.session.state["base_kernel_path"] = os.path.join(
        session_dir, "base_kernel.py"
      )
      logging.info(
        f"[{self.name}] Set base_kernel_path: {ctx.session.state['base_kernel_path']}"
      )

    if "optimized_kernel_path" not in ctx.session.state:
      ctx.session.state["optimized_kernel_path"] = os.path.join(
        session_dir, "optimized_kernel.py"
      )
      logging.info(
        f"[{self.name}] Set optimized_kernel_path: {ctx.session.state['optimized_kernel_path']}"
      )

    if "kernel_plan_path" not in ctx.session.state:
      ctx.session.state["kernel_plan_path"] = os.path.join(
        session_dir, "base_kernel_plan.md"
      )
      logging.info(
        f"[{self.name}] Set kernel_plan_path: {ctx.session.state['kernel_plan_path']}"
      )

    # Yield an event to publish the explicit path state updates to the framework
    yield Event(
      author=self.name,
      actions=EventActions(
        state_delta={
          "workdir": ctx.session.state["workdir"],
          "base_kernel_path": ctx.session.state["base_kernel_path"],
          "optimized_kernel_path": ctx.session.state["optimized_kernel_path"],
          "kernel_plan_path": ctx.session.state["kernel_plan_path"],
        }
      ),
    )
    logging.info(f"[{self.name}] Published explicit path state update Event.")

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

      # Snapshot intermediate result
      kernel_path = ctx.session.state.get("optimized_kernel_path")
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
      test_output = ctx.session.state.get("test_results", {}).get("output", "")
      latency = self._extract_latency(test_output)

      snapshot = {
        "iteration": iteration,
        "code": kernel_code,
        "compiled": ctx.session.state.get("kernel_compilation_status", {}).get(
          "success", False
        ),
        "tests_passed": ctx.session.state.get("test_results", {}).get(
          "success", False
        ),
        "latency": latency,
        "summary": ctx.session.state.get("profiling_summary", ""),
      }
      current_history = ctx.session.state.get("history", [])
      updated_history = current_history + [snapshot]
      ctx.session.state["history"] = (
        updated_history  # Ensure local consistency within the loop
      )

      yield Event(
        author=self.name,
        actions=EventActions(state_delta={"history": updated_history}),
      )
      logging.info(f"[{self.name}] Saved snapshot for iteration {iteration}")

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

    best_solution = await self._apply_best_solution(ctx)

    yield Event(
      author=self.name,
      actions=EventActions(
        state_delta={
          "pipeline_status": "Completed",
          "pipeline_iteration": iteration,
          "best_iteration": best_solution["iteration"] if best_solution else -1,
        }
      ),
    )

  def _extract_latency(self, test_output: str):
    """Extracts execution time from test results output."""
    if not test_output:
      return None
    try:
      import re

      match = re.search(r"PERF_METRICS:\s*([\d.]+)", test_output)
      if match:
        latency = float(match.group(1))
        logging.info(
          f"[{self.name}] Extracted execution time from test results: {latency} ms"
        )
        return latency
    except Exception as e:
      logging.error(
        f"[{self.name}] Failed to parse execution time from test output: {e}"
      )
    return None

  async def _apply_best_solution(self, ctx: InvocationContext):
    """Finds the best solution from history and rolls back the file if needed."""
    history = ctx.session.state.get("history", [])
    valid_solutions = [
      s for s in history if s.get("compiled") and s.get("tests_passed")
    ]

    best_solution = None
    if valid_solutions:
      # Try to sort by latency (lower is better)
      try:
        solutions_with_latency = [
          s for s in valid_solutions if s.get("latency") is not None
        ]
        if solutions_with_latency:
          best_solution = min(
            solutions_with_latency, key=lambda x: x["latency"]
          )
        else:
          logging.warning(
            f"[{self.name}] No latency metrics found. Falling back to LLM selection."
          )
          best_solution = await self._select_best_with_llm(valid_solutions)
      except Exception as e:
        logging.error(
          f"[{self.name}] Error selecting best solution by latency: {e}"
        )
        best_solution = await self._select_best_with_llm(valid_solutions)

    if best_solution:
      logging.info(
        f"[{self.name}] Best solution found from iteration {best_solution['iteration']}"
      )

      # Rollback if needed
      current_code = ""
      kernel_path = ctx.session.state.get("optimized_kernel_path")
      if kernel_path and os.path.exists(kernel_path):
        try:
          with open(kernel_path, "r") as f:
            current_code = f.read()
        except Exception as e:
          logging.error(
            f"[{self.name}] Failed to read current kernel file: {e}"
          )

      if best_solution["code"] != current_code:
        logging.info(
          f"[{self.name}] Reverting kernel file to best solution from iteration {best_solution['iteration']}"
        )
        if kernel_path:
          try:
            with open(kernel_path, "w") as f:
              f.write(best_solution["code"])
          except Exception as e:
            logging.error(
              f"[{self.name}] Failed to write best solution to file: {e}"
            )
      else:
        logging.info(
          f"[{self.name}] Current file is already the best solution."
        )

    return best_solution

  async def _select_best_with_llm(self, valid_solutions):
    """Fallback method to let LLM select the best solution based on summaries."""
    if not valid_solutions:
      return None
    if len(valid_solutions) == 1:
      return valid_solutions[0]

    logging.info(
      f"[{self.name}] Invoking LLM to select best solution from {len(valid_solutions)} candidates."
    )

    prompt = "You are an expert in performance optimization. Compare the following profiling summaries for different iterations of a kernel and decide which one is the best solution (maximizes performance/efficiency). Answer with ONLY the iteration number of the best solution.\n\n"

    for s in valid_solutions:
      prompt += f"Iteration {s['iteration']}:\n{s['summary']}\n\n"

    try:
      model = None
      if hasattr(self.plan_agent, "model"):
        model = self.plan_agent.model
      elif hasattr(self.implement_agent, "model"):
        model = self.implement_agent.model

      if model:
        text = ""
        async for chunk in model.generate_content_async(prompt):
          if hasattr(chunk, "text") and chunk.text:
            text += chunk.text
        text = text.strip()
        import re

        match = re.search(r"\d+", text)
        if match:
          iter_num = int(match.group())
          for s in valid_solutions:
            if s["iteration"] == iter_num:
              return s

      logging.warning(
        f"[{self.name}] LLM selection failed or no model found. Defaulting to last valid solution."
      )
      return valid_solutions[-1]
    except Exception as e:
      logging.error(f"[{self.name}] Error in LLM selection: {e}")
      return valid_solutions[-1]
