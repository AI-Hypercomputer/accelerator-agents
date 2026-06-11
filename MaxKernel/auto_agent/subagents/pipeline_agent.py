"""Autonomous pipeline agent for chaining kernel generation steps."""

import logging
import os
import re
import shutil
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models import LlmRequest
from google.genai import types

from auto_agent.config import WORKDIR


class AutonomousPipelineAgent(BaseAgent):
  """Chains kernel generation sub-agents automatically with an improvement loop.

  Planning phase generates one plan per hypothesis found in idea_store_path.
  Each implementation branch uses exactly one hypothesis plan, in isolation.
  After all branches complete, solutions are ranked and the best is selected.
  """

  plan_agent: BaseAgent
  implement_agent: BaseAgent
  validate_agent: BaseAgent
  test_gen_agent: BaseAgent
  test_run_agent: BaseAgent
  autotune_agent: BaseAgent
  profile_agent: BaseAgent
  max_iterations: int = 2
  num_impl_branches: int = 3
  idea_store_path: str = ""

  def __init__(
    self,
    name: str,
    plan_agent: BaseAgent,
    implement_agent: BaseAgent,
    validate_agent: BaseAgent,
    test_gen_agent: BaseAgent,
    test_run_agent: BaseAgent,
    autotune_agent: BaseAgent,
    profile_agent: BaseAgent,
    max_iterations: int = 2,
    num_impl_branches: int = 3,
    idea_store_path: str = "",
  ):
    super().__init__(
      name=name,
      plan_agent=plan_agent,
      implement_agent=implement_agent,
      validate_agent=validate_agent,
      test_gen_agent=test_gen_agent,
      test_run_agent=test_run_agent,
      autotune_agent=autotune_agent,
      profile_agent=profile_agent,
      max_iterations=max_iterations,
      num_impl_branches=num_impl_branches,
      idea_store_path=idea_store_path,
    )

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    iteration = 0

    yield self._initialize_state(ctx)

    while iteration < self.max_iterations:
      logging.info(
        f"[{self.name}] Starting pipeline iteration {iteration + 1}/{self.max_iterations}"
      )

      # Step 1: Hypothesis-based planning phase — generates one plan per hypothesis.
      logging.info(f"[{self.name}] Running hypothesis-based planning phase...")
      async for event in self._run_hypothesis_planning_phase(ctx):
        yield event

      hypothesis_plans = ctx.session.state.get("hypothesis_plans", [])
      num_branches = len(hypothesis_plans) if hypothesis_plans else self.num_impl_branches
      logging.info(
        f"[{self.name}] Planning complete. Running {num_branches} implementation branch(es)."
      )

      # Initialize branch results for this iteration.
      ctx.session.state["branch_results"] = []
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={"branch_results": []}),
      )

      # Steps 2–7: One implementation branch per hypothesis plan.
      for branch_id in range(num_branches):
        # Assign the branch its dedicated hypothesis plan (isolation: no shared plan).
        if hypothesis_plans and branch_id < len(hypothesis_plans):
          plan_info = hypothesis_plans[branch_id]
          plan_path = plan_info["plan_path"]
          ctx.session.state["kernel_plan_path"] = plan_path
          # Clear hypothesis content so the impl agent only sees its plan file.
          ctx.session.state["current_hypothesis"] = ""
          ctx.session.state["current_hypothesis_id"] = ""
          yield Event(
            author=self.name,
            actions=EventActions(
              state_delta={
                "kernel_plan_path": plan_path,
                "current_hypothesis": "",
                "current_hypothesis_id": "",
              }
            ),
          )
          logging.info(
            f"[{self.name}] Branch {branch_id}: assigned plan for "
            f"hypothesis '{plan_info['hypothesis_file']}' at {plan_path}"
          )

        logging.info(
          f"[{self.name}] Starting implementation branch {branch_id + 1}/{num_branches}..."
        )
        async for event in self._run_impl_branch(ctx, branch_id, iteration):
          yield event

      # Rank all branch results and select the best.
      branch_results = ctx.session.state.get("branch_results", [])
      best_branch = self._select_best_branch(branch_results)

      ranking_summary = self._build_ranking_summary(branch_results, hypothesis_plans)
      logging.info(f"[{self.name}] Branch ranking summary:\n{ranking_summary}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={"branch_ranking_summary": ranking_summary}
        ),
      )

      if best_branch:
        logging.info(
          f"[{self.name}] Best branch: {best_branch['branch_id']} "
          f"with latency {best_branch.get('latency_ms')} ms"
        )
        for event in self._apply_best_branch(ctx, best_branch):
          yield event
      else:
        logging.warning(
          f"[{self.name}] No successful branch found in iteration {iteration}."
        )

      # Snapshot best branch result into history.
      snapshot = {
        "iteration": iteration,
        "kernel_code": best_branch.get("kernel_code", "") if best_branch else "",
        "compilation_status": best_branch.get("compilation_status", {}) if best_branch else {},
        "test_status": best_branch.get("test_status", {}) if best_branch else {},
        "latency_ms": best_branch.get("latency_ms") if best_branch else None,
        "profiling_summary": best_branch.get("profiling_summary", "") if best_branch else "",
      }
      current_history = ctx.session.state.get("history", [])
      updated_history = current_history + [snapshot]
      ctx.session.state["history"] = updated_history
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={"history": updated_history}),
      )
      logging.info(f"[{self.name}] Saved snapshot for iteration {iteration}")

      self._save_iteration_files(ctx, iteration)

      needs_improvement = True
      if not needs_improvement:
        logging.info(
          f"[{self.name}] No further improvement needed. Stopping pipeline."
        )
        break

      logging.info(f"[{self.name}] Looping back to planning...")
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

  # ---------------------------------------------------------------------------
  # Hypothesis-based planning phase
  # ---------------------------------------------------------------------------

  async def _run_hypothesis_planning_phase(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Generates one plan per hypothesis file found in idea_store_path.

    For each hypothesis:
      1. Injects the hypothesis content into session state.
      2. Runs plan_agent, which writes its plan to a hypothesis-specific path.
      3. Records the plan path in hypothesis_plans list.

    Falls back to a single plan_agent run if no idea_store_path or no hypotheses found.
    """
    idea_store = self.idea_store_path
    session_dir = ctx.session.state.get("workdir", "")

    if not idea_store or not os.path.isdir(idea_store):
      logging.warning(
        f"[{self.name}] idea_store_path '{idea_store}' not found or not a directory. "
        "Falling back to single planning run."
      )
      async for event in self.plan_agent.run_async(ctx):
        yield event
      return

    hypothesis_files = sorted([
      f for f in os.listdir(idea_store)
      if f.startswith("Hypothesis_") and f.endswith(".md")
    ])

    if not hypothesis_files:
      logging.warning(
        f"[{self.name}] No Hypothesis_*.md files in {idea_store}. "
        "Falling back to single planning run."
      )
      async for event in self.plan_agent.run_async(ctx):
        yield event
      return

    logging.info(
      f"[{self.name}] Found {len(hypothesis_files)} hypothesis file(s): {hypothesis_files}"
    )

    hypothesis_plans = []

    for i, hyp_file in enumerate(hypothesis_files):
      hyp_path = os.path.join(idea_store, hyp_file)
      try:
        with open(hyp_path, "r") as f:
          hypothesis_content = f.read()
      except Exception as e:
        logging.error(
          f"[{self.name}] Failed to read hypothesis file '{hyp_file}': {e}. Skipping."
        )
        continue

      plan_path = os.path.join(session_dir, f"plan_hypothesis_{i}.md")

      # Inject hypothesis into state so the planning prompt can reference it.
      ctx.session.state["current_hypothesis"] = hypothesis_content
      ctx.session.state["current_hypothesis_id"] = i
      ctx.session.state["kernel_plan_path"] = plan_path

      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            "current_hypothesis": hypothesis_content,
            "current_hypothesis_id": i,
            "kernel_plan_path": plan_path,
          }
        ),
      )

      logging.info(
        f"[{self.name}] Planning for hypothesis {i} ({hyp_file}) → {plan_path}"
      )
      async for event in self.plan_agent.run_async(ctx):
        yield event

      hypothesis_plans.append({
        "hypothesis_id": i,
        "hypothesis_file": hyp_file,
        "plan_path": plan_path,
      })
      logging.info(
        f"[{self.name}] Completed planning for hypothesis {i}. Plan at: {plan_path}"
      )

    # Store all plans and clear transient hypothesis state.
    ctx.session.state["hypothesis_plans"] = hypothesis_plans
    ctx.session.state["current_hypothesis"] = ""
    ctx.session.state["current_hypothesis_id"] = ""

    yield Event(
      author=self.name,
      actions=EventActions(
        state_delta={
          "hypothesis_plans": hypothesis_plans,
          "current_hypothesis": "",
          "current_hypothesis_id": "",
        }
      ),
    )

    logging.info(
      f"[{self.name}] Hypothesis planning phase complete. "
      f"Generated {len(hypothesis_plans)} plan(s)."
    )

  # ---------------------------------------------------------------------------
  # Branch execution
  # ---------------------------------------------------------------------------

  async def _run_impl_branch(
    self, ctx: InvocationContext, branch_id: int, iteration: int
  ) -> AsyncGenerator[Event, None]:
    """Runs the full implementation pipeline for a single branch in an isolated directory."""
    session_dir = ctx.session.state.get("workdir", "")
    branch_dir = os.path.join(session_dir, f"iter_{iteration}_branch_{branch_id}")
    os.makedirs(branch_dir, exist_ok=True)

    # Redirect artifact paths to this branch's directory.
    branch_paths = {
      "optimized_kernel_path": os.path.join(branch_dir, "optimized_kernel.py"),
      "test_file_path": os.path.join(branch_dir, "test_optimized_kernel.py"),
      "profiling_script_path": os.path.join(branch_dir, "profile_optimized_kernel.py"),
      "autotune_specs_path": os.path.join(branch_dir, "autotune_specs.json"),
      "autotune_results_path": os.path.join(branch_dir, "autotune_results.json"),
    }
    ctx.session.state.update(branch_paths)
    yield Event(author=self.name, actions=EventActions(state_delta=branch_paths))

    # Step 2: Implement
    logging.info(f"[{self.name}] Branch {branch_id}: Running ImplementKernelAgent...")
    async for event in self.implement_agent.run_async(ctx):
      yield event

    # Step 3: Validate
    logging.info(f"[{self.name}] Branch {branch_id}: Running ValidateKernelCompilationAgent...")
    async for event in self.validate_agent.run_async(ctx):
      yield event

    compilation_status = ctx.session.state.get("kernel_compilation_status", {})
    if not compilation_status.get("success", False):
      logging.warning(
        f"[{self.name}] Branch {branch_id}: Compilation failed, skipping remaining steps."
      )
      self._record_branch_result(ctx, branch_id, success=False)
      return

    # Step 4: Test Gen
    logging.info(f"[{self.name}] Branch {branch_id}: Running ValidatedTestGenerationAgent...")
    async for event in self.test_gen_agent.run_async(ctx):
      yield event

    validation_status = ctx.session.state.get("validation_loop_status", {})
    if not validation_status.get("success", False):
      logging.warning(
        f"[{self.name}] Branch {branch_id}: Test generation failed, skipping remaining steps."
      )
      self._record_branch_result(ctx, branch_id, success=False)
      return

    # Step 5: Test Run
    logging.info(f"[{self.name}] Branch {branch_id}: Running UnifiedTestAgent...")
    async for event in self.test_run_agent.run_async(ctx):
      yield event

    test_results = ctx.session.state.get("test_results", {})
    if not test_results.get("success", False):
      logging.warning(
        f"[{self.name}] Branch {branch_id}: Tests failed, skipping autotune/profile."
      )
      self._record_branch_result(ctx, branch_id, success=False)
      return

    # Step 6: Autotune
    logging.info(f"[{self.name}] Branch {branch_id}: Running AutotuneAgent...")
    async for event in self.autotune_agent.run_async(ctx):
      yield event

    # Step 7: Profile
    logging.info(f"[{self.name}] Branch {branch_id}: Running ProfileAgentOrchestrator...")
    async for event in self.profile_agent.run_async(ctx):
      yield event

    self._record_branch_result(ctx, branch_id, success=True)

  # ---------------------------------------------------------------------------
  # Branch result tracking
  # ---------------------------------------------------------------------------

  def _record_branch_result(
    self, ctx: InvocationContext, branch_id: int, success: bool
  ):
    """Captures a snapshot of the current branch state into branch_results."""
    kernel_code = ""
    kernel_path = ctx.session.state.get("optimized_kernel_path")
    if kernel_path and os.path.exists(kernel_path):
      try:
        with open(kernel_path, "r") as f:
          kernel_code = f.read()
      except Exception as e:
        logging.error(
          f"[{self.name}] Failed to read kernel for branch {branch_id}: {e}"
        )

    # Attach hypothesis metadata if available.
    hypothesis_plans = ctx.session.state.get("hypothesis_plans", [])
    hyp_info = {}
    if hypothesis_plans and branch_id < len(hypothesis_plans):
      hyp_info = {
        "hypothesis_id": hypothesis_plans[branch_id]["hypothesis_id"],
        "hypothesis_file": hypothesis_plans[branch_id]["hypothesis_file"],
        "plan_path": hypothesis_plans[branch_id]["plan_path"],
      }

    result = {
      "branch_id": branch_id,
      "success": success,
      "kernel_code": kernel_code,
      "kernel_path": kernel_path,
      "test_file_path": ctx.session.state.get("test_file_path"),
      "compilation_status": ctx.session.state.get("kernel_compilation_status", {}),
      "test_status": ctx.session.state.get("test_results", {}),
      "latency_ms": self._extract_latency(ctx),
      "profiling_summary": ctx.session.state.get("profiling_summary", ""),
      **hyp_info,
    }

    branch_results = ctx.session.state.get("branch_results", [])
    branch_results.append(result)
    ctx.session.state["branch_results"] = branch_results
    logging.info(
      f"[{self.name}] Recorded branch {branch_id} result (success={success})"
    )

  def _select_best_branch(self, branch_results: list) -> dict | None:
    """Selects the best branch: lowest latency among fully successful ones."""
    successful = [
      r for r in branch_results
      if r.get("success")
      and r.get("compilation_status", {}).get("success")
      and r.get("test_status", {}).get("success")
    ]
    if not successful:
      return None

    with_latency = [r for r in successful if r.get("latency_ms") is not None]
    if with_latency:
      return min(with_latency, key=lambda x: x["latency_ms"])

    return successful[-1]

  def _build_ranking_summary(
    self, branch_results: list, hypothesis_plans: list
  ) -> str:
    """Builds a human-readable ranking summary of all branches."""
    if not branch_results:
      return "No branch results to rank."

    # Sort: successful + with latency first (ascending), then successful without latency, then failures.
    def sort_key(r):
      success = r.get("success", False)
      latency = r.get("latency_ms")
      if success and latency is not None:
        return (0, latency)
      elif success:
        return (1, 0.0)
      else:
        return (2, 0.0)

    sorted_results = sorted(branch_results, key=sort_key)

    lines = ["=== Branch Ranking Summary ===", ""]
    for rank, result in enumerate(sorted_results, 1):
      branch_id = result.get("branch_id", "?")
      hyp_file = result.get("hypothesis_file", "")
      hyp_label = f" [{hyp_file}]" if hyp_file else ""
      success = result.get("success", False)
      latency = result.get("latency_ms")
      compiled = result.get("compilation_status", {}).get("success", False)
      tested = result.get("test_status", {}).get("success", False)

      status_parts = []
      if not compiled:
        status_parts.append("COMPILATION FAILED")
      elif not tested:
        status_parts.append("TESTS FAILED")
      elif not success:
        status_parts.append("FAILED")
      else:
        status_parts.append("SUCCESS")
        if latency is not None:
          status_parts.append(f"latency={latency:.3f} ms")

      lines.append(
        f"  Rank {rank}: Branch {branch_id}{hyp_label} — {', '.join(status_parts)}"
      )

    lines.append("")
    # Identify the winner.
    best = next((r for r in sorted_results if r.get("success")), None)
    if best:
      hyp_file = best.get("hypothesis_file", "")
      hyp_label = f" [{hyp_file}]" if hyp_file else ""
      latency = best.get("latency_ms")
      lat_str = f" (latency={latency:.3f} ms)" if latency is not None else ""
      lines.append(
        f"  Winner: Branch {best['branch_id']}{hyp_label}{lat_str}"
      )
    else:
      lines.append("  No successful branch found.")

    return "\n".join(lines)

  # ---------------------------------------------------------------------------
  # Artifact management
  # ---------------------------------------------------------------------------

  def _apply_best_branch(
    self, ctx: InvocationContext, best_branch: dict
  ) -> list[Event]:
    """Copies best branch artifacts to canonical paths and restores state keys."""
    session_dir = ctx.session.state.get("workdir", "")
    canonical_kernel = os.path.join(session_dir, "optimized_kernel.py")
    canonical_test = os.path.join(session_dir, "test_optimized_kernel.py")

    branch_kernel = best_branch.get("kernel_path")
    branch_test = best_branch.get("test_file_path")

    if branch_kernel and os.path.exists(branch_kernel):
      shutil.copy2(branch_kernel, canonical_kernel)

    if branch_test and os.path.exists(branch_test):
      shutil.copy2(branch_test, canonical_test)

    state_delta = {
      "optimized_kernel_path": canonical_kernel,
      "test_file_path": canonical_test,
      "kernel_compilation_status": best_branch.get("compilation_status", {}),
      "test_results": best_branch.get("test_status", {}),
      "profiling_summary": best_branch.get("profiling_summary", ""),
    }
    ctx.session.state.update(state_delta)
    return [Event(author=self.name, actions=EventActions(state_delta=state_delta))]

  def _save_iteration_files(
    self,
    ctx: InvocationContext,
    iteration: int,
    keys_to_save: list[str] | None = None,
  ):
    """Saves artifacts with an iteration suffix."""
    if keys_to_save is None:
      keys_to_save = [
        "optimized_kernel_path",
        "test_file_path",
        "autotune_specs_path",
        "autotune_results_path",
      ]
    for path_key in keys_to_save:
      path = ctx.session.state.get(path_key)
      if path and os.path.exists(path):
        directory, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{iteration}{ext}"
        new_path = os.path.join(directory, new_filename)
        try:
          shutil.copy2(path, new_path)
          logging.info(f"[{self.name}] Copied {path_key} to {new_path}")
        except Exception as e:
          logging.error(
            f"[{self.name}] Failed to copy {path_key} to {new_path}: {e}"
          )

  def _initialize_state(self, ctx: InvocationContext) -> Event:
    """Initializes session state with standard paths and returns the event."""
    if "history" not in ctx.session.state:
      ctx.session.state["history"] = []

    session_dir = os.path.join(WORKDIR, ctx.session.id)
    os.makedirs(session_dir, exist_ok=True)

    if "workdir" not in ctx.session.state:
      ctx.session.state["workdir"] = session_dir
      logging.info(f"[{self.name}] Set workdir: {session_dir}")

    if "base_kernel_path" not in ctx.session.state:
      ctx.session.state["base_kernel_path"] = os.path.join(
        session_dir, "base_kernel.py"
      )

    if "optimized_kernel_path" not in ctx.session.state:
      ctx.session.state["optimized_kernel_path"] = os.path.join(
        session_dir, "optimized_kernel.py"
      )

    if "kernel_plan_path" not in ctx.session.state:
      ctx.session.state["kernel_plan_path"] = os.path.join(
        session_dir, "base_kernel_plan.md"
      )

    if "test_file_path" not in ctx.session.state:
      ctx.session.state["test_file_path"] = os.path.join(
        session_dir, "test_optimized_kernel.py"
      )

    if "profiling_script_path" not in ctx.session.state:
      ctx.session.state["profiling_script_path"] = os.path.join(
        session_dir, "profile_optimized_kernel.py"
      )

    if "autotune_specs_path" not in ctx.session.state:
      ctx.session.state["autotune_specs_path"] = os.path.join(
        session_dir, "autotune_specs.json"
      )

    if "autotune_results_path" not in ctx.session.state:
      ctx.session.state["autotune_results_path"] = os.path.join(
        session_dir, "autotune_results.json"
      )

    if "atol" not in ctx.session.state:
      ctx.session.state["atol"] = 1e-2

    if "rtol" not in ctx.session.state:
      ctx.session.state["rtol"] = 1e-2

    # Hypothesis state (cleared initially; populated during planning phase).
    ctx.session.state.setdefault("current_hypothesis", "")
    ctx.session.state.setdefault("current_hypothesis_id", "")
    ctx.session.state.setdefault("hypothesis_plans", [])

    logging.info(f"[{self.name}] State initialized.")
    return Event(
      author=self.name,
      actions=EventActions(
        state_delta={
          "workdir": ctx.session.state["workdir"],
          "base_kernel_path": ctx.session.state["base_kernel_path"],
          "optimized_kernel_path": ctx.session.state["optimized_kernel_path"],
          "kernel_plan_path": ctx.session.state["kernel_plan_path"],
          "test_file_path": ctx.session.state["test_file_path"],
          "profiling_script_path": ctx.session.state["profiling_script_path"],
          "autotune_specs_path": ctx.session.state["autotune_specs_path"],
          "autotune_results_path": ctx.session.state["autotune_results_path"],
          "atol": ctx.session.state["atol"],
          "rtol": ctx.session.state["rtol"],
          "current_hypothesis": "",
          "current_hypothesis_id": "",
          "hypothesis_plans": [],
        }
      ),
    )

  def _extract_latency(self, ctx: InvocationContext):
    """Extracts execution time from autotune results or test results output."""
    autotune_results = ctx.session.state.get("autotune_results", {})
    if autotune_results.get("status") == "success":
      latency = autotune_results.get("best_time_ms")
      if latency is not None:
        logging.info(
          f"[{self.name}] Extracted latency from autotune results: {latency} ms"
        )
        return latency

    test_output = ctx.session.state.get("test_results", {}).get("output", "")
    if not test_output:
      return None
    try:
      match = re.search(r"PERF_METRICS:\s*([\d.]+)", test_output)
      if match:
        latency = float(match.group(1))
        logging.info(
          f"[{self.name}] Extracted execution time from test results: {latency} ms (fallback)"
        )
        return latency
    except Exception as e:
      logging.error(
        f"[{self.name}] Failed to parse execution time from test output: {e}"
      )
    return None

  # ---------------------------------------------------------------------------
  # Cross-iteration best solution selection
  # ---------------------------------------------------------------------------

  async def _apply_best_solution(self, ctx: InvocationContext):
    """Finds the best solution from history and rolls back the file if needed."""
    history = ctx.session.state.get("history", [])
    valid_solutions = [
      s
      for s in history
      if s.get("compilation_status", {}).get("success")
      and s.get("test_status", {}).get("success")
    ]

    best_solution = None
    if valid_solutions:
      solutions_with_latency = [
        s for s in valid_solutions if s.get("latency_ms") is not None
      ]
      if solutions_with_latency:
        try:
          best_solution = min(
            solutions_with_latency, key=lambda x: x["latency_ms"]
          )
          if len(solutions_with_latency) < len(valid_solutions):
            logging.warning(
              f"[{self.name}] {len(valid_solutions) - len(solutions_with_latency)} "
              "valid solution(s) were missing latency metrics and ignored."
            )
        except Exception as e:
          logging.error(
            f"[{self.name}] Error selecting best solution by latency: {e}"
          )
          best_solution = await self._select_best_with_llm(valid_solutions)
      else:
        logging.warning(
          f"[{self.name}] No latency metrics found. Falling back to LLM selection."
        )
        best_solution = await self._select_best_with_llm(valid_solutions)

    if best_solution:
      logging.info(
        f"[{self.name}] Best solution found from iteration {best_solution['iteration']}"
      )

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

      if best_solution["kernel_code"] != current_code:
        logging.info(
          f"[{self.name}] Reverting kernel file to best solution from iteration "
          f"{best_solution['iteration']}"
        )
        if kernel_path:
          try:
            with open(kernel_path, "w") as f:
              f.write(best_solution["kernel_code"])
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
    """Fallback: use LLM to select the best solution based on profiling summaries."""
    if not valid_solutions:
      return None
    if len(valid_solutions) == 1:
      return valid_solutions[0]

    logging.info(
      f"[{self.name}] Invoking LLM to select best solution from {len(valid_solutions)} candidates."
    )

    prompt = (
      "You are an expert in performance optimization. Compare the following profiling "
      "summaries for different iterations of a kernel and decide which one is the best "
      "solution (maximizes performance/efficiency). Answer with ONLY the iteration number "
      "of the best solution.\n\n"
    )
    for s in valid_solutions:
      prompt += f"Iteration {s['iteration']}:\n{s['profiling_summary']}\n\n"

    request = LlmRequest(
      contents=[
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
      ]
    )

    try:
      model = None
      if hasattr(self.plan_agent, "model"):
        model = self.plan_agent.model
      elif hasattr(self.implement_agent, "model"):
        model = self.implement_agent.model

      if model:
        text = ""
        async for chunk in model.generate_content_async(request):
          if chunk.content and chunk.content.parts:
            for part in chunk.content.parts:
              if part.text:
                text += part.text
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
