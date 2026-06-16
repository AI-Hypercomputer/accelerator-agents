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

# State keys preserved across branches (shared inputs)
_INITIAL_STATE_KEYS = frozenset({"workdir", "base_kernel_path", "atol", "rtol"})

# State keys that belong to a single branch and must be cleared before each branch
_PIPELINE_RESET_KEYS = [
  "optimized_kernel_path",
  "kernel_plan_path",
  "test_file_path",
  "profiling_script_path",
  "autotune_specs_path",
  "autotune_results_path",
  "kernel_compilation_status",
  "test_results",
  "validation_loop_status",
  "autotune_results",
  "profiling_summary",
  "history",
  "compilation_history",
  "fix_summary",
  "compilation_results",
  "kernel_code",
  "go_to_end",
  "needs_improvement",
  "_branch_last_iteration",
]


class AutonomousPipelineAgent(BaseAgent):
  """Chains kernel generation sub-agents automatically with an improvement loop.

  When num_branches > 1, runs N independent branches concurrently (sequentially
  in execution, but fully isolated in state and files) and ranks the results.
  Each branch has its own subdirectory and sees no state from other branches.
  """

  plan_agent: BaseAgent
  implement_agent: BaseAgent
  validate_agent: BaseAgent
  test_gen_agent: BaseAgent
  test_run_agent: BaseAgent
  autotune_agent: BaseAgent
  profile_agent: BaseAgent
  max_iterations: int = 2
  num_branches: int = 1

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
    num_branches: int = 1,
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
      num_branches=num_branches,
    )

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if self.num_branches <= 1:
      # --- Single-branch mode: original behavior ---
      yield self._initialize_state(ctx)
      async for event in self._run_pipeline_branch(ctx):
        yield event
      best_solution = await self._apply_best_solution(ctx)
      iteration = ctx.session.state.get("_branch_last_iteration", self.max_iterations)
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
      return

    # --- Multi-branch mode: N isolated branches → ranking ---
    yield self._initialize_base_state(ctx)
    session_dir = ctx.session.state["workdir"]

    # Snapshot the shared initial state and conversation history for restoration
    initial_state = {
      k: v for k, v in ctx.session.state.items() if k in _INITIAL_STATE_KEYS
    }
    events_snapshot = self._snapshot_events(ctx)
    branches_results = []

    for branch_idx in range(self.num_branches):
      logging.info(
        f"[{self.name}] ===== Starting Branch {branch_idx + 1}/{self.num_branches} ====="
      )

      # Restore conversation to the pre-branch snapshot so this branch
      # has no memory of what previous branches wrote or did.
      self._restore_events(ctx, events_snapshot)

      branch_dir = os.path.join(session_dir, f"branch_{branch_idx}")
      os.makedirs(branch_dir, exist_ok=True)
      yield self._setup_branch_state(ctx, initial_state, branch_dir)

      async for event in self._run_pipeline_branch(ctx):
        yield event

      branch_best = await self._apply_best_solution(ctx)
      branches_results.append({
        "branch_idx": branch_idx,
        "branch_dir": branch_dir,
        "best_solution": branch_best,
        "history": list(ctx.session.state.get("history", [])),
      })
      logging.info(
        f"[{self.name}] ===== Branch {branch_idx + 1} Complete ====="
      )

    # Restore conversation after all branches before ranking
    self._restore_events(ctx, events_snapshot)

    best_branch = await self._rank_branches(ctx, branches_results)
    logging.info(
      f"[{self.name}] Best branch: {best_branch['branch_idx'] if best_branch else 'none'}"
    )

    # Copy the best branch's solution to the canonical session-level output path
    final_kernel_path = os.path.join(session_dir, "optimized_kernel.py")
    if best_branch and best_branch.get("best_solution"):
      kernel_code = best_branch["best_solution"].get("kernel_code", "")
      if kernel_code:
        try:
          with open(final_kernel_path, "w") as f:
            f.write(kernel_code)
          logging.info(
            f"[{self.name}] Wrote best solution from branch "
            f"{best_branch['branch_idx']} to {final_kernel_path}"
          )
        except Exception as e:
          logging.error(f"[{self.name}] Failed to write best solution: {e}")

    yield Event(
      author=self.name,
      actions=EventActions(
        state_delta={
          "branches": branches_results,
          "best_branch_idx": best_branch["branch_idx"] if best_branch else -1,
          "optimized_kernel_path": final_kernel_path,
          "pipeline_status": "Completed",
        }
      ),
    )

  async def _run_pipeline_branch(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Runs the full plan→implement→validate→test→autotune→profile loop.

    Called once per branch. Reads and writes only branch-specific state keys
    (already configured by _setup_branch_state before invocation).
    """
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

      test_results = ctx.session.state.get("test_results", {})
      if not test_results.get("success", False):
        logging.error(f"[{self.name}] Tests failed. Looping back to planning.")
        self._save_iteration_files(
          ctx,
          iteration,
          keys_to_save=["optimized_kernel_path", "test_file_path"],
        )
        iteration += 1
        continue

      # Step 6: Autotune
      logging.info(f"[{self.name}] Running AutotuneAgent...")
      async for event in self.autotune_agent.run_async(ctx):
        yield event

      # Step 7: Profile
      logging.info(f"[{self.name}] Running ProfileAgentOrchestrator...")
      async for event in self.profile_agent.run_async(ctx):
        yield event

      # Snapshot this successful iteration into history
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

      latency = self._extract_latency(ctx)
      snapshot = {
        "iteration": iteration,
        "kernel_code": kernel_code,
        "compilation_status": ctx.session.state.get(
          "kernel_compilation_status", {}
        ),
        "test_status": ctx.session.state.get("test_results", {}),
        "latency_ms": latency,
        "profiling_summary": ctx.session.state.get("profiling_summary", ""),
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

      logging.info(
        f"[{self.name}] Improvement needed. Looping back to planning..."
      )
      iteration += 1

    if iteration >= self.max_iterations:
      logging.warning(
        f"[{self.name}] Maximum iterations reached ({self.max_iterations}). Stopping pipeline."
      )

    ctx.session.state["_branch_last_iteration"] = iteration

  # ---------------------------------------------------------------------------
  # State management helpers
  # ---------------------------------------------------------------------------

  def _initialize_base_state(self, ctx: InvocationContext) -> Event:
    """Initializes only the shared state (workdir, base kernel, tolerances).

    Used in multi-branch mode. Branch-specific paths are set later by
    _setup_branch_state before each branch runs.
    """
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
      logging.info(
        f"[{self.name}] Set base_kernel_path: {ctx.session.state['base_kernel_path']}"
      )

    if "atol" not in ctx.session.state:
      ctx.session.state["atol"] = 1e-2

    if "rtol" not in ctx.session.state:
      ctx.session.state["rtol"] = 1e-2

    return Event(
      author=self.name,
      actions=EventActions(
        state_delta={
          "workdir": ctx.session.state["workdir"],
          "base_kernel_path": ctx.session.state["base_kernel_path"],
          "atol": ctx.session.state["atol"],
          "rtol": ctx.session.state["rtol"],
        }
      ),
    )

  def _setup_branch_state(
    self,
    ctx: InvocationContext,
    initial_state: dict,
    branch_dir: str,
  ) -> Event:
    """Resets all pipeline state and sets branch-specific file paths.

    Called once before each branch runs. Ensures the branch starts with a
    clean slate: no knowledge of what other branches wrote or produced.
    """
    for key in _PIPELINE_RESET_KEYS:
      ctx.session.state.pop(key, None)

    # Re-apply the shared initial state (workdir, base_kernel_path, atol, rtol)
    ctx.session.state.update(initial_state)

    branch_state = {
      "optimized_kernel_path": os.path.join(branch_dir, "optimized_kernel.py"),
      "kernel_plan_path": os.path.join(branch_dir, "base_kernel_plan.md"),
      "test_file_path": os.path.join(branch_dir, "test_optimized_kernel.py"),
      "profiling_script_path": os.path.join(
        branch_dir, "profile_optimized_kernel.py"
      ),
      "autotune_specs_path": os.path.join(branch_dir, "autotune_specs.json"),
      "autotune_results_path": os.path.join(branch_dir, "autotune_results.json"),
      "history": [],
    }
    ctx.session.state.update(branch_state)

    logging.info(
      f"[{self.name}] Configured branch state. Branch dir: {branch_dir}"
    )
    return Event(
      author=self.name,
      actions=EventActions(state_delta=branch_state),
    )

  def _snapshot_events(self, ctx: InvocationContext) -> list | None:
    """Returns a copy of the session event list for later restoration."""
    session = ctx.session
    for attr in ("events", "contents", "_events", "_contents"):
      val = getattr(session, attr, None)
      if isinstance(val, list):
        logging.info(
          f"[{self.name}] Snapshotted {len(val)} session events via '{attr}'."
        )
        return list(val)
    logging.warning(
      f"[{self.name}] Could not find a mutable event list on session — "
      "conversation history will not be isolated across branches."
    )
    return None

  def _restore_events(self, ctx: InvocationContext, snapshot: list | None):
    """Restores the session event list to the snapshot, providing conversation isolation."""
    if snapshot is None:
      return
    session = ctx.session
    for attr in ("events", "contents", "_events", "_contents"):
      val = getattr(session, attr, None)
      if isinstance(val, list):
        try:
          val.clear()
          val.extend(snapshot)
          logging.info(
            f"[{self.name}] Restored session events to snapshot ({len(snapshot)} entries)."
          )
          return
        except Exception as e:
          logging.warning(
            f"[{self.name}] Failed to restore events via '{attr}': {e}"
          )
    logging.warning(
      f"[{self.name}] Could not restore session events — "
      "conversation history may accumulate across branches."
    )

  # ---------------------------------------------------------------------------
  # Branch ranking
  # ---------------------------------------------------------------------------

  async def _rank_branches(
    self, ctx: InvocationContext, branches_results: list
  ) -> dict | None:
    """Selects the best branch. Prefers lowest latency; falls back to LLM."""
    valid = [b for b in branches_results if b.get("best_solution")]
    if not valid:
      logging.warning(f"[{self.name}] No valid branch solutions to rank.")
      return None
    if len(valid) == 1:
      return valid[0]

    with_latency = [
      b for b in valid if b["best_solution"].get("latency_ms") is not None
    ]
    if with_latency:
      best = min(with_latency, key=lambda b: b["best_solution"]["latency_ms"])
      logging.info(
        f"[{self.name}] Ranked {len(with_latency)} branches by latency. "
        f"Best: branch {best['branch_idx']} "
        f"({best['best_solution']['latency_ms']} ms)"
      )
      if len(with_latency) < len(valid):
        logging.warning(
          f"[{self.name}] {len(valid) - len(with_latency)} branch(es) had no "
          "latency metrics and were excluded from latency ranking."
        )
      return best

    logging.warning(
      f"[{self.name}] No latency metrics found across branches. "
      "Falling back to LLM ranking."
    )
    return await self._rank_branches_with_llm(valid)

  async def _rank_branches_with_llm(self, valid_branches: list) -> dict | None:
    """Uses the LLM to select the best branch by comparing profiling summaries."""
    if len(valid_branches) == 1:
      return valid_branches[0]

    logging.info(
      f"[{self.name}] Invoking LLM to rank {len(valid_branches)} branches."
    )
    prompt = (
      "You are an expert in performance optimization. "
      "Compare the following profiling summaries for different kernel optimization "
      "branches and decide which one is the best solution. "
      "Answer with ONLY the branch index number.\n\n"
    )
    for b in valid_branches:
      sol = b["best_solution"]
      prompt += (
        f"Branch {b['branch_idx']}:\n"
        f"{sol.get('profiling_summary', 'No profiling data available.')}\n\n"
      )

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
        match = re.search(r"\d+", text)
        if match:
          branch_idx = int(match.group())
          for b in valid_branches:
            if b["branch_idx"] == branch_idx:
              return b

      logging.warning(
        f"[{self.name}] LLM ranking failed or no model found. "
        "Defaulting to last valid branch."
      )
      return valid_branches[-1]
    except Exception as e:
      logging.error(f"[{self.name}] Error in LLM branch ranking: {e}")
      return valid_branches[-1]

  # ---------------------------------------------------------------------------
  # Existing helpers (unchanged)
  # ---------------------------------------------------------------------------

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
    """Initializes session state with standard paths and returns the event.

    Used in single-branch mode. For multi-branch mode, use _initialize_base_state
    followed by _setup_branch_state.
    """
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

    if "test_file_path" not in ctx.session.state:
      ctx.session.state["test_file_path"] = os.path.join(
        session_dir, "test_optimized_kernel.py"
      )
      logging.info(
        f"[{self.name}] Set test_file_path: {ctx.session.state['test_file_path']}"
      )

    if "profiling_script_path" not in ctx.session.state:
      ctx.session.state["profiling_script_path"] = os.path.join(
        session_dir, "profile_optimized_kernel.py"
      )
      logging.info(
        f"[{self.name}] Set profiling_script_path: {ctx.session.state['profiling_script_path']}"
      )

    if "autotune_specs_path" not in ctx.session.state:
      ctx.session.state["autotune_specs_path"] = os.path.join(
        session_dir, "autotune_specs.json"
      )
      logging.info(
        f"[{self.name}] Set autotune_specs_path: {ctx.session.state['autotune_specs_path']}"
      )

    if "autotune_results_path" not in ctx.session.state:
      ctx.session.state["autotune_results_path"] = os.path.join(
        session_dir, "autotune_results.json"
      )
      logging.info(
        f"[{self.name}] Set autotune_results_path: {ctx.session.state['autotune_results_path']}"
      )

    if "atol" not in ctx.session.state:
      ctx.session.state["atol"] = 1e-2
      logging.info(f"[{self.name}] Set atol: {ctx.session.state['atol']}")

    if "rtol" not in ctx.session.state:
      ctx.session.state["rtol"] = 1e-2
      logging.info(f"[{self.name}] Set rtol: {ctx.session.state['rtol']}")

    logging.info(f"[{self.name}] Published explicit path state update Event.")
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
              f"valid solution(s) were missing latency metrics and ignored."
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
          f"[{self.name}] Reverting kernel file to best solution from iteration {best_solution['iteration']}"
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
