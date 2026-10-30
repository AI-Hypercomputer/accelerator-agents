import json
import logging
import os
from typing import Any, Callable, Dict, Optional

from auto_agent.agent import create_root_agent
from auto_agent.agent_client.auto_agent_client import AutoAgentClient
from auto_search.graph import EvaluationResult, Node

logger = logging.getLogger(__name__)


class ADKSessionWorker:
  def __init__(self):
    pass

  async def expand_node(
    self,
    node_id: str,
    parent_node: Node,
    session_dir: str,
    reference_code: str,
    strategy: Optional[str] = None,
    agent_config: Optional[Dict[str, Any]] = None,
    events_compaction: bool = False,
    on_node_update: Optional[Callable[[Node], None]] = None,
  ) -> Node:
    """Expand an ADK session to get a new optimized kernel."""
    os.makedirs(session_dir, exist_ok=True)

    # Prepare inputs files for the agent based on reference_code and parent_node
    self._prepare_inputs(session_dir, parent_node, reference_code)
    # Prepare initial state
    initial_state = self._prepare_initial_state(parent_node)

    # Run the agent and process results
    salvaged_node = None
    active_client_container: list[AutoAgentClient] = []
    try:
      state = await self._run_agent(
        node_id=node_id,
        parent_node=parent_node,
        session_dir=session_dir,
        strategy=strategy,
        agent_config=agent_config,
        initial_state=initial_state,
        events_compaction=events_compaction,
        on_node_update=on_node_update,
        active_client_container=active_client_container,
      )
      node = self._process_results(
        node_id=node_id,
        parent_node=parent_node,
        strategy=strategy,
        state=state,
        session_dir=session_dir,
        execution_status="SUCCESS",
      )
      if on_node_update:
        on_node_update(node)
      return node
    except BaseException as e:
      active_client = (
        active_client_container[0] if active_client_container else None
      )
      if active_client and active_client.session:
        try:
          curr_state = active_client.get_state()
          if curr_state and curr_state.get("history"):
            candidate = self._process_results(
              node_id=node_id,
              parent_node=parent_node,
              strategy=strategy,
              state=curr_state,
              session_dir=session_dir,
              execution_status="INTERRUPTED",
            )
            if candidate.is_valid_candidate:
              salvaged_node = candidate
              if on_node_update:
                on_node_update(salvaged_node)
        except Exception as se:
          logger.warning(f"Failed to salvage node state on interruption: {se}")

      if isinstance(e, Exception):
        logger.exception(f"Node {node_id} expansion failed: {e}")
        if salvaged_node is not None:
          return salvaged_node
        return Node(
          node_id=node_id,
          parent_id=parent_node.node_id,
          session_dir=session_dir,
          code="",
          plan="",
          depth=parent_node.depth + 1,
          strategy_applied=strategy,
          execution_status="FAIL",
          execution_error=str(e),
          evaluation=EvaluationResult(correct=False),
        )
      else:
        # BaseException like KeyboardInterrupt, SystemExit, CancelledError
        raise e

  def _prepare_inputs(
    self, session_dir: str, parent_node: Node, reference_code: str
  ) -> None:
    """Writes the reference code and parent's optimized code (if any) to session directory."""
    base_kernel_path = os.path.join(session_dir, "base_kernel.py")
    with open(base_kernel_path, "w") as f:
      f.write(reference_code)

    if parent_node.depth > 0 and parent_node.code:
      optimized_kernel_path = os.path.join(session_dir, "optimized_kernel.py")
      with open(optimized_kernel_path, "w") as f:
        f.write(parent_node.code)

    kernel_plan_path = None
    if parent_node.plan:
      kernel_plan_path = os.path.join(session_dir, "base_kernel_plan.md")
      with open(kernel_plan_path, "w") as f:
        f.write(parent_node.plan)

  def _prepare_initial_state(self, parent_node: Node) -> Dict[str, Any]:
    initial_state = {}
    if parent_node and parent_node.evaluation:
      initial_state = {
        "kernel_compilation_status": {
          "success": parent_node.evaluation.compiled,
          "message": parent_node.evaluation.compilation_error,
        },
        "test_results": {
          "success": parent_node.evaluation.correct,
          "output": parent_node.evaluation.test_error,
        },
        "profiling_summary": parent_node.evaluation.profiling_summary,
      }
    return initial_state

  async def _run_agent(
    self,
    node_id: str,
    parent_node: Node,
    session_dir: str,
    strategy: Optional[str],
    agent_config: Optional[Dict[str, Any]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    events_compaction: bool = False,
    on_node_update: Optional[Callable[[Node], None]] = None,
    active_client_container: Optional[list] = None,
  ) -> Dict[str, Any]:
    """Sets up a custom AutonomousPipelineAgent and runs the client."""
    agent_config = agent_config or {}

    custom_agent = create_root_agent(
      session_dir=session_dir,
      **agent_config,
    )

    strategy_query = f"Focus on: {strategy}. " if strategy else ""
    query = (
      "Optimize the code for peak performance with pallas kernel. "
      f"{strategy_query}"
      "Base code is at base_kernel.py."
    )
    client = AutoAgentClient(
      user_id="orchestrator",
      session_id=node_id,
      query=query,
      agent=custom_agent,
      events_compaction=events_compaction,
    )
    if active_client_container is not None:
      active_client_container.append(client)

    def handle_event(event: Any) -> None:
      if not on_node_update:
        return
      actions = getattr(event, "actions", None)
      if actions and hasattr(actions, "state_delta") and actions.state_delta:
        state_delta = actions.state_delta
        if "history" in state_delta:
          try:
            curr_state = client.get_state()
            partial_node = self._process_results(
              node_id=node_id,
              parent_node=parent_node,
              strategy=strategy,
              state=curr_state,
              session_dir=session_dir,
              execution_status="IN_PROGRESS",
            )
            on_node_update(partial_node)
          except Exception as err:
            logger.warning(
              f"Error updating intermediate node for {node_id}: {err}"
            )

    await client.create_session(initial_state)
    try:
      await client.run_async(on_event=handle_event)
    finally:
      try:
        session_json_path = os.path.join(session_dir, "session.json")
        with open(session_json_path, "w") as f:
          json.dump(client.get_session_data(), f, indent=2)
        logger.info(f"Saved session data to {session_json_path}")
      except Exception as se:
        logger.warning(f"Failed to save session data for {node_id}: {se}")
    return client.get_state()

  def _process_results(
    self,
    node_id: str,
    parent_node: Node,
    strategy: Optional[str],
    state: Dict[str, Any],
    session_dir: str,
    execution_status: str = "SUCCESS",
  ) -> Node:
    """Reads optimized code/plan and extracts metrics to construct a new Node."""
    history = state.get("history", [])
    best_iter = state.get("best_iteration", -1)

    best_run = {}
    if best_iter != -1:
      best_run = next(
        (run for run in history if run.get("iteration") == best_iter), {}
      )
    elif history:
      # Find best valid run with highest speedup among completed runs
      valid_runs = [
        run
        for run in history
        if run.get("compilation_status", {}).get("success", False)
        and run.get("test_status", {}).get("success", False)
      ]
      if valid_runs:
        best_run = max(
          valid_runs,
          key=lambda r: (
            r.get("speedup") if r.get("speedup") is not None else -float("inf")
          ),
        )
      else:
        best_run = history[-1]

    comp_status = best_run.get("compilation_status") or {}
    compiled = comp_status.get("success", False)
    compilation_error = None
    if not compiled:
      compilation_error = comp_status.get("message") or ""
      final_errors = comp_status.get("final_errors")
      if final_errors:
        compilation_error += "\n\nFinal Errors:\n" + final_errors

    test_status = best_run.get("test_status") or {}
    correct = test_status.get("success", False)
    test_error = test_status.get("output") if not correct else None

    speedup = best_run.get("speedup")
    autotuning_summary = best_run.get("autotuning_summary")
    profiling_summary = best_run.get("profiling_summary")

    # Read optimized code: prefer best_run["kernel_code"], then optimized_kernel_{iteration}.py, then opt_path
    optimized_code = ""
    iteration = best_run.get("iteration")
    if best_run.get("kernel_code"):
      optimized_code = best_run["kernel_code"]
    elif iteration is not None:
      iter_kernel_path = os.path.join(
        session_dir, f"optimized_kernel_{iteration}.py"
      )
      if os.path.exists(iter_kernel_path):
        with open(iter_kernel_path, "r") as f:
          optimized_code = f.read()

    if not optimized_code:
      opt_path = state.get("optimized_kernel_path")
      if opt_path and os.path.exists(opt_path):
        with open(opt_path, "r") as f:
          optimized_code = f.read()

    # Read optimized plan: prefer best_run["plan"], then base_kernel_plan_{iteration}.md, then plan_path
    optimized_plan = ""
    if best_run.get("plan"):
      optimized_plan = best_run["plan"]
    elif iteration is not None:
      iter_plan_path = os.path.join(
        session_dir, f"base_kernel_plan_{iteration}.md"
      )
      if os.path.exists(iter_plan_path):
        with open(iter_plan_path, "r") as f:
          optimized_plan = f.read()

    if not optimized_plan:
      plan_path = state.get("kernel_plan_path")
      if plan_path and os.path.exists(plan_path):
        with open(plan_path, "r") as f:
          optimized_plan = f.read()

    return Node(
      node_id=node_id,
      parent_id=parent_node.node_id,
      session_dir=session_dir,
      code=optimized_code,
      plan=optimized_plan,
      depth=parent_node.depth + 1,
      strategy_applied=strategy,
      execution_status=execution_status,
      evaluation=EvaluationResult(
        compiled=compiled,
        correct=correct,
        speedup=speedup,
        autotuning_summary=autotuning_summary,
        profiling_summary=profiling_summary,
        compilation_error=compilation_error,
        test_error=test_error,
      ),
    )
