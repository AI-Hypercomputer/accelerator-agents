import json
import logging
import os
from typing import Any, Dict, Optional

from auto_agent.agent import root_agent
from auto_agent.agent_client.auto_agent_client import AutoAgentClient
from auto_agent.subagents.pipeline_agent import AutonomousPipelineAgent
from auto_search.graph import EvaluationResult, Node

logger = logging.getLogger(__name__)


class ADKSessionWorker:
  async def expand_node(
    self,
    node_id: str,
    parent_node: Node,
    session_dir: str,
    reference_code: str,
    strategy: Optional[str] = None,
    agent_config: Optional[Dict[str, Any]] = None,
  ) -> Node:
    """Expand an ADK session to get a new optimized kernel."""
    os.makedirs(session_dir, exist_ok=True)

    # Prepare inputs files for the agent based on reference_code and parent_node
    self._prepare_inputs(session_dir, parent_node, reference_code)
    # Prepare initial state
    initial_state = self._prepare_initial_state(parent_node)

    # Run the agent and process results
    try:
      state = await self._run_agent(
        node_id=node_id,
        session_dir=session_dir,
        strategy=strategy,
        agent_config=agent_config,
        initial_state=initial_state,
      )
      return self._process_results(
        node_id=node_id,
        parent_node=parent_node,
        strategy=strategy,
        state=state,
        session_dir=session_dir,
      )
    except Exception as e:
      logger.exception(f"Node {node_id} expansion failed: {e}")
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
    session_dir: str,
    strategy: Optional[str],
    agent_config: Optional[Dict[str, Any]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Sets up a custom AutonomousPipelineAgent and runs the client."""
    agent_config = agent_config or {}

    custom_agent = AutonomousPipelineAgent(
      name="AutonomousPipelineAgent",
      plan_agent=root_agent.plan_agent,
      implement_agent=root_agent.implement_agent,
      validate_agent=root_agent.validate_agent,
      test_gen_agent=root_agent.test_gen_agent,
      test_run_agent=root_agent.test_run_agent,
      autotune_agent=root_agent.autotune_agent,
      profile_agent=root_agent.profile_agent,
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
    )

    await client.create_session(initial_state)
    await client.run_async()
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

    latency_ms = best_run.get("latency_ms")
    autotuning_summary = best_run.get("autotuning_summary")
    profiling_summary = best_run.get("profiling_summary")

    # Read optimized code if it exists
    opt_path = state.get("optimized_kernel_path")
    optimized_code = ""
    if opt_path and os.path.exists(opt_path):
      with open(opt_path, "r") as f:
        optimized_code = f.read()

    # Read optimized plan if it exists
    plan_path = state.get("kernel_plan_path")
    optimized_plan = ""
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
      execution_status="SUCCESS",
      evaluation=EvaluationResult(
        compiled=compiled,
        correct=correct,
        latency_ms=latency_ms,
        autotuning_summary=autotuning_summary,
        profiling_summary=profiling_summary,
        compilation_error=compilation_error,
        test_error=test_error,
      ),
    )
