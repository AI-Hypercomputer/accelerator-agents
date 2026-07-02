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
    session_id: str,
    parent_node: Node,
    session_dir: str,
    strategy: Optional[str] = None,
    agent_config: Optional[Dict[str, Any]] = None,
  ) -> Node:
    """Expand an ADK session to get a new optimized kernel."""
    os.makedirs(session_dir, exist_ok=True)

    # Prepare inputs files for the agent based on parent_node
    self._prepare_inputs(session_dir, parent_node)

    # Run the agent and process results
    try:
      state = await self._run_agent(
        session_id=session_id,
        session_dir=session_dir,
        strategy=strategy,
        agent_config=agent_config,
      )
      return self._process_results(
        session_id=session_id,
        parent_node=parent_node,
        strategy=strategy,
        state=state,
      )
    except Exception as e:
      logger.exception(f"Session {session_id} failed: {e}")
      return Node(
        node_id=session_id,
        parent_id=parent_node.node_id,
        session_id=session_id,
        code="",
        plan="",
        depth=parent_node.depth + 1,
        strategy_applied=strategy,
        execution_status="FAIL",
        execution_error=str(e),
        evaluation=EvaluationResult(correct=False),
      )

  def _prepare_inputs(self, session_dir: str, parent_node: Node) -> None:
    """Writes the parent node code and plan to the session directory."""
    base_kernel_path = os.path.join(session_dir, "base_kernel.py")
    with open(base_kernel_path, "w") as f:
      f.write(parent_node.code)

    optimized_kernel_path = os.path.join(session_dir, "optimized_kernel.py")
    with open(optimized_kernel_path, "w") as f:
      f.write(parent_node.code)

    kernel_plan_path = None
    if parent_node.plan:
      kernel_plan_path = os.path.join(session_dir, "base_kernel_plan.md")
      with open(kernel_plan_path, "w") as f:
        f.write(parent_node.plan)

  async def _run_agent(
    self,
    session_id: str,
    session_dir: str,
    strategy: Optional[str],
    agent_config: Optional[Dict[str, Any]] = None,
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
      session_id=session_id,
      query=query,
      agent=custom_agent,
    )
    await client.create_session()
    await client.run_async()
    return client.get_state()

  def _process_results(
    self,
    session_id: str,
    parent_node: Node,
    strategy: Optional[str],
    state: Dict[str, Any],
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

    comp_status = best_run.get("compilation_status", {})
    compiled = comp_status.get("success", False)
    compilation_error = comp_status.get("message") if not compiled else None

    test_status = best_run.get("test_status", {})
    correct = test_status.get("success", False)
    test_error = test_status.get("output") if not correct else None

    latency_ms = best_run.get("latency_ms")
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
      node_id=session_id,
      parent_id=parent_node.node_id,
      session_id=session_id,
      code=optimized_code,
      plan=optimized_plan,
      depth=parent_node.depth + 1,
      strategy_applied=strategy,
      execution_status="SUCCESS",
      evaluation=EvaluationResult(
        compiled=compiled,
        correct=correct,
        latency_ms=latency_ms,
        profiling_summary=profiling_summary,
        compilation_error=compilation_error,
        test_error=test_error,
      ),
    )
