import asyncio
import logging
from typing import List, Optional, Tuple

from auto_search.graph import Node
from auto_search.orchestrator import SearchOrchestrator
from auto_search.worker import ADKSessionWorker

logger = logging.getLogger(__name__)


class SimpleParallelSearchOrchestrator(SearchOrchestrator):
  """Orchestrates single-iteration parallel kernel explorations from root."""

  def __init__(
    self,
    num_parallel_runs: int = 2,
    strategies: Optional[List[str]] = None,
    agent_config: Optional[dict] = None,
    initial_kernel_code: Optional[str] = None,
    resume_node_id: Optional[str] = None,
    **kwargs,
  ):
    if num_parallel_runs <= 0:
      raise ValueError(
        f"num_parallel_runs must be a positive integer, got {num_parallel_runs}."
      )

    self.num_parallel_runs = num_parallel_runs
    if strategies is not None:
      if len(strategies) > num_parallel_runs:
        raise ValueError(
          f"Number of user specified strategies ({len(strategies)}) cannot"
          f" exceed num_parallel_runs ({num_parallel_runs})."
        )
      self.strategies = [s or "" for s in strategies] + [""] * (
        num_parallel_runs - len(strategies)
      )
    else:
      self.strategies = [""] * num_parallel_runs

    self.remaining_strategies = list(self.strategies)
    self.agent_config = agent_config
    self.worker = ADKSessionWorker()

    super().__init__(
      initial_kernel_code=initial_kernel_code,
      resume_node_id=resume_node_id,
      **kwargs,
    )

  def _resume(self) -> None:
    target_id = None
    if self.resume_node_id:
      if self.graph.get_node(self.resume_node_id):
        target_id = self.resume_node_id
      else:
        logger.warning(
          f"resume_node_id '{self.resume_node_id}' not found in graph. Falling back to root."
        )
    if not target_id:
      target_id = self.graph.root_id

    if not target_id:
      logger.warning("Cannot resume: root_id is not set in graph.")
      return

    existing_children = [
      node for node in self.graph.nodes.values() if node.parent_id == target_id
    ]

    for node in existing_children:
      strategy = node.strategy_applied or ""
      if strategy in self.remaining_strategies:
        self.remaining_strategies.remove(strategy)
    logger.info(
      f"Resumed Simple Parallel Search. Found {len(existing_children)} existing"
      f" evaluations. {len(self.remaining_strategies)} runs remaining."
    )

  def _select_nodes_to_expand(self) -> List[Node]:
    if self._should_terminate():
      return []
    if self.resume_node_id:
      resume_node = self.graph.get_node(self.resume_node_id)
      if resume_node:
        self.resume_node_id = None
        return [resume_node]
      logger.warning(
        f"resume_node_id '{self.resume_node_id}' not found in graph. Falling back to root."
      )
      self.resume_node_id = None
    if not self.graph.root_id:
      logger.error(
        "Cannot select nodes to expand: root_id is not set in graph."
      )
      return []
    root_node = self.graph.get_node(self.graph.root_id)
    return [root_node] if root_node else []

  def _generate_expansion_tasks(
    self, nodes: List[Node]
  ) -> List[Tuple[Node, str]]:
    if not self.remaining_strategies:
      logger.info(
        f"All {self.num_parallel_runs} parallel runs have already completed."
      )
      return []

    target_node = nodes[0] if nodes else self.graph.get_node(self.graph.root_id)
    if not target_node:
      return []

    logger.info(
      f"Launching {len(self.remaining_strategies)} parallel runs for node {target_node.node_id}."
    )
    return [(target_node, strategy) for strategy in self.remaining_strategies]

  def _update_search_state(self, new_nodes: List[Node]) -> None:
    for node in new_nodes:
      strategy = node.strategy_applied or ""
      if strategy in self.remaining_strategies:
        self.remaining_strategies.remove(strategy)

  def _should_terminate(self) -> bool:
    return len(self.remaining_strategies) == 0

  async def _execute_expansions(
    self, tasks: List[Tuple[Node, str]]
  ) -> List[Node]:
    async def run_task(task_idx: int, parent_node: Node, strategy: str) -> Node:
      node_id, base_dir = self.get_next_session_node()
      for attempt in range(1, self.max_worker_retries + 1):
        async with self._semaphore:
          session_dir = f"{base_dir}_attempt_{attempt}"
          node = await self.worker.expand_node(
            node_id,
            parent_node,
            strategy=strategy,
            session_dir=session_dir,
            reference_code=self.reference_code,
            agent_config=self.agent_config,
            events_compaction=self.events_compaction,
            on_node_update=lambda n: self.graph.add_node(n),
          )
          if (
            node.execution_status in ("SUCCESS", "INTERRUPTED")
            or attempt == self.max_worker_retries
          ):
            self.graph.add_node(node)
            return node

        strat_info = f" (strategy: {strategy})" if strategy else ""
        logger.warning(
          f"Task {task_idx}{strat_info} failed attempt"
          f" {attempt}/{self.max_worker_retries}: {node.execution_error}."
          " Retrying..."
        )
        await asyncio.sleep(2**attempt)

    futures = [
      run_task(i, parent, strategy)
      for i, (parent, strategy) in enumerate(tasks)
    ]
    return await asyncio.gather(*futures)
