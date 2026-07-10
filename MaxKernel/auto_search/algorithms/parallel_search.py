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

    super().__init__(**kwargs)

  def _resume(self) -> None:
    if not self.graph.root_id:
      logger.warning("Cannot resume: root_id is not set in graph.")
      return

    existing_children = [
      node
      for node in self.graph.nodes.values()
      if node.parent_id == self.graph.root_id
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

    root_node = self.graph.get_node(self.graph.root_id)
    if not root_node:
      return []

    logger.info(
      f"Launching {len(self.remaining_strategies)} parallel runs for root node."
    )
    return [(root_node, strategy) for strategy in self.remaining_strategies]

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
          )
          if (
            node.execution_status == "SUCCESS"
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
