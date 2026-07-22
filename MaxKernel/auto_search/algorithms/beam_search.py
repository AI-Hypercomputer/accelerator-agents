import asyncio
import logging
import random
from typing import List, Tuple

from auto_search.graph import Node
from auto_search.orchestrator import SearchOrchestrator
from auto_search.strategies import TPU_PALLAS_OPTIMIZATION_STRATEGIES
from auto_search.worker import ADKSessionWorker

logger = logging.getLogger(__name__)


class BeamSearchOrchestrator(SearchOrchestrator):
  def __init__(
    self,
    beam_size: int = 2,
    branches_per_node: int = 2,
    max_depth: int = 2,
    keep_factor: float = 1,
    strategies: List[str] = TPU_PALLAS_OPTIMIZATION_STRATEGIES,
    agent_config: dict = None,
    **kwargs,
  ):
    self._validate_args(beam_size, branches_per_node, max_depth, keep_factor)
    self.beam_size = beam_size
    self.branches_per_node = branches_per_node
    self.strategies = strategies
    self.max_depth = max_depth
    self.keep_factor = keep_factor
    self.agent_config = agent_config or {"max_iterations": 1}
    self.worker = ADKSessionWorker()

    self.current_depth = 0
    self.beam: List[Node] = []

    super().__init__(**kwargs)

    if not self.graph.metadata:
      self.beam = [self.graph.get_node(self.graph.root_id)]
      self.update_metadata("current_depth", self.current_depth)
      self.update_metadata(
        "beam_node_ids", [node.node_id for node in self.beam]
      )

  def _validate_args(
    self,
    beam_size: int,
    branches_per_node: int,
    max_depth: int,
    keep_factor: float,
  ) -> None:
    if beam_size < 1:
      raise ValueError(f"beam_size must be at least 1, got {beam_size}.")
    if branches_per_node < 1:
      raise ValueError(
        f"branches_per_node must be at least 1, got {branches_per_node}."
      )
    if max_depth < 1:
      raise ValueError(f"max_depth must be at least 1, got {max_depth}.")
    if keep_factor <= 0:
      raise ValueError(f"keep_factor must be positive, got {keep_factor}.")

  def _resume(self) -> None:
    self.current_depth = self.graph.metadata.get("current_depth", 0)
    beam_ids = self.graph.metadata.get("beam_node_ids", [])
    self.beam = [
      node
      for node in (self.graph.get_node(node_id) for node_id in beam_ids)
      if node is not None
    ]
    logger.info(
      f"Resumed Beam Search at depth {self.current_depth} with beam: {beam_ids}"
    )

  def _select_nodes_to_expand(self) -> List[Node]:
    return self.beam

  def _generate_expansion_tasks(
    self, nodes: List[Node]
  ) -> List[Tuple[Node, str]]:
    tasks = []
    for node in nodes:
      selected_strategies = random.sample(
        self.strategies,
        min(self.branches_per_node, len(self.strategies)),
      )
      for strategy in selected_strategies:
        tasks.append((node, strategy))
    return tasks

  def _update_search_state(self, new_nodes: List[Node]) -> None:
    candidates = []
    regressed_candidates = []
    for node in new_nodes:
      if not node.is_valid_candidate:
        logger.warning(
          f"Node {node.node_id} failed Validity Check. "
          "Adding to regressed candidates"
        )
        regressed_candidates.append(node)
        continue

      parent = self.graph.get_node(node.parent_id)
      parent_latency = parent.evaluation.latency_ms
      parent_latency = (
        parent_latency if parent_latency is not None else float("inf")
      )

      current_latency = node.evaluation.latency_ms
      current_latency = (
        current_latency if current_latency is not None else float("inf")
      )

      if current_latency < parent_latency * self.keep_factor:
        candidates.append(node)
      else:
        logger.info(
          f"Node {node.node_id} failed Parent Regression Gate. "
          "Adding to regressed candidates"
        )
        regressed_candidates.append(node)

    candidates.sort(key=lambda n: n.evaluation.latency_ms)

    if len(candidates) < self.beam_size and regressed_candidates:
      shortage = self.beam_size - len(candidates)
      logger.warning(
        f"Only {len(candidates)} candidates passed the Parent Regression Gate. "
        f"Padding with the best {shortage} regressed candidates to keep search alive."
      )
      regressed_candidates.sort(
        key=lambda n: (
          n.evaluation.latency_ms
          if n.evaluation.latency_ms is not None
          else float("inf")
        )
      )
      candidates.extend(regressed_candidates)

    self.beam = candidates[: self.beam_size]
    self.update_metadata("beam_node_ids", [n.node_id for n in self.beam])

  def _post_step_hook(self) -> None:
    self.current_depth += 1
    self.update_metadata("current_depth", self.current_depth)

  def _should_terminate(self) -> bool:
    return self.current_depth >= self.max_depth or not self.beam

  async def _execute_expansions(
    self, tasks: List[Tuple[Node, str]]
  ) -> List[Node]:
    async def run_task(task_idx: int, parent_node: Node, strategy: str) -> Node:
      node_id, base_dir = self.get_next_session_node()
      for attempt in range(1, self.max_worker_retries + 1):
        logger.info(
          f"Task {task_idx}: Expanding {parent_node.node_id} using \n"
          f" strategy '{strategy}' -> {node_id}. \n"
          f"Attempt {attempt}/{self.max_worker_retries}."
        )
        async with self._semaphore:
          session_dir = f"{base_dir}_attempt_{attempt}"
          node = await self.worker.expand_node(
            node_id,
            parent_node,
            session_dir=session_dir,
            reference_code=self.reference_code,
            strategy=strategy,
            agent_config=self.agent_config,
          )
          if (
            node.execution_status == "SUCCESS"
            or attempt == self.max_worker_retries
          ):
            self.graph.add_node(node)
            logger.info(
              f"Task {task_idx}: Finished {node_id} with status {node.execution_status} "
              f"(Latency: {node.evaluation.latency_ms} ms)"
            )
            return node

        logger.warning(
          f"Task {task_idx} (strategy: {strategy}) failed attempt"
          f" {attempt}/{self.max_worker_retries}: {node.execution_error}. Retrying..."
        )
        await asyncio.sleep(2**attempt)

    futures = [
      run_task(i, parent, strat) for i, (parent, strat) in enumerate(tasks)
    ]
    return await asyncio.gather(*futures)
