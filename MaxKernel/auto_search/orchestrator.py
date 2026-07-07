import abc
import asyncio
import logging
import os
import time
from typing import Any, List, Optional, Tuple

from auto_search.graph import EvaluationResult, Node, SearchGraph

logger = logging.getLogger(__name__)


class SearchOrchestrator(abc.ABC):
  def __init__(
    self,
    problem_id: str,
    reference_code: str,
    graph_db_path: Optional[str] = None,
    max_concurrency: int = 2,
  ):
    # Resolve graph db path and run directory
    if not graph_db_path:
      workdir = os.environ.get("WORKDIR", os.getcwd())
      timestamp = time.strftime("%Y%m%d_%H%M%S")
      graph_db_path = os.path.join(
        workdir, f"graph_{problem_id}_{timestamp}.json"
      )
      logger.info(f"No graph_db_path provided. Generated: {graph_db_path}")

    # The run directory is the root directory for this search run.
    self.run_dir = os.path.dirname(os.path.abspath(graph_db_path))
    logger.info(f"Using run directory: {self.run_dir}")
    os.makedirs(self.run_dir, exist_ok=True)

    # Initialization of internal states
    self._semaphore = asyncio.Semaphore(max_concurrency)
    self._node_counter = 0

    self.reference_code = reference_code

    # Initialization of search graph
    self.graph = SearchGraph(problem_id, graph_db_path)
    if self.graph.root_id:
      logger.info(
        "Found existing graph. Call `.resume()` explicitly to restore"
        " search state."
      )
    else:
      # Start from scratch, create root node
      node_id, session_dir = self.get_next_session_node()
      os.makedirs(session_dir, exist_ok=True)
      # Write reference code to the root node directory
      with open(os.path.join(session_dir, "base_kernel.py"), "w") as f:
        f.write(reference_code)

      root_node = Node(
        node_id=node_id,
        parent_id=None,
        session_dir=session_dir,
        code=reference_code,
        plan="",
        depth=0,
        strategy_applied="baseline",
        execution_status="SUCCESS",
        evaluation=EvaluationResult(
          compiled=True,
          correct=True,
          latency_ms=float("inf"),
          profiling_summary="",
        ),
      )

      self.graph.add_node(root_node)

  def get_next_session_node(
    self, suffix: Optional[str] = None
  ) -> Tuple[str, str]:
    """Returns (node_id, session_dir) for the next node in the graph."""
    node_id = f"node_{self._node_counter:03d}"
    self._node_counter += 1
    dir_name = f"{node_id}_{suffix}" if suffix else node_id
    session_dir = os.path.join(self.run_dir, "nodes", dir_name)
    return node_id, session_dir

  def resume(self) -> None:
    """Restores the orchestrator state from the loaded graph's metadata."""
    logger.info("Resuming search orchestration from persisted graph state...")

    # Calculate the node counter based on existing nodes in the graph
    existing_indices = [
      int(node_id.split("_")[1])
      for node_id in self.graph.nodes.keys()
      if node_id.startswith("node_") and node_id[5:].isdigit()
    ]
    self._node_counter = max(existing_indices) + 1 if existing_indices else 0

    self._resume()

  def update_metadata(self, key: str, value: Any) -> None:
    """Writes orchestrator metadata to the graph and triggers a save."""
    self.graph.metadata[key] = value
    self.graph.save()

  # --- Template Methods ---

  @abc.abstractmethod
  def _resume(self) -> None:
    """Restores subclass-specific instance variables from metadata.

    Reads from self.graph.metadata to restore internal algorithm state.
    """
    pass

  @abc.abstractmethod
  def _select_nodes_to_expand(self) -> List[Node]:
    """Selects candidate nodes from the search graph for expansion.

    Returns a list of nodes to be expanded in the current step.
    """
    pass

  @abc.abstractmethod
  def _generate_expansion_tasks(self, nodes: List[Node]) -> Any:
    """Generates expansion tasks for the selected nodes.

    Creates tasks to be passed directly into _execute_expansions.
    """
    pass

  @abc.abstractmethod
  def _update_search_state(self, new_nodes: List[Node]) -> None:
    """Updates internal algorithm state using evaluated nodes.

    Updates state such as beam frontier or learnings based on newly evaluated
    nodes from worker expansions.
    """
    pass

  @abc.abstractmethod
  def _should_terminate(self) -> bool:
    """Determines whether the search loop should terminate.

    Returns True if termination criteria (e.g., max depth reached or no
    candidates) are met.
    """
    pass

  def _post_step_hook(self) -> None:
    """Optional hook executed at the end of each search step.

    Can be overridden for cleanup, state updates, or logging.
    """
    pass

  # --- Concrete Flow ---

  @abc.abstractmethod
  async def _execute_expansions(self, tasks: Any) -> List[Node]:
    """Executes expansion tasks to evaluate new kernels.

    Runs tasks (typically in parallel via workers) to compile, test, and profile
    candidate kernels, returning new Nodes.
    """
    pass

  async def run(self) -> Node:
    logger.info(
      f"Starting search orchestration for problem: {self.graph.problem_id}"
    )

    while not self._should_terminate():
      # 1. Select nodes to expand
      nodes_to_expand = self._select_nodes_to_expand()
      if not nodes_to_expand:
        break

      # 2. Generate strategies
      tasks = self._generate_expansion_tasks(nodes_to_expand)
      if not tasks:
        break

      # 3. Parallel Execution (Workers compile, test, and profile)
      new_nodes = await self._execute_expansions(tasks)

      # 4. Add any nodes not already saved during execution
      for node in new_nodes:
        if node.node_id not in self.graph.nodes:
          self.graph.add_node(node)

      # 5. Update Search Algorithm State
      self._update_search_state(new_nodes)

      # Log progress
      best_id = self.graph.best_node_id
      best_latency = "N/A"
      if best_id:
        best_node = self.graph.get_node(best_id)
        if best_node and best_node.evaluation.latency_ms is not None:
          best_latency = f"{best_node.evaluation.latency_ms:.3f} ms"
      logger.info(
        f"Step completed. Total nodes in graph: {len(self.graph.nodes)}. "
        f"Current best latency: {best_latency}"
      )

      # 6. Step Cleanup
      self._post_step_hook()

    best_id = self.graph.best_node_id
    return (
      self.graph.get_node(best_id)
      if best_id
      else self.graph.get_node(self.graph.root_id)
    )
