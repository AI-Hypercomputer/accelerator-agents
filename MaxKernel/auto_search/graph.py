import json
import logging
import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
  compiled: bool = False
  correct: bool = False
  latency_ms: Optional[float] = None
  profiling_summary: Optional[str] = None
  compilation_error: Optional[str] = None
  test_error: Optional[str] = None


class Node(BaseModel):
  node_id: str
  parent_id: Optional[str]
  session_id: str
  code: str
  plan: str
  depth: int
  strategy_applied: Optional[str] = None
  execution_status: str = "FAIL"  # SUCCESS, FAIL
  execution_error: Optional[str] = None
  evaluation: EvaluationResult = Field(default_factory=EvaluationResult)

  @property
  def is_valid_candidate(self) -> bool:
    return (
      self.execution_status == "SUCCESS"
      and self.evaluation.compiled
      and self.evaluation.correct
    )


class SearchGraph:
  def __init__(self, problem_id: str, graph_db_path: Optional[str] = None):
    self.problem_id = problem_id
    self.graph_db_path = graph_db_path
    self.nodes: Dict[str, Node] = {}
    self.root_id: Optional[str] = None
    self.best_node_id: Optional[str] = None
    self.metadata: Dict[str, Any] = {}  # Stores orchestrator state
    logger.info(f"Initializing SearchGraph for problem: {problem_id}")

    if self.graph_db_path and os.path.exists(self.graph_db_path):
      self.load()

  def add_node(self, node: Node) -> None:
    self.nodes[node.node_id] = node
    logger.info(
      f"Added node {node.node_id} (parent: {node.parent_id}, "
      f"depth: {node.depth}, status: {node.execution_status}) to graph."
    )

    if node.parent_id is None:
      self.root_id = node.node_id

    self._update_best_node(node)
    self.save()

  def get_node(self, node_id: str) -> Optional[Node]:
    """Retrieves a node by its id. Returns None if not found."""
    if node_id not in self.nodes:
      logger.warning(f"Node {node_id} not found in graph.")
      return None
    return self.nodes[node_id]

  def _update_best_node(self, node: Node) -> None:
    if not node.is_valid_candidate:
      logger.warning(f"Node {node.node_id} is not a valid candidate. Skip.")
      return

    latency = node.evaluation.latency_ms
    if latency is None:
      logger.warning(f"Node {node.node_id} has no latency.")
      return

    if self.best_node_id is None:
      self.best_node_id = node.node_id
      logger.info(
        f"Set initial best node: {node.node_id} (latency: {latency} ms)"
      )
      return

    best_node = self.nodes[self.best_node_id]
    best_latency = best_node.evaluation.latency_ms
    if best_latency is None or latency < best_latency:
      old_best_id = self.best_node_id
      old_latency = best_latency
      self.best_node_id = node.node_id
      logger.info(
        f"New best node found: {node.node_id} ({latency} ms). "
        f"Previous best: {old_best_id} ({old_latency} ms)."
      )

  def save(self) -> None:
    if not self.graph_db_path:
      return
    logger.info(f"Saving search graph to {self.graph_db_path}...")

    dir_name = os.path.dirname(self.graph_db_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)

    data = {
      "problem_id": self.problem_id,
      "root_id": self.root_id,
      "best_node_id": self.best_node_id,
      "nodes": {nid: node.model_dump() for nid, node in self.nodes.items()},
      "metadata": self.metadata,
    }

    # Atomic write for the graph DB
    tmp_path = self.graph_db_path + ".tmp"
    try:
      with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
      os.replace(tmp_path, self.graph_db_path)
    except Exception as e:
      logger.error(f"Failed to save search graph atomically: {e}")
      if os.path.exists(tmp_path):
        try:
          os.remove(tmp_path)
        except OSError:
          pass

  def load(self) -> None:
    logger.info(f"Loading search graph from {self.graph_db_path}...")
    with open(self.graph_db_path, "r") as f:
      data = json.load(f)
    self.problem_id = data["problem_id"]
    self.root_id = data["root_id"]
    self.best_node_id = data["best_node_id"]
    self.metadata = data.get("metadata", {})
    self.nodes = {
      nid: Node.model_validate(ndat) for nid, ndat in data["nodes"].items()
    }
    logger.info(
      f"Successfully loaded search graph with {len(self.nodes)} nodes."
    )
