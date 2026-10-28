"""Unit tests for incremental search graph updates and persistence."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.events import Event, EventActions
from google.adk.sessions import Session

from auto_agent.agent_client.auto_agent_client import AutoAgentClient
from auto_search.algorithms.parallel_search import (
  SimpleParallelSearchOrchestrator,
)
from auto_search.graph import EvaluationResult, Node, SearchGraph
from auto_search.worker import ADKSessionWorker


@pytest.fixture
def temp_dir():
  with tempfile.TemporaryDirectory() as tmpdir:
    yield tmpdir


def test_auto_agent_client_run_async_triggers_on_event():
  """Test AutoAgentClient.run_async triggers on_event callback."""

  async def _run():
    from google.adk.agents import BaseAgent

    mock_agent = MagicMock(spec=BaseAgent)
    client = AutoAgentClient(
      user_id="test_user",
      session_id="test_session",
      query="test query",
      agent=mock_agent,
    )

    # Mock session service
    client.session_service = AsyncMock()
    fake_session = Session(
      id="test_session",
      user_id="test_user",
      appName="test_app",
    )
    client.session_service.create_session.return_value = fake_session
    client.session_service.get_session.return_value = fake_session

    events_received = []

    def callback(event):
      events_received.append(event)

    mock_event = Event(author="test_agent")

    async def fake_run_async(*args, **kwargs):
      yield mock_event

    with patch(
      "auto_agent.agent_client.auto_agent_client.Runner"
    ) as mock_runner_cls:
      mock_runner_instance = MagicMock()
      mock_runner_instance.run_async = fake_run_async
      mock_runner_cls.return_value = mock_runner_instance

      await client.run_async(on_event=callback)

    assert len(events_received) == 1
    assert events_received[0] == mock_event

  asyncio.run(_run())


def test_worker_triggers_on_node_update_on_intermediate_history_events(
  temp_dir,
):
  """Test ADKSessionWorker triggers on_node_update on intermediate history events."""

  async def _run():
    worker = ADKSessionWorker()
    parent_node = Node(
      node_id="root",
      parent_id=None,
      session_dir=temp_dir,
      code="def base(): pass",
      plan="",
      depth=0,
      strategy_applied="baseline",
      execution_status="SUCCESS",
      evaluation=EvaluationResult(compiled=True, correct=True, speedup=1.0),
    )

    updated_nodes = []

    def on_node_update(node: Node):
      updated_nodes.append(node)

    # Mock AutoAgentClient inside _run_agent
    history_state = {
      "history": [
        {
          "iteration": 0,
          "kernel_code": "def opt_iter0(): pass",
          "plan": "plan 0",
          "compilation_status": {"success": True},
          "test_status": {"success": True},
          "speedup": 1.5,
        }
      ],
      "best_iteration": 0,
    }

    mock_client = MagicMock()
    mock_client.create_session = AsyncMock()
    mock_client.get_session_data = MagicMock(return_value={})
    mock_client.get_state = MagicMock(return_value=history_state)

    event_actions = EventActions(
      state_delta={"history": history_state["history"]}
    )
    mock_event = Event(author="pipeline_agent", actions=event_actions)

    async def mock_run_async(on_event=None):
      if on_event:
        on_event(mock_event)

    mock_client.run_async = mock_run_async

    with (
      patch("auto_search.worker.create_root_agent", return_value=MagicMock()),
      patch("auto_search.worker.AutoAgentClient", return_value=mock_client),
    ):
      result_node = await worker.expand_node(
        node_id="child_1",
        parent_node=parent_node,
        session_dir=temp_dir,
        reference_code="def base(): pass",
        strategy="test_strategy",
        on_node_update=on_node_update,
      )

    assert len(updated_nodes) >= 1
    # The partial node called during event
    assert any(
      n.node_id == "child_1" and n.evaluation.speedup == 1.5
      for n in updated_nodes
    )
    # Result node should also match
    assert result_node.evaluation.speedup == 1.5
    assert result_node.code == "def opt_iter0(): pass"

  asyncio.run(_run())


def test_search_graph_incremental_persistence_and_best_node_update(temp_dir):
  """Test SearchGraph.add_node incrementally updates search_graph.json and updates best_node_id."""
  db_path = os.path.join(temp_dir, "search_graph.json")
  graph = SearchGraph("test_problem", db_path)

  # Root node
  root = Node(
    node_id="root",
    parent_id=None,
    session_dir=temp_dir,
    code="root_code",
    plan="",
    depth=0,
    strategy_applied="baseline",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=1.0),
  )
  graph.add_node(root)

  assert graph.best_node_id == "root"
  assert os.path.exists(db_path)
  with open(db_path) as f:
    saved = json.load(f)
  assert saved["best_node_id"] == "root"
  assert "root" in saved["nodes"]

  # Child node with better speedup
  child = Node(
    node_id="child_1",
    parent_id="root",
    session_dir=temp_dir,
    code="child_code_1",
    plan="",
    depth=1,
    strategy_applied="strat1",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=2.0),
  )
  graph.add_node(child)
  assert graph.best_node_id == "child_1"

  # Updating the SAME node with higher speedup (incremental update during search)
  child_improved = Node(
    node_id="child_1",
    parent_id="root",
    session_dir=temp_dir,
    code="child_code_improved",
    plan="",
    depth=1,
    strategy_applied="strat1",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=3.5),
  )
  graph.add_node(child_improved)
  assert graph.best_node_id == "child_1"
  assert graph.get_node("child_1").evaluation.speedup == 3.5

  with open(db_path) as f:
    saved_after = json.load(f)
  assert saved_after["best_node_id"] == "child_1"
  assert saved_after["nodes"]["child_1"]["evaluation"]["speedup"] == 3.5


def test_worker_interruption_salvages_partial_candidate(temp_dir):
  """Test worker interruption salvages partial candidate node to on_node_update."""

  async def _run():
    worker = ADKSessionWorker()
    parent_node = Node(
      node_id="root",
      parent_id=None,
      session_dir=temp_dir,
      code="def base(): pass",
      plan="",
      depth=0,
      strategy_applied="baseline",
      execution_status="SUCCESS",
      evaluation=EvaluationResult(compiled=True, correct=True, speedup=1.0),
    )

    history_state = {
      "history": [
        {
          "iteration": 0,
          "kernel_code": "def salvaged_kernel(): pass",
          "plan": "salvaged plan",
          "compilation_status": {"success": True},
          "test_status": {"success": True},
          "speedup": 2.5,
        }
      ],
      "best_iteration": 0,
    }

    mock_client = MagicMock()
    mock_client.create_session = AsyncMock()
    mock_client.session = MagicMock()
    mock_client.get_session_data = MagicMock(return_value={})
    mock_client.get_state = MagicMock(return_value=history_state)

    async def mock_run_async_crash(on_event=None):
      raise RuntimeError("Simulated agent timeout / error during execution")

    mock_client.run_async = mock_run_async_crash

    salvaged_nodes = []

    def on_node_update(node: Node):
      salvaged_nodes.append(node)

    with (
      patch("auto_search.worker.create_root_agent", return_value=MagicMock()),
      patch("auto_search.worker.AutoAgentClient", return_value=mock_client),
    ):
      result_node = await worker.expand_node(
        node_id="child_salvage",
        parent_node=parent_node,
        session_dir=temp_dir,
        reference_code="def base(): pass",
        strategy="test_strategy",
        on_node_update=on_node_update,
      )

    # Check that salvaged node was sent to callback and returned
    assert len(salvaged_nodes) >= 1
    assert salvaged_nodes[-1].evaluation.speedup == 2.5
    assert salvaged_nodes[-1].code == "def salvaged_kernel(): pass"
    assert result_node.evaluation.speedup == 2.5
    assert result_node.is_valid_candidate

  asyncio.run(_run())


def test_search_orchestrator_run_finally_saves_graph(temp_dir):
  """Test SearchOrchestrator.run() finally: self.graph.save() saves graph on exit."""

  async def _run():
    db_path = os.path.join(temp_dir, "search_graph.json")

    orch = SimpleParallelSearchOrchestrator(
      num_parallel_runs=1,
      problem_id="test_save_orch",
      reference_code="def base(): pass",
      graph_db_path=db_path,
    )

    # Verify initial root node was saved
    assert os.path.exists(db_path)

    # Track calls to graph.save()
    save_calls = []
    original_save = orch.graph.save

    def tracked_save():
      save_calls.append(len(orch.graph.nodes))
      original_save()

    orch.graph.save = tracked_save

    # Force an exception inside _select_nodes_to_expand to simulate abort/failure during search
    with patch.object(
      orch,
      "_select_nodes_to_expand",
      side_effect=RuntimeError("Search aborted abruptly"),
    ):
      with pytest.raises(RuntimeError, match="Search aborted abruptly"):
        await orch.run()

    # Ensure graph.save() was invoked in the finally block
    assert len(save_calls) >= 1
    assert os.path.exists(db_path)

  asyncio.run(_run())
