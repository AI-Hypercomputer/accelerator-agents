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
from auto_search.algorithms.beam_search import BeamSearchOrchestrator
from auto_search.algorithms.parallel_search import (
  SimpleParallelSearchOrchestrator,
)
from auto_search.graph import EvaluationResult, Node, SearchGraph
from auto_search.run_search import run_search
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
    # Intermediate partial node should have IN_PROGRESS
    intermediate_nodes = [
      n for n in updated_nodes if n.execution_status == "IN_PROGRESS"
    ]
    assert len(intermediate_nodes) >= 1
    # Result node should also match and have SUCCESS
    assert result_node.execution_status == "SUCCESS"
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
    assert salvaged_nodes[-1].execution_status == "INTERRUPTED"
    assert salvaged_nodes[-1].evaluation.speedup == 2.5
    assert salvaged_nodes[-1].code == "def salvaged_kernel(): pass"
    assert result_node.execution_status == "INTERRUPTED"
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


def test_orchestrator_initial_kernel_code_root_seeding(temp_dir):
  """Test SearchOrchestrator correctly seeds root node from initial_kernel_code."""
  db_path = os.path.join(temp_dir, "search_graph.json")
  seed_code = "def seed_kernel(): pass"
  ref_code = "def reference_kernel(): pass"

  orch = SimpleParallelSearchOrchestrator(
    num_parallel_runs=1,
    problem_id="test_seed_orch",
    reference_code=ref_code,
    graph_db_path=db_path,
    initial_kernel_code=seed_code,
  )

  assert orch.graph.root_id is not None
  root_node = orch.graph.get_node(orch.graph.root_id)
  assert root_node is not None
  assert root_node.code == seed_code
  assert root_node.strategy_applied == "seed_kernel"
  assert orch.graph.best_node_id == root_node.node_id

  base_kernel_file = os.path.join(root_node.session_dir, "base_kernel.py")
  assert os.path.exists(base_kernel_file)
  with open(base_kernel_file) as f:
    assert f.read() == seed_code

  # Check persistence
  with open(db_path) as f:
    saved = json.load(f)
  assert saved["best_node_id"] == root_node.node_id
  assert saved["nodes"][root_node.node_id]["strategy_applied"] == "seed_kernel"


def test_worker_prepare_inputs_and_query_with_seed_kernel(temp_dir):
  """Test worker prepares optimized_kernel.py and includes seed_hint in query for seed/optimized parent."""
  worker = ADKSessionWorker()

  # 1. Baseline root node: should NOT create optimized_kernel.py
  baseline_dir = os.path.join(temp_dir, "baseline_session")
  baseline_node = Node(
    node_id="root",
    parent_id=None,
    session_dir=temp_dir,
    code="def base(): pass",
    plan="",
    depth=0,
    strategy_applied="baseline",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=0.0),
  )
  worker._prepare_inputs(baseline_dir, baseline_node, "def base(): pass")
  assert not os.path.exists(os.path.join(baseline_dir, "optimized_kernel.py"))

  # 2. Seed kernel root node: SHOULD create optimized_kernel.py
  seed_dir = os.path.join(temp_dir, "seed_session")
  seed_node = Node(
    node_id="root",
    parent_id=None,
    session_dir=temp_dir,
    code="def seed(): pass",
    plan="",
    depth=0,
    strategy_applied="seed_kernel",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=0.0),
  )
  worker._prepare_inputs(seed_dir, seed_node, "def ref(): pass")
  opt_path = os.path.join(seed_dir, "optimized_kernel.py")
  assert os.path.exists(opt_path)
  with open(opt_path) as f:
    assert f.read() == "def seed(): pass"

  # 3. Evaluated child node with speedup > 0: SHOULD create optimized_kernel.py
  evaluated_dir = os.path.join(temp_dir, "eval_session")
  evaluated_node = Node(
    node_id="eval_0",
    parent_id="root",
    session_dir=temp_dir,
    code="def eval_opt(): pass",
    plan="",
    depth=0,
    strategy_applied="custom",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=1.8),
  )
  worker._prepare_inputs(evaluated_dir, evaluated_node, "def ref(): pass")
  assert os.path.exists(os.path.join(evaluated_dir, "optimized_kernel.py"))

  # 4. Check query contains seed_hint in _run_agent
  async def _check_query():
    captured_query = None

    def fake_client_init(*args, **kwargs):
      nonlocal captured_query
      captured_query = kwargs.get("query", "")
      client = MagicMock()
      client.session = MagicMock()
      client.create_session = AsyncMock()
      client.run_async = AsyncMock()
      client.get_session_data = MagicMock(return_value={})
      client.get_state = MagicMock(return_value={})
      return client

    with (
      patch("auto_search.worker.create_root_agent", return_value=MagicMock()),
      patch("auto_search.worker.AutoAgentClient", side_effect=fake_client_init),
    ):
      await worker._run_agent(
        node_id="child_seed",
        parent_node=seed_node,
        session_dir=seed_dir,
        strategy="tiling",
      )

    expected_hint = "An existing candidate kernel is provided at optimized_kernel.py. Hill-climb and optimize this kernel further."
    assert expected_hint in captured_query
    assert "Focus on: tiling." in captured_query

  asyncio.run(_check_query())


def test_parallel_search_resume_node_id_frontier_selection(temp_dir):
  """Test SimpleParallelSearchOrchestrator selects resume_node_id for expansion."""
  db_path = os.path.join(temp_dir, "search_graph.json")
  graph = SearchGraph("test_parallel_resume", db_path)

  root = Node(
    node_id="node_000",
    parent_id=None,
    session_dir=temp_dir,
    code="def base(): pass",
    plan="",
    depth=0,
    strategy_applied="baseline",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=1.0),
  )
  child = Node(
    node_id="node_001",
    parent_id="node_000",
    session_dir=temp_dir,
    code="def child(): pass",
    plan="",
    depth=1,
    strategy_applied="tiling",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=2.0),
  )
  graph.add_node(root)
  graph.add_node(child)

  orch = SimpleParallelSearchOrchestrator(
    num_parallel_runs=2,
    strategies=["strat_a", "strat_b"],
    problem_id="test_parallel_resume",
    reference_code="def base(): pass",
    graph_db_path=db_path,
    resume_node_id="node_001",
  )

  # Check selection
  selected = orch._select_nodes_to_expand()
  assert len(selected) == 1
  assert selected[0].node_id == "node_001"

  # Check task generation targeting the resumed node
  tasks = orch._generate_expansion_tasks(selected)
  assert len(tasks) == 2
  for parent_node, strategy in tasks:
    assert parent_node.node_id == "node_001"
    assert strategy in ["strat_a", "strat_b"]


def test_beam_search_resume_node_id_frontier_selection(temp_dir):
  """Test BeamSearchOrchestrator resumes from resume_node_id."""
  db_path = os.path.join(temp_dir, "search_graph.json")
  graph = SearchGraph("test_beam_resume", db_path)

  root = Node(
    node_id="node_000",
    parent_id=None,
    session_dir=temp_dir,
    code="def base(): pass",
    plan="",
    depth=0,
    strategy_applied="baseline",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=1.0),
  )
  node1 = Node(
    node_id="node_001",
    parent_id="node_000",
    session_dir=temp_dir,
    code="def n1(): pass",
    plan="",
    depth=1,
    strategy_applied="strat1",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=1.5),
  )
  node2 = Node(
    node_id="node_002",
    parent_id="node_000",
    session_dir=temp_dir,
    code="def n2(): pass",
    plan="",
    depth=1,
    strategy_applied="strat2",
    execution_status="SUCCESS",
    evaluation=EvaluationResult(compiled=True, correct=True, speedup=1.8),
  )
  graph.add_node(root)
  graph.add_node(node1)
  graph.add_node(node2)
  graph.metadata["current_depth"] = 1
  graph.metadata["beam_node_ids"] = ["node_001"]
  graph.save()

  orch = BeamSearchOrchestrator(
    beam_size=2,
    branches_per_node=2,
    max_depth=3,
    problem_id="test_beam_resume",
    reference_code="def base(): pass",
    graph_db_path=db_path,
    resume_node_id="node_002",
  )
  orch.resume()

  assert orch.current_depth == 1
  assert len(orch.beam) == 1
  assert orch.beam[0].node_id == "node_002"

  selected = orch._select_nodes_to_expand()
  assert len(selected) == 1
  assert selected[0].node_id == "node_002"


def test_run_search_with_initial_kernel_path(temp_dir):
  """Test run_search correctly reads initial_kernel_path and seeds search."""

  async def _run():
    ref_file = os.path.join(temp_dir, "ref.py")
    with open(ref_file, "w") as f:
      f.write("def reference(): pass")

    seed_file = os.path.join(temp_dir, "seed.py")
    with open(seed_file, "w") as f:
      f.write("def seeded_solution(): pass")

    db_path = os.path.join(temp_dir, "search_graph.json")

    with patch(
      "auto_search.algorithms.parallel_search.SimpleParallelSearchOrchestrator.run",
      new_callable=AsyncMock,
    ):
      prob_id, status = await run_search(
        reference_file_path=ref_file,
        graph_db_path=db_path,
        initial_kernel_path=seed_file,
        algorithm="parallel",
      )

    with open(db_path) as f:
      saved = json.load(f)

    root_id = saved["root_id"]
    assert saved["nodes"][root_id]["code"] == "def seeded_solution(): pass"
    assert saved["nodes"][root_id]["strategy_applied"] == "seed_kernel"
    assert saved["best_node_id"] == root_id

  asyncio.run(_run())
