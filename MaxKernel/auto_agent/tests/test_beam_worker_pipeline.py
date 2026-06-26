"""Unit tests for BeamWorkerPipeline execution logic."""

import asyncio
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch, mock_open
import pytest

from auto_agent.subagents.beam_worker_pipeline import BeamWorkerPipeline
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext


# Define a dummy agent subclass that inherits from BaseAgent to satisfy Pydantic validations
class DummySubAgent(BaseAgent):
  # Custom mock callback runner
  mock_run: MagicMock = None

  async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
    if self.mock_run:
      async for event in self.mock_run(ctx):
        yield event
    else:
      yield Event(author=self.name)


# Helper to construct basic yield generator
async def default_mock_run(ctx):
  yield Event(author="mock_agent")


@pytest.fixture
def mock_subagents():
  """Construct valid subagent instances using the DummySubAgent subclass."""
  subagents = {
      "plan": DummySubAgent(name="PlanAgent"),
      "implement": DummySubAgent(name="ImplementAgent"),
      "validate": DummySubAgent(name="ValidateAgent"),
      "test_gen": DummySubAgent(name="TestGenAgent"),
      "test_run": DummySubAgent(name="TestRunAgent")
  }
  
  # Set default side effects
  for agent in subagents.values():
    agent.mock_run = MagicMock(side_effect=default_mock_run)

  return subagents


@pytest.fixture
def mock_context():
  """Create a mock ADK InvocationContext."""
  from unittest.mock import AsyncMock

  ctx = MagicMock(spec=InvocationContext)
  
  # Ensure copies of context return the same configured mock context
  ctx.model_copy.return_value = ctx
  ctx.end_invocation = False

  ctx.session = MagicMock()

  ctx.session.id = "mock_session_id"
  ctx.session.state = {
      "history": [],
      "kernel_compilation_status": {},
      "validation_loop_status": {},
      "test_results": {},
      "autotune_results": {}
  }

  # Mock async plugin manager callbacks to be awaitable
  ctx.plugin_manager = MagicMock()
  ctx.plugin_manager.run_before_agent_callback = AsyncMock(return_value=None)
  ctx.plugin_manager.run_after_agent_callback = AsyncMock(return_value=None)

  return ctx



@pytest.fixture
def worker_pipeline(mock_subagents):
  """Instantiate BeamWorkerPipeline with mocked subagents."""
  pipeline = BeamWorkerPipeline(
      name="TestBeamWorker",
      plan_agent=mock_subagents["plan"],
      implement_agent=mock_subagents["implement"],
      validate_agent=mock_subagents["validate"],
      test_gen_agent=mock_subagents["test_gen"],
      test_run_agent=mock_subagents["test_run"],
      max_iterations=3
  )
  return pipeline


@pytest.mark.asyncio
@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("shutil.copy2")
@patch("builtins.open", new_callable=mock_open, read_data="def optimized_kernel(): pass")
async def test_early_exit_on_success(mock_file_open, mock_copy, mock_exists, mock_makedirs, worker_pipeline, mock_context):
  """Test Case 1: Verifies worker pipeline exits early and yields Success event on correctness success."""
  # Setup: Set compilation, test generation, and test execution successes to True
  mock_context.session.state["kernel_compilation_status"] = {"success": True}
  mock_context.session.state["validation_loop_status"] = {"success": True}
  mock_context.session.state["test_results"] = {
      "success": True,
      "output": "PERF_METRICS: 8.52 ms"
  }
  mock_context.session.state["optimized_kernel_path"] = "/dummy/optimized_kernel.py"

  events = []
  async for event in worker_pipeline._run_async_impl(mock_context):
    events.append(event)

  # Check that each subagent's mock callback was invoked once
  assert worker_pipeline.plan_agent.mock_run.call_count == 1
  assert worker_pipeline.implement_agent.mock_run.call_count == 1
  assert worker_pipeline.validate_agent.mock_run.call_count == 1
  assert worker_pipeline.test_gen_agent.mock_run.call_count == 1
  assert worker_pipeline.test_run_agent.mock_run.call_count == 1

  # Check final emitted success state delta
  delta_event = events[-1]
  assert delta_event.actions.state_delta["worker_status"] == "Success"
  assert delta_event.actions.state_delta["pipeline_status"] == "Completed"
  assert delta_event.actions.state_delta["latency_ms"] == 8.52
  assert delta_event.actions.state_delta["kernel_code"] == "def optimized_kernel(): pass"


@pytest.mark.asyncio
@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("shutil.copy2")
@patch("builtins.open", new_callable=mock_open, read_data="def code(): pass")
async def test_iteration_loop_retries(mock_file_open, mock_copy, mock_exists, mock_makedirs, worker_pipeline, mock_context):
  """Test Case 2: Verifies pipeline loops back to planning and retries on failures."""
  states_history = [
      # Iteration 0 (Compilation failure)
      {
          "kernel_compilation_status": {"success": False},
          "validation_loop_status": {},
          "test_results": {}
      },
      # Iteration 1 (Test generation failure)
      {
          "kernel_compilation_status": {"success": True},
          "validation_loop_status": {"success": False},
          "test_results": {}
      },
      # Iteration 2 (Correctness Success)
      {
          "kernel_compilation_status": {"success": True},
          "validation_loop_status": {"success": True},
          "test_results": {"success": True, "output": "PERF_METRICS: 9.1 ms"}
      }
  ]

  # Mock run_async of plan_agent to dynamically adjust state context
  call_count = 0
  async def plan_side_effect(ctx):
    nonlocal call_count
    # Inject state variables for this iteration
    ctx.session.state["kernel_compilation_status"] = states_history[call_count]["kernel_compilation_status"]
    ctx.session.state["validation_loop_status"] = states_history[call_count]["validation_loop_status"]
    ctx.session.state["test_results"] = states_history[call_count]["test_results"]
    ctx.session.state["optimized_kernel_path"] = "/dummy/optimized_kernel.py"
    call_count += 1
    yield Event(author="PlanAgent")

  worker_pipeline.plan_agent.mock_run = MagicMock(side_effect=plan_side_effect)

  events = []
  async for event in worker_pipeline._run_async_impl(mock_context):
    events.append(event)

  # Asserts it ran exactly 3 iterations (0, 1, 2)
  assert call_count == 3
  assert worker_pipeline.plan_agent.mock_run.call_count == 3

  # Check final status
  delta_event = events[-1]
  assert delta_event.actions.state_delta["worker_status"] == "Success"
  assert delta_event.actions.state_delta["pipeline_status"] == "Completed"
  assert delta_event.actions.state_delta["latency_ms"] == 9.1


@pytest.mark.asyncio
@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("shutil.copy2")
@patch("builtins.open", new_callable=mock_open, read_data="def code(): pass")
async def test_exhaust_iterations_failure(mock_file_open, mock_copy, mock_exists, mock_makedirs, worker_pipeline, mock_context):
  """Test Case 3: Verifies failure event emitted when iteration limit is hit without correctness success."""
  # Set max_iterations = 2
  worker_pipeline.max_iterations = 2

  # Set compilation and test generation success, but correctness tests always fail
  mock_context.session.state["kernel_compilation_status"] = {"success": True}
  mock_context.session.state["validation_loop_status"] = {"success": True}
  mock_context.session.state["test_results"] = {"success": False}
  mock_context.session.state["optimized_kernel_path"] = "/dummy/optimized_kernel.py"

  events = []
  async for event in worker_pipeline._run_async_impl(mock_context):
    events.append(event)

  # Check final emitted event is failure
  delta_event = events[-1]
  assert delta_event.actions.state_delta["worker_status"] == "Failed"
  assert delta_event.actions.state_delta["pipeline_status"] == "Failed"
