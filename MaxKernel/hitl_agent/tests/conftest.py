"""Shared fixtures and test utilities for HITL agent tests."""

import tempfile
from typing import AsyncGenerator, Callable, Optional
from unittest.mock import Mock

import pytest
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.sessions import Session
from google.genai.types import Content, Part
from pydantic import PrivateAttr


@pytest.fixture
def temp_workdir():
  """Create a temporary work directory for tests."""
  with tempfile.TemporaryDirectory() as tmpdir:
    yield tmpdir


@pytest.fixture
def mock_session(temp_workdir):
  """Create a proper ADK session with initialized state."""
  session = Session(
    id="test-session-123", user_id="test-user", appName="test-hitl-agent"
  )
  # Initialize state with test values
  session.state["workdir"] = temp_workdir
  session.state["tpu_version"] = "v5e"
  return session


@pytest.fixture
def mock_invocation_context(mock_session):
  """Create a mock invocation context."""
  ctx = Mock()
  ctx.session = mock_session
  ctx.invocation_id = "test-invocation"
  ctx.branch = "main"

  # Create a mock plugin manager with async methods
  mock_plugin_manager = Mock()

  async def mock_before_agent_callback(agent, callback_context):
    return None

  async def mock_after_agent_callback(agent, callback_context):
    return None

  mock_plugin_manager.run_before_agent_callback = mock_before_agent_callback
  mock_plugin_manager.run_after_agent_callback = mock_after_agent_callback

  ctx.plugin_manager = mock_plugin_manager

  # Mock model_copy to return a new context that shares the same session
  def model_copy_func(update=None):
    new_ctx = Mock()
    new_ctx.session = ctx.session  # Share the same session/state
    new_ctx.invocation_id = ctx.invocation_id
    new_ctx.branch = ctx.branch
    new_ctx.plugin_manager = ctx.plugin_manager
    new_ctx.model_copy = model_copy_func
    # Apply any updates if provided
    if update:
      for key, value in update.items():
        setattr(new_ctx, key, value)
    return new_ctx

  ctx.model_copy = model_copy_func

  return ctx


@pytest.fixture
def kernel_code_valid():
  """Valid kernel code that should compile."""
  return """
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def matmul_kernel(x_ref, y_ref, o_ref):
    # Simple matrix multiplication
    o_ref[...] = x_ref[...] @ y_ref[...]

def matmul(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
    )(x, y)
"""


@pytest.fixture
def kernel_code_syntax_error():
  """Kernel code with syntax error."""
  return """
import jax
import jax.numpy as jnp

def broken_kernel(x_ref, y_ref, o_ref):
    # Missing closing parenthesis
    o_ref[...] = x_ref[...] @ y_ref[...
"""


@pytest.fixture
def kernel_code_import_error():
  """Kernel code with import error."""
  return """
import jax
import nonexistent_module  # This will fail

def kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] @ y_ref[...]
"""


# Shared test helper classes


class MockAgent(BaseAgent):
  """Generic mock agent for tests."""

  _run_func: Optional[Callable] = PrivateAttr(default=None)
  _call_count: int = PrivateAttr(default=0)

  def __init__(self, name: str, run_func=None):
    super().__init__(name=name)
    self._run_func = run_func
    self._call_count = 0

  @property
  def call_count(self):
    return self._call_count

  async def run_async(self, ctx):
    """Override run_async to directly call _run_async_impl for tests."""
    async for event in self._run_async_impl(ctx):
      yield event

  async def _run_async_impl(self, ctx):
    """Implement _run_async_impl like a normal agent."""
    self._call_count += 1
    if self._run_func:
      async for event in self._run_func(ctx):
        yield event
    else:
      yield Event(
        author=self.name,
        content=Content(parts=[Part(text=f"Mock {self.name} run")]),
      )


class MockFixAgent(MockAgent):
  """Mock fix agent that simulates fixing compilation errors."""

  def __init__(self, name: str, fix_func=None):
    super().__init__(name=name, run_func=fix_func)


class MockCompilationChecker(BaseAgent):
  """Mock compilation checker that returns predefined results."""

  _output_key: str = PrivateAttr()
  _results: list = PrivateAttr()
  _call_count: int = PrivateAttr(default=0)

  def __init__(self, name, output_key, results):
    super().__init__(name=name)
    self._output_key = output_key
    self._results = results if isinstance(results, list) else [results]
    self._call_count = 0

  async def run_async(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Override run_async to directly call _run_async_impl for tests."""
    async for event in self._run_async_impl(ctx):
      yield event

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if self._call_count < len(self._results):
      result = self._results[self._call_count]
    else:
      result = self._results[-1]

    self._call_count += 1

    # Update state directly so the loop sees it immediately
    ctx.session.state[self._output_key] = result

    yield Event(
      author=self.name,
      actions=EventActions(state_delta={self._output_key: result}),
    )


class CompilationCheckerWrapper(BaseAgent):
  """Wrapper that calls a real KernelCompilationChecker's _run_async_impl directly."""

  def __init__(self, real_checker):
    super().__init__(name=real_checker.name)
    self._real_checker = real_checker

  async def run_async(self, ctx):
    """Call the real checker's implementation directly."""
    async for event in self._real_checker._run_async_impl(ctx):
      yield event
