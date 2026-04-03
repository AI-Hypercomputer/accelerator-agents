"""Integration tests for compilation validation loop using real TPU compilation.

These tests use real KernelCompilationChecker to actually compile code on TPU,
testing the compilation validation loop component. The fix agents are mocked
(no real LLM responses needed).

The KernelCompilationChecker is configured to auto-manage servers (auto_manage_servers=True),
so it will start the necessary TPU and eval servers if they are not running.

These are slower integration tests - use pytest markers to control execution:
  pytest -m integration       # Run only these tests
  pytest -m "not integration" # Skip these tests
"""

import pytest
import os
from google.adk.events import Event
from google.genai.types import Content, Part

from hitl_agent.subagents.kernel_writing.kernel_compilation import KernelCompilationChecker
from hitl_agent.subagents.kernel_writing.agent import (
    KernelCompilationValidationLoop as _KernelCompilationValidationLoop,)
from conftest import MockFixAgent, CompilationCheckerWrapper


# Test-only wrapper that bypasses BaseAgent.run_async scaffolding for simplified test setup
class KernelCompilationValidationLoop(_KernelCompilationValidationLoop):
  """Test wrapper that overrides run_async to directly call _run_async_impl."""

  async def run_async(self, ctx):
    async for event in self._run_async_impl(ctx):
      yield event


class TestCompilationValidationLoop:
  """Integration tests using real TPU compilation with mocked LLM agents."""

  @pytest.mark.asyncio
  @pytest.mark.integration
  async def test_compilation_success(self, mock_invocation_context,
                                     temp_workdir, kernel_code_valid):
    """Test compilation validation - successful case."""
    # Setup: Create a valid kernel file
    kernel_path = os.path.join(temp_workdir, "test_kernel.py")
    with open(kernel_path, "w") as f:
      f.write(kernel_code_valid)

    # Set up state
    mock_invocation_context.session.state["optimized_kernel_path"] = kernel_path
    mock_invocation_context.session.state["kernel_code"] = kernel_code_valid

    # Use real KernelCompilationChecker with auto_manage_servers
    compilation_checker = KernelCompilationChecker(
        name="RealCompilationChecker",
        input_key="kernel_code",
        output_key="compilation_results",
        auto_manage_servers=True  # Let the checker manage servers
    )
    # Wrap it to call _run_async_impl directly
    wrapped_checker = CompilationCheckerWrapper(compilation_checker)

    # Mock fix agent (should not be called for valid code)
    fix_called = [False]

    async def mock_fix(ctx):
      fix_called[0] = True
      yield Event(author="Fix", content=Content(parts=[Part(text="Fixed")]))

    mock_fix_agent = MockFixAgent(name="MockFixAgent", fix_func=mock_fix)

    # Create validation loop
    validation_loop = KernelCompilationValidationLoop(
        name="TestValidationLoop",
        compilation_checker=wrapped_checker,
        fix_agent=mock_fix_agent,
        max_retries=4)

    # Run validation
    events = []
    async for event in validation_loop._run_async_impl(mock_invocation_context):
      events.append(event)
      # Apply state_delta if present
      if hasattr(event, 'actions') and event.actions and hasattr(
          event.actions, 'state_delta'):
        if event.actions.state_delta:
          mock_invocation_context.session.state.update(
              event.actions.state_delta)

    # Assertions
    assert len(events) > 0, "Should have received events"

    # Check compilation succeeded
    compilation_result = mock_invocation_context.session.state.get(
        "compilation_results")
    assert compilation_result == "Success", f"Expected Success but got: {compilation_result}"

    # Fix agent should not have been called
    assert fix_called[
        0] is False, "Fix agent should not be called for valid code"

    # Check final status
    status = mock_invocation_context.session.state.get(
        "kernel_compilation_status")
    assert status is not None
    assert status["success"] is True
    assert status["retries"] == 0

  @pytest.mark.asyncio
  @pytest.mark.integration
  async def test_compilation_syntax_error(self, mock_invocation_context,
                                          temp_workdir,
                                          kernel_code_syntax_error):
    """Test compilation validation - syntax error case."""
    # Setup: Create a kernel file with syntax error
    kernel_path = os.path.join(temp_workdir, "test_kernel.py")
    with open(kernel_path, "w") as f:
      f.write(kernel_code_syntax_error)

    # Set up state
    mock_invocation_context.session.state["optimized_kernel_path"] = kernel_path
    mock_invocation_context.session.state[
        "kernel_code"] = kernel_code_syntax_error

    # Use real KernelCompilationChecker
    compilation_checker = KernelCompilationChecker(
        name="RealCompilationChecker",
        input_key="kernel_code",
        output_key="compilation_results",
        auto_manage_servers=True)
    # Wrap it to call _run_async_impl directly
    wrapped_checker = CompilationCheckerWrapper(compilation_checker)

    # Mock fix agent that doesn't actually fix (to test retry exhaustion)
    fix_call_count = [0]

    async def mock_fix(ctx):
      fix_call_count[0] += 1
      # Don't actually fix the code, just yield an event
      yield Event(author="Fix",
                  content=Content(parts=[Part(text="Attempted fix")]))

    mock_fix_agent = MockFixAgent(name="MockFixAgent", fix_func=mock_fix)

    # Create validation loop with limited retries
    validation_loop = KernelCompilationValidationLoop(
        name="TestValidationLoop",
        compilation_checker=wrapped_checker,
        fix_agent=mock_fix_agent,
        max_retries=2  # Limit retries for faster test
    )

    # Run validation
    events = []
    async for event in validation_loop._run_async_impl(mock_invocation_context):
      events.append(event)
      # Apply state_delta if present
      if hasattr(event, 'actions') and event.actions and hasattr(
          event.actions, 'state_delta'):
        if event.actions.state_delta:
          mock_invocation_context.session.state.update(
              event.actions.state_delta)

    # Assertions
    assert len(events) > 0, "Should have received events"

    # Check compilation failed
    compilation_result = mock_invocation_context.session.state.get(
        "compilation_results")
    assert compilation_result != "Success", "Should have failed compilation"
    assert "syntax" in compilation_result.lower(
    ) or "error" in compilation_result.lower()

    # Fix agent should have been called multiple times
    assert fix_call_count[0] > 0, "Fix agent should have been called"

    # Check final status shows failure
    status = mock_invocation_context.session.state.get(
        "kernel_compilation_status")
    assert status is not None
    assert status["success"] is False
    assert status["retries"] == 1  # Should have retried once before giving up

  @pytest.mark.asyncio
  @pytest.mark.integration
  async def test_compilation_retry_success(self, mock_invocation_context,
                                           temp_workdir,
                                           kernel_code_syntax_error,
                                           kernel_code_valid):
    """Test compilation validation - retry success case (initially wrong, then fixed)."""
    # Setup: Create a kernel file with syntax error initially
    kernel_path = os.path.join(temp_workdir, "test_kernel.py")
    with open(kernel_path, "w") as f:
      f.write(kernel_code_syntax_error)

    # Set up state
    mock_invocation_context.session.state["optimized_kernel_path"] = kernel_path
    mock_invocation_context.session.state[
        "kernel_code"] = kernel_code_syntax_error

    # Use real KernelCompilationChecker
    compilation_checker = KernelCompilationChecker(
        name="RealCompilationChecker",
        input_key="kernel_code",
        output_key="compilation_results",
        auto_manage_servers=True)
    # Wrap it to call _run_async_impl directly
    wrapped_checker = CompilationCheckerWrapper(compilation_checker)

    # Mock fix agent that fixes the code
    fix_call_count = [0]

    async def mock_fix(ctx):
      fix_call_count[0] += 1
      # Fix the code in state and file
      ctx.session.state["kernel_code"] = kernel_code_valid
      with open(kernel_path, "w") as f:
        f.write(kernel_code_valid)

      yield Event(author="Fix",
                  content=Content(parts=[Part(text="Fixed the code")]))

    mock_fix_agent = MockFixAgent(name="MockFixAgent", fix_func=mock_fix)

    # Create validation loop
    validation_loop = KernelCompilationValidationLoop(
        name="TestValidationLoop",
        compilation_checker=wrapped_checker,
        fix_agent=mock_fix_agent,
        max_retries=2)

    # Run validation
    events = []
    async for event in validation_loop._run_async_impl(mock_invocation_context):
      events.append(event)
      # Apply state_delta if present
      if hasattr(event, 'actions') and event.actions and hasattr(
          event.actions, 'state_delta'):
        if event.actions.state_delta:
          mock_invocation_context.session.state.update(
              event.actions.state_delta)

    # Assertions
    assert len(events) > 0

    # Check compilation succeeded
    compilation_result = mock_invocation_context.session.state.get(
        "compilation_results")
    assert compilation_result == "Success"

    # Fix agent should have been called once
    assert fix_call_count[0] == 1

    # Check final status
    status = mock_invocation_context.session.state.get(
        "kernel_compilation_status")
    assert status is not None
    assert status["success"] is True
    assert status["retries"] == 1
