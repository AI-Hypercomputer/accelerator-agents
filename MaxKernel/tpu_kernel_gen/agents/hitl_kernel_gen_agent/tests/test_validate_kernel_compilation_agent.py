"""Unit tests for ValidateKernelCompilationAgent with mocked compilation.

These tests verify the full ValidateKernelCompilationAgent workflow including:
- Reading kernel code
- Running the validation loop
- Cleanup operations
- Summary generation

All components including the compilation checker are mocked for fast execution.
For integration tests with real TPU compilation, see test_compilation_validation_loop.py.
"""

import os

import pytest
from conftest import MockAgent, MockCompilationChecker, MockFixAgent
from google.adk.events import Event
from google.genai.types import Content, Part
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.kernel_writing.agent import (
  KernelCompilationValidationLoop as _KernelCompilationValidationLoop,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.kernel_writing.agent import (
  ValidateKernelCompilationAgent as _ValidateKernelCompilationAgent,
)


# Test-only wrappers that bypass BaseAgent.run_async scaffolding for simplified test setup
class KernelCompilationValidationLoop(_KernelCompilationValidationLoop):
  """Test wrapper that overrides run_async to directly call _run_async_impl."""

  async def run_async(self, ctx):
    async for event in self._run_async_impl(ctx):
      yield event


class ValidateKernelCompilationAgent(_ValidateKernelCompilationAgent):
  """Test wrapper that overrides run_async to directly call _run_async_impl."""

  async def run_async(self, ctx):
    async for event in self._run_async_impl(ctx):
      yield event


class TestValidateKernelCompilationAgent:
  """Unit tests for ValidateKernelCompilationAgent with all components mocked."""

  @pytest.mark.asyncio
  @pytest.mark.unit
  async def test_full_agent_compilation_success(self, mock_invocation_context, temp_workdir, kernel_code_valid):
    """Test compilation validation - successful case."""
    # Setup: Create a valid kernel file
    kernel_path = os.path.join(temp_workdir, "test_kernel.py")
    with open(kernel_path, "w") as f:
      f.write(kernel_code_valid)

    # Set up state
    mock_invocation_context.session.state["optimized_kernel_path"] = kernel_path
    mock_invocation_context.session.state["kernel_code"] = kernel_code_valid

    # Use Mock KernelCompilationChecker
    compilation_checker = MockCompilationChecker(
      name="MockCompilationChecker", output_key="compilation_results", results="Success"
    )

    # Mock fix agent (should not be called for valid code)
    fix_called = [False]

    async def mock_fix(ctx):
      fix_called[0] = True
      yield Event(author="Fix", content=Content(parts=[Part(text="Fixed")]))

    mock_fix_agent = MockFixAgent(name="MockFixAgent", fix_func=mock_fix)

    # Create validation loop
    validation_loop = KernelCompilationValidationLoop(
      name="TestValidationLoop", compilation_checker=compilation_checker, fix_agent=mock_fix_agent, max_retries=4
    )

    # Create other mock agents
    mock_read_file = MockAgent("MockReadFile")
    mock_cleanup = MockAgent("MockCleanup")
    mock_summary = MockAgent("MockSummary")

    # Create ValidateKernelCompilationAgent
    validate_agent = ValidateKernelCompilationAgent(
      name="TestValidateAgent",
      read_file_agent=mock_read_file,
      validation_loop_agent=validation_loop,
      cleanup_agent=mock_cleanup,
      summary_agent=mock_summary,
    )

    # Run validation
    events = []
    async for event in validate_agent._run_async_impl(mock_invocation_context):
      events.append(event)
      # Apply state_delta if present
      if hasattr(event, "actions") and event.actions and hasattr(event.actions, "state_delta"):
        if event.actions.state_delta:
          mock_invocation_context.session.state.update(event.actions.state_delta)

    # Assertions
    assert len(events) > 0, "Should have received events"

    # Check compilation succeeded
    compilation_result = mock_invocation_context.session.state.get("compilation_results")
    assert compilation_result == "Success", f"Expected Success but got: {compilation_result}"

    # Fix agent should not have been called
    assert fix_called[0] is False, "Fix agent should not be called for valid code"

    # Check final status
    status = mock_invocation_context.session.state.get("kernel_compilation_status")
    assert status is not None
    assert status["success"] is True
    assert status["retries"] == 0

    # Check that cleanup and summary were called
    assert mock_cleanup.call_count == 1
    assert mock_summary.call_count == 1
    # read_file should not be called because path is in state and exists
    assert mock_read_file.call_count == 0

  @pytest.mark.asyncio
  @pytest.mark.unit
  async def test_full_agent_compilation_failure(self, mock_invocation_context, temp_workdir, kernel_code_syntax_error):
    """Test full agent with compilation failure and retry exhaustion."""
    # Setup: Create a kernel file with syntax error
    kernel_path = os.path.join(temp_workdir, "test_kernel.py")
    with open(kernel_path, "w") as f:
      f.write(kernel_code_syntax_error)

    # Set up state
    mock_invocation_context.session.state["optimized_kernel_path"] = kernel_path
    mock_invocation_context.session.state["kernel_code"] = kernel_code_syntax_error

    # Use Mock KernelCompilationChecker
    compilation_checker = MockCompilationChecker(
      name="MockCompilationChecker", output_key="compilation_results", results="Syntax Error: missing parenthesis"
    )

    # Mock fix agent that doesn't actually fix (to test retry exhaustion)
    fix_call_count = [0]

    async def mock_fix(ctx):
      fix_call_count[0] += 1
      # Don't actually fix the code, just yield an event
      yield Event(author="Fix", content=Content(parts=[Part(text="Attempted fix")]))

    mock_fix_agent = MockFixAgent(name="MockFixAgent", fix_func=mock_fix)

    # Create validation loop with limited retries
    validation_loop = KernelCompilationValidationLoop(
      name="TestValidationLoop",
      compilation_checker=compilation_checker,
      fix_agent=mock_fix_agent,
      max_retries=2,  # Limit retries for faster test
    )

    # Create other mock agents
    mock_read_file = MockAgent("MockReadFile")
    mock_cleanup = MockAgent("MockCleanup")
    mock_summary = MockAgent("MockSummary")

    # Create ValidateKernelCompilationAgent
    validate_agent = ValidateKernelCompilationAgent(
      name="TestValidateAgent",
      read_file_agent=mock_read_file,
      validation_loop_agent=validation_loop,
      cleanup_agent=mock_cleanup,
      summary_agent=mock_summary,
    )

    # Run validation
    events = []
    async for event in validate_agent._run_async_impl(mock_invocation_context):
      events.append(event)
      # Apply state_delta if present
      if hasattr(event, "actions") and event.actions and hasattr(event.actions, "state_delta"):
        if event.actions.state_delta:
          mock_invocation_context.session.state.update(event.actions.state_delta)

    # Assertions
    assert len(events) > 0, "Should have received events"

    # Check compilation failed
    compilation_result = mock_invocation_context.session.state.get("compilation_results")
    assert compilation_result != "Success", "Should have failed compilation"
    assert "syntax" in compilation_result.lower() or "error" in compilation_result.lower()

    # Fix agent should have been called multiple times
    assert fix_call_count[0] > 0, "Fix agent should have been called"

    # Check final status shows failure
    status = mock_invocation_context.session.state.get("kernel_compilation_status")
    assert status is not None
    assert status["success"] is False
    assert status["retries"] == 1  # Should have retried once before giving up

    # Check that cleanup and summary were called
    assert mock_cleanup.call_count == 1
    assert mock_summary.call_count == 1

  @pytest.mark.asyncio
  @pytest.mark.unit
  async def test_full_agent_retry_success(
    self, mock_invocation_context, temp_workdir, kernel_code_syntax_error, kernel_code_valid
  ):
    """Test full agent with retry success (initially fails, then succeeds after fix)."""
    # Setup: Create a kernel file with syntax error initially
    kernel_path = os.path.join(temp_workdir, "test_kernel.py")
    with open(kernel_path, "w") as f:
      f.write(kernel_code_syntax_error)

    # Set up state
    mock_invocation_context.session.state["optimized_kernel_path"] = kernel_path
    mock_invocation_context.session.state["kernel_code"] = kernel_code_syntax_error

    # Use Mock KernelCompilationChecker with sequential results
    compilation_checker = MockCompilationChecker(
      name="MockCompilationChecker", output_key="compilation_results", results=["Syntax Error", "Success"]
    )

    # Mock fix agent that fixes the code
    fix_call_count = [0]

    async def mock_fix(ctx):
      fix_call_count[0] += 1
      # Fix the code in state and file
      ctx.session.state["kernel_code"] = kernel_code_valid
      with open(kernel_path, "w") as f:
        f.write(kernel_code_valid)

      yield Event(author="Fix", content=Content(parts=[Part(text="Fixed the code")]))

    mock_fix_agent = MockFixAgent(name="MockFixAgent", fix_func=mock_fix)

    # Create validation loop
    validation_loop = KernelCompilationValidationLoop(
      name="TestValidationLoop", compilation_checker=compilation_checker, fix_agent=mock_fix_agent, max_retries=2
    )

    # Create other mock agents
    mock_read_file = MockAgent("MockReadFile")
    mock_cleanup = MockAgent("MockCleanup")
    mock_summary = MockAgent("MockSummary")

    # Create ValidateKernelCompilationAgent
    validate_agent = ValidateKernelCompilationAgent(
      name="TestValidateAgent",
      read_file_agent=mock_read_file,
      validation_loop_agent=validation_loop,
      cleanup_agent=mock_cleanup,
      summary_agent=mock_summary,
    )

    # Run validation
    events = []
    async for event in validate_agent._run_async_impl(mock_invocation_context):
      events.append(event)
      # Apply state_delta if present
      if hasattr(event, "actions") and event.actions and hasattr(event.actions, "state_delta"):
        if event.actions.state_delta:
          mock_invocation_context.session.state.update(event.actions.state_delta)

    # Assertions
    assert len(events) > 0

    # Check compilation succeeded
    compilation_result = mock_invocation_context.session.state.get("compilation_results")
    assert compilation_result == "Success"

    # Fix agent should have been called once
    assert fix_call_count[0] == 1

    # Check final status
    status = mock_invocation_context.session.state.get("kernel_compilation_status")
    assert status is not None
    assert status["success"] is True
    assert status["retries"] == 1

    # Check that cleanup and summary were called
    assert mock_cleanup.call_count == 1
    assert mock_summary.call_count == 1
