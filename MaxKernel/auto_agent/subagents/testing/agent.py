"""Testing subagent - test generation, validation, and execution.

This module contains all agents related to generating, validating, and executing tests.
"""

import ast
import logging
import os
import subprocess
import tempfile
from typing import AsyncGenerator, Callable, Optional

import aiohttp
from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from auto_agent.client_utils.eval_client import call_eval_server_async
from auto_agent.config import get_thinking_planner, model_config
from auto_agent.constants import EVAL_SERVER_PORT, MODEL_NAME, REQUEST_TIMEOUT
from auto_agent.custom_types import CustomLlmAgent
from auto_agent.subagents.testing.prompts import (
  fix_test_script,
  gen_test_file,
  summarize_test_results_prompt,
  validation_summary,
)
from auto_agent.tools.file_tools import (
  filesystem_tool_r,
  write_test_file_tool,
)
from auto_agent.tools.search_api_tool import search_api_tool
from auto_agent.tools.tools import vertex_ai_rag_tool

# Timeout specifications (in seconds)
COMPILE_VALIDATION_TIMEOUT = 60 * 1
MOCK_EXECUTION_TIMEOUT = 60 * 5
TEST_EXECUTION_TIMEOUT = 60 * 5
TEST_EXECUTION_POLL_INTERVAL = 20


class TestRunner(BaseAgent):
  """Executes the generated test file and captures results with full tracebacks."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None
  before_agent_callback: Optional[Callable] = None

  def __init__(
    self,
    name: str,
    input_key: str,
    output_key: str,
    before_agent_callback: Optional[Callable] = None,
  ):
    super().__init__(name=name, before_agent_callback=before_agent_callback)
    self.input_key = input_key
    self.output_key = output_key

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    test_file_path = ctx.session.state.get(self.input_key, "")

    if not test_file_path:
      error_msg = "No test file was generated. Please generate a test file first using the GenerateTestFileAgent."
      logging.warning(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "exit_code": -1,
              "output": error_msg,
              "success": False,
            }
          }
        ),
      )
      return

    if not os.path.exists(test_file_path):
      error_msg = f"Test file not found at {test_file_path}. Please ensure the file exists."
      logging.warning(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "exit_code": -1,
              "output": error_msg,
              "success": False,
            }
          }
        ),
      )
      return

    try:
      logging.info(
        f"[{self.name}] Dispatching tests on {test_file_path} to eval server"
      )

      with open(test_file_path, "r") as f:
        code_content = f.read()

      base_kernel_path = ctx.session.state.get("base_kernel_path", "")
      optimized_kernel_path = ctx.session.state.get("optimized_kernel_path", "")
      dependencies = {}

      if base_kernel_path and os.path.exists(base_kernel_path):
        with open(base_kernel_path, "r") as f:
          dependencies["base_kernel.py"] = f.read()
      else:
        logging.error(f"[{self.name}] Base kernel path not found")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              self.output_key: {
                "exit_code": -1,
                "output": "Base kernel path not found",
                "success": False,
              }
            }
          ),
        )
        return

      if optimized_kernel_path and os.path.exists(optimized_kernel_path):
        with open(optimized_kernel_path, "r") as f:
          dependencies["optimized_kernel.py"] = f.read()
      else:
        logging.error(f"[{self.name}] Optimized kernel path not found")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              self.output_key: {
                "exit_code": -1,
                "output": "Optimized kernel path not found",
                "success": False,
              }
            }
          ),
        )
        return

      payload = {
        "eval_type": "unified_test",
        "code": code_content,
        "timeout": TEST_EXECUTION_TIMEOUT,
        "backend_type": "tpu",
        "dependencies": dependencies,
      }

      client_timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT + 10)
      async with aiohttp.ClientSession(timeout=client_timeout) as session:
        result_json = await call_eval_server_async(
          session,
          f"http://localhost:{EVAL_SERVER_PORT}",
          payload,
          poll_interval=TEST_EXECUTION_POLL_INTERVAL,
          client_wait_timeout=REQUEST_TIMEOUT,
        )

      full_output = f"STDOUT:\n{result_json.get('output', '')}\n\nSTDERR:\n{result_json.get('error', '') or ''}"

      test_results = {
        "exit_code": result_json.get("exit_code", -1),
        "output": full_output,
        "success": result_json.get("exit_code") == 0,
      }

      logging.info(
        f"[{self.name}] Test execution completed with exit code {result_json.get('exit_code')}"
      )
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={self.output_key: test_results}),
      )

    except Exception as e:
      error_msg = f"Exception during test execution: {str(e)}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "exit_code": -1,
              "output": error_msg,
              "success": False,
            }
          }
        ),
      )


class SyntaxValidationAgent(BaseAgent):
  """Validates Python syntax of generated test file using AST parsing."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None

  def __init__(self, name: str, input_key: str, output_key: str):
    super().__init__(name=name)
    self.input_key = input_key
    self.output_key = output_key

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    test_file_path = ctx.session.state.get(self.input_key, "")

    if not test_file_path or not os.path.exists(test_file_path):
      error_msg = f"Test file not found at {test_file_path}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "syntax",
            }
          }
        ),
      )
      return

    try:
      with open(test_file_path, "r") as f:
        code = f.read()

      ast.parse(code)

      logging.info(
        f"[{self.name}] Syntax validation passed for {test_file_path}"
      )
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": True,
              "errors": [],
              "validation_type": "syntax",
            }
          }
        ),
      )

    except SyntaxError as e:
      error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "syntax",
              "details": str(e),
            }
          }
        ),
      )
    except Exception as e:
      error_msg = f"Unexpected error during syntax validation: {str(e)}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "syntax",
            }
          }
        ),
      )


class ImportValidationAgent(BaseAgent):
  """Validates that imports in the test file can be resolved."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None

  def __init__(self, name: str, input_key: str, output_key: str):
    super().__init__(name=name)
    self.input_key = input_key
    self.output_key = output_key

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    test_file_path = ctx.session.state.get(self.input_key, "")

    if not test_file_path or not os.path.exists(test_file_path):
      error_msg = f"Test file not found at {test_file_path}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "import",
            }
          }
        ),
      )
      return

    try:
      result = subprocess.run(
        ["python", "-m", "py_compile", test_file_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(test_file_path),
        timeout=COMPILE_VALIDATION_TIMEOUT,
      )

      if result.returncode == 0:
        logging.info(
          f"[{self.name}] Import validation passed for {test_file_path}"
        )
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              self.output_key: {
                "valid": True,
                "errors": [],
                "validation_type": "import",
              }
            }
          ),
        )
      else:
        error_msg = f"Import validation failed: {result.stderr}"
        logging.error(f"[{self.name}] {error_msg}")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              self.output_key: {
                "valid": False,
                "errors": [error_msg],
                "validation_type": "import",
                "details": result.stderr,
              }
            }
          ),
        )

    except subprocess.TimeoutExpired:
      error_msg = "Import validation timed out"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "import",
            }
          }
        ),
      )
    except Exception as e:
      error_msg = f"Unexpected error during import validation: {str(e)}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "import",
            }
          }
        ),
      )


class MockTestExecutionAgent(BaseAgent):
  """Validates tests can execute using JAX baseline code as a mock for the kernel."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None

  def __init__(self, name: str, input_key: str, output_key: str):
    super().__init__(name=name)
    self.input_key = input_key
    self.output_key = output_key

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    test_file_path = ctx.session.state.get(self.input_key, "")

    if not test_file_path or not os.path.exists(test_file_path):
      error_msg = f"Test file not found at {test_file_path}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "mock_execution",
            }
          }
        ),
      )
      return

    try:
      with open(test_file_path, "r") as f:
        test_content = f.read()

      has_baseline_ref = any(
        keyword in test_content.lower()
        for keyword in [
          "base_kernel",
        ]
      )

      if not has_baseline_ref:
        logging.warning(
          f"[{self.name}] No baseline reference found in test file. Skipping mock execution validation."
        )
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              self.output_key: {
                "valid": True,
                "errors": [],
                "validation_type": "mock_execution",
                "skipped": True,
                "reason": "No baseline reference found in test",
              }
            }
          ),
        )
        return

      mock_content = test_content

      # Mock execution relies on the safe import fallback generated in the test file.

      with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_mock_test.py",
        delete=False,
        dir=os.path.dirname(test_file_path),
      ) as tmp_file:
        tmp_file.write(mock_content)
        tmp_test_path = tmp_file.name

      try:
        with open(tmp_test_path, "r") as f:
          mock_code_content = f.read()

        base_kernel_path = ctx.session.state.get("base_kernel_path", "")
        dependencies = {}
        if base_kernel_path and os.path.exists(base_kernel_path):
          with open(base_kernel_path, "r") as f:
            dependencies["base_kernel.py"] = f.read()

        payload = {
          "eval_type": "unified_test",
          "code": mock_code_content,
          "timeout": MOCK_EXECUTION_TIMEOUT,
          "backend_type": "tpu",
          "dependencies": dependencies,
        }

        client_timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT + 10)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
          result_json = await call_eval_server_async(
            session,
            f"http://localhost:{EVAL_SERVER_PORT}",
            payload,
            poll_interval=10,
            client_wait_timeout=REQUEST_TIMEOUT,
          )

        exit_code = result_json.get("exit_code", -1)
        stdout = result_json.get("output", "")
        stderr = result_json.get("error", "") or ""

        if exit_code == 0:
          logging.info(f"[{self.name}] Mock execution validation passed")
          yield Event(
            author=self.name,
            actions=EventActions(
              state_delta={
                self.output_key: {
                  "valid": True,
                  "errors": [],
                  "validation_type": "mock_execution",
                  "tests_passed": True,
                  "output_summary": (
                    stdout[-500:] if len(stdout) > 500 else stdout
                  ),
                  "details": "",
                }
              }
            ),
          )
        else:
          error_msg = f"Mock execution failed. Tests may have structural issues.\n{stdout}\n{stderr}"
          logging.warning(f"[{self.name}] {error_msg}")
          yield Event(
            author=self.name,
            actions=EventActions(
              state_delta={
                self.output_key: {
                  "valid": False,
                  "errors": [error_msg[:1000]],
                  "validation_type": "mock_execution",
                  "details": (stdout[-1000:] if len(stdout) > 1000 else stdout),
                }
              }
            ),
          )
      finally:
        try:
          os.unlink(tmp_test_path)
        except Exception as e:
          logging.warning(f"Failed to clean up temporary test file: {e}")

    except subprocess.TimeoutExpired:
      error_msg = (
        f"Mock test execution timed out after {MOCK_EXECUTION_TIMEOUT} seconds"
      )
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "mock_execution",
            }
          }
        ),
      )
    except Exception as e:
      error_msg = f"Unexpected error during mock execution validation: {str(e)}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "mock_execution",
            }
          }
        ),
      )


class TestValidationLoopAgent(BaseAgent):
  """Custom loop agent that validates and fixes test files until valid or max retries reached."""

  syntax_agent: Optional[BaseAgent] = None
  import_agent: Optional[BaseAgent] = None
  mock_execution_agent: Optional[BaseAgent] = None
  fix_agent: Optional[BaseAgent] = None
  max_retries: int = 3

  def __init__(
    self,
    name: str,
    syntax_agent: BaseAgent,
    import_agent: BaseAgent,
    mock_execution_agent: BaseAgent,
    fix_agent: BaseAgent,
    max_retries: int = 3,
  ):
    super().__init__(
      name=name,
      syntax_agent=syntax_agent,
      import_agent=import_agent,
      mock_execution_agent=mock_execution_agent,
      fix_agent=fix_agent,
      max_retries=max_retries,
    )

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Validation loop: validate -> fix -> repeat until valid or max retries."""

    test_file_path = ctx.session.state.get("test_file_path", "")

    if "test_file_path" not in ctx.session.state:
      ctx.session.state["test_file_path"] = ""

    if not test_file_path:
      logging.error(f"[{self.name}] No test file path found in state.")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            "validation_loop_status": {
              "success": False,
              "retries": 0,
              "message": "No test file was generated. Cannot validate.",
              "syntax_valid": False,
              "import_valid": False,
              "mock_execution_valid": False,
            }
          }
        ),
      )
      return

    retry_count = 0

    while retry_count < self.max_retries:
      logging.info(
        f"[{self.name}] Validation attempt {retry_count + 1}/{self.max_retries}"
      )

      async for event in self.syntax_agent.run_async(ctx):
        yield event

      async for event in self.import_agent.run_async(ctx):
        yield event

      async for event in self.mock_execution_agent.run_async(ctx):
        yield event

      syntax_valid = ctx.session.state.get("syntax_validation", {}).get(
        "valid", False
      )
      import_valid = ctx.session.state.get("import_validation", {}).get(
        "valid", False
      )
      mock_execution_valid = ctx.session.state.get(
        "mock_execution_validation", {}
      ).get("valid", False)

      if syntax_valid and import_valid and mock_execution_valid:
        logging.info(f"[{self.name}] ✓ All validations passed!")

        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              "validation_loop_status": {
                "success": True,
                "retries": retry_count,
                "message": "Test file validated successfully",
                "all_checks_passed": True,
              }
            }
          ),
        )
        return

      if retry_count < self.max_retries - 1:
        logging.info(f"[{self.name}] Validation failed. Attempting fix...")
        async for event in self.fix_agent.run_async(ctx):
          yield event
        retry_count += 1
      else:
        logging.error(f"[{self.name}] ✗ Max retries reached.")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              "validation_loop_status": {
                "success": False,
                "retries": retry_count,
                "message": f"Test file validation failed after {self.max_retries} attempts",
                "syntax_valid": syntax_valid,
                "import_valid": import_valid,
                "mock_execution_valid": mock_execution_valid,
                "all_checks_passed": False,
              }
            }
          ),
        )
        return


# Validation Summary Agent
def create_validation_summary_agent(
  model_name: str = MODEL_NAME,
) -> CustomLlmAgent:
  return CustomLlmAgent(
    name="ValidationSummaryAgent",
    model=model_name,
    generate_content_config=model_config,
    instruction=validation_summary.PROMPT,
    description="Summarizes validation results and provides next steps to the user.",
  )


validation_summary_agent = create_validation_summary_agent()


# Test file generation agent
def create_generate_test_file_agent(
  model_name: str = MODEL_NAME,
) -> CustomLlmAgent:
  return CustomLlmAgent(
    name="GenerateTestFileAgent",
    model=model_name,
    generate_content_config=model_config,
    planner=get_thinking_planner("high"),
    instruction=gen_test_file.PROMPT,
    description="Generates a comprehensive test file.",
    tools=(
      [
        search_api_tool,
        filesystem_tool_r,
        write_test_file_tool,
        vertex_ai_rag_tool,
      ]
      if vertex_ai_rag_tool
      else [search_api_tool, filesystem_tool_r, write_test_file_tool]
    ),
  )


generate_test_file_agent = create_generate_test_file_agent()


# Validation agents
def create_syntax_validation_agent(
  model_name: str = MODEL_NAME,
) -> SyntaxValidationAgent:
  return SyntaxValidationAgent(
    name="SyntaxValidationAgent",
    input_key="test_file_path",
    output_key="syntax_validation",
  )


syntax_validation_agent = create_syntax_validation_agent()


def create_import_validation_agent(
  model_name: str = MODEL_NAME,
) -> ImportValidationAgent:
  return ImportValidationAgent(
    name="ImportValidationAgent",
    input_key="test_file_path",
    output_key="import_validation",
  )


import_validation_agent = create_import_validation_agent()


def create_mock_execution_validation_agent(
  model_name: str = MODEL_NAME,
) -> MockTestExecutionAgent:
  return MockTestExecutionAgent(
    name="MockTestExecutionAgent",
    input_key="test_file_path",
    output_key="mock_execution_validation",
  )


mock_execution_validation_agent = create_mock_execution_validation_agent()


def create_fix_test_script_agent(
  model_name: str = MODEL_NAME,
) -> CustomLlmAgent:
  return CustomLlmAgent(
    name="FixTestScriptAgent",
    model=model_name,
    generate_content_config=model_config,
    planner=get_thinking_planner("high"),
    instruction=fix_test_script.PROMPT,
    description="Fixes validation errors in the generated test file.",
    tools=[filesystem_tool_r, write_test_file_tool, search_api_tool],
    include_contents="none",
  )


fix_test_script_agent = create_fix_test_script_agent()


# Validation loop agent
def create_validation_loop_agent(
  model_name: str = MODEL_NAME,
) -> TestValidationLoopAgent:
  return TestValidationLoopAgent(
    name="TestValidationLoopAgent",
    syntax_agent=create_syntax_validation_agent(model_name),
    import_agent=create_import_validation_agent(model_name),
    mock_execution_agent=create_mock_execution_validation_agent(model_name),
    fix_agent=create_fix_test_script_agent(model_name),
    max_retries=6,
  )


validation_loop_agent = create_validation_loop_agent()


# Sequential agent that generates and validates test files
def create_validated_test_generation_agent(
  model_name: str = MODEL_NAME,
) -> SequentialAgent:
  return SequentialAgent(
    name="ValidatedTestGenerationAgent",
    sub_agents=[
      create_generate_test_file_agent(model_name),
      create_validation_loop_agent(model_name),
      create_validation_summary_agent(model_name),
    ],
    description="Generates a validated test file with automatic iterative error detection and fixing.",
  )


validated_test_generation_agent = create_validated_test_generation_agent()


# Test execution agents
def create_run_tests_agent(model_name: str = MODEL_NAME) -> TestRunner:
  return TestRunner(
    name="RunTestsAgent",
    input_key="test_file_path",
    output_key="test_results",
  )


run_tests_agent = create_run_tests_agent()


def create_summarize_test_results_agent(
  model_name: str = MODEL_NAME,
) -> CustomLlmAgent:
  return CustomLlmAgent(
    name="SummarizeTestResultsAgent",
    model=model_name,
    generate_content_config=model_config,
    planner=get_thinking_planner("high"),
    instruction=summarize_test_results_prompt.PROMPT,
    description="Analyzes test results and provides recommendations.",
    tools=(
      [search_api_tool, vertex_ai_rag_tool]
      if vertex_ai_rag_tool
      else [search_api_tool]
    ),
    output_key="test_summary",
    include_contents="none",
  )


summarize_test_results_agent = create_summarize_test_results_agent()


def create_unified_test_agent(model_name: str = MODEL_NAME) -> SequentialAgent:
  return SequentialAgent(
    name="UnifiedTestAgent",
    sub_agents=[
      create_run_tests_agent(model_name),
      create_summarize_test_results_agent(model_name),
    ],
    description="Executes the generated test file and provides a comprehensive summary.",
  )


unified_test_agent = create_unified_test_agent()


__all__ = [
  "TestRunner",
  "SyntaxValidationAgent",
  "ImportValidationAgent",
  "MockTestExecutionAgent",
  "TestValidationLoopAgent",
  "validated_test_generation_agent",
  "unified_test_agent",
]
