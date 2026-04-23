"""Testing subagent - test generation, validation, and execution.

This module contains all agents related to generating, validating, and executing tests.
"""

import ast
import logging
import os
import re
import subprocess
import tempfile
from typing import AsyncGenerator, Callable, Optional

import aiohttp
from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from auto_agent.config import model_config, thinking_planner
from auto_agent.constants import EVAL_SERVER_PORT, MODEL_NAME
from auto_agent.custom_types import CustomLlmAgent
from auto_agent.subagents.testing.prompts import (
  fix_test_script,
  gen_test_file,
  read_file_prompt,
  summarize_test_results_prompt,
  validation_summary,
)
from auto_agent.tools.search_api_tool import search_api_tool
from auto_agent.tools.tools import filesystem_tool_rw, vertex_ai_rag_tool

# Timeout specifications (in seconds)
COMPILE_VALIDATION_TIMEOUT = 60 * 1
MOCK_EXECUTION_TIMEOUT = 60 * 3
TEST_EXECUTION_TIMEOUT = 60 * 5


class TestRunner(BaseAgent):
  """Executes pytest on a generated test file and captures results with full tracebacks."""

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
        f"[{self.name}] Dispatching pytest on {test_file_path} to eval server"
      )

      with open(test_file_path, "r") as f:
        code_content = f.read()

      base_kernel_path = ctx.session.state.get("base_kernel_path", "")
      optimized_kernel_path = ctx.session.state.get("optimized_kernel_path", "")
      dependencies = {}

      if base_kernel_path and os.path.exists(base_kernel_path):
        with open(base_kernel_path, "r") as f:
          dependencies[os.path.basename(base_kernel_path)] = f.read()

      if optimized_kernel_path and os.path.exists(optimized_kernel_path):
        with open(optimized_kernel_path, "r") as f:
          dependencies[os.path.basename(optimized_kernel_path)] = f.read()

      payload = {
        "eval_type": "unified_test",
        "code": code_content,
        "timeout": TEST_EXECUTION_TIMEOUT,
        "backend_type": "tpu",
        "dependencies": dependencies,
      }

      async with aiohttp.ClientSession() as session:
        async with session.post(
          f"http://localhost:{EVAL_SERVER_PORT}/evaluate",
          json=payload,
        ) as response:
          if response.status != 200:
            error_text = await response.text()
            raise Exception(
              f"Eval server returned status {response.status}: {error_text}"
            )
          result_json = await response.json()

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

    except subprocess.TimeoutExpired:
      error_msg = "Test execution timed out after 5 minutes"
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


class TestStructureValidationAgent(BaseAgent):
  """Validates pytest test structure and conventions by attempting test setup."""

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
              "validation_type": "structure",
            }
          }
        ),
      )
      return

    try:
      collect_result = subprocess.run(
        ["pytest", test_file_path, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(test_file_path),
        timeout=30,
      )

      if (
        "no tests ran" in collect_result.stdout.lower()
        or collect_result.returncode != 0
      ):
        error_msg = f"No valid pytest tests found or collection failed: {collect_result.stdout}"
        logging.warning(f"[{self.name}] {error_msg}")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              self.output_key: {
                "valid": False,
                "errors": [error_msg],
                "validation_type": "structure",
                "details": collect_result.stdout + "\n" + collect_result.stderr,
              }
            }
          ),
        )
        return

      setup_result = subprocess.run(
        ["pytest", test_file_path, "--setup-only", "-q"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(test_file_path),
        timeout=30,
      )

      if setup_result.returncode != 0:
        error_msg = f"Test setup failed (imports or fixtures broken): {setup_result.stdout}"
        logging.error(f"[{self.name}] {error_msg}")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              self.output_key: {
                "valid": False,
                "errors": [error_msg],
                "validation_type": "structure",
                "details": setup_result.stdout + "\n" + setup_result.stderr,
              }
            }
          ),
        )
      else:
        logging.info(
          f"[{self.name}] Test structure validation passed for {test_file_path}"
        )
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              self.output_key: {
                "valid": True,
                "errors": [],
                "validation_type": "structure",
                "tests_collected": collect_result.stdout,
              }
            }
          ),
        )

    except subprocess.TimeoutExpired:
      error_msg = "Test structure validation timed out"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "structure",
            }
          }
        ),
      )
    except Exception as e:
      error_msg = f"Unexpected error during structure validation: {str(e)}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: {
              "valid": False,
              "errors": [error_msg],
              "validation_type": "structure",
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
          "baseline",
          "jax_baseline",
          "reference_impl",
          "converted_jax",
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

      # Programmatically comment out optimized_kernel import if uncommented
      mock_content = re.sub(
        r"^([ \t]*from\s+optimized_kernel\s+import\s+.+)",
        r"# \1",
        mock_content,
        flags=re.MULTILINE,
      )
      mock_content = re.sub(
        r"^([ \t]*import\s+optimized_kernel)",
        r"# \1",
        mock_content,
        flags=re.MULTILINE,
      )

      # Ensure optimized_kernel is aliased to base_kernel for the mock run
      if "optimized_kernel = base_kernel" not in mock_content:
        mock_content = (
          "import base_kernel\noptimized_kernel = base_kernel\n" + mock_content
        )

      mock_prefix = """
# Mock setup: Temporarily disable kernel imports to test with baseline
import sys
from unittest.mock import MagicMock

# If kernel import fails, tests should fall back to baseline
"""

      mock_content = mock_prefix + mock_content

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
            dependencies[os.path.basename(base_kernel_path)] = f.read()

        payload = {
          "eval_type": "unified_test",
          "code": mock_code_content,
          "timeout": MOCK_EXECUTION_TIMEOUT,
          "backend_type": "cpu",
          "dependencies": dependencies,
        }

        async with aiohttp.ClientSession() as session:
          async with session.post(
            f"http://localhost:{EVAL_SERVER_PORT}/evaluate",
            json=payload,
          ) as response:
            if response.status != 200:
              error_text = await response.text()
              raise Exception(
                f"Eval server returned status {response.status}: {error_text}"
              )
            result_json = await response.json()

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
  structure_agent: Optional[BaseAgent] = None
  mock_execution_agent: Optional[BaseAgent] = None
  fix_agent: Optional[BaseAgent] = None
  max_retries: int = 3

  def __init__(
    self,
    name: str,
    syntax_agent: BaseAgent,
    import_agent: BaseAgent,
    structure_agent: BaseAgent,
    mock_execution_agent: BaseAgent,
    fix_agent: BaseAgent,
    max_retries: int = 3,
  ):
    super().__init__(
      name=name,
      syntax_agent=syntax_agent,
      import_agent=import_agent,
      structure_agent=structure_agent,
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
              "structure_valid": False,
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

      async for event in self.structure_agent.run_async(ctx):
        yield event

      async for event in self.mock_execution_agent.run_async(ctx):
        yield event

      syntax_valid = ctx.session.state.get("syntax_validation", {}).get(
        "valid", False
      )
      import_valid = ctx.session.state.get("import_validation", {}).get(
        "valid", False
      )
      structure_valid = ctx.session.state.get("structure_validation", {}).get(
        "valid", False
      )
      mock_execution_valid = ctx.session.state.get(
        "mock_execution_validation", {}
      ).get("valid", False)

      if (
        syntax_valid
        and import_valid
        and structure_valid
        and mock_execution_valid
      ):
        logging.info(f"[{self.name}] ✓ All validations passed!")

        if test_file_path and os.path.exists(test_file_path):
          try:
            with open(test_file_path, "r") as f:
              content = f.read()

            content = re.sub(
              r"# (from .+ import .+ as optimized_kernel)", r"\1", content
            )

            content = re.sub(
              r"\noptimized_kernel = base_kernel.*(?=\n)", "", content
            )

            with open(test_file_path, "w") as f:
              f.write(content)

            logging.info(
              f"[{self.name}] Successfully uncommented optimized kernel import"
            )
          except Exception as e:
            logging.warning(
              f"[{self.name}] Failed to uncomment kernel import: {e}"
            )

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
                "structure_valid": structure_valid,
                "mock_execution_valid": mock_execution_valid,
                "all_checks_passed": False,
              }
            }
          ),
        )
        return


# Validation Summary Agent
validation_summary_agent = CustomLlmAgent(
  name="ValidationSummaryAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=validation_summary.PROMPT,
  description="Summarizes validation results and provides next steps to the user.",
)

# Test file generation agent
generate_test_file_agent = CustomLlmAgent(
  name="GenerateTestFileAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=gen_test_file.PROMPT,
  description="Generates a comprehensive pytest test file.",
  tools=(
    [search_api_tool, filesystem_tool_rw, vertex_ai_rag_tool]
    if vertex_ai_rag_tool
    else [search_api_tool, filesystem_tool_rw]
  ),
)

# Validation agents
syntax_validation_agent = SyntaxValidationAgent(
  name="SyntaxValidationAgent",
  input_key="test_file_path",
  output_key="syntax_validation",
)

import_validation_agent = ImportValidationAgent(
  name="ImportValidationAgent",
  input_key="test_file_path",
  output_key="import_validation",
)

structure_validation_agent = TestStructureValidationAgent(
  name="TestStructureValidationAgent",
  input_key="test_file_path",
  output_key="structure_validation",
)

mock_execution_validation_agent = MockTestExecutionAgent(
  name="MockTestExecutionAgent",
  input_key="test_file_path",
  output_key="mock_execution_validation",
)

fix_test_script_agent = CustomLlmAgent(
  name="FixTestScriptAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=fix_test_script.PROMPT,
  description="Fixes validation errors in the generated test file.",
  tools=[filesystem_tool_rw],
  include_contents="none",
)

# Validation loop agent
validation_loop_agent = TestValidationLoopAgent(
  name="TestValidationLoopAgent",
  syntax_agent=syntax_validation_agent,
  import_agent=import_validation_agent,
  structure_agent=structure_validation_agent,
  mock_execution_agent=mock_execution_validation_agent,
  fix_agent=fix_test_script_agent,
  max_retries=6,
)

# Sequential agent that generates and validates test files
validated_test_generation_agent = SequentialAgent(
  name="ValidatedTestGenerationAgent",
  sub_agents=[
    generate_test_file_agent,
    validation_loop_agent,
    validation_summary_agent,
  ],
  description="Generates a validated pytest test file with automatic iterative error detection and fixing.",
)

# Test execution agents
read_file_for_testing_agent = CustomLlmAgent(
  name="ReadFileForTestingAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=read_file_prompt.PROMPT,
  description="Reads the test file mentioned by the user.",
  tools=[filesystem_tool_rw],
)

run_tests_agent = TestRunner(
  name="RunTestsAgent",
  input_key="test_file_path",
  output_key="test_results",
)

summarize_test_results_agent = CustomLlmAgent(
  name="SummarizeTestResultsAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  instruction=summarize_test_results_prompt.PROMPT,
  description="Analyzes pytest test results and provides recommendations.",
  tools=(
    [search_api_tool, vertex_ai_rag_tool]
    if vertex_ai_rag_tool
    else [search_api_tool]
  ),
  output_key="test_summary",
  include_contents="none",
)

unified_test_agent = SequentialAgent(
  name="UnifiedTestAgent",
  sub_agents=[
    read_file_for_testing_agent,
    run_tests_agent,
    summarize_test_results_agent,
  ],
  description="Executes the generated pytest test file and provides a comprehensive summary.",
)

__all__ = [
  "TestRunner",
  "SyntaxValidationAgent",
  "ImportValidationAgent",
  "TestStructureValidationAgent",
  "MockTestExecutionAgent",
  "TestValidationLoopAgent",
  "validated_test_generation_agent",
  "unified_test_agent",
]
