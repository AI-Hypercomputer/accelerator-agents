import logging
import os
from typing import AsyncGenerator, Optional

import aiohttp
from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tpu_kernel_gen.agents.kernel_gen_agent.constants import (
  EVAL_SERVER_PORT,
  REQUEST_TIMEOUT,
  TPU_TIMEOUT,
)


class KernelCorrectnessChecker(BaseAgent):
  """Checks whether kernel is computationally correct and escalates stop the loop if grade is 'pass'."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None
  before_agent_callback: Optional[callable] = None
  raise_exception_upon_success: bool = True

  def __init__(
    self,
    name: str,
    input_key: str,
    output_key: str,
    before_agent_callback: Optional[callable] = None,
    raise_exception_upon_success: bool = True,
  ):
    super().__init__(name=name, before_agent_callback=before_agent_callback)
    self.input_key = input_key
    self.output_key = output_key
    self.raise_exception_upon_success = raise_exception_upon_success

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    correctness_test_code = ctx.session.state.get(self.input_key, "")
    if not correctness_test_code:
      logging.warning(f"[{self.name}] No correctness_test_code found in context")
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={self.output_key: None}),
      )
      return

    # Retrieve optional dependency files from state
    files = {}
    kernel_code = ctx.session.state.get("kernel_code", None)
    kernel_file_path = ctx.session.state.get("kernel_file_path", None)

    if kernel_code and kernel_file_path:
      # Use the basename of the kernel file as the filename on the server
      kernel_filename = os.path.basename(kernel_file_path)
      files[kernel_filename] = kernel_code
      logging.info(f"[{self.name}] Including kernel file: {kernel_filename}")

    try:
      # Call the TPU server to execute the code
      logging.info(f"[{self.name}] Running code")
      async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
        async with session.post(
          f"http://localhost:{EVAL_SERVER_PORT}/evaluate",
          json={
            "eval_type": "correctness_test",
            "code": correctness_test_code,
            "timeout": TPU_TIMEOUT,
            "files": files if files else None,
          },
        ) as response:
          if response.status == 200:
            result = await response.json()
            logging.info(f"[{self.name}] Correctness test result: {result}")
            if "Identical" in result["output"]:
              logging.info(f"[{self.name}] Correctness test passed. Escalating to stop loop.")
              yield Event(
                author=self.name,
                actions=EventActions(
                  escalate=True if self.raise_exception_upon_success else False,
                  state_delta={self.output_key: "Success"},
                ),
              )
            elif "Different" in result["output"]:
              logging.info(f"[{self.name}] Correctness test failed. Loop will continue.")
              error_msg = (
                "There is a correctness issue with the code, the output is not identical to the expected result."
              )
              yield Event(
                author=self.name,
                actions=EventActions(state_delta={self.output_key: error_msg}),
              )
            else:
              logging.info(f"[{self.name}] Correctness test failed. Loop will continue.")
              # Use 'or' to handle None case - result.get() returns None if key exists with None value
              error_msg = (
                result.get("error") or result.get("output") or "Unknown error: No error message or output available"
              )
              # Add diagnostic logging when error field is None or empty
              if result.get("error") is None:
                logging.warning(
                  f"[{self.name}] Error field is None. "
                  f"exit_code: {result.get('exit_code')}, "
                  f"output length: {len(result.get('output', ''))}, "
                  f"Using output as fallback: {result.get('output', '')[:200]}"
                )
              yield Event(
                author=self.name,
                actions=EventActions(state_delta={self.output_key: error_msg}),
              )
          else:
            error_detail = await response.text()
            logging.error(f"[{self.name}] HTTP error {response.status}: {error_detail}")
            yield Event(
              author=self.name,
              actions=EventActions(state_delta={self.output_key: f"HTTP error {response.status}: {error_detail}"}),
            )
    except Exception as e:
      logging.error(f"[{self.name}] Exception during code execution: {str(e)}")
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={self.output_key: f"Exception during code execution: {str(e)}"}),
      )


def check_whether_to_test(callback_context: CallbackContext, compilation_key: str) -> types.Content:
  if callback_context.state.get(compilation_key, None) != "Success":
    return types.Content(
      role="model",
      parts=[types.Part(text="The kernel code has compilation errors, so correctness testing is not applicable.")],
    )
