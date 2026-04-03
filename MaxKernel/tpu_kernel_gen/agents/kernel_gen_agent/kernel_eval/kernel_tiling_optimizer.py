import logging
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


class KernelTilingOptimizer(BaseAgent):
  """Checks whether kernel is performing well and escalates to stop the loop if grade is 'pass'."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None
  before_agent_callback: Optional[callable] = None
  raise_exception_upon_success: bool = False

  def __init__(
    self,
    name: str,
    input_key: str,
    output_key: str,
    before_agent_callback: Optional[callable] = None,
    raise_exception_upon_success: bool = False,
  ):
    super().__init__(name=name, before_agent_callback=before_agent_callback)
    self.input_key = input_key
    self.output_key = output_key
    self.raise_exception_upon_success = raise_exception_upon_success

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    performance_test_code = ctx.session.state.get(self.input_key, "")
    if not performance_test_code:
      logging.warning(f"[{self.name}] No performance_test_code found in context")
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={self.output_key: None}),
      )
      return

    try:
      # Call the TPU server to execute the code
      logging.info(f"[{self.name}] Running code")
      async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
        async with session.post(
          f"http://localhost:{EVAL_SERVER_PORT}/evaluate",
          json={
            "eval_type": "performance_test",
            "code": performance_test_code,
            "timeout": TPU_TIMEOUT * 2,  # Allow more time for performance testing
          },
        ) as response:
          if response.status == 200:
            result = await response.json()
            logging.info(f"[{self.name}] Tiling optimization test result: {result}")
            output_lines = result.get("output", "").strip().split("\n")
            avg_base_time = None
            avg_best_optimized_time = None
            best_block_config = None

            for line in output_lines:
              if "Base JAX implementation time:" in line:
                avg_base_time = float(line.split(":")[1].strip().split()[0])
              elif "Best block configuration:" in line:
                best_block_config = line.split(":")[1].strip()

              elif "Best Pallas kernel time:" in line:
                avg_best_optimized_time = float(line.split(":")[1].strip().split()[0])

            if avg_base_time is not None and avg_best_optimized_time is not None:
              speedup = avg_base_time / avg_best_optimized_time
              yield Event(
                author=self.name,
                actions=EventActions(
                  escalate=True if self.raise_exception_upon_success else False,
                  state_delta={
                    self.output_key: {
                      "speedup": speedup,
                      "base_time": avg_base_time,
                      "optimized_time": avg_best_optimized_time,
                      "best_block_config": best_block_config,
                    }
                  },
                ),
              )
            else:
              logging.error(f"[{self.name}] Could not parse timing results")
              yield Event(
                author=self.name,
                actions=EventActions(state_delta={self.output_key: "Could not parse timing results"}),
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


def check_whether_to_test(
  callback_context: CallbackContext, compilation_key: str, correctness_key: str
) -> types.Content:
  if callback_context.state.get(compilation_key, None) != "Success":
    return types.Content(
      role="model",
      parts=[types.Part(text="The kernel code has compilation errors, so performance testing is not applicable.")],
    )
  if callback_context.state.get(correctness_key, None) != "Success":
    return types.Content(
      role="model",
      parts=[types.Part(text="The kernel code has correctness issues, so performance testing is not applicable.")],
    )
