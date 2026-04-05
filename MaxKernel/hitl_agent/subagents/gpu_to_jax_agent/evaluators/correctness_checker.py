"""Correctness checker for JAX conversions - validates numerical accuracy."""

import logging
from typing import AsyncGenerator, Optional

import aiohttp
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from hitl_agent.subagents.gpu_to_jax_agent.constants import (
    EVAL_SERVER_PORT,
    CONVERSION_TIMEOUT,
    NUMERICAL_TOLERANCE,
    PREFERRED_BACKEND,
)
from hitl_agent.server_utils.server_manager_mixin import ServerManagerMixin


class JaxCorrectnessChecker(ServerManagerMixin, BaseAgent):
  """Checks whether JAX conversion produces numerically correct results."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None
  auto_manage_servers: bool = False

  def __init__(self,
               name: str,
               input_key: str,
               output_key: str,
               auto_manage_servers: bool = False):
    super().__init__(name=name)
    self.input_key = input_key
    self.output_key = output_key
    self.auto_manage_servers = auto_manage_servers
    self._servers_started = []  # Track which servers this instance started

  async def _run_async_impl(
      self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    # Ensure servers are running if auto_manage_servers is True
    await self._ensure_servers_running()

    test_code = ctx.session.state.get(self.input_key, "")
    if not test_code:
      logging.warning(f"[{self.name}] No {self.input_key} found in context")
      yield Event(
          author=self.name,
          actions=EventActions(state_delta={
              self.output_key: "No test code provided for correctness check"
          }),
      )
      return

    try:
      # Call the eval server to run the correctness test
      logging.info(f"[{self.name}] Running correctness test")
      async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
          total=CONVERSION_TIMEOUT)) as session:
        async with session.post(
            f"http://localhost:{EVAL_SERVER_PORT}/evaluate",
            json={
                "eval_type": "correctness_test",
                "code": test_code,
                "timeout": CONVERSION_TIMEOUT,
                "backend_type": PREFERRED_BACKEND,
            },
        ) as response:
          if response.status == 200:
            result = await response.json()
            logging.info(f"[{self.name}] Correctness test result: {result}")

            output = result.get("output", "")
            error = result.get("error", "")

            # Check for success indicators
            if "Identical" in output or "PASSED" in output.upper():
              logging.info(
                  f"[{self.name}] Correctness test passed - outputs are identical"
              )
              yield Event(
                  author=self.name,
                  actions=EventActions(
                      state_delta={
                          self.output_key:
                              f"Success: Numerical outputs match within tolerance ({NUMERICAL_TOLERANCE})"
                      }),
              )
            elif "Different" in output or "FAILED" in output.upper():
              logging.warning(
                  f"[{self.name}] Correctness test failed - outputs differ")
              error_msg = (
                  f"Correctness test failed: Outputs are not identical.\n"
                  f"Expected tolerance: {NUMERICAL_TOLERANCE}\n"
                  f"Details: {output}")
              yield Event(
                  author=self.name,
                  actions=EventActions(
                      state_delta={self.output_key: error_msg}),
              )
            elif result["exit_code"] == 0 and not error:
              # Success with no specific marker
              logging.info(
                  f"[{self.name}] Correctness test passed (exit code 0)")
              success_msg = f"Success: Test executed without errors"
              if output:
                success_msg += f"\nOutput: {output}"
              yield Event(
                  author=self.name,
                  actions=EventActions(
                      state_delta={self.output_key: success_msg}),
              )
            else:
              # Error case
              logging.error(
                  f"[{self.name}] Correctness test encountered an error")
              error_msg = f"Correctness test failed with errors:\n"
              if error:
                error_msg += f"Error: {error}\n"
              if output:
                error_msg += f"Output: {output}\n"
              error_msg += f"Exit code: {result['exit_code']}"

              yield Event(
                  author=self.name,
                  actions=EventActions(
                      state_delta={self.output_key: error_msg}),
              )
          else:
            error_detail = await response.text()
            logging.error(
                f"[{self.name}] HTTP error {response.status}: {error_detail}")
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={
                        self.output_key:
                            f"HTTP error {response.status}: {error_detail}"
                    }),
            )
    except aiohttp.ClientConnectorError:
      error_msg = (
          f"Cannot connect to evaluation server at localhost:{EVAL_SERVER_PORT}. "
          "Make sure the eval server is running.")
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
          author=self.name,
          actions=EventActions(state_delta={self.output_key: error_msg}),
      )
    except Exception as e:
      error_msg = f"Exception during correctness check: {str(e)}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
          author=self.name,
          actions=EventActions(state_delta={self.output_key: error_msg}),
      )
    finally:
      await self._cleanup_servers()
