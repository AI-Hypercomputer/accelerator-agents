"""Compilation checker for JAX code - wraps existing KernelCompilationChecker."""

import logging
from typing import AsyncGenerator, Optional

import aiohttp
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from hitl_agent.subagents.gpu_to_jax_agent.constants import (
    EVAL_SERVER_PORT,
    CONVERSION_TIMEOUT,
    PREFERRED_BACKEND,
)
from hitl_agent.server_utils.server_manager_mixin import ServerManagerMixin


class JaxCompilationChecker(ServerManagerMixin, BaseAgent):
  """Checks whether JAX code compiles successfully."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None
  auto_manage_servers: bool = False

  def __init__(
      self,
      name: str,
      input_key: str,
      output_key: str,
      auto_manage_servers: bool = False,
  ):
    super().__init__(name=name)
    self.input_key = input_key
    self.output_key = output_key
    self.auto_manage_servers = auto_manage_servers
    self._servers_started = []  # Track which servers this instance started

  async def _run_async_impl(
      self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    # Ensure servers are running if auto_manage_servers is True
    await self._ensure_servers_running()

    code = ctx.session.state.get(self.input_key, "")
    if not code:
      logging.warning(f"[{self.name}] No {self.input_key} found in context")
      yield Event(
          author=self.name,
          actions=EventActions(state_delta={self.output_key: None}),
      )
      return

    try:
      # Call the eval server to compile and run the code
      logging.info(f"[{self.name}] Compiling JAX code")
      async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
          total=CONVERSION_TIMEOUT)) as session:
        async with session.post(
            f"http://localhost:{EVAL_SERVER_PORT}/evaluate",
            json={
                "eval_type": "compilation_test",
                "code": code,
                "timeout": CONVERSION_TIMEOUT,
                "backend_type": PREFERRED_BACKEND,
            },
        ) as response:
          if response.status == 200:
            result = await response.json()
            logging.info(f"[{self.name}] Compilation test result: {result}")
            if result["exit_code"] == 0:
              logging.info(f"[{self.name}] JAX code compilation successful.")
              yield Event(
                  author=self.name,
                  actions=EventActions(
                      state_delta={self.output_key: "Success"}),
              )
            elif (result["error"] is None and result["output"] == "" and
                  result["exit_code"] == 1):
              logging.info(
                  f"[{self.name}] Code execution had exit code 1, but no error, indicating success."
              )
              yield Event(
                  author=self.name,
                  actions=EventActions(
                      state_delta={self.output_key: "Success"}),
              )
            else:
              logging.error(f"[{self.name}] JAX code compilation failed.")
              error_msg = result.get("error", "Unknown compilation error")
              output_msg = result.get("output", "")
              full_error = f"Compilation Error:\n{error_msg}"
              if output_msg:
                full_error += f"\n\nOutput:\n{output_msg}"

              yield Event(
                  author=self.name,
                  actions=EventActions(
                      state_delta={self.output_key: full_error}),
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
      error_msg = f"Exception during compilation check: {str(e)}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
          author=self.name,
          actions=EventActions(state_delta={self.output_key: error_msg}),
      )
    finally:
      await self._cleanup_servers()
