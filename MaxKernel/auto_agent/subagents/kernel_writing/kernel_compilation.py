import logging
from typing import AsyncGenerator, Callable, Optional

import aiohttp
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from auto_agent.constants import (
  EVAL_SERVER_PORT,
  REQUEST_TIMEOUT,
  TPU_TIMEOUT,
)
from auto_agent.server_utils.server_manager_mixin import (
  ServerManagerMixin,
)


class KernelCompilationChecker(ServerManagerMixin, BaseAgent):
  """Checks whether kernel compiles and escalates to stop the loop if grade is 'pass'.

  Automatically manages eval server lifecycle:
  - Starts TPU and eval servers if not running
  - Runs compilation check
  - Tears down servers after completion if auto_manage_servers is True
  """

  input_key: Optional[str] = None
  output_key: Optional[str] = None
  before_agent_callback: Optional[Callable] = None
  auto_manage_servers: bool = (
    False  # Default to False to preserve existing behavior
  )

  def __init__(
    self,
    name: str,
    input_key: str,
    output_key: str,
    before_agent_callback: Optional[Callable] = None,
    auto_manage_servers: bool = False,
  ):
    super().__init__(name=name, before_agent_callback=before_agent_callback)
    self.input_key = input_key
    self.output_key = output_key
    self.auto_manage_servers = auto_manage_servers
    self._servers_started = []  # Track which servers we started

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    code = ctx.session.state.get(self.input_key, "")
    if not code:
      logging.warning(f"[{self.name}] No {self.input_key} found in context")
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={self.output_key: None}),
      )
      return

    try:
      # Ensure servers are running before compilation check
      servers_ok, error_msg = await self._ensure_servers_running()
      if not servers_ok:
        logging.error(f"[{self.name}] Server startup failed: {error_msg}")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={self.output_key: f"Server startup failed: {error_msg}"}
          ),
        )
        return
      # Call the TPU server to execute the code
      logging.info(f"[{self.name}] Running code")
      async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
      ) as session:
        async with session.post(
          f"http://localhost:{EVAL_SERVER_PORT}/evaluate",
          json={
            "eval_type": "compilation_test",
            "code": code,
            "timeout": TPU_TIMEOUT,
            "backend_type": "tpu",
          },
        ) as response:
          if response.status == 200:
            result = await response.json()
            logging.info(f"[{self.name}] Compilation test result: {result}")
            if result["exit_code"] == 0:
              logging.info(f"[{self.name}] Code execution successful.")
              yield Event(
                author=self.name,
                actions=EventActions(state_delta={self.output_key: "Success"}),
              )
            elif (
              result["error"] is None
              and result["output"] == ""
              and result["exit_code"] == 1
            ):
              logging.info(
                f"[{self.name}] Code execution had exit code 1, but no error, indicating success."
              )
              yield Event(
                author=self.name,
                actions=EventActions(state_delta={self.output_key: "Success"}),
              )
            else:
              logging.info(
                f"[{self.name}] Code execution failed. Loop will continue."
              )
              # Use 'or' to handle None case - result.get() returns None if key exists with None value
              error_msg = (
                result.get("error")
                or result.get("output")
                or "Unknown error: No error message or output available"
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
            logging.error(
              f"[{self.name}] HTTP error {response.status}: {error_detail}"
            )
            ctx.session.state[self.output_key] = (
              f"HTTP error {response.status}: {error_detail}"
            )
            yield Event(author=self.name)
    except Exception as e:
      logging.error(f"[{self.name}] Exception during code execution: {str(e)}")
      ctx.session.state[self.output_key] = (
        f"Exception during code execution: {str(e)}"
      )
      yield Event(author=self.name)
    finally:
      # Cleanup servers if we started them
      await self._cleanup_servers()
