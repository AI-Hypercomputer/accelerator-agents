import logging
from typing import AsyncGenerator, Callable, Optional

import aiohttp
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from hitl_agent.constants import (
  EVAL_SERVER_PORT,
  REQUEST_TIMEOUT,
  TPU_TIMEOUT,
)
from hitl_agent.server_utils.server_manager_mixin import ServerManagerMixin


class KernelProfiler(ServerManagerMixin, BaseAgent):
  """Profiles the kernel to identify performance bottlenecks.

  Automatically manages eval server lifecycle:
  - Starts TPU and eval servers if not running
  - Runs profiling
  - Tears down servers after completion if auto_manage_servers is True
  """

  input_key: Optional[str] = None
  output_key: Optional[str] = None
  before_agent_callback: Optional[Callable] = None
  raise_exception_upon_success: bool = True
  auto_manage_servers: bool = (
    False  # Default to False to preserve existing behavior
  )

  def __init__(
    self,
    name: str,
    input_key: str,
    output_key: str,
    before_agent_callback: Optional[Callable] = None,
    raise_exception_upon_success: bool = True,
    auto_manage_servers: bool = False,
  ):
    super().__init__(name=name, before_agent_callback=before_agent_callback)
    self.input_key = input_key
    self.output_key = output_key
    self.raise_exception_upon_success = raise_exception_upon_success
    self.auto_manage_servers = auto_manage_servers
    self._servers_started = []  # Track which servers we started

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    profile_code = ctx.session.state.get(self.input_key, "")
    if not profile_code:
      logging.warning(f"[{self.name}] No profile_code found in context")
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={self.output_key: None}),
      )
      return

    try:
      # Ensure servers are running before profiling
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
            "eval_type": "profile",
            "code": profile_code,
            "timeout": TPU_TIMEOUT,
          },
        ) as response:
          if response.status == 200:
            result = await response.json()
            logging.info(f"[{self.name}] Profiling result: {result}")

            # Check if profiling was successful based on exit code and output
            exit_code = result.get("exit_code", 0)
            output = result.get("output", "")
            error_msg = result.get("error", "")

            # Profiling succeeds if exit_code is 0 and we have output
            # Stderr may contain warnings (like TensorFlow import warnings) which are not failures
            if exit_code != 0:
              full_error = f"Profiling script failed with exit code {exit_code}"
              if error_msg:
                full_error += f": {error_msg}"
              logging.error(f"[{self.name}] {full_error}")
              yield Event(
                author=self.name,
                actions=EventActions(state_delta={self.output_key: full_error}),
              )
            elif not output or output.strip() == "":
              full_error = "Profiling script produced no output"
              if error_msg:
                full_error += f". Stderr: {error_msg}"
              logging.error(f"[{self.name}] {full_error}")
              yield Event(
                author=self.name,
                actions=EventActions(state_delta={self.output_key: full_error}),
              )
            else:
              # Successful profiling - parse the ratio and xplane path
              try:
                try:
                  import json

                  res = json.loads(output.strip())
                  ratio = float(res.get("ratio", 0))
                  xplane_path = res.get("xplane_path", "")
                except (ValueError, json.JSONDecodeError):
                  # Fallback for old servers returning raw ratio
                  ratio = float(output.strip())
                  xplane_path = ""

                # Log warnings if present, but don't fail
                if error_msg:
                  logging.warning(
                    f"[{self.name}] Profiling succeeded but had warnings in stderr: {error_msg[:200]}"
                  )

                logging.info(
                  f"[{self.name}] Profiling succeeded with ratio: {ratio}, xplane_path: {xplane_path}"
                )
                yield Event(
                  author=self.name,
                  actions=EventActions(
                    escalate=False,
                    state_delta={
                      self.output_key: {
                        "DMAs_and_memory_transfers_ratio": ratio,
                        "compute_ratio": 1 - ratio,
                        "xplane_path": xplane_path,
                      }
                    },
                  ),
                )
              except (ValueError, KeyError) as e:
                error_msg_full = (
                  f"Failed to parse profiling output: '{output}'. Error: {e}"
                )
                logging.error(f"[{self.name}] {error_msg_full}")
                yield Event(
                  author=self.name,
                  actions=EventActions(
                    state_delta={self.output_key: error_msg_full}
                  ),
                )
          else:
            error_detail = await response.text()
            logging.error(
              f"[{self.name}] HTTP error {response.status}: {error_detail}"
            )
            yield Event(
              author=self.name,
              actions=EventActions(
                state_delta={
                  self.output_key: f"HTTP error {response.status}: {error_detail}"
                }
              ),
            )
    except Exception as e:
      logging.error(f"[{self.name}] Exception during code execution: {str(e)}")
      yield Event(
        author=self.name,
        actions=EventActions(
          state_delta={
            self.output_key: f"Exception during code execution: {str(e)}"
          }
        ),
      )
    finally:
      # Cleanup servers if we started them
      await self._cleanup_servers()
