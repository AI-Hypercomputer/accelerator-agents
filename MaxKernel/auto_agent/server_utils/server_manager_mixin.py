"""Mixin class for managing TPU and eval server lifecycle in kernel evaluation agents."""

import asyncio
import logging
import os
import subprocess

import aiohttp


class ServerManagerMixin:
  """Mixin that provides server lifecycle management for evaluation agents.

  This mixin adds the ability to automatically start and stop TPU and eval servers
  before and after agent execution. Agents that inherit from this mixin should:

  1. Set auto_manage_servers=True to enable automatic server management
  2. Initialize self._servers_started = [] in their __init__
  3. Call await self._ensure_servers_running() before operations that need servers
  4. Call await self._cleanup_servers() in a finally block after operations complete

  The mixin tracks which servers it started and only tears down those servers,
  so it won't interfere with already-running servers or servers started by other agents.

  Attributes:
      auto_manage_servers: Should be set to True in child classes to enable server management
      _servers_started: List tracking which servers this instance started (for cleanup)
  """

  def _is_server_running(self, server_name: str) -> bool:
    """Check if a server process is running.

    Args:
        server_name: Name of the server process to check (e.g., "tpu_server.py")

    Returns:
        True if the server is running, False otherwise
    """
    try:
      result = subprocess.run(
        ["pgrep", "-f", server_name],
        capture_output=True,
        text=True,
      )
      return result.returncode == 0
    except Exception as e:
      logging.error(f"Error checking if {server_name} is running: {e}")
      return False

  async def _wait_for_server_ready(
    self,
    port: int,
    server_type: str,
    max_retries: int = 30,
    retry_delay: float = 1.0,
  ) -> bool:
    """Wait for a server's HTTP health endpoint to be ready.

    Args:
        port: Port number the server is listening on
        server_type: Type of server for logging (e.g., "cpu", "tpu", "eval")
        max_retries: Maximum number of health check attempts
        retry_delay: Delay in seconds between retry attempts

    Returns:
        True if server health check succeeds, False otherwise
    """
    health_url = f"http://localhost:{port}/health"

    for attempt in range(max_retries):
      try:
        async with aiohttp.ClientSession() as session:
          async with session.get(
            health_url, timeout=aiohttp.ClientTimeout(total=2)
          ) as response:
            if response.status == 200:
              data = await response.json()
              if data.get("status") == "healthy":
                logging.info(
                  f"[ServerManager] {server_type} server health check passed on attempt {attempt + 1}"
                )
                return True
      except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        if attempt < max_retries - 1:
          logging.debug(
            f"[ServerManager] {server_type} server not ready yet (attempt {attempt + 1}/{max_retries}), retrying..."
          )
          await asyncio.sleep(retry_delay)
        else:
          logging.error(
            f"[ServerManager] {server_type} server health check failed after {max_retries} attempts: {e}"
          )
      except Exception as e:
        logging.error(
          f"[ServerManager] Unexpected error during {server_type} health check: {e}"
        )
        return False

    return False

  async def _start_server(self, server_type: str, setup_script: str) -> bool:
    """Start a specific server (tpu, cpu, or eval).

    Args:
        server_type: Type of server to start ("tpu", "cpu", or "eval")
        setup_script: Path to the setup.sh script

    Returns:
        True if server started successfully, False otherwise
    """
    try:
      logging.info(f"Starting {server_type} server...")
      # Determine the directory containing the setup script
      setup_dir = os.path.dirname(setup_script)

      process = await asyncio.create_subprocess_exec(
        "bash",
        setup_script,
        f"--start-{server_type}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=setup_dir,  # Run from the directory containing setup.sh
      )
      await process.wait()

      # Initial brief wait for process to start
      await asyncio.sleep(1)

      # Verify server process started
      server_name = f"{server_type}_server.py"
      if not self._is_server_running(server_name):
        logging.error(f"Failed to start {server_type} server process")
        return False

      # Map server type to port (from constants.py)
      port_map = {
        "tpu": 5463,  # TPU_SERVER_PORT
        "cpu": 5464,  # CPU_SERVER_PORT
        "eval": 1245,  # EVAL_SERVER_PORT
      }

      port = port_map.get(server_type)
      if port is None:
        logging.error(f"Unknown server type: {server_type}")
        return False

      # Wait for HTTP health endpoint to be ready
      logging.info(
        f"Waiting for {server_type} server HTTP endpoint to be ready..."
      )
      if await self._wait_for_server_ready(port, server_type):
        logging.info(f"{server_type} server started successfully and is ready")
        self._servers_started.append(server_type)
        return True
      else:
        logging.error(
          f"Failed to start {server_type} server - health check failed"
        )
        return False
    except Exception as e:
      logging.error(f"Exception starting {server_type} server: {e}")
      return False

  def _stop_server_sync(self, process_name: str):
    """Stop a specific server synchronously using pkill.

    Args:
        process_name: Name of the server process to stop (e.g., "tpu_server.py")
    """
    try:
      logging.info(f"Stopping {process_name}...")
      result = subprocess.run(
        ["pkill", "-f", process_name],
        capture_output=True,
        text=True,
        timeout=5,
      )
      if (
        result.returncode == 0 or result.returncode == 1
      ):  # 1 means no process found
        logging.info(f"{process_name} stopped")
      else:
        logging.warning(
          f"pkill returned {result.returncode} for {process_name}"
        )
    except Exception as e:
      logging.error(f"Exception stopping {process_name}: {e}")

  async def _ensure_servers_running(self) -> tuple[bool, str]:
    """Ensure TPU and eval servers are running.

    Checks if servers are running and starts them if needed. Only starts servers
    if auto_manage_servers is True. Tracks which servers were started by this agent
    in self._servers_started so they can be cleaned up later.

    Returns:
        Tuple of (success: bool, error_message: str)
        - success: True if servers are running or were started successfully
        - error_message: Empty string on success, error description on failure
    """
    logging.info(
      f"[ServerManager] _ensure_servers_running called, auto_manage_servers={getattr(self, 'auto_manage_servers', 'NOT_SET')}"
    )

    if not getattr(self, "auto_manage_servers", False):
      logging.info(
        "[ServerManager] auto_manage_servers is False, skipping server management"
      )
      return True, ""

    # Find setup script
    setup_script = os.path.join(os.path.dirname(__file__), "setup.sh")
    setup_script = os.path.abspath(setup_script)
    logging.info(f"[ServerManager] Looking for setup script at: {setup_script}")

    if not os.path.exists(setup_script):
      error_msg = f"Setup script not found at {setup_script}"
      logging.error(error_msg)
      return False, error_msg

    # Check and start CPU server (required by eval server)
    if self._is_server_running("cpu_server.py"):
      logging.info(
        "CPU server already running, restarting to ensure fresh state"
      )
      self._stop_server_sync("cpu_server.py")
      await asyncio.sleep(1)  # Wait for graceful shutdown

    success = await self._start_server("cpu", setup_script)
    if not success:
      return False, "Failed to start CPU server"

    # Check and start TPU server
    if self._is_server_running("tpu_server.py"):
      logging.info(
        "TPU server already running, restarting to ensure fresh state"
      )
      self._stop_server_sync("tpu_server.py")
      await asyncio.sleep(1)  # Wait for graceful shutdown

    success = await self._start_server("tpu", setup_script)
    if not success:
      return False, "Failed to start TPU server"

    # Check and start eval server
    if self._is_server_running("eval_server.py"):
      logging.info(
        "Eval server already running, restarting to ensure fresh state"
      )
      self._stop_server_sync("eval_server.py")
      await asyncio.sleep(1)  # Wait for graceful shutdown

    success = await self._start_server("eval", setup_script)
    if not success:
      return False, "Failed to start eval server"

    return True, ""

  async def _cleanup_servers(self):
    """Stop servers that were started by this agent.

    Only stops servers that this agent instance started (tracked in self._servers_started).
    This ensures we don't accidentally tear down servers that were already running
    or started by other agents.
    """
    if not self._servers_started:
      return

    for server_type in self._servers_started:
      process_name = f"{server_type}_server.py"
      self._stop_server_sync(process_name)
      await asyncio.sleep(0.5)  # Brief pause between stops
