"""Tool for dynamically updating the active workspace directory."""

import logging
import os
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager

import hitl_agent.config as config
from hitl_agent.tools.filesystem_tools import filesystem_tool_r, filesystem_tool_rw


async def set_working_directory_fn(
    path: str, tool_context: ToolContext, persist: bool = False
) -> str:
  """Set the active workspace/working directory path for the session.

  All filesystem tools (list_directory, read_file, write_file) will use
  the new path.

  Args:
      path: The absolute path of the directory.
      persist: Whether to permanently save this change to the configuration file (default is False).

  Returns:
      A string containing the operation result description:
      - If successful: "Successfully switched working directory to: <abs_path> (persisted: <persist>)"
      - If failed: "Error: Path '<path>' does not exist." or "Error: Path '<path>' is not a directory."
  """
  if not os.path.exists(path):
    return f"Error: Path '{path}' does not exist."
  if not os.path.isdir(path):
    return f"Error: Path '{path}' is not a directory."

  abs_path = os.path.abspath(path)

  # Update in-memory config variables
  config.WORKDIR = abs_path
  os.environ["WORKDIR"] = abs_path

  # Update .env file if persist is requested
  if persist:
    try:
      env_path = ".env"
      if os.path.exists(env_path):
        with open(env_path, "r") as f:
          lines = f.readlines()
        with open(env_path, "w") as f:
          updated = False
          for line in lines:
            if line.strip().startswith("WORKDIR="):
              f.write(f'WORKDIR="{abs_path}"\n')
              updated = True
            else:
              f.write(line)
          if not updated:
            f.write(f'WORKDIR="{abs_path}"\n')
    except Exception as e:
      logging.error(f"Failed to update .env: {e}")

  # Update active filesystem tools
  # Update read-only filesystem tool
  params_r = filesystem_tool_r._connection_params
  server_params_r = getattr(params_r, "server_params", None) or params_r
  if hasattr(server_params_r, "args") and len(server_params_r.args) >= 2:
    server_params_r.args[-1] = abs_path
    await filesystem_tool_r.close()
    filesystem_tool_r._mcp_session_manager = MCPSessionManager(
        connection_params=filesystem_tool_r._connection_params,
        errlog=filesystem_tool_r._errlog,
        sampling_callback=filesystem_tool_r._sampling_callback,
        sampling_capabilities=filesystem_tool_r._sampling_capabilities,
    )

  # Update read-write filesystem tool
  params_rw = filesystem_tool_rw._connection_params
  server_params_rw = getattr(params_rw, "server_params", None) or params_rw
  if hasattr(server_params_rw, "args") and len(server_params_rw.args) >= 2:
    server_params_rw.args[-1] = abs_path
    await filesystem_tool_rw.close()
    filesystem_tool_rw._mcp_session_manager = MCPSessionManager(
        connection_params=filesystem_tool_rw._connection_params,
        errlog=filesystem_tool_rw._errlog,
        sampling_callback=filesystem_tool_rw._sampling_callback,
        sampling_capabilities=filesystem_tool_rw._sampling_capabilities,
    )

  return f"Successfully switched working directory to: {abs_path} (persisted: {persist})"


set_working_directory = FunctionTool(set_working_directory_fn)
