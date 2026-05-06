"""File-related tools for subagents."""

import os
from pathlib import Path

from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters

from auto_agent.config import WORKDIR

# Read-only filesystem tool for orchestration agent (no write access)
filesystem_tool_r = MCPToolset(
  connection_params=StdioConnectionParams(
    server_params=StdioServerParameters(
      command="npx",
      args=[
        "-y",  # Argument for npx to auto-confirm install
        "@modelcontextprotocol/server-filesystem@0.5.1",
        os.path.abspath(WORKDIR),
      ],
    ),
  ),
  # Optional: Filter which tools from the MCP server are exposed
  tool_filter=["list_directory", "read_file"],
)

# Read-write filesystem tool for sub-agents
filesystem_tool_rw = MCPToolset(
  connection_params=StdioConnectionParams(
    server_params=StdioServerParameters(
      command="npx",
      args=[
        "-y",  # Argument for npx to auto-confirm install
        "@modelcontextprotocol/server-filesystem@0.5.1",
        os.path.abspath(WORKDIR),
      ],
    ),
  ),
  # Optional: Filter which tools from the MCP server are exposed
  tool_filter=["list_directory", "read_file", "write_file"],
)


def restricted_write_file(state_key: str, description: str) -> FunctionTool:
  """Creates a tool that writes content to a path stored in session state.

  Args:
      state_key: The key in tool_context.state that holds the target file path.
      description: The description of the tool for the agent.
  """

  def _write_file(content: str, tool_context: ToolContext) -> str:
    target_path = tool_context.state.get(state_key)
    if not target_path:
      return f"Error: Path variable '{state_key}' not found in session state."

    base = Path(WORKDIR).resolve()
    target = Path(target_path).resolve()

    try:
      if not target.is_relative_to(base):
        return f"Error: Access denied. Path is outside {WORKDIR}"
    except ValueError:
      return "Error: Invalid path or access denied."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Successfully wrote to {target}"

  _write_file.__name__ = "restricted_write_file"
  _write_file.__doc__ = description
  return FunctionTool(_write_file)


write_test_file_tool = restricted_write_file(
  "test_file_path", "Writes the generated pytest file."
)
write_optimized_kernel_tool = restricted_write_file(
  "optimized_kernel_path", "Writes the optimized Pallas kernel file."
)
write_optimization_plan_tool = restricted_write_file(
  "kernel_plan_path", "Writes the optimization plan."
)
write_profiling_script_tool = restricted_write_file(
  "profiling_script_path", "Writes the profiling script."
)

__all__ = [
  "filesystem_tool_r",
  "filesystem_tool_rw",
  "write_test_file_tool",
  "write_optimized_kernel_tool",
  "write_optimization_plan_tool",
  "write_profiling_script_tool",
]
