"""File-related tools for subagents."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

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
    logging.info(f"[restricted_write_file] Called for {state_key}. Target path from state: {target_path}")
    if not target_path:
      err = f"Error: Path variable '{state_key}' not found in session state."
      logging.error(f"[restricted_write_file] {err}")
      return err

    base = Path(WORKDIR).resolve()
    target = Path(target_path).resolve()

    try:
      if not target.is_relative_to(base):
        err = f"Error: Access denied. Path {target} is outside base {base}"
        logging.error(f"[restricted_write_file] {err}")
        return err
    except ValueError as e:
      err = f"Error: Invalid path or access denied: {e}"
      logging.error(f"[restricted_write_file] {err}")
      return err

    try:
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_text(content)
      res = f"Successfully wrote to {target}"
      logging.info(f"[restricted_write_file] {res}")
      return res
    except Exception as e:
      err = f"Error writing file {target}: {e}"
      logging.error(f"[restricted_write_file] {err}")
      return err

  _write_file.__name__ = "restricted_write_file"
  _write_file.__doc__ = description
  return FunctionTool(_write_file)


def write_autotune_specs_tool_fn(
  kernel_name: str,
  code_template: str,
  search_space: Dict[str, List[Any]],
  tool_context: ToolContext,
) -> str:
  """Writes the structured autotuning specifications to autotune_specs_path in session state.

  Args:
      kernel_name: The name of the Pallas kernel.
      code_template: The kernel source code template with placeholders like {BLOCK_M}.
      search_space: Dictionary mapping placeholder names to lists of suggested tuning values.
  """
  target_path = tool_context.state.get("autotune_specs_path")
  if not target_path:
    return (
      "Error: Path variable 'autotune_specs_path' not found in session state."
    )

  base = Path(WORKDIR).resolve()
  target = Path(target_path).resolve()

  try:
    if not target.is_relative_to(base):
      return f"Error: Access denied. Path is outside {WORKDIR}"
  except ValueError:
    return "Error: Invalid path or access denied."

  content_dict = {
    "kernel_name": kernel_name,
    "code_template": code_template,
    "search_space": search_space,
  }

  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(content_dict, indent=2))
  return f"Successfully wrote structured autotuning specs to {target}"


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
write_autotune_specs_tool_fn.__name__ = "restricted_write_file"
write_autotune_specs_tool = FunctionTool(write_autotune_specs_tool_fn)

__all__ = [
  "filesystem_tool_r",
  "filesystem_tool_rw",
  "write_test_file_tool",
  "write_optimized_kernel_tool",
  "write_optimization_plan_tool",
  "write_profiling_script_tool",
  "write_autotune_specs_tool",
]
