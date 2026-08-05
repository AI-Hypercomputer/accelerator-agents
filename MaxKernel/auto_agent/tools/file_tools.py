"""File-related tools for subagents."""

import ast
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters

from auto_agent.config import WORKDIR
from auto_agent.isolate_object import ImportCollector, ObjectExtractor
from auto_agent.tools.test_harness import TEST_TEMPLATE

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


def write_test_file_tool_fn(
  content: str,
  kernel_name: str,
  tool_context: ToolContext,
  atol: float = 1e-2,
  rtol: float = 1e-2,
) -> str:
  """Writes the generated input generation snippet to the test file using the rigorous harness template.

  Args:
      content: The Python code snippet containing the `get_inputs()` function.
      kernel_name: The exact function name of the base kernel entry point (e.g., "computation").
      atol: Absolute tolerance for numerical correctness checks. Default 1e-2. Set lower for F32.
      rtol: Relative tolerance for numerical correctness checks. Default 1e-2. Set lower for F32.
  """
  target_path = tool_context.state.get("test_file_path")
  if not target_path:
    return "Error: Path variable 'test_file_path' not found in session state."

  base = Path(WORKDIR).resolve()
  target = Path(target_path).resolve()

  try:
    if not target.is_relative_to(base):
      return f"Error: Access denied. Path is outside {WORKDIR}"
  except ValueError:
    return "Error: Invalid path or access denied."

  target.parent.mkdir(parents=True, exist_ok=True)

  import re

  content = re.sub(r"^```(python)?\n", "", content.strip())
  content = re.sub(r"\n```$", "", content)

  # Save kernel_name and tolerances to state for downstream agents
  tool_context.state["kernel_name"] = kernel_name
  tool_context.state["atol"] = atol
  tool_context.state["rtol"] = rtol

  full_content = TEST_TEMPLATE.format(
    input_gen_code=content, atol=atol, rtol=rtol, kernel_name=kernel_name
  )

  target.write_text(full_content)
  return f"Successfully wrote to {target}"


write_test_file_tool_fn.__name__ = "restricted_write_file"
write_test_file_tool = FunctionTool(write_test_file_tool_fn)
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
write_base_kernel_tool = restricted_write_file(
  "base_kernel_path", "Writes the base kernel file."
)


def write_dependency_file_fn(
  path: str, content: str, tool_context: ToolContext
) -> str:
  """Writes a helper or dependency Python file to the workspace directory
     and registers it in state['dependencies'].

  Args:
      path: Relative filename for the dependency (e.g. 'utils.py' or
        'common_helpers.py').
      content: The Python code content of the dependency file.
  """
  workdir = tool_context.state.get("workdir", WORKDIR)
  base = Path(workdir).resolve()
  target = (base / path).resolve()

  try:
    if not target.is_relative_to(base):
      return f"Error: Access denied. Path is outside {workdir}"
  except ValueError:
    return "Error: Invalid path or access denied."

  os.makedirs(target.parent, exist_ok=True)
  try:
    with open(target, "w") as f:
      f.write(content)
    if "dependencies" not in tool_context.state:
      tool_context.state["dependencies"] = {}
    rel_name = os.path.relpath(target, base)
    tool_context.state["dependencies"][rel_name] = content
    return (
      f"Successfully wrote dependency file to {rel_name} and"
      f" registered in state['dependencies']."
    )
  except Exception as e:
    return f"Error writing dependency file: {e}"


write_dependency_file_fn.__name__ = "write_dependency_file"
write_dependency_file_tool = FunctionTool(write_dependency_file_fn)


def discover_kernel_dependencies_fn(
  source_file_path: str, tool_context: ToolContext = None
) -> str:
  """Automatically discovers and registers all local Python dependency
     files for a kernel file into state['dependencies'].

  Args:
      source_file_path: The path to the original reference kernel source file
        on disk (e.g. '/repo/tpu_commons/attention.py').
  """
  if not source_file_path or not os.path.exists(source_file_path):
    return (
      "No reference source file path provided or found on disk; skipping"
      " dependency discovery."
    )

  try:
    abs_source_file = os.path.abspath(source_file_path)
    tool_context.state["source_file_path"] = abs_source_file

    workspace_root = ObjectExtractor._find_workspace_root(
      None, os.path.dirname(abs_source_file)
    )
    tool_context.state["source_dir"] = workspace_root

    source_base = Path(workspace_root).resolve()
    source_file_dir = Path(abs_source_file).parent.resolve()

    extractor = ObjectExtractor(abs_source_file, debug=False)

    with open(abs_source_file, "r", encoding="utf-8") as f:
      tree = ast.parse(f.read())
    collector = ImportCollector()
    collector.visit(tree)

    local_files = set(extractor.get_local_import_files(list(collector.imports)))

    for import_stmt in collector.imports:
      try:
        import_tree = ast.parse(import_stmt)
        for node in ast.walk(import_tree):
          module_names = []
          if isinstance(node, ast.ImportFrom) and node.module:
            module_names.append(node.module)
          elif isinstance(node, ast.Import):
            for alias in node.names:
              module_names.append(alias.name)

          for mod_name in module_names:
            if mod_name.startswith("."):
              mod_name = mod_name.lstrip(".")
            parts = mod_name.split(".")
            rel_path_py = Path(*parts).with_suffix(".py")
            rel_path_init = Path(*parts) / "__init__.py"

            for cand_base in [source_file_dir, source_base]:
              if (cand_base / rel_path_py).exists():
                local_files.add(str((cand_base / rel_path_py).resolve()))
              if (cand_base / rel_path_init).exists():
                local_files.add(str((cand_base / rel_path_init).resolve()))
      except Exception as e:
        logging.warning(
          f"Failed to inspect import statement {import_stmt}: {e}"
        )

    dependencies = tool_context.state.setdefault("dependencies", {})

    registered = set()

    def _get_rel_name(file_path: Path) -> Optional[str]:
      if source_file_dir and file_path.is_relative_to(source_file_dir):
        return os.path.relpath(file_path, source_file_dir)
      elif file_path.is_relative_to(source_base):
        return os.path.relpath(file_path, source_base)
      return None

    def _register_file(file_path: Path):
      rel_name = _get_rel_name(file_path)
      if rel_name:
        if rel_name not in dependencies:
          with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
          dependencies[rel_name] = content
          try:
            dest = (base / rel_name).resolve()
            os.makedirs(dest.parent, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as out_f:
              out_f.write(content)
          except Exception as e:
            logging.warning(
              f"Failed to copy dependency {rel_name} to {base}: {e}"
            )
        registered.add(rel_name)

    for abs_path in local_files:
      abs_p = Path(abs_path).resolve()
      try:
        if abs_p != Path(abs_source_file).resolve() and _get_rel_name(abs_p):
          _register_file(abs_p)
          parent = abs_p.parent
          while parent != source_base and parent.is_relative_to(source_base):
            init_file = parent / "__init__.py"
            if init_file.exists():
              _register_file(init_file)
            parent = parent.parent
      except Exception as e:
        logging.warning(f"Failed to read dependency {abs_path}: {e}")

    if not registered:
      return (
        "No local workspace dependency files were found for"
        f" {source_file_path}."
      )
    registered_list = sorted(registered)
    return (
      f"Successfully discovered and registered {len(registered_list)} local"
      f" dependency files: {', '.join(registered_list)}"
    )

  except Exception as e:
    return f"Error discovering dependencies for {source_file_path}: {e}"


discover_kernel_dependencies_fn.__name__ = "discover_kernel_dependencies"
discover_kernel_dependencies_tool = FunctionTool(
  discover_kernel_dependencies_fn
)

__all__ = [
  "filesystem_tool_r",
  "filesystem_tool_rw",
  "write_test_file_tool",
  "write_optimized_kernel_tool",
  "write_optimization_plan_tool",
  "write_profiling_script_tool",
  "write_autotune_specs_tool",
  "write_base_kernel_tool",
  "write_dependency_file_tool",
  "discover_kernel_dependencies_tool",
]
