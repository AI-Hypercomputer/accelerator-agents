"""Tool for dynamically updating the max compilation retries setting."""

import logging
import os
from google.adk.tools import FunctionTool, ToolContext


async def set_max_compilation_retries_fn(retries: int, tool_context: ToolContext) -> str:
  """Set the maximum number of compilation validation and auto-fixing attempts.

  Args:
      retries: The maximum number of attempts (must be positive).

  Returns:
      A string containing the operation result description:
      - If successful: "Successfully updated maximum compilation retries to: <retries>"
      - If failed: "Error: The number of retries must be a positive integer."
  """
  if retries <= 0:
    return "Error: The number of retries must be a positive integer."

  # Update in-memory config variables
  import hitl_agent.config as hitl_cfg
  import auto_agent.config as auto_cfg
  hitl_cfg.MAX_COMPILATION_RETRIES = retries
  auto_cfg.MAX_COMPILATION_RETRIES = retries
  os.environ["MAX_COMPILATION_RETRIES"] = str(retries)

  # Update .env file
  try:
    env_path = ".env"
    if os.path.exists(env_path):
      with open(env_path, "r") as f:
        lines = f.readlines()
      with open(env_path, "w") as f:
        updated = False
        for line in lines:
          if line.strip().startswith("MAX_COMPILATION_RETRIES="):
            f.write(f"MAX_COMPILATION_RETRIES={retries}\n")
            updated = True
          else:
            f.write(line)
        if not updated:
          f.write(f"MAX_COMPILATION_RETRIES={retries}\n")
  except Exception as e:
    logging.error(f"Failed to update .env: {e}")

  # Update the active validation loops in-memory
  try:
    from hitl_agent.subagents.kernel_writing.agent import kernel_compilation_validation_loop as hitl_loop
    hitl_loop.max_retries = retries
  except Exception as e:
    logging.warning(f"Could not update hitl_loop max_retries: {e}")

  try:
    from auto_agent.subagents.kernel_writing.agent import kernel_compilation_validation_loop as auto_loop
    auto_loop.max_retries = retries
  except Exception as e:
    logging.warning(f"Could not update auto_loop max_retries: {e}")

  return f"Successfully updated maximum compilation retries to: {retries}"


set_max_compilation_retries = FunctionTool(set_max_compilation_retries_fn)
