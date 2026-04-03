"""Callback utilities for HITL kernel generation agents."""

import json
import logging
import os
from typing import Any, Dict, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import BaseTool, ToolContext

from tpu_kernel_gen.agents.hitl_kernel_gen_agent.config import TPU_VERSION, WORKDIR


def create_path_saver(state_key: str):
  """
  Factory function that creates a callback to save file paths to a specific state key.

  Args:
      state_key: The key in tool_context.state where the file path will be saved.

  Returns:
      A callback function compatible with after_tool_callback signature.
  """

  def save_path(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Optional[Dict]
  ) -> Optional[Dict]:
    # MCP filesystem tools may have different naming patterns
    # Check for both snake_case and potential prefixed versions
    if "read" in tool.name.lower() or "write" in tool.name.lower():
      file_path = args.get("path", None)
      if file_path:
        tool_context.state[state_key] = file_path
        logging.info(f"Saved file path to {state_key}: {file_path} (from tool: {tool.name})")
    return None

  return save_path


def save_kernel_file_paths(
  tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Optional[Dict]
) -> Optional[Dict]:
  """
  Saves kernel file paths with semantic naming based on read order.
  First file read = base_kernel_path, Second file read = optimized_kernel_path.
  This callback is used by agents that need to compare two kernels.
  """
  if tool.name == "read_file":
    file_path = args.get("path", None)

    # If base_kernel_path not set, this is the first file (base)
    if "base_kernel_path" not in tool_context.state or not tool_context.state["base_kernel_path"]:
      tool_context.state["base_kernel_path"] = file_path
      logging.info(f"Set base kernel path: {file_path}")
    # Otherwise, this is the second file (optimized)
    else:
      tool_context.state["optimized_kernel_path"] = file_path
      logging.info(f"Set optimized kernel path: {file_path}")

  return None


def save_kernel_and_plan_paths(
  tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Optional[Dict]
) -> Optional[Dict]:
  """Saves both optimized_kernel_path and kernel_plan_path during implementation."""
  if "read" in tool.name.lower() or "write" in tool.name.lower():
    file_path = args.get("path", None)
    if file_path:
      # Check if this is a plan file based on filename or path
      if "plan" in file_path.lower() and file_path.endswith(".md"):
        tool_context.state["kernel_plan_path"] = file_path
        logging.info(f"Saved plan path to kernel_plan_path: {file_path} (from tool: {tool.name})")
      # Otherwise assume it's the kernel file
      else:
        tool_context.state["optimized_kernel_path"] = file_path
        logging.info(f"Saved kernel path to optimized_kernel_path: {file_path} (from tool: {tool.name})")
  return None


def load_single_kernel_to_state(callback_context: CallbackContext):
  """
  Loads a single kernel file content into state.
  Uses kernel_file_path to find the file.
  Stores content in 'kernel_code' for use by compilation/profiling agents.
  """
  file_path = callback_context.state.get("kernel_file_path", None)

  if file_path:
    try:
      with open(file_path, "r") as f:
        kernel_code = f.read()
      callback_context.state["kernel_code"] = kernel_code
      logging.info(f"Loaded kernel code from {file_path}")
    except Exception as e:
      logging.error(f"Failed to read kernel file: {e}")
      callback_context.state["kernel_code"] = None
  else:
    logging.warning("No kernel file path found in state")


def load_profiling_script_to_state(callback_context: CallbackContext):
  """
  Loads profiling script file content into state.
  Uses profiling_script_path to find the file.
  Stores content in 'profiling_script' for use by profiling execution agent.
  """
  file_path = callback_context.state.get("profiling_script_path", None)

  if file_path:
    try:
      with open(file_path, "r") as f:
        profiling_code = f.read()
      callback_context.state["profiling_script"] = profiling_code
      logging.info(f"Loaded profiling script from {file_path}")
    except Exception as e:
      logging.error(f"Failed to read profiling script file: {e}")
      callback_context.state["profiling_script"] = None
  else:
    logging.warning("No profiling script path found in state")


def load_two_kernels_to_state(callback_context: CallbackContext):
  """
  Loads two kernel files (base and optimized) into state for comparison.
  Reads from base_kernel_path and optimized_kernel_path.
  Stores contents in base_kernel_code and optimized_kernel_code.
  """
  base_path = callback_context.state.get("base_kernel_path", None)
  optimized_path = callback_context.state.get("optimized_kernel_path", None)

  if base_path:
    try:
      with open(base_path, "r") as f:
        base_code = f.read()
      callback_context.state["base_kernel_code"] = base_code
      logging.info(f"Loaded base kernel code from {base_path}")
    except Exception as e:
      logging.error(f"Failed to read base kernel file: {e}")
      callback_context.state["base_kernel_code"] = None
  else:
    logging.warning("No base kernel path found in state")

  if optimized_path:
    try:
      with open(optimized_path, "r") as f:
        optimized_code = f.read()
      callback_context.state["optimized_kernel_code"] = optimized_code
      logging.info(f"Loaded optimized kernel code from {optimized_path}")
    except Exception as e:
      logging.error(f"Failed to read optimized kernel file: {e}")
      callback_context.state["optimized_kernel_code"] = None
  else:
    logging.warning("No optimized kernel path found in state")


def load_kernel_and_plan_to_state(callback_context: CallbackContext):
  """
  Loads kernel file and optimization plan into state for compilation fixing.
  Uses optimized_kernel_path and kernel_plan_path to find files.
  Stores content in 'kernel_code' and 'kernel_plan' for use by fix agent.
  Also formats compilation_history for better readability.
  """
  # Load kernel code
  kernel_path = callback_context.state.get("optimized_kernel_path", None)
  if kernel_path and os.path.exists(kernel_path):
    try:
      with open(kernel_path, "r") as f:
        kernel_code = f.read()
      callback_context.state["kernel_code"] = kernel_code
      logging.info(f"Loaded kernel code from {kernel_path}")
    except Exception as e:
      logging.error(f"Failed to read kernel file: {e}")
      callback_context.state["kernel_code"] = None
  else:
    logging.warning("No kernel file path found in state or file does not exist")
    callback_context.state["kernel_code"] = None

  # Load optimization plan if exists
  plan_path = callback_context.state.get("kernel_plan_path", None)
  if plan_path and os.path.exists(plan_path):
    try:
      with open(plan_path, "r") as f:
        kernel_plan = f.read()
      callback_context.state["kernel_plan"] = kernel_plan
      logging.info(f"Loaded optimization plan from {plan_path}")
    except Exception as e:
      logging.error(f"Failed to read plan file: {e}")
      callback_context.state["kernel_plan"] = None
  else:
    logging.info("No optimization plan path found (this is okay for some workflows)")
    callback_context.state["kernel_plan"] = None

  # Format compilation history for readability
  history = callback_context.state.get("compilation_history", [])
  if history:
    formatted_history = []
    for record in history:
      attempt_num = record.get("attempt", "?")
      success = record.get("success", False)
      result = record.get("result", "Unknown")
      fix_summary = record.get("fix_summary", None)

      status = "✓ SUCCESS" if success else "✗ FAILED"
      formatted_history.append(f"**Attempt {attempt_num}:** {status}")

      if not success:
        # Include the fix that was attempted (if available)
        if fix_summary:
          formatted_history.append(f"Fix Applied: {fix_summary}")

      formatted_history.append("")  # Blank line

    # Store formatted version in a separate key for the prompt
    callback_context.state["compilation_history_formatted"] = "\n".join(formatted_history)
  else:
    callback_context.state["compilation_history_formatted"] = "No previous attempts (this is the first attempt)"


def get_tpu_version_callback(callback_context: CallbackContext):
  """Load TPU version and specifications into state."""
  tpu_version = TPU_VERSION

  callback_context.state["tpu_version"] = tpu_version
  logging.info(f"Detected TPU version: {tpu_version}")

  try:
    with open("hitl_kernel_gen_agent/tpu_specs.json", "r") as f:
      tpu_specs = json.load(f)

    if tpu_version in tpu_specs:
      callback_context.state["tpu_specs"] = tpu_specs[tpu_version]
    else:
      callback_context.state["tpu_specs"] = "TPU specs not found for detected version."
    logging.info(f"Loaded TPU specs for {tpu_version}")
  except Exception as e:
    logging.error(f"Failed to load TPU specs: {e}")
    callback_context.state["tpu_specs"] = None


def add_workdir_callback(callback_context: CallbackContext):
  """Add working directory to state."""
  callback_context.state["workdir"] = WORKDIR
  logging.info(f"Set working directory to: {WORKDIR}")


def extract_fix_summary(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse:
  """Extract the agent's response and store it as the fix summary.

  This is an after_model_callback that receives the LlmResponse directly.
  """
  if llm_response.content is None or not llm_response.content.parts:
    logging.warning("No content in LlmResponse to extract fix summary from")
    return llm_response

  # Collect all text parts from the response
  text_parts = []
  for part in llm_response.content.parts:
    if hasattr(part, "text") and part.text:
      text_parts.append(part.text)

  if text_parts:
    # Join all text parts and store as fix summary
    fix_summary = "\n".join(text_parts).strip()
    callback_context.state["fix_summary"] = fix_summary
    logging.info(f"Captured fix summary ({len(fix_summary)} chars)")
  else:
    logging.warning("No text parts found in LlmResponse")

  return llm_response


__all__ = [
  "create_path_saver",
  "save_kernel_file_paths",
  "save_kernel_and_plan_paths",
  "load_single_kernel_to_state",
  "load_profiling_script_to_state",
  "load_two_kernels_to_state",
  "load_kernel_and_plan_to_state",
  "get_tpu_version_callback",
  "add_workdir_callback",
  "extract_fix_summary",
]
