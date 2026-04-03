"""GPU to JAX conversion agent - converts CUDA/Triton/PyTorch CUDA code to JAX."""

import logging
import warnings
from typing import Any, Dict, Optional

# Suppress all experimental feature warnings
warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*")
warnings.filterwarnings("ignore", category=UserWarning)

import os

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import AgentTool, BaseTool, FunctionTool, ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.genai import types
from mcp import StdioServerParameters

from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.gpu_to_jax_agent.constants import (
  MODEL_NAME,
  TOP_K,
  TOP_P,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.gpu_to_jax_agent.evaluators import (
  JaxCompilationChecker,
  JaxCorrectnessChecker,
  JaxSyntaxChecker,
  ShapeValidator,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.gpu_to_jax_agent.prompts import (
  analyze_plan_prompt,
  convert_simplified_to_jax_prompt,
  fix_conversion_extended_prompt,
  fix_conversion_prompt,
  generate_summary_extended_prompt,
  generate_test_extended_prompt,
  generate_test_prompt,
  identify_framework_prompt,
  orchestrator_prompt,
  run_test_routing_prompt,
  simplify_gpu_code_prompt,
  summary_prompt,
  validate_compilation_routing_prompt,
  validate_shapes_routing_prompt,
  validate_syntax_routing_prompt,
  write_readme_prompt,
)
from tpu_kernel_gen.agents.kernel_gen_agent.agent import CustomLlmAgent

# Model configuration
model_config = types.GenerateContentConfig(
  temperature=0.1,
  top_p=TOP_P,
  top_k=TOP_K,
)

WORKDIR = os.environ.get("WORKDIR", os.path.dirname(os.path.abspath(__file__)))

# Filesystem tools for reading/writing GPU and JAX code
# Note: MCP stdio has known issues with Python 3.13's stricter async task validation
# We wrap this to handle errors gracefully
try:
  filesystem_tool_rw = MCPToolset(
    connection_params=StdioConnectionParams(
      server_params=StdioServerParameters(
        command="npx",
        args=[
          "-y",
          "@modelcontextprotocol/server-filesystem@0.5.1",
          os.path.abspath(WORKDIR),
        ],
        env={**os.environ, "MCP_LOG_LEVEL": "error"},  # Suppress info messages
      ),
    ),
    tool_filter=["list_directory", "read_file", "write_file"],
  )
  logging.info("MCP filesystem toolset initialized successfully")
except Exception as e:
  logging.error(f"Failed to initialize MCP filesystem toolset: {e}")
  logging.warning("MCP tools may not be available. write_file_direct will be used as fallback.")
  # Create a minimal placeholder - agents should prefer write_file_direct anyway
  filesystem_tool_rw = None


# Helper function to determine output directory
def get_output_directory(tool_context: ToolContext, file_path: str) -> str:
  """Determine the output directory for generated files.

  Priority:
  1. Directory of original GPU code file (if available and file_path is a simple filename)
  2. WORKDIR (fallback for paths with subdirectories)

  Args:
      tool_context: Context containing state with original_gpu_code_path
      file_path: The relative path being written (to check if it has subdirectories)

  Returns:
      Absolute path to output directory
  """
  # Check if we have an original GPU code path
  original_gpu_path = tool_context.state.get("original_gpu_code_path")

  # Only use GPU file directory if:
  # 1. We have an original GPU path
  # 2. The GPU path exists
  # 3. The file being written is a simple filename (no subdirectories)
  #    This prevents doubling paths like /path/to/gpu/dir/subdir/file.py
  if original_gpu_path and os.path.exists(original_gpu_path):
    # Check if file_path has directory components
    if os.path.dirname(file_path) == "":
      # Simple filename - use GPU file's directory
      output_dir = os.path.dirname(os.path.abspath(original_gpu_path))
      logging.info(f"Using output directory from original GPU file: {output_dir}")
      return output_dir
    else:
      # Path has subdirectories - use WORKDIR to avoid path duplication
      logging.info(f"File path has subdirectories, using WORKDIR: {WORKDIR}")
      return WORKDIR

  # Fallback to WORKDIR
  logging.info(f"Using WORKDIR as output directory: {WORKDIR}")
  return WORKDIR


# Direct file writing tool (bypasses MCP for reliability)
def write_file_direct(path: str, content: str, tool_context: ToolContext) -> str:
  """Write content to a file. Writes to the directory of the original GPU file if available, otherwise WORKDIR."""
  try:
    # Ensure path is relative
    if os.path.isabs(path):
      error_msg = "Error: Absolute paths not allowed. Use relative path like 'filename.md'"
      logging.error(error_msg)
      return error_msg

    # Get the appropriate output directory (GPU file dir or WORKDIR)
    output_dir = get_output_directory(tool_context, path)
    full_path = os.path.join(output_dir, path)

    # Security: ensure we're still within allowed directories
    abs_full_path = os.path.abspath(full_path)
    abs_workdir = os.path.abspath(WORKDIR)

    # Allow writes to WORKDIR or the directory containing the original GPU file
    original_gpu_path = tool_context.state.get("original_gpu_code_path")
    allowed_dirs = [abs_workdir]
    if original_gpu_path:
      abs_gpu_dir = os.path.dirname(os.path.abspath(original_gpu_path))
      allowed_dirs.append(abs_gpu_dir)

    # Check if path is within any allowed directory
    is_allowed = any(abs_full_path.startswith(allowed_dir) for allowed_dir in allowed_dirs)
    if not is_allowed:
      error_msg = "Error: Path escapes allowed directories"
      logging.error(error_msg)
      return error_msg

    # Try to create parent directories and write file
    try:
      os.makedirs(os.path.dirname(full_path), exist_ok=True)

      with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

      # Save to state
      tool_context.state["most_recent_file_path"] = full_path

      success_msg = f"Successfully wrote {len(content)} characters to {full_path}"
      logging.info(success_msg)
      return success_msg

    except (PermissionError, OSError) as perm_error:
      # If we failed to write to the GPU file directory, try WORKDIR as fallback
      if output_dir != WORKDIR:
        logging.warning(f"Failed to write to {output_dir}, falling back to WORKDIR: {perm_error}")
        fallback_path = os.path.join(WORKDIR, path)

        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        with open(fallback_path, "w", encoding="utf-8") as f:
          f.write(content)

        tool_context.state["most_recent_file_path"] = fallback_path

        success_msg = f"Successfully wrote {len(content)} characters to {fallback_path} (fallback location)"
        logging.info(success_msg)
        return success_msg
      else:
        # Already tried WORKDIR, re-raise
        raise

  except Exception as e:
    error_msg = f"Error writing file: {str(e)}"
    logging.error(f"write_file_direct failed: {error_msg}")
    logging.exception("Full traceback:")
    return error_msg


write_file_tool = FunctionTool(write_file_direct)


# Framework detection tool (saves framework to state)
def save_framework_detection(framework: str, tool_context: ToolContext) -> str:
  """Save the detected GPU framework to state for use by downstream agents.

  Args:
      framework: The detected framework name (e.g., 'CUDA', 'Triton', 'PyTorch CUDA')
      tool_context: Context for accessing agent state

  Returns:
      Success message confirming the framework was saved
  """
  try:
    tool_context.state["framework_detected"] = framework
    success_msg = f"Framework '{framework}' saved to state"
    logging.info(success_msg)
    return success_msg
  except Exception as e:
    error_msg = f"Error saving framework to state: {str(e)}"
    logging.error(error_msg)
    return error_msg


save_framework_tool = FunctionTool(save_framework_detection)


def get_available_tools(*tools):
  """Filter out None tools and return list of available tools."""
  return [t for t in tools if t is not None]


def save_path_from_tool_run(
  tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Optional[Dict]
) -> Optional[Dict]:
  """Save file path to state after file operations and log tool execution results."""
  # Log tool execution for debugging
  tool_name = tool.name
  logging.info(f"Tool '{tool_name}' executed with args: {args}")

  # Check for errors in tool response
  if tool_response:
    # If response is a string that contains "Error" or "error", log it
    if isinstance(tool_response, str) and "error" in tool_response.lower():
      logging.error(f"Tool '{tool_name}' returned error: {tool_response}")
    # If response is a dict with error field, log it
    elif isinstance(tool_response, dict) and "error" in tool_response:
      logging.error(f"Tool '{tool_name}' returned error: {tool_response.get('error')}")
    else:
      logging.info(f"Tool '{tool_name}' response: {str(tool_response)[:200]}")

  if tool.name == "read_file" or tool.name == "write_file":
    file_path = args.get("path", None)
    tool_context.state["most_recent_file_path"] = file_path

    # Preserve the original GPU code file path (only set once, when reading GPU files)
    if tool.name == "read_file" and file_path and not tool_context.state.get("original_gpu_code_path"):
      # Check if this looks like a GPU source file (not a plan or output file)
      # Include common GPU source/header extensions: CUDA (.cu, .cuh), HIP (.hip), C/C++ (.c, .cpp, .h, .hpp), Python (.py)
      gpu_extensions = (".cu", ".cuh", ".py", ".cpp", ".c", ".h", ".hpp", ".hip", ".cc", ".cxx")
      if file_path.endswith(gpu_extensions) and "PLAN" not in file_path.upper() and "SUMMARY" not in file_path.upper():
        # Check if the tool response indicates success (no error)
        is_success = True
        if tool_response:
          # Check for error indicators in the response
          if isinstance(tool_response, dict):
            # Check isError field if present
            if tool_response.get("isError", False):
              is_success = False
            # Check if content contains error messages
            elif "content" in tool_response:
              content = tool_response["content"]
              if isinstance(content, list) and len(content) > 0:
                first_content = content[0]
                if isinstance(first_content, dict) and "text" in first_content:
                  text = first_content["text"]
                  if text.startswith("Error:"):
                    is_success = False

        # Only save the path if the read was successful
        if is_success:
          # Normalize to absolute path - if relative, join with WORKDIR
          if not os.path.isabs(file_path):
            abs_file_path = os.path.join(WORKDIR, file_path)
          else:
            abs_file_path = file_path

          # Verify the file actually exists before saving
          if os.path.exists(abs_file_path):
            tool_context.state["original_gpu_code_path"] = abs_file_path
            logging.info(f"Preserved original GPU code path: {abs_file_path}")
          else:
            logging.warning(f"File does not exist, not saving path: {abs_file_path}")
        else:
          logging.warning(f"Read file failed, not saving path: {file_path}")

  return None


def save_gpu_code_to_state(callback_context: CallbackContext):
  """Save GPU code from original GPU source file to state. Only loads once and caches."""
  # If gpu_code already exists, we've already loaded it - don't reload
  if "gpu_code" in callback_context.state and callback_context.state["gpu_code"]:
    logging.info("GPU code already loaded in state, using cached version")
    return

  # Use original_gpu_code_path (preserved from initial file read), not most_recent_file_path
  # most_recent_file_path can change as files are written (e.g., SIMPLIFICATION_PLAN.md)
  file_path = callback_context.state.get("original_gpu_code_path") or callback_context.state.get("gpu_code_file_path")

  if file_path:
    try:
      with open(file_path, "r") as f:
        gpu_code = f.read()
      callback_context.state["gpu_code"] = gpu_code
      callback_context.state["gpu_code_file_path"] = file_path  # Store path for future reference
      logging.info(f"Loaded GPU code from {file_path} and cached in state")
    except Exception as e:
      logging.error(f"Failed to read GPU code file: {e}")
      callback_context.state["gpu_code"] = None
  else:
    logging.warning("No original GPU code path found in state - cannot load GPU code")


def save_jax_code_to_state(callback_context: CallbackContext):
  """Save JAX code from most recent file to state."""
  file_path = callback_context.state.get("most_recent_file_path", None)
  if file_path:
    try:
      with open(file_path, "r") as f:
        jax_code = f.read()
      callback_context.state["jax_code"] = jax_code
      logging.info(f"Loaded JAX code from {file_path}")
    except Exception as e:
      logging.error(f"Failed to read JAX code file: {e}")
      callback_context.state["jax_code"] = None


def save_test_code_to_state(callback_context: CallbackContext):
  """Save correctness test code from file to state."""
  # Determine the output directory (GPU file dir or WORKDIR)
  original_gpu_path = callback_context.state.get("original_gpu_code_path")
  if original_gpu_path and os.path.exists(original_gpu_path):
    output_dir = os.path.dirname(os.path.abspath(original_gpu_path))
  else:
    output_dir = WORKDIR

  test_file_path = os.path.join(output_dir, "test_correctness.py")
  if os.path.exists(test_file_path):
    try:
      with open(test_file_path, "r") as f:
        test_code = f.read()
      callback_context.state["correctness_test_code"] = test_code
      logging.info(f"Loaded test code from {test_file_path}")
    except Exception as e:
      logging.error(f"Failed to read test code file: {e}")
  else:
    logging.warning(f"Test file not found at {test_file_path}")


def ensure_summary_state_defaults(callback_context: CallbackContext):
  """Ensure all state variables needed for summary generation have defaults."""
  defaults = {
    "framework_detected": "Unknown",
    "compilation_results": "Not available",
    "syntax_validation_results": "Not available",
    "shape_validation_results": "Not available",
    "correctness_test_results": "Not available",
  }
  for key, default_value in defaults.items():
    if key not in callback_context.state or not callback_context.state[key]:
      callback_context.state[key] = default_value
      logging.info(f"Set default value for {key}: {default_value}")


def load_simplification_plan_from_file(callback_context: CallbackContext):
  """Load SIMPLIFICATION_PLAN.md from file into state (to capture user edits)."""
  # Determine the output directory (GPU file dir or WORKDIR)
  original_gpu_path = callback_context.state.get("original_gpu_code_path")
  if original_gpu_path and os.path.exists(original_gpu_path):
    output_dir = os.path.dirname(os.path.abspath(original_gpu_path))
  else:
    output_dir = WORKDIR

  plan_file_path = os.path.join(output_dir, "SIMPLIFICATION_PLAN.md")

  # Check if file exists (it may not exist on first run)
  if os.path.exists(plan_file_path):
    try:
      with open(plan_file_path, "r") as f:
        plan_content = f.read()
      callback_context.state["simplification_plan"] = plan_content
      logging.info(f"Loaded simplification plan from {plan_file_path} (includes any user edits)")
    except Exception as e:
      logging.error(f"Failed to read simplification plan file: {e}")
  else:
    logging.info(f"No existing SIMPLIFICATION_PLAN.md found at {plan_file_path}")


# Step 1: Read GPU code and identify framework
identify_framework_agent = CustomLlmAgent(
  name="IdentifyFrameworkAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=identify_framework_prompt.PROMPT,
  description="Reads GPU code file and identifies which GPU framework is being used",
  output_key="framework_detected",
  tools=get_available_tools(filesystem_tool_rw, save_framework_tool),
  after_tool_callback=save_path_from_tool_run,
)

# Step 2: Analyze, plan, and write plan to file (combined)
analyze_plan_and_write_agent = CustomLlmAgent(
  name="AnalyzePlanAndWriteAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=analyze_plan_prompt.PROMPT,
  description="Analyzes GPU code, creates simplification plan, and writes it to file",
  tools=get_available_tools(filesystem_tool_rw, write_file_tool),  # Include both read and write tools
  before_agent_callback=load_simplification_plan_from_file,  # Load existing plan from file before regenerating
)

# Step 3: Execute simplification based on approved plan and write to file
organize_gpu_code_agent = CustomLlmAgent(
  name="OrganizeGpuCodeAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=simplify_gpu_code_prompt.PROMPT,
  description="Simplifies GPU code based on the approved plan and writes it to file with appropriate extension",
  output_key="organized_code",
  tools=get_available_tools(filesystem_tool_rw, write_file_tool),
  after_tool_callback=save_path_from_tool_run,
)

# Step 4: Write simplification README documenting the process
write_simplification_readme_agent = CustomLlmAgent(
  name="WriteSimplificationReadmeAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=write_readme_prompt.PROMPT,
  description="Writes README explaining original code and simplification steps",
  tools=get_available_tools(filesystem_tool_rw, write_file_tool),
)

# Step 5: Convert to JAX and write to file (combined)
convert_to_jax_agent = CustomLlmAgent(
  name="ConvertToJaxAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=convert_simplified_to_jax_prompt.PROMPT,
  description="Converts organized code to JAX and writes it to converted_jax.py",
  tools=[write_file_tool],
  after_tool_callback=save_path_from_tool_run,
)

# Step 6: Validate syntax (evaluator wrapped as tool)
_syntax_checker = JaxSyntaxChecker(
  name="check_jax_syntax",
  input_key="jax_code",
  output_key="syntax_validation_results",
)
syntax_checker_tool = AgentTool(agent=_syntax_checker)

# Step 6b: Syntax validation with routing logic
validate_syntax_agent = CustomLlmAgent(
  name="ValidateSyntaxAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  tools=[syntax_checker_tool],
  instruction=validate_syntax_routing_prompt.PROMPT,
  description="Validates JAX syntax and routes to fix or compilation based on results",
  before_agent_callback=save_jax_code_to_state,  # Load JAX code from file before validation
)

# Step 7: Fix conversion errors
fix_conversion_agent = CustomLlmAgent(
  name="FixConversionAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=fix_conversion_prompt.PROMPT.replace("{jax_code}", "{jax_code}").replace(
    "{error_messages}", "{syntax_validation_results}"
  )
  + fix_conversion_extended_prompt.PROMPT,
  description="Fixes syntax errors in JAX conversion and writes to converted_jax.py",
  tools=[write_file_tool],
)

# Step 8: Validate compilation (evaluator wrapped as tool)
_compilation_checker = JaxCompilationChecker(
  name="check_jax_compilation",
  input_key="jax_code",
  output_key="compilation_results",
  auto_manage_servers=True,
)
compilation_checker_tool = AgentTool(agent=_compilation_checker)

# Step 8b: Compilation validation with routing
validate_compilation_agent = CustomLlmAgent(
  name="ValidateCompilationAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  tools=[compilation_checker_tool],
  instruction=validate_compilation_routing_prompt.PROMPT,
  description="Validates JAX compilation and proceeds to shape validation",
)

# Step 9: Validate shapes (evaluator wrapped as tool)
_shape_validator = ShapeValidator(
  name="validate_shapes",
  input_key="jax_code",
  output_key="shape_validation_results",
)
shape_validator_tool = AgentTool(agent=_shape_validator)

# Step 9b: Shape validation with routing
validate_shapes_agent = CustomLlmAgent(
  name="ValidateShapesAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  tools=[shape_validator_tool],
  instruction=validate_shapes_routing_prompt.PROMPT,
  description="Validates tensor shapes and proceeds to test generation",
)

# Step 10: Generate correctness test and write to file (combined)
generate_correctness_test_agent = CustomLlmAgent(
  name="GenerateCorrectnessTestAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=generate_test_prompt.PROMPT + "\n" + generate_test_extended_prompt.PROMPT,
  description="Generates a validation test for JAX code and writes it to test_correctness.py",
  # Note: No output_key - test is written to file and read by RunCorrectnessTestAgent from file
  tools=[write_file_tool],
  after_tool_callback=save_path_from_tool_run,
)

# Step 11: Run correctness test (evaluator wrapped as tool)
_correctness_checker = JaxCorrectnessChecker(
  name="run_correctness_test",
  input_key="correctness_test_code",
  output_key="correctness_test_results",
  auto_manage_servers=True,
)
correctness_checker_tool = AgentTool(agent=_correctness_checker)

# Step 11b: Run test with routing
run_correctness_test_agent = CustomLlmAgent(
  name="RunCorrectnessTestAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  tools=[correctness_checker_tool],
  instruction=run_test_routing_prompt.PROMPT,
  description="Loads test from file, runs correctness check, and proceeds to summary generation",
  before_agent_callback=save_test_code_to_state,  # Load test file into state before running
)

# Step 12: Generate summary and write to file (combined)
generate_and_write_summary_agent = CustomLlmAgent(
  name="GenerateAndWriteSummaryAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=summary_prompt.PROMPT.replace("{framework_detected}", "{framework_detected}")
  .replace("{conversion_status}", "Success")
  .replace(
    "{test_results}",
    """Compilation: {compilation_results}
Syntax Validation: {syntax_validation_results}
Shape Validation: {shape_validation_results}
Numerical Correctness: {correctness_test_results}""",
  )
  + generate_summary_extended_prompt.PROMPT,
  description="Generates conversion summary and writes it to CONVERSION_SUMMARY.md",
  output_key="conversion_summary",
  tools=[write_file_tool],
  before_agent_callback=ensure_summary_state_defaults,
)

# Main GPU-to-JAX orchestrator agent
gpu_to_jax_agent = CustomLlmAgent(
  name="GpuToJaxAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=orchestrator_prompt.PROMPT,
  description="Routes user requests to appropriate GPU-to-JAX conversion workflow phase",
  sub_agents=[
    # Framework identification and planning
    identify_framework_agent,
    analyze_plan_and_write_agent,
    # Simplification execution
    organize_gpu_code_agent,
    write_simplification_readme_agent,
    # JAX conversion and validation
    convert_to_jax_agent,
    validate_syntax_agent,
    fix_conversion_agent,
    validate_compilation_agent,
    validate_shapes_agent,
    # Testing and summary
    generate_correctness_test_agent,
    run_correctness_test_agent,
    generate_and_write_summary_agent,
  ],
)
