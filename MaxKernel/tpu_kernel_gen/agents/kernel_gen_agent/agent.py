# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import partial
from typing import AsyncGenerator, Literal

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models import LlmResponse
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.genai.types import Part
from pydantic import BaseModel, Field

from tpu_kernel_gen.agents.kernel_gen_agent.constants import (
  MAX_ITERATIONS,
  MODEL_NAME,
  TEMPERATURE,
  TOP_K,
  TOP_P,
)
from tpu_kernel_gen.agents.kernel_gen_agent.kernel_eval import (
  KernelCompilationChecker,
  KernelCorrectnessChecker,
  KernelTilingOptimizer,
  check_whether_to_test_jax,
  check_whether_to_test_kernel_correctness,
  check_whether_to_test_kernel_tiling,
  jax_compilation_checker,
  jax_conversion_checker,
)
from tpu_kernel_gen.agents.kernel_gen_agent.prompts import (
  add_kernel_tiling,
  adjust_kernel_input,
  convert_to_jax,
  fix_base_kernel_code,
  fix_jax_code,
  fix_kernel_tiling_code,
  gen_correctness_test,
  gen_tile_tuning_script,
  judge_jax_conversion,
  organize_code,
  pallas_docs,
  pallas_profiling_docs,
  summary,
  write_basic_kernel,
)
from tpu_kernel_gen.agents.kernel_gen_agent.tools import search_api_tool

model_config = types.GenerateContentConfig(
  temperature=TEMPERATURE,
  top_p=TOP_P,
  top_k=TOP_K,
)


class Result(BaseModel):
  """Model for providing evaluation feedback on research quality."""

  result: Literal["pass", "fail"] = Field(
    description="Assessment result. 'pass' if the conversion is logically correct, 'fail' if it needs revision."
  )


class CustomLlmAgent(LlmAgent):
  """Agent that allows early exit from the loop if a condition is met.

  Automatically uses gemini_model (with retry support) when a string model name is provided.
  """

  def __init__(self, *args, **kwargs):
    """Initialize CustomLlmAgent with automatic Gemini model (with retry) wrapping."""
    # If model is a string, use the pre-configured gemini_model with retry support
    if "model" in kwargs and isinstance(kwargs["model"], str):
      gemini_model = Gemini(
        model=MODEL_NAME,
        retry_options=types.HttpRetryOptions(
          initial_delay=1,
          attempts=5,
        ),
      )
      kwargs["model"] = gemini_model
    super().__init__(*args, **kwargs)

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    # Reset go_to_end flag when new user input is detected
    if hasattr(ctx.session, "contents") and ctx.session.contents and len(ctx.session.contents) > 0:
      last_message = ctx.session.contents[-1]
      # Check if the last message is from the user (role='user')
      if hasattr(last_message, "role") and last_message.role == "user":
        if ctx.session.state.get("go_to_end", False):
          logging.info(f"[{self.name}] New user input detected. Resetting go_to_end flag.")
        ctx.session.state["go_to_end"] = False

    if ctx.session.state.get("go_to_end", False):
      logging.info(f"[{self.name}] Early exit condition met. Skipping loop.")
      yield Event(
        author=self.name,
        actions=EventActions(escalate=True),
      )
    else:
      # Delegate to parent implementation (with native retry support at API level)
      async for event in super()._run_async_impl(ctx):
        yield event


def add_pallas_docs(callback_context: CallbackContext):
  """Adds the full Pallas documentation to the callback context."""
  callback_context.state["pallas_docs"] = pallas_docs.PROMPT
  callback_context.state["pallas_profiling_docs"] = pallas_profiling_docs.PROMPT


def fix_kernel_code_agent_preprocess(
  callback_context: CallbackContext,
  compilation_key,
  correctness_key,
  prev_attempts_key,
  code_key,
  iter_key,
):
  compilation_result = callback_context.state.get(compilation_key, None)
  correctness_result = callback_context.state.get(correctness_key, None)

  logging.info(
    f"Determining what to fix: compilation_result={compilation_result}, correctness_result={correctness_result}"
  )

  callback_context.state["error"] = compilation_result if compilation_result != "Success" else correctness_result

  prev_attempts = callback_context.state.get(prev_attempts_key, "")
  code = callback_context.state.get(code_key, None)
  error = compilation_result if compilation_result != "Success" else correctness_result
  fix_loop_iter = callback_context.state.get(iter_key, 0) + 1
  callback_context.state[iter_key] = fix_loop_iter

  if fix_loop_iter == MAX_ITERATIONS:
    logging.info("Reached maximum iterations for this loop. Will proceed to end")
    callback_context.state["go_to_end"] = True

  prev_attempts += f"\nIteration {fix_loop_iter} script:\n{code}\n\n"
  prev_attempts += f"Observed Error with script from iteration {fix_loop_iter}:\n```\n{error}```\n"

  callback_context.state[prev_attempts_key] = prev_attempts


def check_whether_to_skip(callback_context: CallbackContext) -> types.Content:
  if callback_context.state.get("go_to_end", False):
    return types.Content(
      role="model",
      parts=[Part(text="Skipping this loop as the maximum number of iterations has been reached in a previous step.")],
    )


def filter_code_parts_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse:
  logging.info(f"llm_response: {llm_response.content}")
  if llm_response.content is None or not llm_response.content.parts:
    return llm_response

  filtered_parts = []
  for part in llm_response.content.parts:
    if part.text is not None:
      if part.text.startswith("```"):
        filtered_parts.append(part)
    else:
      filtered_parts.append(part)

  llm_response.content = types.Content(role="model", parts=filtered_parts)

  return llm_response


# --- Step 1: Organize Code ---
organize_code_agent = CustomLlmAgent(
  name="OrganizeCodeAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=organize_code.PROMPT,
  description="Isolates and organizes essential code into imports, initialization, and computation sections.",
  output_key="organized_code",  # Store output for next agent
  include_contents="none",
  after_model_callback=filter_code_parts_callback,
)


# --- Step 2: Convert to JAX (Iterative Loop) ---
convert_to_jax_agent = CustomLlmAgent(
  name="ConvertToJaxAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=convert_to_jax.PROMPT,
  description="Rewrites initialization logic using JAX and Flax based on isolated code.",
  output_key="jax_base_code",  # Store output for later agents
  include_contents="none",
  after_model_callback=filter_code_parts_callback,
)


judge_jax_conversion_agent = CustomLlmAgent(
  name="JudgeJaxConversionsAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=judge_jax_conversion.PROMPT,
  description="Assesses if the JAX conversion is computationally correct.",
  disallow_transfer_to_parent=True,
  disallow_transfer_to_peers=True,
  output_schema=Result,
  output_key="jax_base_code_correctness_result",  # Store judgement for loop control
  include_contents="none",
  before_agent_callback=check_whether_to_test_jax,
)

fix_jax_base_code_agent = CustomLlmAgent(
  name="FixJaxBaseCodeAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=fix_jax_code.PROMPT,
  description="Identifies and corrects issues in the JAX base code to ensure it compiles successfully and is computationally correct.",
  output_key="jax_base_code",  # Store corrected code for next iteration
  tools=[search_api_tool],
  include_contents="none",
  before_agent_callback=partial(
    fix_kernel_code_agent_preprocess,
    compilation_key="jax_base_code_compilation_result",
    correctness_key="jax_base_code_correctness_result",
    prev_attempts_key="jax_conversion_prev_attempts",
    code_key="jax_base_code",
    iter_key="fix_jax_base_code_loop_iter",
  ),
  after_model_callback=filter_code_parts_callback,
)

jax_conversion_loop = LoopAgent(
  name="JaxConversionLoop",
  sub_agents=[
    jax_compilation_checker,
    judge_jax_conversion_agent,
    jax_conversion_checker,
    fix_jax_base_code_agent,
  ],
  before_agent_callback=check_whether_to_skip,
  description="Loop to refine JAX conversion until it meets requirements.",
  max_iterations=MAX_ITERATIONS,  # Limit iterations to prevent infinite loops
)

# --- Step 3: Write Base Kernel Code (Iterative Loop) ---

write_base_kernel_agent = CustomLlmAgent(
  name="WriteBaseKernelAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=write_basic_kernel.PROMPT,
  description="Writes a basic Pallas kernel based on the provided JAX base code.",
  output_key="base_kernel_code",  # Store output for next agent
  include_contents="none",
  tools=[
    search_api_tool
  ],  # gkroiz: TODO: Using this too here causes code 400 error: Please ensure that function call turn comes immediately after a user turn or after a function response turn.
  before_agent_callback=[add_pallas_docs, check_whether_to_skip],
  after_model_callback=filter_code_parts_callback,
)

base_kernel_compilation_checker = KernelCompilationChecker(
  name="BaseKernelCompilationChecker",
  input_key="base_kernel_code",
  output_key="base_kernel_compilation_result",
)


gen_base_kernel_correctness_test_agent = CustomLlmAgent(
  name="GenBaseKernelCorrectnessTestAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=gen_correctness_test.PROMPT.replace("{base_code}", "{jax_base_code}").replace(
    "{optimized_code}", "{base_kernel_code}"
  ),
  description="Generates a JAX script to test the correctness of the Pallas kernel.",
  output_key="base_kernel_correctness_test_code",  # Store generated test script for next agent
  include_contents="none",
  before_agent_callback=partial(
    check_whether_to_test_kernel_correctness,
    compilation_key="base_kernel_compilation_result",
  ),
)

base_kernel_correctness_checker = KernelCorrectnessChecker(
  name="BaseKernelCorrectnessChecker",
  input_key="base_kernel_correctness_test_code",
  output_key="base_kernel_correctness_result",
  before_agent_callback=partial(
    check_whether_to_test_kernel_correctness,
    compilation_key="base_kernel_compilation_result",
  ),
)

fix_base_kernel_code_agent = CustomLlmAgent(
  name="FixBaseKernelCodeAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=fix_base_kernel_code.PROMPT,
  description="Identifies and corrects issues in the base Pallas kernel code to ensure it compiles successfully and is computationally correct.",
  output_key="base_kernel_code",
  tools=[search_api_tool],
  include_contents="none",
  before_agent_callback=partial(
    fix_kernel_code_agent_preprocess,
    compilation_key="base_kernel_compilation_result",
    correctness_key="base_kernel_correctness_result",
    prev_attempts_key="base_kernel_prev_attempts",
    code_key="base_kernel_code",
    iter_key="fix_base_kernel_loop_iter",
  ),
  after_model_callback=filter_code_parts_callback,
)

base_kernel_refinement_loop = LoopAgent(
  name="BaseKernelRefinementLoop",
  sub_agents=[
    base_kernel_compilation_checker,
    gen_base_kernel_correctness_test_agent,
    base_kernel_correctness_checker,
    fix_base_kernel_code_agent,
  ],
  before_agent_callback=check_whether_to_skip,
  description="Loop to refine the kernel code.",
  max_iterations=MAX_ITERATIONS,
)

# --- Step 3.5: Adjust kernel code to use original input shapes ---

adjust_base_kernel_input_shapes_agent = CustomLlmAgent(
  name="AdjustBaseKernelInputShapesAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=adjust_kernel_input.PROMPT,
  description="Adjusts the Pallas kernel to use the original input shapes from the JAX base code.",
  output_key="base_kernel_code",  # Overwrite the base kernel code with adjusted shapes
  include_contents="none",
  before_agent_callback=check_whether_to_skip,
  after_model_callback=filter_code_parts_callback,
)

# --- Step 4: Add Kernel Tiling (Iterative Loop) ---

add_kernel_tiling_agent = CustomLlmAgent(
  name="AddKernelTilingAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=add_kernel_tiling.PROMPT,
  description="Enhances the Pallas kernel by adding tiling logic to optimize performance on TPU hardware.",
  output_key="tiled_kernel_code",
  include_contents="none",
  tools=[search_api_tool],
  before_agent_callback=check_whether_to_skip,
  after_model_callback=filter_code_parts_callback,
)

add_kernel_tiling_compilation_checker = KernelCompilationChecker(
  name="AddKernelTilingCompilationChecker",
  input_key="tiled_kernel_code",
  output_key="tiled_kernel_compilation_result",
)

gen_add_kernel_tiling_correctness_test_agent = CustomLlmAgent(
  name="GenAddKernelTilingCorrectnessTestAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=gen_correctness_test.PROMPT.replace("{base_code}", "{jax_base_code}").replace(
    "{optimized_code}", "{tiled_kernel_code}"
  ),
  description="Generates a JAX script to test the correctness of the tiled Pallas kernel.",
  output_key="tiled_kernel_correctness_test_code",
  include_contents="none",
  before_agent_callback=partial(
    check_whether_to_test_kernel_correctness,
    compilation_key="tiled_kernel_compilation_result",
  ),
  after_model_callback=filter_code_parts_callback,
)

tiled_kernel_correctness_checker = KernelCorrectnessChecker(
  name="TiledKernelCorrectnessChecker",
  input_key="tiled_kernel_correctness_test_code",
  output_key="tiled_kernel_correctness_result",
  before_agent_callback=partial(
    check_whether_to_test_kernel_correctness,
    compilation_key="tiled_kernel_compilation_result",
  ),
)

fix_tiled_kernel_code_agent = CustomLlmAgent(
  name="FixTiledKernelCodeAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=fix_kernel_tiling_code.PROMPT,
  description="Identifies and corrects issues in the tiled Pallas kernel code to ensure it compiles successfully and is computationally correct.",
  output_key="tiled_kernel_code",
  tools=[search_api_tool],
  include_contents="none",
  before_agent_callback=partial(
    fix_kernel_code_agent_preprocess,
    compilation_key="tiled_kernel_compilation_result",
    correctness_key="tiled_kernel_correctness_result",
    prev_attempts_key="tiled_kernel_prev_attempts",
    code_key="tiled_kernel_code",
    iter_key="fix_tiled_kernel_loop_iter",
  ),
  after_model_callback=filter_code_parts_callback,
)

tiled_kernel_refinement_loop = LoopAgent(
  name="TiledKernelRefinementLoop",
  sub_agents=[
    add_kernel_tiling_compilation_checker,
    gen_add_kernel_tiling_correctness_test_agent,
    tiled_kernel_correctness_checker,
    fix_tiled_kernel_code_agent,
  ],
  before_agent_callback=check_whether_to_skip,
  description="Loop to refine the tiled kernel code.",
  max_iterations=MAX_ITERATIONS,
)


# --- Step 5: Generate Performance Test ---

# generate_performance_test_agent = CustomLlmAgent(
#     name="GeneratePerformanceTestAgent",
#     model=MODEL_NAME,
#     generate_content_config=model_config,
#     instruction=gen_performance_test.PROMPT.replace(
#         "{kernel_code}", "{tiled_kernel_code}"
#     ),
#     description="Generates a performance test script to benchmark the kernel code.",
#     output_key="performance_test_code",
#     include_contents="none",
#     before_agent_callback=partial(
#         check_whether_to_test_kernel_performance,
#         compilation_key="base_kernel_compilation_result",
#         correctness_key="base_kernel_correctness_result",
#     ),
# )

# --- Step 5: Tune blocks & run performance test ---
generate_tuning_script_agent = CustomLlmAgent(
  name="GenerateTuningScriptAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=gen_tile_tuning_script.PROMPT.replace("{kernel_code}", "{tiled_kernel_code}"),
  description="Generates a performance test script to benchmark the kernel code across different tiling configurations.",
  output_key="tile_tuning_script",
  include_contents="none",
  before_agent_callback=check_whether_to_skip,
  after_model_callback=filter_code_parts_callback,
)


kernel_tiling_optimizer = KernelTilingOptimizer(
  name="KernelTilingOptimizer",
  input_key="tile_tuning_script",
  output_key="kernel_tiling_optimization_result",
  before_agent_callback=partial(
    check_whether_to_test_kernel_tiling,
    compilation_key="tiled_kernel_compilation_result",
    correctness_key="tiled_kernel_correctness_result",
  ),
)

# --- Sequential Agent Workflow ---


summary_agent = CustomLlmAgent(
  name="SummaryAgent",
  description="Generates a summary of the kernel generation process.",
  output_key="summary",
  model=MODEL_NAME,
  generate_content_config=model_config,
  instruction=summary.PROMPT,
  include_contents="none",
)

root_agent = SequentialAgent(
  name="PallasKernelGenerationWorkflow",
  sub_agents=[
    organize_code_agent,
    convert_to_jax_agent,
    jax_conversion_loop,
    write_base_kernel_agent,
    base_kernel_refinement_loop,
    adjust_base_kernel_input_shapes_agent,
    add_kernel_tiling_agent,
    tiled_kernel_refinement_loop,
    # generate_performance_test_agent,
    generate_tuning_script_agent,
    kernel_tiling_optimizer,
    summary_agent,
  ],
  description="Sequential workflow for converting code to Jax with Pallas Kernel.",
)
