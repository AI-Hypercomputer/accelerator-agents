import logging
from functools import partial
from typing import AsyncGenerator, Literal

from google.adk.agents import LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from google.genai.types import Part
from pydantic import BaseModel, Field

from tpu_kernel_gen.agents.kernel_gen_agent.agent import (
  CustomLlmAgent,
  add_pallas_docs,
  filter_code_parts_callback,
)
from tpu_kernel_gen.agents.kernel_gen_agent.constants import MODEL_NAME, TOP_K, TOP_P
from tpu_kernel_gen.agents.kernel_gen_agent.kernel_eval import (
  KernelCompilationChecker,
  KernelCorrectnessChecker,
  KernelPerformanceChecker,
)
from tpu_kernel_gen.agents.kernel_gen_agent.prompts import (
  gen_correctness_test,
  gen_performance_test,
)
from tpu_kernel_gen.agents.kernel_gen_agent.tools import search_api_tool
from tpu_kernel_gen.agents.open_optimization_agent.prompts import (
  idea_prompt,
  judge_prompt,
  summarize_evals_prompt,
  writer_prompt,
)

MAX_LOOP_ITERATIONS = 500
MAX_IDEA_LOOPS = 5

model_config_creative = types.GenerateContentConfig(
  temperature=0.5,
  top_p=TOP_P,
  top_k=TOP_K,
)

model_config_straightforward = types.GenerateContentConfig(
  temperature=0.1,
  top_p=TOP_P,
  top_k=TOP_K,
)


def add_starting_code(callback_context: CallbackContext):
  if callback_context.state.get("base_code", None) is not None:
    return

  user_content = callback_context.user_content
  code = user_content.parts[0].text.split("```")[1]
  logging.info(f"Defining base code as:\n{code}")
  callback_context.state["base_code"] = code
  callback_context.state["kernel_code"] = code


def iter_begining_callback(callback_context: CallbackContext):
  callback_context.state["skip_iter"] = False


def whether_to_skip(
  callback_context: CallbackContext,
  result_key: str = None,
) -> types.Content:
  """Check if the user has provided a base JAX code to skip the kernel generation."""
  if result_key is not None:
    result = callback_context.state.get(result_key, "")
    if result != "Success":
      callback_context.state["skip_iter"] = True
      logging.info(f"Skipping iteration due to result: {result}")

  if callback_context.state.get("skip_iter", False):
    return types.Content(
      parts=[
        Part(
          text="Skipping this agent due to previous agent's decision to skip.",
        )
      ]
    )


class Result(BaseModel):
  """Model for providing evaluation feedback on research quality."""

  result: Literal["good", "retry"] = Field(
    description="Judge's evaluation of the idea quality. 'good' means the idea is acceptable, 'retry' means the idea needs to be improved."
  )
  feedback: str = Field(description="Feedback from the judge on how to improve the idea if it is not good.")


class NeverExitLoopAgent(LoopAgent):
  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    times_looped = 0
    while not self.max_iterations or times_looped < self.max_iterations:
      for sub_agent in self.sub_agents:
        should_exit = False
        async with Aclosing(sub_agent.run_async(ctx)) as agen:
          async for event in agen:
            yield event

      times_looped += 1
    return


class JudgeAgent(CustomLlmAgent):
  """Agent that judges whether the idea is good or needs improvement."""

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    async with Aclosing(super()._run_async_impl(ctx)) as agen:
      async for event in agen:
        logging.info(f"[{self.name}] Event HERE: {event}")
        yield event
        if '"good"' in event.content.parts[0].text:
          yield Event(
            author=self.name,
            actions=EventActions(escalate=True),
          )


def clear_previous_plans(callback_context: CallbackContext):
  """Clear previous plans from the callback context."""
  if "idea" in callback_context.state:
    logging.info("Clearing previous plans from the callback context.")
    callback_context.state["idea"] = None
  if "judgement" in callback_context.state:
    logging.info("Clearing previous judgement from the callback context.")
    callback_context.state["judgement"] = None


idea_agent = CustomLlmAgent(
  name="IdeaAgent",
  model=MODEL_NAME,
  generate_content_config=model_config_creative,
  instruction=idea_prompt.PROMPT,
  description="Generates a Pallas kernel based on the user's request.",
  tools=[search_api_tool],
  output_key="idea",
  before_agent_callback=iter_begining_callback,
)

judge_agent = JudgeAgent(
  name="JudgeAgent",
  model=MODEL_NAME,
  generate_content_config=model_config_straightforward,
  instruction=judge_prompt.PROMPT,
  description="Judges whether the idea is good or needs improvement.",
  output_schema=Result,
  output_key="judgement",
  disallow_transfer_to_parent=True,
  disallow_transfer_to_peers=True,
  include_contents="none",
)

idea_generation_agent = LoopAgent(
  name="IdeaGenerationAgent",
  sub_agents=[
    idea_agent,
    judge_agent,
  ],
  before_agent_callback=clear_previous_plans,
  max_iterations=MAX_IDEA_LOOPS,
)

kernel_writer_agent = CustomLlmAgent(
  name="KernelWriterAgent",
  model=MODEL_NAME,
  generate_content_config=model_config_creative,
  instruction=writer_prompt.PROMPT,
  description="Writes a Pallas kernel based on the user's request.",
  tools=[search_api_tool],
  after_model_callback=filter_code_parts_callback,
  output_key="kernel_code",
)

eval_compilation_agent = KernelCompilationChecker(
  name="EvalCompilationAgent",
  input_key="kernel_code",
  output_key="compilation_results",
)


gen_correctness_test_agent = CustomLlmAgent(
  name="GenCorrectnessTestAgent",
  model=MODEL_NAME,
  generate_content_config=model_config_straightforward,
  instruction=gen_correctness_test.PROMPT.replace("{optimized_code}", "{kernel_code}"),
  description="Generates a correctness test for the Pallas kernel.",
  output_key="correctness_test_code",
  include_contents="none",
  before_agent_callback=partial(whether_to_skip, result_key="compilation_results"),
  after_model_callback=filter_code_parts_callback,
)

eval_correctness_agent = KernelCorrectnessChecker(
  name="EvalCorrectnessAgent",
  input_key="correctness_test_code",
  output_key="correctness_test_results",
  before_agent_callback=whether_to_skip,
  raise_exception_upon_success=False,
)

generate_performance_test_agent = CustomLlmAgent(
  name="GeneratePerformanceTestAgent",
  model=MODEL_NAME,
  generate_content_config=model_config_straightforward,
  instruction=gen_performance_test.PROMPT.replace("{jax_base_code}", "{base_code}"),
  description="Generates a performance test script for the Pallas kernel.",
  output_key="performance_test_script",
  include_contents="none",
  before_agent_callback=partial(
    whether_to_skip,
    result_key="correctness_test_results",
  ),
  after_model_callback=filter_code_parts_callback,
)

eval_performance_agent = KernelPerformanceChecker(
  name="EvalPerformanceAgent",
  input_key="performance_test_script",
  output_key="performance_test_results",
  before_agent_callback=whether_to_skip,
)

summarize_agent = CustomLlmAgent(
  name="SummarizeEvaluationAgent",
  model=MODEL_NAME,
  generate_content_config=model_config_straightforward,
  description="Summarizes the performance results of the kernel tuning.",
  output_key="eval_summary",
  include_contents="none",
  instruction=summarize_evals_prompt.PROMPT,
)

eval_agent = SequentialAgent(
  name="EvalPerformanceAgent",
  sub_agents=[
    eval_compilation_agent,
    gen_correctness_test_agent,
    eval_correctness_agent,
    generate_performance_test_agent,
    eval_performance_agent,
    summarize_agent,
  ],
)

root_agent = NeverExitLoopAgent(
  name="KernelOptimizerAgent",
  before_agent_callback=[add_pallas_docs, add_starting_code],
  sub_agents=[
    idea_generation_agent,
    kernel_writer_agent,
    eval_agent,
  ],
  max_iterations=MAX_LOOP_ITERATIONS,
)
