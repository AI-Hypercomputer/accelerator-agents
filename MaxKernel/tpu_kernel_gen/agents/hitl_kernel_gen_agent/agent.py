"""Main orchestration agent for HITL kernel generation.

This module contains the root orchestrator that coordinates all subagents
for the human-in-the-loop kernel generation process.
"""

from google.adk.models.anthropic_llm import Claude
from google.adk.models.registry import LLMRegistry

LLMRegistry.register(Claude)

from tpu_kernel_gen.agents.hitl_kernel_gen_agent.callbacks import (
  add_workdir_callback,
  get_tpu_version_callback,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.config import (
  model_config,
  thinking_planner,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.prompts import interactive_prompt
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.explanation import explanation_agent
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.gpu_to_jax_agent.agent import (
  gpu_to_jax_agent,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.kernel_writing import (
  implement_kernel_agent,
  plan_kernel_agent,
  validate_kernel_compilation_agent,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.profiling import profile_agent
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.testing import (
  unified_test_agent,
  validated_test_generation_agent,
)
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.tools import filesystem_tool_r
from tpu_kernel_gen.agents.kernel_gen_agent.agent import (
  CustomLlmAgent,
  add_pallas_docs,
)
from tpu_kernel_gen.agents.kernel_gen_agent.constants import MODEL_NAME

# Root orchestration agent
root_agent = CustomLlmAgent(
  name="KernelGenerationOrchestrationAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  before_agent_callback=[add_pallas_docs, get_tpu_version_callback, add_workdir_callback],
  sub_agents=[
    explanation_agent,  # Provides explanations
    plan_kernel_agent,  # Step 1: Create/revise plan
    implement_kernel_agent,  # Step 2: Implement kernel
    validate_kernel_compilation_agent,  # Step 3: Validate compilation
    validated_test_generation_agent,  # Step 4: Generate and validate tests
    unified_test_agent,  # Step 5: Run tests and provide summary
    profile_agent,  # Step 6: Profile for bottlenecks
    gpu_to_jax_agent,  # GPU-to-JAX conversion
  ],
  tools=[filesystem_tool_r],  # Read-only access - orchestrator delegates writes to sub-agents
  instruction=interactive_prompt.PROMPT,
  description="Orchestrates the human-in-the-loop kernel generation process with GPU to JAX conversion capability.",
)


__all__ = ["root_agent"]
