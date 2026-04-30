"""Main orchestration agent for HITL kernel generation.

This module contains the root orchestrator that coordinates all subagents
for the human-in-the-loop kernel generation process.
"""

from hitl_agent.callbacks import (
  add_pallas_docs,
  add_workdir_callback,
  get_tpu_version_callback,
)
from hitl_agent.config import (
  model_config,
  thinking_planner,
)
from hitl_agent.constants import MODEL_NAME
from hitl_agent.custom_types import CustomLlmAgent
from hitl_agent.prompts import interactive_prompt
from hitl_agent.subagents.autotuning.agent import autotune_agent
from hitl_agent.subagents.explanation import explanation_agent
from hitl_agent.subagents.gpu_to_jax_agent.agent import (
  gpu_to_jax_agent,
)
from hitl_agent.subagents.kernel_writing import (
  implement_kernel_agent,
  plan_kernel_agent,
  validate_kernel_compilation_agent,
)
from hitl_agent.subagents.profiling import profile_agent
from hitl_agent.subagents.testing import (
  unified_test_agent,
  validated_test_generation_agent,
)
from hitl_agent.tools.tools import filesystem_tool_r

# Root orchestration agent
root_agent = CustomLlmAgent(
  name="KernelGenerationOrchestrationAgent",
  model=MODEL_NAME,
  generate_content_config=model_config,
  planner=thinking_planner,
  before_agent_callback=[
    add_pallas_docs,
    get_tpu_version_callback,
    add_workdir_callback,
  ],
  sub_agents=[
    explanation_agent,  # Provides explanations
    plan_kernel_agent,  # Step 1: Create/revise plan
    implement_kernel_agent,  # Step 2: Implement kernel
    validate_kernel_compilation_agent,  # Step 3: Validate compilation
    validated_test_generation_agent,  # Step 4: Generate and validate tests
    unified_test_agent,  # Step 5: Run tests and provide summary
    profile_agent,  # Step 6: Profile for bottlenecks
    gpu_to_jax_agent,  # GPU-to-JAX conversion
    autotune_agent,  # Step 7: Auto-tune kernel
  ],
  tools=[
    filesystem_tool_r
  ],  # Read-only access - orchestrator delegates writes to sub-agents
  instruction=interactive_prompt.PROMPT,
  description="Orchestrates the human-in-the-loop kernel generation process with GPU to JAX conversion capability.",
)

__all__ = ["root_agent"]
