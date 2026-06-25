"""Beam Search Worker Agent registration.

This module instantiates the correctness-only pipeline agent
used to generate and verify kernel candidates for Beam Search.
"""

from auto_agent.subagents.beam_worker_pipeline import BeamWorkerPipeline
from auto_agent.subagents.kernel_writing import (
  implement_kernel_agent,
  plan_kernel_agent,
  validate_kernel_compilation_agent,
)
from auto_agent.subagents.testing import (
  unified_test_agent,
  validated_test_generation_agent,
)

beam_worker_agent = BeamWorkerPipeline(
  name="BeamWorkerPipeline",
  plan_agent=plan_kernel_agent,
  implement_agent=implement_kernel_agent,
  validate_agent=validate_kernel_compilation_agent,
  test_gen_agent=validated_test_generation_agent,
  test_run_agent=unified_test_agent,
  max_iterations=5,
)

__all__ = ["beam_worker_agent"]
