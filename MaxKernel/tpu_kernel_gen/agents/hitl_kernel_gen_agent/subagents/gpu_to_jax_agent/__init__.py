"""GPU to JAX conversion agent.

This agent converts GPU-optimized code (CUDA, Triton, PyTorch CUDA, CuDNN/cuBLAS)
to clean, algorithmic JAX code, stripping hardware-specific optimizations.
"""

from tpu_kernel_gen.agents.hitl_kernel_gen_agent.subagents.gpu_to_jax_agent.agent import (
  gpu_to_jax_agent,
)

__version__ = "1.0.0"
__all__ = ["gpu_to_jax_agent"]
