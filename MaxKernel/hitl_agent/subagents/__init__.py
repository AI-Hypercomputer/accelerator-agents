"""Subagents for HITL kernel generation.

This package contains all subagents organized by task:
- kernel_writing: Planning, implementation, and compilation validation
- testing: Test generation, validation, and execution
- profiling: Performance profiling and analysis
- explanation: Explanations for the kernel generation process
- gpu_to_jax: GPU-to-JAX code conversion
"""

from . import kernel_writing
from . import testing
from . import profiling
from . import explanation
from . import gpu_to_jax_agent

__all__ = [
    'kernel_writing',
    'testing',
    'profiling',
    'explanation',
    'gpu_to_jax_agent',
]
