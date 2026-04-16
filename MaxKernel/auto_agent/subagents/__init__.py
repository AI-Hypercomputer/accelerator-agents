"""Subagents for autonomous kernel generation.

This package contains all subagents organized by task:
- kernel_writing: Planning, implementation, and compilation validation
- testing: Test generation, validation, and execution
- profiling: Performance profiling and analysis
- explanation: Explanations for the kernel generation process
"""

from . import explanation, kernel_writing, profiling, testing

__all__ = [
  "kernel_writing",
  "testing",
  "profiling",
  "explanation",
]
