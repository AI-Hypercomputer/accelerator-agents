"""Prompts for kernel writing subagent."""

from . import (
  add_debug_statements,
  ask_validation_prompt,
  cleanup_debug_statements,
  fix_kernel_compilation,
  kernel_compilation_summary,
  kernel_implementation_prompt,
  kernel_planning_prompt,
  read_file_prompt,
  summary_prompt,
)

__all__ = [
  "kernel_planning_prompt",
  "kernel_implementation_prompt",
  "ask_validation_prompt",
  "fix_kernel_compilation",
  "kernel_compilation_summary",
  "add_debug_statements",
  "cleanup_debug_statements",
  "summary_prompt",
  "read_file_prompt",
]
