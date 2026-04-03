"""Prompts for kernel writing subagent."""

from . import (
    kernel_planning_prompt,
    kernel_implementation_prompt,
    ask_validation_prompt,
    fix_kernel_compilation,
    kernel_compilation_summary,
    add_debug_statements,
    cleanup_debug_statements,
    summary_prompt,
    read_file_prompt,
)

__all__ = [
    'kernel_planning_prompt',
    'kernel_implementation_prompt',
    'ask_validation_prompt',
    'fix_kernel_compilation',
    'kernel_compilation_summary',
    'add_debug_statements',
    'cleanup_debug_statements',
    'summary_prompt',
    'read_file_prompt',
]
