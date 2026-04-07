"""Root-level prompts for the HITL kernel generation agent.

This module only contains the interactive prompt used by the root orchestrator.
All other prompts have been moved to their respective subagent directories.
"""

from . import interactive_prompt

__all__ = ["interactive_prompt"]
