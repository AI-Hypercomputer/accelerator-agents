"""Smoke tests validating imports and dependency cycles of core modules."""

import importlib
from typing import List

import pytest

CORE_MODULES: List[str] = [
  "auto_search.graph",
  "auto_search.orchestrator",
  "auto_search.worker",
  "auto_search.run_search",
  "auto_search.algorithms.beam_search",
  "auto_search.algorithms.parallel_search",
  "auto_search.utils.analyze_timing",
  "auto_search.utils.parse_search_distribution",
  "auto_agent.agent_client.auto_agent_client",
  "auto_agent.agent_client.run_batch_agent_call",
  "auto_agent.config",
  "auto_agent.subagents.pipeline_agent",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_import(module_name: str):
  """Verify that each core module imports cleanly without circular dependencies or syntax errors."""
  try:
    mod = importlib.import_module(module_name)
    assert mod is not None, f"Module {module_name} resolved to None."
  except Exception as e:
    pytest.fail(f"Failed to import core module '{module_name}': {e}")
