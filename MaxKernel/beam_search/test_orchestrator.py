"""Unit tests for AgenticSearchOrchestrator helper functions."""

import ast
import json
import os
import unittest
from unittest.mock import MagicMock, patch, mock_open
import pytest

from beam_search.orchestrator import AgenticSearchOrchestrator

from pathlib import Path

# Setup dummy paths for constructor instantiation
DUMMY_BASELINE_PATH = "dummy_baseline.py"
DUMMY_REFERENCE_PATH = "dummy_reference.py"
DUMMY_TASK_YAML_PATH = "dummy_task.yaml"
DUMMY_OUTPUT_DIR = Path("dummy_output")



@pytest.fixture
def mock_orchestrator():
  """Fixture to instantiate AgenticSearchOrchestrator with mocked dependencies."""
  with patch("builtins.open", mock_open(read_data="def computation(x): return x")):
    with patch("os.path.exists", return_value=True):
      with patch("pathlib.Path.mkdir"):
        orchestrator = AgenticSearchOrchestrator(
            baseline_code_path=DUMMY_BASELINE_PATH,
            reference_code_path=DUMMY_REFERENCE_PATH,
            task_yaml_path=DUMMY_TASK_YAML_PATH,
            output_dir=DUMMY_OUTPUT_DIR,
            mock_mode=True,  # Use fake evaluator internally
        )
        # Mock the internal evaluator completely for safety
        orchestrator.evaluator = MagicMock()
        return orchestrator


def test_apply_parent_regression_gate(mock_orchestrator):
  """Test Case 1: Verifies that regression gate correctly filters based on parent limits."""
  candidates = [
      {"path": "c0", "latency_ms": 10.0, "parent_latency_ms": 12.0},  # Improved
      {"path": "c1", "latency_ms": 13.0, "parent_latency_ms": 12.0},  # Regressed
      {"path": "c2", "latency_ms": 9.2, "parent_latency_ms": 9.5},    # Improved
      {"path": "c3", "latency_ms": 10.0, "parent_latency_ms": 9.5},   # Regressed
  ]

  # Scenario 1: keep_factor = 1.0 (strict)
  survivors_strict = mock_orchestrator._apply_parent_regression_gate(candidates, keep_factor=1.0)
  assert len(survivors_strict) == 2
  assert survivors_strict[0]["path"] == "c0"
  assert survivors_strict[1]["path"] == "c2"

  # Scenario 2: keep_factor = 1.1 (10% margin, limit for c1 is 13.2 ms, limit for c3 is 10.45 ms)
  survivors_loose = mock_orchestrator._apply_parent_regression_gate(candidates, keep_factor=1.1)
  assert len(survivors_loose) == 4


def test_deduplicate_candidates_ast(mock_orchestrator):
  """Test Case 2: Verifies AST deduplication preserves unique candidates."""
  candidates = [
      {"path": "c0", "code": "def computation(x):\n    return x + 1", "latency_ms": 10.0},
      {"path": "c1", "code": "def computation(x):\n    # Some comment\n    return x + 1", "latency_ms": 11.0}, # AST duplicate of c0
      {"path": "c2", "code": "def computation(x):\n    return x + 2", "latency_ms": 9.5}, # Unique
  ]

  deduped = mock_orchestrator._deduplicate_candidates_ast(candidates)
  assert len(deduped) == 2
  assert deduped[0]["path"] == "c0"
  assert deduped[1]["path"] == "c2"


@patch("os.path.exists")
@patch("builtins.open")
def test_evaluate_candidate_performance_normal(mock_open, mock_exists, mock_orchestrator):
  """Test Case 3: Verifies evaluator is called normally when no timing files exist."""
  mock_orchestrator.evaluator.evaluate = MagicMock()
  mock_orchestrator.evaluator.reference_time_ms = 11.4
  mock_orchestrator.evaluator.evaluate.return_value = MagicMock(optimized_time_ms=8.5)

  # Mock file existence: only evaluation_metrics.json exists (written by evaluator)
  # but NOT autotune_results.json (prior to eval)
  def side_effect_exists(path):
    if "autotune_results.json" in path:
      return False
    if "evaluation_metrics.json" in path:
      return True
    return True
  mock_exists.side_effect = side_effect_exists

  metrics_content = json.dumps({
      "estimated_hbm_utilization": 22.5,
      "estimated_compute_utilization": 14.8,
      "estimated_computation_density_flops_per_byte": 1.25,
      "performance_analysis": "Memory access pattern optimized"
  })

  def mock_open_helper(filename, *args, **kwargs):
    content = ""
    if "evaluation_metrics.json" in filename:
      content = metrics_content
    elif filename.endswith(".py"):
      content = "def optimized_code(): pass"
    m = MagicMock()
    m.__enter__.return_value.read.return_value = content
    return m
  mock_open.side_effect = mock_open_helper

  res = mock_orchestrator._evaluate_candidate_performance(
      path="dummy_candidate_path.py",
      parent_latency=9.5,
      fallback_analysis="Unit test run"
  )

  # Verify evaluate was called
  mock_orchestrator.evaluator.evaluate.assert_called_once()
  assert res["latency_ms"] == 8.5
  assert res["parent_latency_ms"] == 9.5
  assert res["hbm_utilization_pct"] == 22.5
  assert res["compute_utilization_pct"] == 14.8
  assert res["computation_density_flops_per_byte"] == 1.25
  assert res["analysis"] == "Memory access pattern optimized"
  assert res["code"] == "def optimized_code(): pass"


@patch("os.path.exists")
@patch("builtins.open")
def test_evaluate_candidate_performance_bypassed_autotune(mock_open, mock_exists, mock_orchestrator):
  """Test Case 3b: Verifies evaluator is bypassed when autotune_results.json exists with best_time_ms."""
  mock_orchestrator.evaluator.evaluate = MagicMock()

  # Mock file existence: autotune_results.json exists
  def side_effect_exists(path):
    if "autotune_results.json" in path:
      return True
    return False
  mock_exists.side_effect = side_effect_exists

  autotune_content = json.dumps({
      "best_time_ms": 6.02,
      "best_config": {"BLOCK_M": 64}
  })

  def mock_open_helper(filename, *args, **kwargs):
    content = ""
    if "autotune_results.json" in filename:
      content = autotune_content
    elif filename.endswith(".py"):
      content = "def optimized_code(): pass"
    m = MagicMock()
    m.__enter__.return_value.read.return_value = content
    return m
  mock_open.side_effect = mock_open_helper

  res = mock_orchestrator._evaluate_candidate_performance(
      path="dummy_candidate_path.py",
      parent_latency=9.5,
      fallback_analysis="Unit test run"
  )

  # Verify evaluate was NOT called
  mock_orchestrator.evaluator.evaluate.assert_not_called()
  assert res["latency_ms"] == 6.02
  assert res["parent_latency_ms"] == 9.5
  assert res["hbm_utilization_pct"] == 10.0  # Mocked metrics
  assert res["compute_utilization_pct"] == 2.0
  assert res["computation_density_flops_per_byte"] == 0.5
  assert "Latency: 6.020 ms" in res["analysis"]
  assert res["code"] == "def optimized_code(): pass"


@patch("os.path.exists")
@patch("builtins.open")
def test_evaluate_candidate_performance_bypassed_metrics(mock_open, mock_exists, mock_orchestrator):
  """Test Case 3c: Verifies evaluator is bypassed when evaluation_metrics.json contains parsed latency."""
  mock_orchestrator.evaluator.evaluate = MagicMock()

  # Mock file existence: autotune_results.json does NOT exist, but evaluation_metrics.json exists
  def side_effect_exists(path):
    if "autotune_results.json" in path:
      return False
    if "evaluation_metrics.json" in path:
      return True
    return False
  mock_exists.side_effect = side_effect_exists

  metrics_content = json.dumps({
      "estimated_hbm_utilization": 12.0,
      "estimated_compute_utilization": 3.0,
      "estimated_computation_density_flops_per_byte": 0.6,
      "performance_analysis": "Fallback decay latency: 7.185 ms"
  })

  def mock_open_helper(filename, *args, **kwargs):
    content = ""
    if "evaluation_metrics.json" in filename:
      content = metrics_content
    elif filename.endswith(".py"):
      content = "def optimized_code(): pass"
    m = MagicMock()
    m.__enter__.return_value.read.return_value = content
    return m
  mock_open.side_effect = mock_open_helper

  res = mock_orchestrator._evaluate_candidate_performance(
      path="dummy_candidate_path.py",
      parent_latency=9.5,
      fallback_analysis="Unit test run"
  )

  # Verify evaluate was NOT called
  mock_orchestrator.evaluator.evaluate.assert_not_called()
  assert res["latency_ms"] == 7.185
  assert res["parent_latency_ms"] == 9.5
  assert res["hbm_utilization_pct"] == 10.0  # Mocked defaults for mock mode bypassing
  assert res["compute_utilization_pct"] == 2.0
  assert res["computation_density_flops_per_byte"] == 0.5
  assert "Latency: 7.185 ms" in res["analysis"]
  assert res["code"] == "def optimized_code(): pass"


@patch("os.path.exists")
@patch("builtins.open")
def test_evaluate_candidate_performance_bypassed_prod_metrics(mock_open, mock_exists):
  """Test Case 3d: Verifies evaluator is bypassed in production mode and real metrics are loaded."""
  with patch("builtins.open", mock_open(read_data="def computation(x): return x")):
    with patch("os.path.exists", return_value=True):
      with patch("pathlib.Path.mkdir"):
        orchestrator = AgenticSearchOrchestrator(
            baseline_code_path=DUMMY_BASELINE_PATH,
            reference_code_path=DUMMY_REFERENCE_PATH,
            task_yaml_path=DUMMY_TASK_YAML_PATH,
            output_dir=DUMMY_OUTPUT_DIR,
            mock_mode=False,  # Production mode
        )
        orchestrator.evaluator = MagicMock()

  # Mock file existence: autotune_results.json exists, and evaluation_metrics.json exists
  def side_effect_exists(path):
    if "autotune_results.json" in path:
      return True
    if "evaluation_metrics.json" in path:
      return True
    return False
  mock_exists.side_effect = side_effect_exists

  autotune_content = json.dumps({
      "best_time_ms": 5.4,
      "best_config": {"BLOCK_M": 128}
  })
  metrics_content = json.dumps({
      "estimated_hbm_utilization": 45.0,
      "estimated_compute_utilization": 15.0,
      "estimated_computation_density_flops_per_byte": 2.5,
      "performance_analysis": "Real GPU profiling results"
  })

  def mock_open_helper(filename, *args, **kwargs):
    content = ""
    if "autotune_results.json" in filename:
      content = autotune_content
    elif "evaluation_metrics.json" in filename:
      content = metrics_content
    elif filename.endswith(".py"):
      content = "def optimized_code(): pass"
    m = MagicMock()
    m.__enter__.return_value.read.return_value = content
    return m
  mock_open.side_effect = mock_open_helper

  res = orchestrator._evaluate_candidate_performance(
      path="dummy_candidate_path.py",
      parent_latency=9.5,
      fallback_analysis="Unit test run"
  )

  # Verify evaluate was NOT called
  orchestrator.evaluator.evaluate.assert_not_called()
  assert res["latency_ms"] == 5.4
  assert res["parent_latency_ms"] == 9.5
  assert res["hbm_utilization_pct"] == 45.0
  assert res["compute_utilization_pct"] == 15.0
  assert res["computation_density_flops_per_byte"] == 2.5
  assert res["analysis"] == "Real GPU profiling results"
  assert res["code"] == "def optimized_code(): pass"


def test_run_sequential_fallback_evaluation(mock_orchestrator):
  """Test Case 4: Verifies sequential evaluations over a list of results."""
  valid_results = [
      {"path": "p0", "parent_latency_ms": 11.4},
      {"path": "p1", "parent_latency_ms": 10.0}
  ]

  mock_metrics = {
      "latency_ms": 9.2,
      "parent_latency_ms": 11.4,
      "code": "def code(): pass",
      "hbm_utilization_pct": 10.0,
      "compute_utilization_pct": 5.0,
      "computation_density_flops_per_byte": 0.5,
      "analysis": "fallback"
  }

  with patch.object(mock_orchestrator, "_evaluate_candidate_performance", return_value=mock_metrics) as mock_eval:
    res = mock_orchestrator._run_sequential_fallback_evaluation(valid_results)
    assert len(res) == 2
    assert res[0]["latency_ms"] == 9.2
    assert mock_eval.call_count == 2


def test_prepare_worker_tasks(mock_orchestrator):
  """Test Case 5: Verifies tasks prepare correctly and round-robin allocations are mapped correctly."""
  beam = [
      {"code": "optimized_code_a", "latency_ms": 9.5},  # Rank 1
      {"code": "optimized_code_b", "latency_ms": 10.2}  # Rank 2
  ]

  mock_orchestrator.run_worker_session = MagicMock(return_value="mock_coroutine")

  tasks = mock_orchestrator._prepare_worker_tasks(
      round_idx=2,
      beam=beam,
      beam_size=4,
      dropout_menu_options=0.5
  )

  assert len(tasks) == 4
  assert tasks == ["mock_coroutine"] * 4

  # Extract arguments passed to run_worker_session mock calls
  call_args = mock_orchestrator.run_worker_session.call_args_list
  assert len(call_args) == 4

  # Verify round-robin parent candidate allocation
  # Task 0 -> Candidate A (idx % 2 == 0)
  assert call_args[0].kwargs["base_code"] == "optimized_code_a"
  assert call_args[0].kwargs["parent_latency"] == 9.5
  # Task 1 -> Candidate B (idx % 2 == 1)
  assert call_args[1].kwargs["base_code"] == "optimized_code_b"
  assert call_args[1].kwargs["parent_latency"] == 10.2
  # Task 2 -> Candidate A (idx % 2 == 0)
  assert call_args[2].kwargs["base_code"] == "optimized_code_a"
  assert call_args[2].kwargs["parent_latency"] == 9.5
  # Task 3 -> Candidate B (idx % 2 == 1)
  assert call_args[3].kwargs["base_code"] == "optimized_code_b"
  assert call_args[3].kwargs["parent_latency"] == 10.2


def test_beam_sorting_and_pruning(mock_orchestrator):
  """Test Case 6: Verifies beam candidates are sorted (lowest latency first) and pruned to beam_size."""
  candidates = [
      {"path": "p0", "latency_ms": 12.1},
      {"path": "p1", "latency_ms": 9.8},
      {"path": "p2", "latency_ms": 11.5},
      {"path": "p3", "latency_ms": 8.4},
      {"path": "p4", "latency_ms": 10.0},
  ]

  # Sort by latency
  candidates.sort(key=lambda x: x["latency_ms"])
  # Prune to beam_size = 3
  beam = candidates[:3]

  assert len(beam) == 3
  assert beam[0]["latency_ms"] == 8.4
  assert beam[1]["latency_ms"] == 9.8
  assert beam[2]["latency_ms"] == 10.0


def test_incumbent_preservation_on_total_regression(mock_orchestrator):
  """Test Case 7: Verifies that incumbents are preserved and carried forward when new round candidates regress."""
  incumbents = [{"path": "p0", "latency_ms": 9.5, "code": "code_p0"}]
  
  # New candidates all regress compared to parent
  new_candidates = [
      {"path": "c0", "latency_ms": 11.5, "parent_latency_ms": 9.5}
  ]

  # Apply parent regression gate (strict 1.0)
  survivors = mock_orchestrator._apply_parent_regression_gate(new_candidates, keep_factor=1.0)
  assert len(survivors) == 0  # c0 is pruned

  # Merging survivors with incumbents
  all_candidates = incumbents + survivors
  assert len(all_candidates) == 1
  assert all_candidates[0]["path"] == "p0"
  assert all_candidates[0]["latency_ms"] == 9.5
