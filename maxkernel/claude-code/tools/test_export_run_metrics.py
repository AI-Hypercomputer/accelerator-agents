#!/usr/bin/env python3
"""Unit tests for export_run_metrics.py."""

import json
import os
import shutil
import sys
import tempfile
import unittest

tools_dir = os.path.dirname(os.path.abspath(__file__))
if tools_dir not in sys.path:
  sys.path.insert(0, tools_dir)

from export_run_metrics import (
  extract_run_metrics,
  main,
  parse_state_file,
  parse_stream_tokens,
  RunMetricsSummary,
)


class TestExportRunMetrics(unittest.TestCase):
  """Tests parsing of state.json and claude_stream.json."""

  def setUp(self):
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def test_complete_run(self):
    state_data = {
      "run_id": "test_run_123",
      "iteration": 5,
      "best_speedup": 2.35,
      "base_time_ms": 10.5,
      "best_optimized_time": 4.468,
      "best_code_path": "/tmp/workspace/best.py",
      "stopping_recommendations": {
        "recommend_stop": True,
        "reason": "Max iterations reached",
      },
    }
    with open(
      os.path.join(self.test_dir, "state.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(state_data, f)

    stream_events = [
      {
        "type": "message_start",
        "message": {
          "usage": {
            "input_tokens": 1000,
            "cache_read_input_tokens": 300,
            "output_tokens": 50,
          }
        },
      },
      {"type": "message_delta", "usage": {"output_tokens": 150}},
      {
        "type": "message_start",
        "message": {
          "usage": {
            "input_tokens": 500,
            "cache_read_input_tokens": 200,
            "output_tokens": 25,
          }
        },
      },
      {"type": "message_delta", "usage": {"output_tokens": 75}},
    ]
    with open(
      os.path.join(self.test_dir, "claude_stream.json"), "w", encoding="utf-8"
    ) as f:
      for event in stream_events:
        f.write(json.dumps(event) + "\n")

    summary = extract_run_metrics(self.test_dir)
    self.assertEqual(summary.run_id, "test_run_123")
    self.assertEqual(summary.iteration_count, 5)
    self.assertAlmostEqual(summary.best_speedup, 2.35)
    self.assertAlmostEqual(summary.base_time_ms, 10.5)
    self.assertAlmostEqual(summary.best_optimized_time, 4.468)
    self.assertEqual(summary.best_code_path, "/tmp/workspace/best.py")
    self.assertTrue(summary.stopping_recommendations["recommend_stop"])
    self.assertEqual(summary.input_tokens, 1500)
    self.assertEqual(summary.output_tokens, 300)
    self.assertEqual(summary.cache_read_tokens, 500)
    self.assertEqual(summary.total_tokens, 2300)

  def test_single_iteration_run(self):
    state_data = {
      "run_id": "single_iter_run",
      "iteration": 1,
      "best_speedup": 1.15,
      "base_time_ms": 12.0,
      "best_optimized_time": 10.43,
      "best_code_path": "/tmp/workspace/iter1.py",
      "history": [{"iteration": 1, "speedup": 1.15}],
    }
    with open(
      os.path.join(self.test_dir, "state.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(state_data, f)

    # No stream file provided; token metrics should be None
    summary = extract_run_metrics(self.test_dir)
    self.assertEqual(summary.iteration_count, 1)
    self.assertAlmostEqual(summary.best_speedup, 1.15)
    self.assertIsNone(summary.input_tokens)
    self.assertIsNone(summary.output_tokens)
    self.assertIsNone(summary.cache_read_tokens)
    self.assertIsNone(summary.total_tokens)
    self.assertIsNone(summary.stopping_recommendations)

  def test_early_stopping(self):
    state_data = {
      "run_id": "early_stop_run",
      "iteration": 2,
      "best_speedup": 3.85,
      "base_time_ms": 20.0,
      "best_optimized_time": 5.19,
      "best_code_path": "/tmp/workspace/opt_iter2.py",
      "stopping": {
        "recommend_stop": True,
        "reason": "Target roofline speedup reached",
        "roofline_attained_pct": 92.5,
      },
    }
    with open(
      os.path.join(self.test_dir, "state.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(state_data, f)

    summary = extract_run_metrics(self.test_dir)
    self.assertEqual(summary.iteration_count, 2)
    self.assertEqual(
      summary.stopping_recommendations["reason"],
      "Target roofline speedup reached",
    )
    self.assertEqual(
      summary.stopping_recommendation["reason"],
      "Target roofline speedup reached",
    )

  def test_stream_tokens_with_result_event(self):
    stream_file = os.path.join(self.test_dir, "claude_stream.json")
    stream_events = [
      {
        "type": "message_start",
        "message": {"usage": {"input_tokens": 100, "output_tokens": 20}},
      },
      {
        "type": "result",
        "subtype": "success",
        "usage": {
          "input_tokens": 4500,
          "output_tokens": 1200,
          "cache_read_input_tokens": 800,
          "total_tokens": 6500,
        },
      },
    ]
    with open(stream_file, "w", encoding="utf-8") as f:
      for ev in stream_events:
        f.write(json.dumps(ev) + "\n")

    tokens = parse_stream_tokens(stream_file)
    self.assertEqual(tokens["input_tokens"], 4500)
    self.assertEqual(tokens["output_tokens"], 1200)
    self.assertEqual(tokens["cache_read_tokens"], 800)
    self.assertEqual(tokens["total_tokens"], 6500)

  def test_missing_state_file_fail_closed(self):
    nonexistent_dir = os.path.join(self.test_dir, "nonexistent")
    with self.assertRaises(FileNotFoundError):
      extract_run_metrics(nonexistent_dir)

  def test_invalid_json_state_file_fail_closed(self):
    state_file = os.path.join(self.test_dir, "state.json")
    with open(state_file, "w", encoding="utf-8") as f:
      f.write("{invalid json content")
    with self.assertRaises(ValueError):
      extract_run_metrics(self.test_dir)

  def test_empty_state_file_fail_closed(self):
    state_file = os.path.join(self.test_dir, "state.json")
    with open(state_file, "w", encoding="utf-8") as f:
      f.write("")
    with self.assertRaises(ValueError):
      extract_run_metrics(self.test_dir)

  def test_cli_execution_to_output_file(self):
    state_data = {
      "run_id": "cli_run_test",
      "iteration": 3,
      "best_speedup": 1.75,
      "base_time_ms": 15.0,
      "best_optimized_time": 8.57,
      "best_code_path": "/tmp/workspace/best.py",
    }
    with open(
      os.path.join(self.test_dir, "state.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(state_data, f)

    output_path = os.path.join(self.test_dir, "summary_metrics.json")
    orig_argv = sys.argv
    try:
      sys.argv = [
        "export_run_metrics.py",
        "--run_dir",
        self.test_dir,
        "--output",
        output_path,
      ]
      exit_code = main()
      self.assertEqual(exit_code, 0)
      self.assertTrue(os.path.exists(output_path))
      with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      self.assertEqual(data["run_id"], "cli_run_test")
      self.assertEqual(data["iteration_count"], 3)
      self.assertAlmostEqual(data["best_speedup"], 1.75)
    finally:
      sys.argv = orig_argv


if __name__ == "__main__":
  unittest.main()
