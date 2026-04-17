import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

from evaluation.custom_types.evaluation_result import EvaluationResult
from evaluation.custom_types.kernel_task import KernelTask
from evaluation.evaluation_utils import (
  load_kernel_task_from_yaml,
  print_eval_result,
  summarize_results,
  write_kernel_task_to_yaml,
)


class TestEvaluationUtils(unittest.TestCase):
  @patch("builtins.open", new_callable=mock_open)
  def test_write_kernel_task_to_yaml(self, mock_file):
    task = KernelTask(
      task_id="test_task_1",
      description="A test description.",
      input_gen_code="def get_inputs():\n    return [], []\n",
    )

    write_kernel_task_to_yaml(task, "dummy_path.yaml")

    mock_file.assert_called_once_with("dummy_path.yaml", "w", encoding="utf-8")
    handle = mock_file()
    # Assert that write was called; PyYAML makes multiple write calls,
    # so we can combine them to check the final output string
    written_content = "".join(
      call.args[0] for call in handle.write.call_args_list
    )
    self.assertIn("test_task_1", written_content)

  @patch("evaluation.evaluation_utils.os.path.exists", return_value=True)
  def test_load_kernel_task_from_yaml(self, mock_exists):
    mock_yaml_content = "task_id: test_task_1\ndescription: A test description.\ninput_gen_code: |\n  def get_inputs():\n      return [], []\n"
    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
      loaded_task = load_kernel_task_from_yaml("dummy_path.yaml")

    self.assertEqual(loaded_task.task_id, "test_task_1")
    self.assertIn("def get_inputs():", loaded_task.input_gen_code)

  def test_load_missing_yaml(self):
    with self.assertRaises(FileNotFoundError):
      load_kernel_task_from_yaml("non_existent_file.yaml")

  def test_print_eval_result_success(self):
    result = EvaluationResult(
      task_id="test_1",
      compiled_successfully=True,
      numerically_correct=True,
      max_abs_diff=1.0e-3,
      max_rel_diff=1.0e-3,
      reference_time_ms=10.0,
      optimized_time_ms=5.0,
      error_trace=None,
    )

    f = io.StringIO()
    with redirect_stdout(f):
      print_eval_result(result)

    output = f.getvalue()
    self.assertIn("Correctness:    [PASS]", output)
    self.assertIn("Max Absolute Difference: 1.000000e-03", output)
    self.assertIn("Max Relative Difference: 1.000000e-03", output)
    self.assertIn("Reference time:       10.000 ms", output)
    self.assertIn("Optimized time:       5.000 ms", output)
    self.assertIn("Speedup:        2.00x", output)

  def test_print_eval_result_error(self):
    result = EvaluationResult(
      task_id="test_2", error_trace="Compilation failed due to syntax error."
    )

    f = io.StringIO()
    with redirect_stdout(f):
      print_eval_result(result)

    output = f.getvalue()
    self.assertIn(
      "Error:          Compilation failed due to syntax error.", output
    )

  @patch("evaluation.evaluation_utils.logger.info")
  def test_summarize_results(self, mock_logger_info):
    results = [
      {"task_id": "1", "compiled_successfully": False},
      {
        "task_id": "2",
        "compiled_successfully": True,
        "numerically_correct": False,
      },
      {
        "task_id": "3",
        "compiled_successfully": True,
        "numerically_correct": True,
        "speedup": 2.0,
      },
      {
        "task_id": "4",
        "compiled_successfully": True,
        "numerically_correct": True,
        "speedup": 0.5,
      },
      {
        "task_id": "5",
        "compiled_successfully": True,
        "numerically_correct": True,
        "speedup": 8.0,
      },
    ]

    summarize_results(results)

    log_messages = [call.args[0] for call in mock_logger_info.call_args_list]
    log_text = "\n".join(log_messages)

    self.assertIn("- Attempted / Evaluated:     5", log_text)
    self.assertIn("- Compiled Successfully: 4 (80.00%)", log_text)
    self.assertIn("- Failed Compilation/Runtime: 1", log_text)
    self.assertIn("- Numerically Correct: 3 (60.00%)", log_text)
    self.assertIn("- Improved (Speedup > 1x): 2 (40.00%)", log_text)
    self.assertIn("- Slower (Speedup < 1x):   1 (20.00%)", log_text)
    self.assertIn("- Max Speedup:             8.00x", log_text)
    self.assertIn("- Average Speedup/Slowdown (Arithmetic): 3.50x", log_text)
    self.assertIn("- Average Speedup/Slowdown (Geometric):  2.00x", log_text)

  @patch("evaluation.evaluation_utils.logger.info")
  def test_summarize_results_empty(self, mock_logger_info):
    summarize_results([])
    mock_logger_info.assert_called_once_with(
      "No results found to summarize and no tasks discovered."
    )

  @patch("evaluation.evaluation_utils.os.makedirs")
  @patch("builtins.open", new_callable=mock_open)
  @patch("evaluation.evaluation_utils.logger.info")
  def test_summarize_results_with_output_dir(
    self, mock_logger_info, mock_file, mock_makedirs
  ):
    results = [
      {"task_id": "1", "compiled_successfully": False},
      {
        "task_id": "2",
        "compiled_successfully": True,
        "numerically_correct": False,
      },
      {
        "task_id": "3",
        "compiled_successfully": True,
        "numerically_correct": True,
        "speedup": 2.0,
      },
      {
        "task_id": "4",
        "compiled_successfully": True,
        "numerically_correct": True,
        "speedup": 0.5,
      },
      {
        "task_id": "5",
        "compiled_successfully": True,
        "numerically_correct": True,
        "speedup": 8.0,
      },
    ]

    summarize_results(results, output_dir="/fake/dir")

    mock_makedirs.assert_called_once_with("/fake/dir", exist_ok=True)
    mock_file.assert_any_call(
      os.path.join("/fake/dir", "summary.txt"), "w", encoding="utf-8"
    )
    mock_file.assert_any_call(
      os.path.join("/fake/dir", "summary.json"), "w", encoding="utf-8"
    )

    handle = mock_file()
    written_content = "".join(
      call.args[0] for call in handle.write.call_args_list
    )
    self.assertIn('"total_attempted": 5', written_content)
    self.assertIn('"compiled_successfully": 4', written_content)
    self.assertIn('"numerically_correct": 3', written_content)
    self.assertIn('"improved": 2', written_content)
