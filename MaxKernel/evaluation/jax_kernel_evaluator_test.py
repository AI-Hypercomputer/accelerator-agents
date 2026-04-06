import json
import subprocess
import unittest
from unittest.mock import MagicMock, mock_open, patch

from evaluation.custom_types.kernel_task import KernelTask
from evaluation.jax_kernel_evaluator import JAXKernelEvaluator


class TestJAXKernelEvaluator(unittest.TestCase):
  @patch("evaluation.jax_kernel_evaluator.TPUVMClient")
  def test_init_remote_success(self, mock_client):
    evaluator = JAXKernelEvaluator(
      local=False,
      tpu_name="test-tpu",
      project="test-proj",
      zone="us-central1-a",
    )
    self.assertFalse(evaluator.local)
    mock_client.assert_called_once_with(
      project="test-proj", zone="us-central1-a", tpu_name="test-tpu"
    )

  def test_init_remote_missing_args(self):
    with self.assertRaises(ValueError):
      JAXKernelEvaluator(local=False)

  def test_init_local(self):
    evaluator = JAXKernelEvaluator(local=True)
    self.assertTrue(evaluator.local)
    self.assertIsNone(evaluator.client)

  @patch.object(JAXKernelEvaluator, "_evaluate_local")
  def test_evaluate_routing_local(self, mock_eval_local):
    evaluator = JAXKernelEvaluator(local=True)
    evaluator.evaluate("ref.py", "opt.py", "task.yaml")
    mock_eval_local.assert_called_once_with(
      "ref.py", "opt.py", "task.yaml", None, 300, True, 1e-3, 1e-3
    )

  @patch.object(JAXKernelEvaluator, "_evaluate_remote")
  @patch("evaluation.jax_kernel_evaluator.TPUVMClient")
  def test_evaluate_routing_remote(self, mock_client, mock_eval_remote):
    evaluator = JAXKernelEvaluator(
      local=False, tpu_name="t", project="p", zone="z"
    )
    evaluator.evaluate("ref.py", "opt.py", "task.yaml")
    mock_eval_remote.assert_called_once_with(
      "ref.py", "opt.py", "task.yaml", None, 300, True, 1e-3, 1e-3
    )

  @patch("evaluation.jax_kernel_evaluator.load_kernel_task_from_yaml")
  @patch("evaluation.jax_kernel_evaluator.shutil")
  @patch("evaluation.jax_kernel_evaluator.tempfile")
  @patch("evaluation.jax_kernel_evaluator.subprocess.run")
  @patch("os.path.exists")
  @patch("evaluation.jax_kernel_evaluator.print_eval_result")
  def test_evaluate_local_success(
    self,
    mock_print,
    mock_exists,
    mock_subproc,
    mock_tempfile,
    mock_shutil,
    mock_load_task,
  ):
    evaluator = JAXKernelEvaluator(local=True)
    mock_load_task.return_value = KernelTask(
      task_id="test_task", description="", input_gen_code=""
    )
    mock_tempfile.mkdtemp.return_value = "/fake/dir"
    mock_exists.return_value = True  # for os.path.exists(result_local)

    result_data = {
      "compiled_successfully": True,
      "numerically_correct": True,
      "reference_time_ms": 10.0,
      "optimized_time_ms": 5.0,
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(result_data))):
      with (
        patch.object(evaluator, "_build_task_json"),
        patch.object(evaluator, "_build_harness_code"),
      ):
        result = evaluator._evaluate_local("ref.py", "opt.py", "task.yaml")

    self.assertTrue(result.compiled_successfully)
    self.assertTrue(result.numerically_correct)
    self.assertEqual(result.speedup, 2.0)
    mock_subproc.assert_called_once()
    mock_shutil.rmtree.assert_called_once_with("/fake/dir", ignore_errors=True)

  @patch("evaluation.jax_kernel_evaluator.load_kernel_task_from_yaml")
  @patch("evaluation.jax_kernel_evaluator.shutil")
  @patch("evaluation.jax_kernel_evaluator.tempfile")
  @patch("evaluation.jax_kernel_evaluator.subprocess.run")
  def test_evaluate_local_timeout(
    self, mock_subproc, mock_tempfile, mock_shutil, mock_load_task
  ):
    evaluator = JAXKernelEvaluator(local=True)
    mock_load_task.return_value = KernelTask(
      task_id="test_task", description="", input_gen_code=""
    )
    mock_tempfile.mkdtemp.return_value = "/fake/dir"
    mock_subproc.side_effect = subprocess.TimeoutExpired(
      cmd="python3 harness.py", timeout=300
    )

    with (
      patch("builtins.open", mock_open()),
      patch.object(evaluator, "_build_task_json"),
      patch.object(evaluator, "_build_harness_code"),
    ):
      result = evaluator._evaluate_local(
        "ref.py", "opt.py", "task.yaml", timeout_seconds=300
      )

    self.assertIn("timed out", result.error_trace)

  @patch("evaluation.jax_kernel_evaluator.load_kernel_task_from_yaml")
  @patch("evaluation.jax_kernel_evaluator.shutil")
  @patch("evaluation.jax_kernel_evaluator.tempfile")
  @patch("evaluation.jax_kernel_evaluator.subprocess.run")
  def test_evaluate_local_crash(
    self, mock_subproc, mock_tempfile, mock_shutil, mock_load_task
  ):
    evaluator = JAXKernelEvaluator(local=True)
    mock_load_task.return_value = KernelTask(
      task_id="test_task", description="", input_gen_code=""
    )
    mock_tempfile.mkdtemp.return_value = "/fake/dir"
    mock_subproc.side_effect = subprocess.CalledProcessError(
      returncode=-11, cmd="python3 harness.py", stderr="Segmentation fault"
    )

    with (
      patch("builtins.open", mock_open()),
      patch.object(evaluator, "_build_task_json"),
      patch.object(evaluator, "_build_harness_code"),
    ):
      result = evaluator._evaluate_local("ref.py", "opt.py", "task.yaml")

    self.assertIn("Command failed with exit code -11", result.error_trace)

  def test_evaluate_local_missing_task_yaml(self):
    evaluator = JAXKernelEvaluator(local=True)

    # Test with adapt but missing task output
    with patch.object(evaluator, "_adapt_inputs") as mock_adapt:
      mock_adapt.return_value = ("ref.py", "opt.py", None)
      with self.assertRaises(ValueError) as context:
        evaluator._evaluate_local(
          "ref.py", "opt.py", task_yaml_path=None, adapt=["reference_code"]
        )
      self.assertIn("task_yaml_path is required", str(context.exception))

    # Test without adapt
    with self.assertRaises(ValueError) as context:
      evaluator._evaluate_local("ref.py", "opt.py", task_yaml_path=None)
    self.assertIn("task_yaml_path is required", str(context.exception))

  @patch("evaluation.jax_kernel_evaluator.load_kernel_task_from_yaml")
  @patch("evaluation.jax_kernel_evaluator.tempfile")
  @patch("evaluation.jax_kernel_evaluator.os")
  @patch("evaluation.jax_kernel_evaluator.print_eval_result")
  @patch("evaluation.jax_kernel_evaluator.TPUVMClient")
  def test_evaluate_remote_success(
    self, mock_client_cls, mock_print, mock_os, mock_tempfile, mock_load_task
  ):
    mock_client_instance = mock_client_cls.return_value
    mock_client_instance.tpu_name = "test-tpu"

    mock_cat_result = MagicMock()
    mock_cat_result.stdout = '{"compiled_successfully": true, "numerically_correct": true, "reference_time_ms": 15.0, "optimized_time_ms": 10.0}'
    mock_client_instance.execute_ssh_command.side_effect = [
      MagicMock(),  # mkdir
      MagicMock(),  # run script
      mock_cat_result,  # cat result.json
      MagicMock(),  # rm -rf (cleanup)
    ]

    evaluator = JAXKernelEvaluator(
      local=False, tpu_name="t", project="p", zone="z"
    )
    evaluator.client = mock_client_instance

    mock_load_task.return_value = KernelTask(
      task_id="test_task", description="", input_gen_code=""
    )
    mock_tempfile.mkstemp.side_effect = [
      (1, "/tmp/harness.py"),
      (2, "/tmp/task.json"),
    ]
    mock_os.path.exists.return_value = True

    with (
      patch.object(evaluator, "_build_task_json"),
      patch.object(evaluator, "_build_harness_code"),
    ):
      result = evaluator._evaluate_remote("ref.py", "opt.py", "task.yaml")

    self.assertTrue(result.compiled_successfully)
    self.assertTrue(result.numerically_correct)
    self.assertEqual(result.speedup, 1.5)
    self.assertEqual(mock_client_instance.execute_ssh_command.call_count, 4)
    mock_client_instance.upload_file.assert_called()

  @patch("evaluation.jax_kernel_evaluator.load_kernel_task_from_yaml")
  @patch("evaluation.jax_kernel_evaluator.tempfile")
  @patch("evaluation.jax_kernel_evaluator.os")
  @patch("evaluation.jax_kernel_evaluator.print_eval_result")
  @patch("evaluation.jax_kernel_evaluator.TPUVMClient")
  def test_evaluate_remote_timeout(
    self, mock_client_cls, mock_print, mock_os, mock_tempfile, mock_load_task
  ):
    mock_client_instance = mock_client_cls.return_value
    mock_client_instance.tpu_name = "test-tpu"

    def side_effect(*args, **kwargs):
      if "harness.py" in args[0] and "pkill" not in args[0]:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=300)
      return MagicMock()

    mock_client_instance.execute_ssh_command.side_effect = side_effect

    evaluator = JAXKernelEvaluator(
      local=False, tpu_name="t", project="p", zone="z"
    )
    evaluator.client = mock_client_instance

    mock_load_task.return_value = KernelTask(
      task_id="test_task", description="", input_gen_code=""
    )
    mock_tempfile.mkstemp.side_effect = [(1, "/tmp/h.py"), (2, "/tmp/t.json")]

    with (
      patch.object(evaluator, "_build_task_json"),
      patch.object(evaluator, "_build_harness_code"),
      patch("jax_kernel_evaluator.time.sleep"),
    ):  # Mock sleep to avoid delay
      result = evaluator._evaluate_remote("ref.py", "opt.py", "task.yaml")

    self.assertIn("timed out", result.error_trace)
    # Verify pkill was called to cleanup the runaway process
    pkill_calls = [
      call
      for call in mock_client_instance.execute_ssh_command.call_args_list
      if "pkill" in call.args[0]
    ]
    self.assertTrue(len(pkill_calls) > 0)

  @patch("evaluation.jax_kernel_evaluator.load_kernel_task_from_yaml")
  @patch("evaluation.jax_kernel_evaluator.tempfile")
  @patch("evaluation.jax_kernel_evaluator.os")
  @patch("evaluation.jax_kernel_evaluator.print_eval_result")
  @patch("evaluation.jax_kernel_evaluator.TPUVMClient")
  def test_evaluate_remote_runtime_error(
    self, mock_client_cls, mock_print, mock_os, mock_tempfile, mock_load_task
  ):
    mock_client_instance = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
      if "harness.py" in args[0]:
        raise RuntimeError("Segmentation fault (core dumped)")
      return MagicMock()

    mock_client_instance.execute_ssh_command.side_effect = side_effect
    evaluator = JAXKernelEvaluator(
      local=False, tpu_name="t", project="p", zone="z"
    )
    evaluator.client = mock_client_instance
    mock_load_task.return_value = KernelTask(
      task_id="test_task", description="", input_gen_code=""
    )
    mock_tempfile.mkstemp.side_effect = [(1, "/tmp/h.py"), (2, "/tmp/t.json")]

    with (
      patch.object(evaluator, "_build_task_json"),
      patch.object(evaluator, "_build_harness_code"),
    ):
      result = evaluator._evaluate_remote("ref.py", "opt.py", "task.yaml")

    self.assertIn("Segmentation fault", result.error_trace)
