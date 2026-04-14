"""A Evaluator for JAX kernels on remote TPU VMs, leveraging the TPUVMClient for orchestration."""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

from google import genai

from evaluation.code_adapter import code_adapter
from evaluation.custom_types.evaluation_result import EvaluationResult
from evaluation.custom_types.kernel_task import KernelTask
from evaluation.evaluation_utils import (
  load_kernel_task_from_yaml,
  print_eval_result,
  write_kernel_task_to_yaml,
)
from evaluation.harness_code import HARNESS_TEMPLATE
from evaluation.remote_client.tpu_client import TPUVMClient

logger = logging.getLogger(__name__)  # Get a logger for this module


class JAXKernelEvaluator:
  """Orchestrates the evaluation of JAX kernels on a remote TPU VM."""

  def __init__(
    self,
    local: bool = False,
    tpu_name: Optional[str] = None,
    project: Optional[str] = None,
    zone: Optional[str] = None,
    venv_path: Optional[str] = None,
    remote_base_dir: Optional[str] = None,
  ):
    """Initializes the JAXKernelEvaluator.

    Args:
        local: If True, runs evaluation locally instead of on a remote TPU VM.
        tpu_name: Name of the TPU VM to connect to for remote evaluation.
        project: Google Cloud project ID.
        zone: Google Cloud zone.
        venv_path: Path to the Python virtual environment locally or on the remote TPU.
        remote_base_dir: Base directory on the remote TPU for storing temporary files.
                        If None, a unique directory will be created for each evaluation
                        run.
    """
    if (not local) and (not tpu_name or not project or not zone):
      raise ValueError(
        "tpu_name, project, and zone must be provided for remote evaluation."
      )
    self.local = local
    if not self.local:
      self.client = TPUVMClient(project=project, zone=zone, tpu_name=tpu_name)
    else:
      self.client = None
    self.venv_path = venv_path
    self.remote_base_dir = remote_base_dir
    self.adapter = None

  def evaluate(
    self,
    reference_code_path: str,
    optimized_code_path: str,
    task_yaml_path: Optional[str] = None,
    adapt: Optional[List[str]] = None,
    timeout_seconds: int = 300,  # Default timeout of 5 minutes
    cleanup: bool = True,
    atol: float = 1e-3,
    rtol: float = 1e-3,
  ) -> EvaluationResult:
    """Runs the evaluation pipeline for a given reference and kernel script."""
    if self.local:
      return self._evaluate_local(
        reference_code_path,
        optimized_code_path,
        task_yaml_path,
        adapt,
        timeout_seconds,
        cleanup,
        atol,
        rtol,
      )
    else:
      return self._evaluate_remote(
        reference_code_path,
        optimized_code_path,
        task_yaml_path,
        adapt,
        timeout_seconds,
        cleanup,
        atol,
        rtol,
      )

  def _evaluate_local(
    self,
    reference_code_path: str,
    optimized_code_path: str,
    task_yaml_path: Optional[str] = None,
    adapt: Optional[List[str]] = None,
    timeout_seconds: int = 300,
    cleanup: bool = True,
    atol: float = 1e-3,
    rtol: float = 1e-3,
  ) -> EvaluationResult:
    """Runs the evaluation pipeline locally on the current machine."""
    # Adapt the provided code and generate kernel task if necessary
    if adapt:
      reference_code_path, optimized_code_path, task_yaml_path = (
        self._adapt_inputs(
          reference_code_path, optimized_code_path, adapt, task_yaml_path
        )
      )

    if not task_yaml_path:
      raise ValueError(
        "task_yaml_path is required when 'kernel_task' is not in adapt"
      )

    task = load_kernel_task_from_yaml(task_yaml_path)
    eval_result = EvaluationResult(task_id=task.task_id)

    # Create a temporary directory for the local evaluation run
    local_base_dir = tempfile.mkdtemp(prefix=f"eval_run_{int(time.time())}_")

    try:
      # Prepare file paths
      task_json_local = os.path.join(local_base_dir, "task.json")
      harness_local = os.path.join(local_base_dir, "harness.py")
      ref_local = os.path.join(local_base_dir, "reference.py")
      opt_local = os.path.join(local_base_dir, "optimized.py")
      result_local = os.path.join(local_base_dir, "result.json")
      xprof_src = os.path.join(os.path.dirname(__file__), "xprof_utils.py")
      xprof_local = os.path.join(local_base_dir, "xprof_utils.py")

      # Build JSON and harness
      self._build_task_json(task, task_json_local)
      self._build_harness_code(harness_local, atol, rtol)

      # Copy reference, optimized code and xprof_utils.py
      shutil.copy(reference_code_path, ref_local)
      shutil.copy(optimized_code_path, opt_local)
      shutil.copy(xprof_src, xprof_local)

      logger.info(f"Starting local evaluation in {local_base_dir}...")

      # Construct the command
      cmd = ["python3", "harness.py"]
      env = os.environ.copy()

      # Activate virtual environment if specified by prepending to PATH
      if self.venv_path:
        env["PATH"] = f"{self.venv_path}/bin:" + env.get("PATH", "")

      # Execute locally with timeout handling
      try:
        subprocess.run(
          cmd,
          cwd=local_base_dir,
          env=env,
          timeout=timeout_seconds,
          check=True,
          capture_output=True,
          text=True,
        )
      except subprocess.TimeoutExpired:
        logger.error(
          f"Local execution timed out after {timeout_seconds} seconds for "
          f"task {task.task_id}."
        )
        eval_result.error_trace = (
          f"Local execution timed out after {timeout_seconds} seconds."
        )
        return eval_result
      except subprocess.CalledProcessError as e:
        # Handle crashes like Segfaults
        if e.returncode in [-11, -4]:  # SIGSEGV, SIGILL
          logger.error(
            "CRITICAL: Local interpreter crashed. This is likely due to an "
            "invalid Pallas kernel performing an illegal memory access."
          )
        eval_result.error_trace = (
          f"Command failed with exit code {e.returncode}. Stderr: {e.stderr}"
        )

      # Parse the result
      data = None
      if os.path.exists(result_local):
        try:
          with open(result_local, "r") as f:
            data = json.load(f)
        except Exception as e:
          logger.error(f"Failed to read or parse result.json: {e}")

      if not data:
        error_msg = "No data found in result.json. Local execution may have failed without producing output."
        logger.error(error_msg)
        if not eval_result.error_trace:
          eval_result.error_trace = error_msg
      else:
        if "error" in data:
          error_msg = data.get("error")
          eval_result.error_trace = data.get("traceback", error_msg)

        # Populate the eval result
        for key, value in data.items():
          if hasattr(eval_result, key):
            setattr(eval_result, key, value)

      print_eval_result(eval_result)
      return eval_result

    finally:
      if cleanup:
        logger.info(f"Cleaning up local directory: {local_base_dir}")
        shutil.rmtree(local_base_dir, ignore_errors=True)
      else:
        logger.info(f"Skipping cleanup. Local files are in {local_base_dir}")

  def _evaluate_remote(
    self,
    reference_code_path: str,
    optimized_code_path: str,
    task_yaml_path: Optional[str] = None,
    adapt: Optional[List[str]] = None,
    timeout_seconds: int = 300,  # Default timeout of 5 minutes
    cleanup: bool = True,
    atol: float = 1e-3,
    rtol: float = 1e-3,
  ) -> EvaluationResult:
    """Runs the evaluation pipeline for a given reference and kernel script.

    Args:
        reference_code_path: Local path to the reference JAX script.
        optimized_code_path: Local path to the optimized JAX script.
        task_yaml_path: Path to the kernel_task.yaml file. If None, the adapter is used to generate task.
        adapt: A list of components to adapt. Can contain 'reference_code', 'optimized_code', 'kernel_task'.
        timeout_seconds: Maximum time in seconds to wait for the remote execution to complete
        cleanup: If True, deletes all temporary files from the remote TPU VM after evaluation.
    """
    # Adapt the provided code and generate kernel task if necessary
    if adapt:
      reference_code_path, optimized_code_path, task_yaml_path = (
        self._adapt_inputs(
          reference_code_path, optimized_code_path, adapt, task_yaml_path
        )
      )

    if not task_yaml_path:
      raise ValueError(
        "task_yaml_path is required when 'kernel_task' is not in adapt"
      )

    # Prepare temporary files for harness and task JSON for remote execution
    fd, harness_local = tempfile.mkstemp(suffix=".py", prefix="tpu_harness_")
    os.close(fd)
    fd_json, task_json_local = tempfile.mkstemp(suffix=".json", prefix="task_")
    os.close(fd_json)

    try:
      remote_base_dir = None
      remote_base_dir_created = False

      task = load_kernel_task_from_yaml(task_yaml_path)
      self._build_task_json(task, task_json_local)

      eval_result = EvaluationResult(task_id=task.task_id)
      self._build_harness_code(harness_local, atol, rtol)

      xprof_src = os.path.join(os.path.dirname(__file__), "xprof_utils.py")
      files_to_upload = {
        "reference.py": reference_code_path,
        "optimized.py": optimized_code_path,
        "harness.py": harness_local,
        "task.json": task_json_local,
        "xprof_utils.py": xprof_src,
      }

      safe_task_id = str(task.task_id).replace(" ", "_")
      # Create a unique directory for this evaluation run on the remote machine
      if self.remote_base_dir is None:
        remote_base_dir = f"eval_run_{int(time.time())}_{safe_task_id}"
      else:
        remote_base_dir = (
          self.remote_base_dir
          + "/"
          + f"eval_run_{int(time.time())}_{safe_task_id}"
        )
      # Create the remote directory
      self.client.execute_ssh_command(f"mkdir -p {remote_base_dir}")
      remote_base_dir_created = (
        True  # Flag to track if the remote directory was created successfully
      )

      # Upload files to the remote directory
      for filename, local_path in files_to_upload.items():
        remote_path = os.path.join(remote_base_dir, filename)
        self.client.upload_file(local_path, remote_path)

      logger.info(
        f"Starting remote evaluation in {self.client.tpu_name}:{remote_base_dir}..."
      )

      # Construct the command to run inside the remote directory
      # 1. Change to the remote directory
      cmd_parts = [f"cd {remote_base_dir}"]
      # 2. Activate virtual environment if specified
      if self.venv_path:
        cmd_parts.append(
          f"source {TPUVMClient.quote_path(self.venv_path)}/bin/activate"
        )
      # 3. Run the harness script
      cmd_parts.append("python3 harness.py")
      full_cmd = " && ".join(cmd_parts)

      # Execute the command remotely with timeout handling
      try:
        result = self.client.execute_ssh_command(
          full_cmd, timeout=timeout_seconds
        )
      except subprocess.TimeoutExpired:
        logger.error(
          f"Remote execution timed out after {timeout_seconds} seconds for "
          f"task {task.task_id}."
        )
        # Attempt to kill the runaway process on the remote VM to release the TPU.
        logger.warning(
          f"Attempting to kill runaway process on {self.client.tpu_name}..."
        )
        try:
          kill_cmd = "pkill -f 'harness.py'"
          self.client.execute_ssh_command(kill_cmd, timeout=30)
          logger.info(
            f"Successfully sent kill signal to remote process for task "
            f"{task.task_id}."
          )
        except Exception as kill_e:
          logger.error(
            f"Failed to kill remote process, TPU may remain locked: {kill_e}"
          )
        eval_result.error_trace = (
          f"Remote execution timed out after {timeout_seconds} seconds."
        )
        # Give some time for the kill to take effect
        time.sleep(60)
        return eval_result
      except RuntimeError as e:
        eval_result.error_trace = str(e)
        # Detect if the remote interpreter crashed (e.g., Segfault from Pallas)
        if any(
          sig in str(e).lower()
          for sig in ["segmentation fault", "sigsegv", "illegal instruction"]
        ):
          logger.error(
            "CRITICAL: Remote interpreter crashed. This is likely due to an "
            "invalid Pallas kernel performing an illegal hardware operation."
          )
        return eval_result

      # Parse and handle the result
      # Read the structured JSON result from the remote file
      data = None
      try:
        cat_result = self.client.execute_ssh_command(
          f"cat {remote_base_dir}/result.json"
        )
        data = json.loads(cat_result.stdout)
      except Exception as e:
        logger.error(f"Failed to read or parse result.json: {e}")

      if not data:
        error_msg = "No data found in result.json. Remote execution may have failed without producing output."
        logger.error(error_msg)
        eval_result.error_trace = error_msg
      else:
        if "error" in data:
          error_msg = data.get("error")
          eval_result.error_trace = data.get("traceback", error_msg)

        # Populate the eval result
        for key, value in data.items():
          if hasattr(eval_result, key):
            setattr(eval_result, key, value)

      print_eval_result(eval_result)
      return eval_result

    finally:
      # Cleanup remote directory if requested
      if cleanup and remote_base_dir_created:
        logger.info(f"Cleaning up remote directory: {remote_base_dir}")
        # Protect cleanup so network drops don't shadow actual evaluation outcomes
        try:
          self.client.execute_ssh_command(f"rm -rf {remote_base_dir}")
        except Exception as e:
          logger.warning(
            f"Failed to cleanup remote directory {remote_base_dir}: {e}"
          )
      elif not remote_base_dir_created:
        logger.info(
          "No remote directory was created, so no cleanup is necessary."
        )
      else:
        logger.info(
          f"Skipping cleanup. Remote files are in {self.client.tpu_name}:{remote_base_dir}"
        )

      # Cleanup local temporary files
      if os.path.exists(harness_local):
        os.remove(harness_local)
      if os.path.exists(task_json_local):
        os.remove(task_json_local)

  def _build_task_json(self, task: KernelTask, local_path: str):
    """Creates a JSON file with task input specifications.

    Args:
        task: KernelTask object containing input specifications.
        local_path: Local path to save the JSON file.
    """
    task_info = {
      "input_gen_code": task.input_gen_code,
    }
    with open(local_path, "w") as f:
      json.dump(task_info, f)

  def _build_harness_code(
    self, local_path: str, atol: float = 1e-3, rtol: float = 1e-3
  ):
    """Creates the remote evaluation harness script locally.

    Args:
        local_path: Local path to save the harness script.
        atol: Absolute tolerance for correctness check.
        rtol: Relative tolerance for correctness check.
    """
    harness_content = HARNESS_TEMPLATE.template
    harness_content = harness_content.replace("{atol}", str(atol))
    harness_content = harness_content.replace("{rtol}", str(rtol))

    with open(local_path, "w", encoding="utf-8") as f_write:
      f_write.write(harness_content)

  def _adapt_inputs(
    self,
    reference_code_path: str,
    optimized_code_path: str,
    adapt: List[str],
    task_yaml_path: Optional[str] = None,
  ) -> Tuple[str, str, Optional[str]]:
    """Handles adapting code using the LLM CodeAdapter."""
    task = None

    if self.adapter is None:
      api_key = os.environ.get("GEMINI_API_KEY")
      if not api_key:
        raise ValueError(
          "GEMINI_API_KEY environment variable not set. Unable to adapt code."
        )
      genai_client = genai.Client(api_key=api_key)
      self.adapter = code_adapter.CodeAdapter(client=genai_client)

    # The order of adaptation matters - the optimized code adaptation relies on the
    # get_inputs code generated in the kernel task, which in turn relies on the
    # reference code. So we adapt in this order:
    # reference code -> kernel task -> optimized code.
    if "reference_code" in adapt:
      logging.info("Adapting reference code ...")
      with open(reference_code_path, "r", encoding="utf-8") as f:
        ref_code = f.read()
      reference_code_adapted = self.adapter.adapt(ref_code)
      base, ext = os.path.splitext(reference_code_path)
      ref_adapted_path = f"{base}_adapted{ext}"
      with open(ref_adapted_path, "w", encoding="utf-8") as f:
        f.write(reference_code_adapted)
      reference_code_path = ref_adapted_path
      logging.info(f"Wrote adapted reference code to {reference_code_path}")

    if "kernel_task" in adapt:
      logging.info("Creating kernel task ...")
      with open(reference_code_path, "r", encoding="utf-8") as f:
        current_ref_code = f.read()
      task_id = os.path.basename(os.path.dirname(reference_code_path))
      task = self.adapter.generate_kernel_task(
        task_id, task_id, current_ref_code
      )
      if task_yaml_path:
        base, ext = os.path.splitext(task_yaml_path)
        task_adapted_path = f"{base}_adapted{ext}"
      else:
        base, ext = os.path.splitext(reference_code_path)
        task_adapted_path = f"{base}_task_adapted.yaml"
      write_kernel_task_to_yaml(task, task_adapted_path)
      logging.info(f"Wrote kernel task to {task_adapted_path}")
      task_yaml_path = task_adapted_path

    if "optimized_code" in adapt:
      logging.info("Adapting optimized code ...")
      with open(optimized_code_path, "r", encoding="utf-8") as f:
        opt_code = f.read()
      if not task_yaml_path or not os.path.exists(task_yaml_path):
        raise FileNotFoundError(
          "task_yaml_path must be valid if optimized_code is being adapted without kernel_task adaptation."
        )
      task = load_kernel_task_from_yaml(task_yaml_path)
      get_inputs_code = task.input_gen_code
      if not get_inputs_code:
        logging.warning(
          "No get_inputs code found in task. The optimized code adaptation is less robust without it."
        )
        optimized_code_adapted = self.adapter.adapt(opt_code)
      else:
        optimized_code_adapted = self.adapter.adapt(
          opt_code, adapt_optimized=True, get_inputs_code=get_inputs_code
        )
      base, ext = os.path.splitext(optimized_code_path)
      opt_adapted_path = f"{base}_adapted{ext}"
      with open(opt_adapted_path, "w", encoding="utf-8") as f:
        f.write(optimized_code_adapted)
      optimized_code_path = opt_adapted_path
      logging.info(f"Wrote adapted optimized code to {optimized_code_path}")

    return reference_code_path, optimized_code_path, task_yaml_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Evaluate a single JAX kernel on a remote TPU VM or locally."
  )
  parser.add_argument(
    "--reference_code_path",
    type=str,
    required=True,
    help="Local path to the reference JAX script.",
  )
  parser.add_argument(
    "--optimized_code_path",
    type=str,
    required=True,
    help="Local path to the optimized JAX script.",
  )
  parser.add_argument(
    "--task_yaml_path",
    type=str,
    default=None,
    help="Path to the kernel_task.yaml file.",
  )
  parser.add_argument(
    "--local",
    action="store_true",
    help="Run evaluation locally instead of on a remote TPU.",
  )
  parser.add_argument(
    "--tpu_name",
    type=str,
    help="The name of the target TPU VM.",
  )
  parser.add_argument(
    "--zone",
    type=str,
    help="The Google Cloud zone where the TPU is located.",
  )
  parser.add_argument(
    "--project",
    type=str,
    help="The Google Cloud project ID.",
  )
  parser.add_argument(
    "--venv_path",
    type=str,
    help="The absolute path to the Python virtual environment.",
  )
  parser.add_argument(
    "--remote_base_dir",
    type=str,
    help="Base directory on remote TPU for storing temporary files.",
  )
  parser.add_argument(
    "--adapt",
    type=str,
    nargs="*",
    default=None,
    help="Optional list specifying which components to adapt via LLM. Can contain 'reference_code', 'optimized_code', 'kernel_task'.",
  )
  parser.add_argument(
    "--timeout_seconds",
    type=int,
    default=300,
    help="Maximum time in seconds to wait for execution.",
  )
  parser.add_argument(
    "--no_cleanup",
    action="store_false",
    dest="cleanup",
    help="Skip cleanup of temporary files.",
  )
  parser.add_argument(
    "--atol",
    type=float,
    default=1e-3,
    help="Absolute tolerance for correctness check.",
  )
  parser.add_argument(
    "--rtol",
    type=float,
    default=1e-3,
    help="Relative tolerance for correctness check.",
  )

  args = parser.parse_args()

  # Configure logging if not already configured
  if not logging.getLogger().hasHandlers():
    logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
    )

  evaluator = JAXKernelEvaluator(
    local=args.local,
    tpu_name=args.tpu_name,
    project=args.project,
    zone=args.zone,
    venv_path=args.venv_path,
    remote_base_dir=args.remote_base_dir,
  )

  evaluator.evaluate(
    reference_code_path=args.reference_code_path,
    optimized_code_path=args.optimized_code_path,
    task_yaml_path=args.task_yaml_path,
    adapt=args.adapt,
    timeout_seconds=args.timeout_seconds,
    cleanup=args.cleanup,
    atol=args.atol,
    rtol=args.rtol,
  )
