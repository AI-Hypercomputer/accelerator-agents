"""A modular client for interacting with GCP TPU VMs via gcloud, along with a high-level function to run local scripts on remote TPUs."""

import logging
import os
import shlex
import subprocess
from typing import List, Optional

import google.auth

logger = logging.getLogger(__name__)


class TPUVMClient:
  """A modular client for interacting with GCP TPU VMs via gcloud."""

  def __init__(
    self, project: Optional[str] = None, zone: str = "", tpu_name: str = ""
  ):
    self.project = project
    self.zone = zone
    self.tpu_name = tpu_name
    self._authenticate()

  def _authenticate(self):
    """Validates GCP authentication and sets default project if needed."""
    try:
      credentials, project_id = google.auth.default()
      if not self.project:
        self.project = project_id
      logger.info(f"Authenticated for project: {self.project}")
    except google.auth.exceptions.DefaultCredentialsError:
      logger.error(
        "GCP credentials not found. Please run 'gcloud auth application-default login'."
      )
      raise

  def _run_gcloud(
    self, args: List[str], check: bool = True, timeout: Optional[int] = None
  ) -> subprocess.CompletedProcess:
    """Helper to run gcloud commands with standard project and zone flags.

    Args:
        args: List of arguments for the gcloud command.
        check: Whether to raise an exception on non-zero exit code.

    Returns:
        CompletedProcess instance with the command result.
    """
    full_cmd = (
      ["gcloud", "compute", "tpus", "tpu-vm"]
      + args
      + [
        f"--zone={self.zone}",
        f"--project={self.project}",
      ]
    )
    try:
      return subprocess.run(
        full_cmd, check=check, capture_output=True, text=True, timeout=timeout
      )
    except subprocess.CalledProcessError as e:
      logger.error(f"GCP Command failed: {' '.join(e.cmd)}")
      logger.error(f"Stderr: {e.stderr}")
      raise RuntimeError(f"Failed to execute gcloud command: {e.stderr}") from e

  def upload_file(self, local_path: str, remote_path: str):
    """Uploads a local file to the TPU VM."""
    if not os.path.exists(local_path):
      raise FileNotFoundError(f"Local script not found: {local_path}")

    logger.info(f"Uploading {local_path} to {self.tpu_name}:{remote_path}")

    scp_args = ["scp", local_path, f"{self.tpu_name}:{remote_path}"]
    self._run_gcloud(scp_args)

  def execute_ssh_command(
    self, command_str: str, timeout: Optional[int] = None
  ) -> subprocess.CompletedProcess:
    """Executes a command string on the TPU VM via SSH."""
    logger.info(f"Executing remote command on {self.tpu_name}")
    return self._run_gcloud(
      ["ssh", self.tpu_name, f"--command={command_str}"], timeout=timeout
    )

  def delete_file(self, remote_path: str):
    """Removes a file from the TPU VM."""
    logger.info(f"Cleaning up {remote_path} on {self.tpu_name}")
    self.execute_ssh_command(f"rm -f {self.quote_path(remote_path)}")

  @staticmethod
  def quote_path(path: str) -> str:
    """Quotes a path for shell use, handling tilde expansion."""
    if path.startswith("~/"):
      return f"$HOME/{shlex.quote(path[2:])}"
    return shlex.quote(path)


# High-level function to run a local script on a remote TPU VM
def run_script_on_tpu_vm(
  local_script_path: str,
  tpu_name: str,
  zone: str,
  project: Optional[str] = None,
  vm_dest_path: Optional[str] = None,
  venv_path: Optional[str] = None,
  script_args: Optional[List[str]] = None,
  cleanup_script: bool = True,
):
  """
  High-level function to run a local script on a remote TPU VM.

  Args:
  - local_script_path: Path to the local script to run.
  - tpu_name: Name of the TPU VM.
  - zone: GCP zone of the TPU VM.
  - project: GCP project ID (optional, will use default if not provided).
  - vm_dest_path: Destination path on the TPU VM (optional, defaults to /home/username/uploaded_script.py).
  - venv_path: Path to a virtual environment on the TPU VM to activate before running the script (optional).
  - script_args: List of arguments to pass to the script (optional).
  - cleanup_script: Whether to delete the script from the TPU VM after execution (default: True).

  Raises:
  - FileNotFoundError: If the local script is not found.
  """
  client = TPUVMClient(project=project, zone=zone, tpu_name=tpu_name)

  if vm_dest_path is None:
    vm_dest_path = f"uploaded_{os.path.basename(local_script_path)}"

  try:
    # 1. Copy the script
    client.upload_file(local_script_path, vm_dest_path)

    # 2. Construct execution command
    execution_parts = []
    if venv_path:
      execution_parts.append(
        f"source {TPUVMClient.quote_path(venv_path)}/bin/activate"
      )

    python_call = ["python3", TPUVMClient.quote_path(vm_dest_path)]
    if script_args:
      python_call.extend([shlex.quote(arg) for arg in script_args])

    execution_parts.append(" ".join(python_call))
    full_command = " && ".join(execution_parts)

    # 3. Execute
    result = client.execute_ssh_command(full_command)
    logger.info("Execution completed successfully.")
    logger.info(f"STDOUT:\n{result.stdout}")
    if result.stderr:
      if "error" in result.stderr.lower() or "failed" in result.stderr.lower():
        logger.warning(f"Remote STDERR:\n{result.stderr}")
      else:
        logger.info(f"Remote Status (STDERR):\n{result.stderr}")

  except Exception as e:
    logger.error(f"An error occurred during remote execution: {e}")
    raise
  finally:
    # 4. Cleanup
    if cleanup_script:
      try:
        client.delete_file(vm_dest_path)
      except Exception as e:
        logger.warning(f"Cleanup failed: {e}")
