import subprocess
import unittest
from unittest.mock import MagicMock, patch

from google.auth.exceptions import DefaultCredentialsError

from evaluation.remote_client.tpu_client import (
  TPUVMClient,
  run_script_on_tpu_vm,
)


class TestTPUVMClient(unittest.TestCase):
  def setUp(self):
    self.auth_patcher = patch("remote_client.tpu_client.google.auth.default")
    self.mock_auth = self.auth_patcher.start()
    self.addCleanup(self.auth_patcher.stop)
    self.mock_auth.return_value = (MagicMock(), "test-proj")

  def test_authenticate_success(self):
    self.mock_auth.return_value = (MagicMock(), "default-project")
    client = TPUVMClient(zone="us-east1-d", tpu_name="my-tpu")
    self.assertEqual(client.project, "default-project")
    self.assertEqual(client.zone, "us-east1-d")
    self.assertEqual(client.tpu_name, "my-tpu")

  def test_authenticate_provided_project(self):
    self.mock_auth.return_value = (MagicMock(), "default-project")
    client = TPUVMClient(
      project="my-project", zone="us-east1-d", tpu_name="my-tpu"
    )
    self.assertEqual(client.project, "my-project")

  def test_authenticate_failure(self):
    self.mock_auth.side_effect = DefaultCredentialsError("No creds")
    with self.assertRaises(DefaultCredentialsError):
      TPUVMClient()

  @patch("evaluation.remote_client.tpu_client.subprocess.run")
  def test_run_gcloud_success(self, mock_subprocess_run):
    client = TPUVMClient(zone="test-zone", tpu_name="test-tpu")

    mock_result = MagicMock()
    mock_subprocess_run.return_value = mock_result

    res = client._run_gcloud(["arg1", "arg2"])

    self.assertEqual(res, mock_result)
    mock_subprocess_run.assert_called_once_with(
      [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "arg1",
        "arg2",
        "--zone=test-zone",
        "--project=test-proj",
      ],
      check=True,
      capture_output=True,
      text=True,
      timeout=None,
    )

  @patch("evaluation.remote_client.tpu_client.subprocess.run")
  def test_run_gcloud_failure(self, mock_subprocess_run):
    client = TPUVMClient(zone="test-zone", tpu_name="test-tpu")

    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
      returncode=1, cmd="cmd", stderr="error message"
    )
    with self.assertRaisesRegex(
      RuntimeError, "Failed to execute gcloud command"
    ):
      client._run_gcloud(["arg"])

  @patch("evaluation.remote_client.tpu_client.os.path.exists")
  @patch.object(TPUVMClient, "_run_gcloud")
  def test_upload_file_success(self, mock_run_gcloud, mock_exists):
    mock_exists.return_value = True
    client = TPUVMClient(zone="test-zone", tpu_name="test-tpu")

    client.upload_file("local_script.py", "remote_script.py")
    mock_run_gcloud.assert_called_once_with(
      ["scp", "local_script.py", "test-tpu:remote_script.py"]
    )

  @patch("evaluation.remote_client.tpu_client.os.path.exists")
  def test_upload_file_not_found(self, mock_exists):
    mock_exists.return_value = False
    client = TPUVMClient(zone="test-zone", tpu_name="test-tpu")

    with self.assertRaises(FileNotFoundError):
      client.upload_file("missing.py", "remote.py")

  @patch.object(TPUVMClient, "_run_gcloud")
  def test_execute_ssh_command(self, mock_run_gcloud):
    client = TPUVMClient(zone="test-zone", tpu_name="test-tpu")

    client.execute_ssh_command("ls -la", timeout=15)
    mock_run_gcloud.assert_called_once_with(
      ["ssh", "test-tpu", "--command=ls -la"], timeout=15
    )

  @patch.object(TPUVMClient, "execute_ssh_command")
  def test_delete_file(self, mock_ssh):
    client = TPUVMClient(zone="test-zone", tpu_name="test-tpu")

    client.delete_file("remote_script.py")
    mock_ssh.assert_called_once_with("rm -f remote_script.py")

  def test_quote_path(self):
    self.assertEqual(TPUVMClient.quote_path("/path/to/file"), "/path/to/file")
    self.assertEqual(TPUVMClient.quote_path("~/my_file.py"), "$HOME/my_file.py")
    self.assertEqual(TPUVMClient.quote_path("my file.py"), "'my file.py'")


class TestRunScriptOnTPUVM(unittest.TestCase):
  @patch("evaluation.remote_client.tpu_client.TPUVMClient")
  def test_run_script_success(self, mock_client_class):
    mock_client_class.quote_path.side_effect = lambda x: x
    mock_client = mock_client_class.return_value
    mock_result = MagicMock()
    mock_result.stdout = "output"
    mock_result.stderr = ""
    mock_client.execute_ssh_command.return_value = mock_result

    run_script_on_tpu_vm(
      local_script_path="local.py",
      tpu_name="test-tpu",
      zone="test-zone",
      venv_path="/path/to/venv",
      script_args=["arg1", "arg2"],
      cleanup_script=True,
    )

    mock_client.upload_file.assert_called_once_with(
      "local.py", "uploaded_local.py"
    )
    mock_client.execute_ssh_command.assert_called_once()

    cmd = mock_client.execute_ssh_command.call_args[0][0]
    self.assertIn("source /path/to/venv/bin/activate", cmd)
    self.assertIn("python3 uploaded_local.py", cmd)
    self.assertIn("arg1 arg2", cmd)

    mock_client.delete_file.assert_called_once_with("uploaded_local.py")
