import json
import os
import sys
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

tools_dir = os.path.dirname(__file__)
sys.path.insert(0, tools_dir)

import tpu_client


class TestTPUClient(unittest.TestCase):

  @patch("urllib.request.urlopen")
  def test_submit_job_code(self, mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "job_id": "job_123",
        "status": "queued",
        "queue_position": 1,
        "action": "correctness_test",
        "created_at": 100.0,
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    res = tpu_client.submit_job(
        "correctness_test", "print('hello')", 60, port=8000
    )
    self.assertIsNotNone(res)
    self.assertEqual(res["job_id"], "job_p8000_job_123")
    self.assertEqual(res["status"], "queued")
    self.assertEqual(res["queue_position"], 1)

  @patch("urllib.request.urlopen")
  def test_check_job_status(self, mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "job_id": "job_123",
        "status": "completed",
        "action": "correctness_test",
        "result": {"output": "hello\n", "exit_code": 0},
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    res = tpu_client.check_job_status("job_p8000_job_123")
    self.assertIsNotNone(res)
    self.assertEqual(res["job_id"], "job_p8000_job_123")
    self.assertEqual(res["status"], "completed")
    self.assertEqual(res["result"]["exit_code"], 0)

  @patch("urllib.request.urlopen")
  def test_check_job_status_404_not_found(self, mock_urlopen):
    mock_error = urllib.error.HTTPError(
        url="http://127.0.0.1:8000/job/job_123",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None,
    )
    mock_urlopen.side_effect = mock_error

    res = tpu_client.check_job_status("job_p8000_job_123")
    self.assertIsNotNone(res)
    self.assertEqual(res["status"], "not_found")
    self.assertEqual(res["http_code"], 404)

  @patch("urllib.request.urlopen")
  def test_get_queue_info(self, mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "total_jobs": 2,
        "queued_count": 1,
        "running_count": 1,
        "queued_jobs": [
            {"job_id": "job_2", "action": "autotune", "position": 1}
        ],
        "running_jobs": [{"job_id": "job_1", "action": "correctness_test"}],
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    res = tpu_client.get_queue_info(port=8000)
    self.assertIsNotNone(res)
    self.assertEqual(res["total_jobs"], 2)
    self.assertEqual(res["queued_count"], 1)
    self.assertEqual(res["running_count"], 1)

  @patch("tpu_client.check_health")
  def test_start_server_idempotent_already_healthy(self, mock_health):
    mock_health.return_value = True
    self.assertTrue(tpu_client.start_server_idempotent(mode="local"))

  @patch("builtins.open", new_callable=MagicMock)
  @patch("subprocess.Popen")
  @patch("tpu_client.check_health")
  @patch("tpu_client.get_tpu_config")
  def test_start_server_idempotent_local_mode(
      self, mock_config, mock_health, mock_popen, mock_open
  ):
    mock_health.side_effect = [False, False, False, False, True]
    mock_config.return_value = [{"mode": "local", "local_port": 8000}]
    res = tpu_client.start_server_idempotent(mode="local")
    self.assertTrue(res)
    mock_popen.assert_called_once()
    cmd_args = mock_popen.call_args[0][0]
    self.assertIn("tpu_server.py", cmd_args[1])

  @patch("tpu_client.get_tpu_config")
  def test_start_server_idempotent_remote_missing_config(self, mock_config):
    mock_config.return_value = []
    with patch("tpu_client.check_health", return_value=False):
      self.assertFalse(tpu_client.start_server_idempotent(mode="remote"))

  @patch("os.path.exists", return_value=True)
  @patch("builtins.open", new_callable=MagicMock)
  @patch("json.load")
  def test_multi_tpu_config_parsing(
      self, mock_json_load, mock_open, mock_exists
  ):
    mock_json_load.return_value = {
        "tpus": [
            {"tpu_name": "tpu-1", "zone": "zone-a", "project": "proj-a"},
            {"tpu_name": "tpu-2", "zone": "zone-b", "project": "proj-b"},
        ]
    }
    configs = tpu_client.get_tpu_config()
    self.assertEqual(len(configs), 2)
    self.assertEqual(configs[0]["local_port"], 8000)
    self.assertEqual(configs[1]["local_port"], 8001)

  def test_shard_search_space(self):
    search_space = {
        "BLOCK_M": [64, 128],
        "BLOCK_N": [32, 64, 128],
    }
    shards = tpu_client.shard_search_space(search_space, 2)
    self.assertEqual(len(shards), 2)
    combo_count = 0
    for s in shards:
      combos = 1
      for v in s.values():
        combos *= len(v)
      combo_count += combos
    self.assertEqual(combo_count, 6)

  @patch("tpu_client.get_queue_info")
  @patch("tpu_client.check_health")
  def test_select_best_tpu_server(self, mock_health, mock_queue):
    configs = [
        {"tpu_name": "tpu-1", "local_port": 8000},
        {"tpu_name": "tpu-2", "local_port": 8001},
    ]
    mock_health.return_value = True
    mock_queue.side_effect = lambda port: (
        {"queued_count": 2, "running_count": 1}
        if port == 8000
        else {"queued_count": 0, "running_count": 0}
    )

    port, cfg = tpu_client.select_best_tpu_server(tpu_configs=configs)
    self.assertEqual(port, 8001)
    self.assertEqual(cfg["tpu_name"], "tpu-2")

  def test_infer_tpu_version(self):
    self.assertEqual(
        tpu_client.infer_tpu_version({"tpu_spec": {"device_kind": "TPU v6 lite"}}),
        "TPU v6e",
    )
    self.assertEqual(
        tpu_client.infer_tpu_version({"tpu_spec": {"device_kind": "TPU v5 lite"}}),
        "TPU v5e",
    )
    self.assertEqual(
        tpu_client.infer_tpu_version({"tpu_spec": {"device_kind": "TPU v5p"}}),
        "TPU v5p",
    )
    self.assertEqual(
        tpu_client.infer_tpu_version({"tpu_spec": {"device_kind": "TPU v7x"}}),
        "TPU v7x",
    )
    self.assertEqual(
        tpu_client.infer_tpu_version({"tpu_name": "tpu-v6e-8-1"}),
        "TPU v6e",
    )
    self.assertEqual(
        tpu_client.infer_tpu_version({"tpu_name": "tpu-v5p-16-1"}),
        "TPU v5p",
    )
    self.assertEqual(
        tpu_client.infer_tpu_version({"tpu_name": "tpu-v7x-32-1"}),
        "TPU v7x",
    )
    self.assertEqual(
        tpu_client.infer_tpu_version({"tpu_version": "TPU v7x"}),
        "TPU v7x",
    )


if __name__ == "__main__":
  unittest.main()
