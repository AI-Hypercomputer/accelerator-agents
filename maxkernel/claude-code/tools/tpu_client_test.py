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
    mock_response.read.return_value = json.dumps(
      {
        "job_id": "job_123",
        "status": "queued",
        "queue_position": 1,
        "action": "correctness_test",
        "created_at": 100.0,
      }
    ).encode("utf-8")
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
    mock_response.read.return_value = json.dumps(
      {
        "job_id": "job_123",
        "status": "completed",
        "action": "correctness_test",
        "result": {"output": "hello\n", "exit_code": 0},
      }
    ).encode("utf-8")
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
    mock_response.read.return_value = json.dumps(
      {
        "total_jobs": 2,
        "queued_count": 1,
        "running_count": 1,
        "queued_jobs": [
          {"job_id": "job_2", "action": "autotune", "position": 1}
        ],
        "running_jobs": [{"job_id": "job_1", "action": "correctness_test"}],
      }
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    res = tpu_client.get_queue_info(port=8000)
    self.assertIsNotNone(res)
    self.assertEqual(res["total_jobs"], 2)
    self.assertEqual(res["queued_count"], 1)
    self.assertEqual(res["running_count"], 1)

  @patch("tpu_client.ensure_tpu_specs_cached")
  @patch("tpu_client.check_health")
  @patch("tpu_client.get_tpu_config")
  def test_start_server_idempotent_already_healthy(
    self, mock_config, mock_health, mock_ensure
  ):
    mock_config.return_value = [{"mode": "local", "local_port": 8000}]
    mock_health.return_value = True
    self.assertTrue(tpu_client.start_server_idempotent(mode="local"))
    mock_ensure.assert_called_once()

  @patch("tpu_client.ensure_tpu_specs_cached")
  @patch("builtins.open", new_callable=MagicMock)
  @patch("subprocess.Popen")
  @patch("tpu_client.check_health")
  @patch("tpu_client.get_tpu_config")
  def test_start_server_idempotent_local_mode(
    self, mock_config, mock_health, mock_popen, mock_open, mock_ensure
  ):
    mock_health.side_effect = [False, False, False, False, True]
    mock_config.return_value = [{"mode": "local", "local_port": 8000}]
    res = tpu_client.start_server_idempotent(mode="local")
    self.assertTrue(res)
    mock_ensure.assert_called_once()
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
      tpu_client.infer_tpu_version(
        {"tpu_spec": {"device_kind": "TPU v6 lite"}}
      ),
      "TPU v6e",
    )
    self.assertEqual(
      tpu_client.infer_tpu_version(
        {"tpu_spec": {"device_kind": "TPU v5 lite"}}
      ),
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

  def test_version_from_spec_ignores_cached_version(self):
    self.assertEqual(
      tpu_client.version_from_spec(
        {
          "tpu_version": "TPU v6e",
          "tpu_spec": {"device_kind": "TPU v5 lite"},
        }
      ),
      "TPU v5e",
    )

  @patch("tpu_client.fetch_tpu_spec")
  @patch("tpu_client.check_health")
  def test_ensure_tpu_specs_cached_fetches_real_spec(
    self, mock_health, mock_fetch
  ):
    import tempfile

    mock_health.return_value = True
    mock_fetch.return_value = {"device_kind": "TPU v5 lite", "device_count": 4}
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "tpu_config.json")
      with open(path, "w") as f:
        json.dump({"tpus": [{"mode": "local", "local_port": 8000}]}, f)
      tpu_client.ensure_tpu_specs_cached(config_path_override=path)
      with open(path) as f:
        entry = json.load(f)["tpus"][0]
    self.assertEqual(entry["tpu_spec"]["device_count"], 4)
    self.assertEqual(entry["tpu_version"], "TPU v5e")

  @patch("tpu_client.check_health")
  def test_ensure_tpu_specs_cached_unreachable_writes_no_guess(
    self, mock_health
  ):
    import tempfile

    mock_health.return_value = False
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "tpu_config.json")
      with open(path, "w") as f:
        json.dump({"tpus": [{"mode": "local", "local_port": 8000}]}, f)
      tpu_client.ensure_tpu_specs_cached(config_path_override=path)
      with open(path) as f:
        entry = json.load(f)["tpus"][0]
    self.assertNotIn("tpu_spec", entry)
    self.assertNotIn("tpu_version", entry)

  @patch("tpu_client.fetch_tpu_spec")
  @patch("tpu_client.check_health")
  @patch("tpu_client.start_server_for_tpu")
  def test_add_tpu_refreshes_existing_entry(
    self, mock_start, mock_health, mock_fetch
  ):
    import tempfile

    mock_health.return_value = True
    mock_fetch.return_value = {"device_kind": "TPU v5 lite", "device_count": 4}
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "tpu_config.json")
      stale = {
        "mode": "local",
        "local_port": 8000,
        "tpu_version": "TPU v6e",
        "tpu_spec": {"device_kind": "TPU v6 lite", "device_count": 8},
      }
      with open(path, "w") as f:
        json.dump({"tpus": [stale]}, f)
      self.assertTrue(
        tpu_client.add_tpu_to_config(
          {"mode": "local"}, config_path_override=path
        )
      )
      with open(path) as f:
        tpus = json.load(f)["tpus"]
    self.assertEqual(len(tpus), 1)
    self.assertEqual(tpus[0]["tpu_spec"]["device_count"], 4)
    self.assertEqual(tpus[0]["tpu_version"], "TPU v5e")

  @patch("os.path.exists", return_value=False)
  def test_get_tpu_config_no_file_falls_back_to_local(self, mock_exists):
    self.assertEqual(
      tpu_client.get_tpu_config(), [{"mode": "local", "local_port": 8000}]
    )

  @patch("tpu_client.submit_job")
  @patch("tpu_client.check_job_status")
  def test_fetch_tpu_spec_rejects_non_tpu(self, mock_status, mock_submit):
    mock_submit.return_value = {"job_id": "job_1"}
    mock_status.return_value = {
      "status": "completed",
      "result": {"output": "DEVICE_KIND:cpu\nDEVICE_COUNT:1\n"},
    }
    self.assertIsNone(tpu_client.fetch_tpu_spec(port=8000))

  def _run_ensure_without_config(self, spec):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
      fake_tools = os.path.join(tmp, "tools")
      os.makedirs(fake_tools)
      with (
        patch.object(
          tpu_client, "__file__", os.path.join(fake_tools, "tpu_client.py")
        ),
        patch("tpu_client.check_health", return_value=True),
        patch("tpu_client.fetch_tpu_spec", return_value=spec),
      ):
        tpu_client.ensure_tpu_specs_cached()
      path = os.path.join(tmp, "tpu_config.json")
      if not os.path.exists(path):
        return None
      with open(path) as f:
        return json.load(f)

  def test_ensure_tpu_specs_cached_creates_config_for_local_tpu(self):
    raw = self._run_ensure_without_config(
      {"device_kind": "TPU v6 lite", "device_count": 1}
    )
    self.assertEqual(
      raw,
      {
        "tpus": [
          {
            "mode": "local",
            "local_port": 8000,
            "tpu_spec": {"device_kind": "TPU v6 lite", "device_count": 1},
            "tpu_spec_verified": True,
            "tpu_version": "TPU v6e",
          }
        ]
      },
    )

  def test_ensure_tpu_specs_cached_no_tpu_writes_no_config(self):
    self.assertIsNone(self._run_ensure_without_config(None))

  def _run_ensure_with_entry(self, entry, spec):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "tpu_config.json")
      with open(path, "w") as f:
        json.dump({"tpus": [entry]}, f)
      with (
        patch("tpu_client.check_health", return_value=True),
        patch("tpu_client.fetch_tpu_spec", return_value=spec) as mock_fetch,
      ):
        tpu_client.ensure_tpu_specs_cached(config_path_override=path)
      with open(path) as f:
        return json.load(f)["tpus"][0], mock_fetch

  def test_ensure_tpu_specs_cached_corrects_hand_written_spec(self):
    entry, mock_fetch = self._run_ensure_with_entry(
      {
        "mode": "local",
        "local_port": 8000,
        "tpu_version": "TPU v6e",
        "tpu_spec": {"device_kind": "TPU v6 lite", "device_count": 8},
      },
      {"device_kind": "TPU v6 lite", "device_count": 1},
    )
    mock_fetch.assert_called_once()
    self.assertEqual(entry["tpu_spec"]["device_count"], 1)
    self.assertTrue(entry["tpu_spec_verified"])

  def test_ensure_tpu_specs_cached_skips_verified_spec(self):
    verified = {
      "mode": "local",
      "local_port": 8000,
      "tpu_version": "TPU v6e",
      "tpu_spec": {"device_kind": "TPU v6 lite", "device_count": 1},
      "tpu_spec_verified": True,
    }
    entry, mock_fetch = self._run_ensure_with_entry(dict(verified), None)
    mock_fetch.assert_not_called()
    self.assertEqual(entry, verified)

  def test_ensure_tpu_specs_cached_keeps_unverified_spec_if_unreachable(self):
    hand = {
      "mode": "local",
      "local_port": 8000,
      "tpu_spec": {"device_kind": "TPU v6 lite", "device_count": 8},
    }
    entry, _ = self._run_ensure_with_entry(dict(hand), None)
    self.assertEqual(entry, hand)


if __name__ == "__main__":
  unittest.main()
