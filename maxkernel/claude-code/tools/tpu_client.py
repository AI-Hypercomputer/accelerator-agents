"""TPU execution client wrapper for submitting jobs to TPU servers."""

import argparse
import base64
import copy
import fcntl
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SERVER_URL = "http://127.0.0.1:8000"


def extract_trace_artifacts(res, output_dir=None, code_file=None, run_dir=None):
  """Extracts base64 encoded tar.gz trace archive if present in job result dict."""
  if not isinstance(res, dict) or not res.get("trace_tar_b64"):
    return None

  target_dir = output_dir
  if not target_dir and code_file:
    target_dir = os.path.dirname(os.path.abspath(code_file))
  if not target_dir and run_dir:
    target_dir = os.path.abspath(run_dir)
  if not target_dir:
    target_dir = os.getcwd()

  profile_dir = os.path.join(target_dir, "profile")
  os.makedirs(profile_dir, exist_ok=True)

  try:
    b64_data = res["trace_tar_b64"]
    raw_bytes = base64.b64decode(b64_data)
    buf = io.BytesIO(raw_bytes)

    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
      tar.extractall(path=profile_dir)

    logging.info("Extracted trace artifacts to: %s", profile_dir)

    xplane_path = None
    for root, _, files in os.walk(profile_dir):
      for f in files:
        if f.endswith(".xplane.pb"):
          xplane_path = os.path.join(root, f)
          break
      if xplane_path:
        break

    print("\n" + "=" * 40)
    print("Trace Artifacts Downloaded:")
    print("=" * 40)
    print(f"Profile Directory: {profile_dir}")
    if xplane_path:
      print(f"Local XPlane Trace File: {xplane_path}")
    print("=" * 40)

    res["trace_tar_b64"] = None
    return profile_dir
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Failed to extract trace artifacts: %s", e)
    res["trace_tar_b64"] = None
    return None


def get_tpu_config(config_path_override=None, run_dir=None):
  """Read TPU config file(s), resolving single or multi-TPU configurations."""
  config_path = None
  if config_path_override and os.path.exists(config_path_override):
    config_path = config_path_override
  elif run_dir:
    run_config = os.path.join(run_dir, "tpu_config.json")
    if os.path.exists(run_config):
      config_path = run_config

  if not config_path:
    default_path = os.path.join(
      os.path.dirname(__file__), "..", "tpu_config.json"
    )
    if os.path.exists(default_path):
      config_path = default_path

  raw_data = {}
  if config_path and os.path.exists(config_path):
    try:
      with open(config_path, "r") as f:
        raw_data = json.load(f) or {}
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error(
        "Failed to read tpu_config.json from %s: %s", config_path, e
      )

  tpu_list = []
  if isinstance(raw_data, dict):
    if "tpus" in raw_data and isinstance(raw_data["tpus"], list):
      tpu_list = raw_data["tpus"]
    elif raw_data:
      tpu_list = [raw_data]
  elif isinstance(raw_data, list):
    tpu_list = raw_data

  normalized_configs = []
  for idx, item in enumerate(tpu_list):
    cfg = copy.deepcopy(item)
    if "local_port" not in cfg or not cfg["local_port"]:
      cfg["local_port"] = 8000 + idx
    if "mode" not in cfg or not cfg["mode"]:
      cfg["mode"] = "remote" if "tpu_name" in cfg else "local"
    normalized_configs.append(cfg)

  return normalized_configs


def check_health(port=8000, timeout=5):
  """Check if the TPU server on local_port is running and reachable."""
  req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
  try:
    with urllib.request.urlopen(req, timeout=timeout) as response:
      if response.status != 200:
        return False
      data = json.loads(response.read().decode())
      return data.get("status") == "healthy"
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def ssh_cmd(config, cmd, bg=False):
  """Run an SSH command on the TPU VM."""
  base = [
    "/usr/bin/gcloud",
    "compute",
    "tpus",
    "tpu-vm",
    "ssh",
    config["tpu_name"],
    "--zone",
    config["zone"],
    "--project",
    config["project"],
    "--command",
    cmd,
  ]
  if bg:
    return subprocess.Popen(
      base,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      start_new_session=True,
    )
  return subprocess.run(base, capture_output=True, text=True, check=False)


def scp_cmd(config, local_path, remote_path):
  """SCP a file to the TPU VM."""
  base = [
    "/usr/bin/gcloud",
    "compute",
    "tpus",
    "tpu-vm",
    "scp",
    local_path,
    f"{config['tpu_name']}:{remote_path}",
    "--zone",
    config["zone"],
    "--project",
    config["project"],
  ]
  return subprocess.run(base, capture_output=True, text=True, check=False)


def ensure_ssh_tunnel(config):
  """Ensure local SSH port-forwarding tunnel to the TPU VM is established and healthy."""
  if not all(k in config for k in ("tpu_name", "zone", "project")):
    return False

  port = config.get("local_port", 8000)
  logging.info(
    "Establishing/restoring SSH tunnel for %s on local port %s...",
    config["tpu_name"],
    port,
  )
  tunnel_cmd = [
    "/usr/bin/gcloud",
    "compute",
    "tpus",
    "tpu-vm",
    "ssh",
    config["tpu_name"],
    "--zone",
    config["zone"],
    "--project",
    config["project"],
    "--",
    "-N",
    "-L",
    f"{port}:localhost:8000",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "ExitOnForwardFailure=yes",
  ]
  subprocess.Popen(
    tunnel_cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
  )

  for _ in range(15):
    if check_health(port=port, timeout=3):
      logging.info(
        "SSH tunnel for %s connected successfully on port %s.",
        config["tpu_name"],
        port,
      )
      return True
    time.sleep(1)
  return False


def start_server_for_tpu(config, mode=None):
  """Start a single TPU server lazily if not running."""
  port = config.get("local_port", 8000)
  if check_health(port=port, timeout=5):
    return True

  for _ in range(2):
    time.sleep(2)
    if check_health(port=port, timeout=5):
      logging.info("TPU server on port %s responded on retry.", port)
      return True

  target_mode = mode or config.get("mode")
  if not target_mode:
    target_mode = "remote" if "tpu_name" in config else "local"

  if target_mode not in ("local", "remote"):
    logging.error(
      "Invalid mode '%s' for TPU server on port %s.", target_mode, port
    )
    return False

  tpu_id = config.get("tpu_name", f"port_{port}")
  lock_file = f"/tmp/maxkernel_tpu_setup_{tpu_id}.lock"
  lock_fd = os.open(lock_file, os.O_RDWR | os.O_CREAT)
  try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not check_health(port=port, timeout=5):
      if target_mode == "local":
        logging.info(
          "TPU Server on port %s is down. Initiating local lazy start...",
          port,
        )
        server_path = os.path.abspath(
          os.path.join(
            os.path.dirname(__file__), "..", "server", "tpu_server.py"
          )
        )
        log_file_path = os.path.abspath(
          os.path.join(os.path.dirname(__file__), "..", f"server_{port}.log")
        )
        logging.info(
          "Booting local server daemon on port %s: %s", port, server_path
        )
        with open(log_file_path, "a") as log_f:
          subprocess.Popen(
            [sys.executable, server_path],
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,
          )
        for _ in range(15):
          if check_health(port=port, timeout=5):
            logging.info("Local TPU Server on port %s is up!", port)
            break
          time.sleep(1)
        else:
          logging.error("Local TPU Server on port %s failed to boot.", port)
          return False
      else:
        if not all(k in config for k in ("tpu_name", "zone", "project")):
          logging.error(
            "Remote TPU config missing required keys for %s.", config
          )
          return False

        logging.info(
          "Testing remote SSH tunnel for %s on port %s...",
          config["tpu_name"],
          port,
        )
        if ensure_ssh_tunnel(config):
          if check_health(port=port, timeout=5):
            logging.info(
              "Remote server on port %s restored via SSH tunnel!", port
            )
            return True

        logging.info(
          "Remote TPU Server for %s is down. Initiating remote start...",
          config["tpu_name"],
        )
        req_path = os.path.join(
          os.path.dirname(__file__), "..", "requirements.txt"
        )
        server_path = os.path.join(
          os.path.dirname(__file__), "..", "server", "tpu_server.py"
        )

        ssh_cmd(config, "mkdir -p ~/maxkernel_deploy")
        if os.path.exists(req_path):
          scp_cmd(config, req_path, "~/maxkernel_deploy/requirements.txt")
        scp_cmd(config, server_path, "~/maxkernel_deploy/tpu_server.py")

        setup_script = """
                cd ~/maxkernel_deploy
                if [ ! -d "$HOME/maxkernel_venv" ]; then
                    echo "Setting up venv..."
                    sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
                    python3.12 -m venv ~/maxkernel_venv
                    source ~/maxkernel_venv/bin/activate
                    python3 -m pip install --upgrade pip
                    if [ -f requirements.txt ]; then
                        python3 -m pip install -r requirements.txt --quiet
                    else
                        python3 -m pip install jax==0.11.0 jaxlib==0.11.0 libtpu==0.0.47 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
                    fi
                fi
                source ~/maxkernel_venv/bin/activate
                pkill -9 -f run_cod[e].py || true
                fuser -k 8000/tcp || true
                sudo rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*lock*
                nohup python3 tpu_server.py > server.log 2>&1 < /dev/null &
                """
        logging.info(
          "Executing setup on remote TPU VM %s...", config["tpu_name"]
        )
        res = ssh_cmd(config, setup_script)
        if res.returncode != 0:
          logging.error("Remote setup failed: %s", res.stderr)
          return False

        if not ensure_ssh_tunnel(config):
          logging.error(
            "Server on port %s failed to boot or tunnel failed.", port
          )
          return False
        logging.info("Remote Server on port %s is up and connected!", port)

  except BlockingIOError:
    logging.info("Setup lock active for %s. Waiting...", tpu_id)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    if not check_health(port=port, timeout=5):
      logging.error("Setup lock released, but server on port %s is down.", port)
      return False
  finally:
    try:
      fcntl.flock(lock_fd, fcntl.LOCK_UN)
      os.close(lock_fd)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  return True


def fetch_tpu_spec(port=8000):
  """Fetch device_kind and device_count from TPU server running on specified port."""
  spec_code = (
    "import jax\n"
    "devs = jax.devices()\n"
    "print(f'DEVICE_KIND:{devs[0].device_kind}')\n"
    "print(f'DEVICE_COUNT:{len(devs)}')\n"
  )
  res = submit_job("correctness_test", spec_code, timeout=60, port=port)
  if not res or "job_id" not in res:
    return None
  job_id = res["job_id"]
  start_t = time.time()
  while time.time() - start_t < 90:
    j_info = check_job_status(job_id)
    if isinstance(j_info, dict) and j_info.get("status") in (
      "completed",
      "failed",
    ):
      r = j_info.get("result") or {}
      out = r.get("output", "")
      kind_match = re.search(r"DEVICE_KIND:(.+)", out)
      count_match = re.search(r"DEVICE_COUNT:(\d+)", out)
      if kind_match and count_match:
        return {
          "device_kind": kind_match.group(1).strip(),
          "device_count": int(count_match.group(1).strip()),
        }
      break
    time.sleep(2)
  return None


def infer_tpu_version(tpu_entry):
  """Infer standardized TPU version string (e.g., 'TPU v5p', 'TPU v6e', 'TPU v7x') from TPU config entry."""
  if not isinstance(tpu_entry, dict):
    return "TPU v6e"
  if tpu_entry.get("tpu_version"):
    return tpu_entry["tpu_version"]
  spec = tpu_entry.get("tpu_spec") or {}
  device_kind = spec.get("device_kind", "")
  tpu_name = tpu_entry.get("tpu_name", "")

  dk_lower = device_kind.lower()
  if "v6 lite" in dk_lower or "v6e" in dk_lower:
    return "TPU v6e"
  elif "v5 lite" in dk_lower or "v5e" in dk_lower:
    return "TPU v5e"
  elif "v5p" in dk_lower:
    return "TPU v5p"
  elif "v7x" in dk_lower or "v7" in dk_lower:
    return "TPU v7x"
  elif "v4" in dk_lower:
    return "TPU v4"

  name_lower = tpu_name.lower()
  if "v6e" in name_lower or "v6-lite" in name_lower:
    return "TPU v6e"
  elif "v5p" in name_lower:
    return "TPU v5p"
  elif "v5e" in name_lower or "v5-lite" in name_lower:
    return "TPU v5e"
  elif "v7x" in name_lower:
    return "TPU v7x"
  elif "v4" in name_lower:
    return "TPU v4"

  if device_kind:
    return device_kind
  return "TPU v6e"


def ensure_tpu_specs_cached(config_path_override=None, run_dir=None):
  """Populate missing tpu_spec and tpu_version fields in tpu_config.json if TPU servers are reachable."""
  config_path = config_path_override
  if not config_path and run_dir:
    run_config = os.path.join(run_dir, "tpu_config.json")
    if os.path.exists(run_config):
      config_path = run_config

  if not config_path:
    default_path = os.path.join(
      os.path.dirname(__file__), "..", "tpu_config.json"
    )
    if os.path.exists(default_path):
      config_path = default_path

  if not config_path or not os.path.exists(config_path):
    return

  try:
    with open(config_path, "r") as f:
      raw = json.load(f) or {}

    tpu_list = []
    if (
      isinstance(raw, dict) and "tpus" in raw and isinstance(raw["tpus"], list)
    ):
      tpu_list = raw["tpus"]
    elif isinstance(raw, dict) and raw:
      tpu_list = [raw]
    elif isinstance(raw, list):
      tpu_list = raw

    updated = False
    for idx, item in enumerate(tpu_list):
      if isinstance(item, dict):
        port = item.get("local_port") or (8000 + idx)
        if "tpu_spec" not in item:
          if check_health(port=port, timeout=3):
            logging.info("Auto-fetching tpu_spec for TPU on port %s...", port)
            spec = fetch_tpu_spec(port=port)
            if spec:
              item["tpu_spec"] = spec
              updated = True
        if "tpu_version" not in item:
          item["tpu_version"] = infer_tpu_version(item)
          updated = True

    if updated:
      with open(config_path, "w") as f:
        json.dump(raw, f, indent=2)
      logging.info(
        "Updated %s with auto-cached tpu_spec/tpu_version.", config_path
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("Could not auto-update tpu_spec in config: %s", e)


def cleanup_stale_temp_files(config_dir, max_age_seconds=600):
  """Prune abandoned .tpu_config.json.tmp.* files older than max_age_seconds."""
  try:
    if not os.path.exists(config_dir):
      return
    now = time.time()
    for fname in os.listdir(config_dir):
      if fname.startswith(".tpu_config.json.tmp."):
        fpath = os.path.join(config_dir, fname)
        try:
          if now - os.path.getmtime(fpath) > max_age_seconds:
            os.remove(fpath)
            logging.info("Pruned stale temp config file: %s", fpath)
        except Exception:  # pylint: disable=broad-exception-caught
          pass
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("Temp file cleanup error: %s", e)


def add_tpu_to_config(new_tpu_input, config_path_override=None, run_dir=None):
  """Safely add a TPU configuration entry to tpu_config.json with file locking, deduplication, auto spec fetching, and atomic replace."""
  if isinstance(new_tpu_input, str):
    try:
      new_tpu = json.loads(new_tpu_input)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Invalid JSON input for --add_tpu: %s", e)
      return False
  elif isinstance(new_tpu_input, dict):
    new_tpu = new_tpu_input
  else:
    logging.error("Invalid input type for add_tpu_to_config.")
    return False

  if not isinstance(new_tpu, dict) or (
    "tpu_name" not in new_tpu and new_tpu.get("mode") != "local"
  ):
    logging.error(
      "TPU entry must be a dict containing 'tpu_name' or 'mode': 'local'."
    )
    return False

  config_path = config_path_override
  if not config_path and run_dir:
    run_config = os.path.join(run_dir, "tpu_config.json")
    if os.path.exists(run_config):
      config_path = run_config

  if not config_path:
    config_path = os.path.join(
      os.path.dirname(__file__), "..", "tpu_config.json"
    )

  lock_file = "/tmp/maxkernel_tpu_config_update.lock"
  lock_fd = os.open(lock_file, os.O_RDWR | os.O_CREAT)
  try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    raw_data = {}
    if os.path.exists(config_path):
      try:
        with open(config_path, "r") as f:
          raw_data = json.load(f) or {}
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
          "Could not read existing config at %s: %s", config_path, e
        )

    if (
      isinstance(raw_data, dict)
      and "tpus" in raw_data
      and isinstance(raw_data["tpus"], list)
    ):
      tpu_list = raw_data["tpus"]
    elif isinstance(raw_data, dict) and raw_data:
      tpu_list = [raw_data]
    elif isinstance(raw_data, list):
      tpu_list = raw_data
    else:
      tpu_list = []

    if new_tpu.get("mode") == "local":
      tpu_identifier = "local"
      existing = next(
        (
          item
          for item in tpu_list
          if isinstance(item, dict) and item.get("mode") == "local"
        ),
        None,
      )
    else:
      tpu_identifier = new_tpu.get("tpu_name", "unknown")
      existing = next(
        (
          item
          for item in tpu_list
          if isinstance(item, dict) and item.get("tpu_name") == tpu_identifier
        ),
        None,
      )

    if existing:
      logging.info(
        "TPU '%s' is already present in %s. No duplicate added.",
        tpu_identifier,
        config_path,
      )
      target_entry = existing
    else:
      tpu_list.append(new_tpu)
      target_entry = new_tpu
      logging.info("Added TPU '%s' to config list.", tpu_identifier)

    new_config_structure = {"tpus": tpu_list}

    dir_name = os.path.dirname(os.path.abspath(config_path))
    os.makedirs(dir_name, exist_ok=True)
    cleanup_stale_temp_files(dir_name)
    tmp_file = os.path.join(
      dir_name,
      f".tpu_config.json.tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}",
    )
    with open(tmp_file, "w") as f:
      json.dump(new_config_structure, f, indent=2)
    os.replace(tmp_file, config_path)
    logging.info("Safely updated %s.", config_path)

    if "tpu_version" not in target_entry:
      target_entry["tpu_version"] = infer_tpu_version(target_entry)

    port = target_entry.get("local_port") or (8000 + (len(tpu_list) - 1))
    target_entry["local_port"] = port
    start_server_for_tpu(target_entry)
    if check_health(port=port, timeout=5):
      if "tpu_spec" not in target_entry:
        logging.info(
          "Fetching tpu_spec for newly added TPU '%s'...", tpu_identifier
        )
        spec = fetch_tpu_spec(port=port)
        if spec:
          target_entry["tpu_spec"] = spec
          target_entry["tpu_version"] = infer_tpu_version(target_entry)
          tmp_file2 = os.path.join(
            dir_name,
            f".tpu_config.json.tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}",
          )
          with open(tmp_file2, "w") as f:
            json.dump(new_config_structure, f, indent=2)
          os.replace(tmp_file2, config_path)
          logging.info(
            "Updated %s with tpu_spec/tpu_version in %s.",
            tpu_identifier,
            config_path,
          )

    return True
  finally:
    try:
      fcntl.flock(lock_fd, fcntl.LOCK_UN)
      os.close(lock_fd)
    except Exception:  # pylint: disable=broad-exception-caught
      pass


def start_server_idempotent(
  mode=None, tpu_configs=None, config_path_override=None, run_dir=None
):
  """Lazily start TPU servers for all configured TPUs and auto-cache specs."""
  if tpu_configs is not None:
    configs = tpu_configs
  else:
    configs = get_tpu_config(
      config_path_override=config_path_override, run_dir=run_dir
    )
  if not configs and mode == "local":
    configs = [{"mode": "local", "local_port": 8000, "tpu_version": "TPU v6e"}]
  if not configs:
    logging.error("No TPU configurations found.")
    return False

  success_count = 0
  for cfg in configs:
    if start_server_for_tpu(cfg, mode=mode):
      success_count += 1

  if success_count > 0:
    ensure_tpu_specs_cached(
      config_path_override=config_path_override, run_dir=run_dir
    )

  return success_count > 0


def get_queue_info(port=8000):
  """Query server queue info on specified port."""
  endpoint = f"http://127.0.0.1:{port}/queue"
  req = urllib.request.Request(endpoint, method="GET")
  try:
    with urllib.request.urlopen(req, timeout=10) as response:
      return json.loads(response.read().decode())
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Error checking queue on port %s: %s", port, e)
    return {
      "error": f"Error checking queue: {str(e)}",
      "total_jobs": 0,
      "queued_count": 0,
      "running_count": 0,
    }


def cancel_jobs(port=8000, job_id=None):
  """Cancels a specific job, or all active jobs, on the server at `port`."""
  endpoint = f"http://127.0.0.1:{port}/cancel"
  if job_id:
    endpoint += f"?job_id={urllib.parse.quote(job_id)}"
  req = urllib.request.Request(endpoint, method="POST")
  try:
    with urllib.request.urlopen(req, timeout=30) as response:
      return json.loads(response.read().decode())
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Error cancelling jobs on port %s: %s", port, e)
    return {"error": f"Error cancelling jobs: {str(e)}", "cancelled_count": 0}


def select_best_tpu_server(tpu_configs=None):
  """Select the TPU server with the lowest queue/execution load."""
  configs = tpu_configs if tpu_configs is not None else get_tpu_config()
  if not configs:
    return None, None

  healthy_candidates = []
  for cfg in configs:
    port = cfg.get("local_port", 8000)
    if check_health(port=port, timeout=3):
      q_info = get_queue_info(port=port)
      if "error" not in q_info or not q_info.get("error"):
        queued = q_info.get("queued_count", 0)
        running = q_info.get("running_count", 0)
        score = (queued * 10) + (1 if running > 0 else 0)
        healthy_candidates.append((score, port, cfg))

  if not healthy_candidates:
    start_server_idempotent(tpu_configs=configs)
    for cfg in configs:
      port = cfg.get("local_port", 8000)
      if check_health(port=port, timeout=3):
        q_info = get_queue_info(port=port)
        if "error" not in q_info or not q_info.get("error"):
          queued = q_info.get("queued_count", 0)
          running = q_info.get("running_count", 0)
          score = (queued * 10) + (1 if running > 0 else 0)
          healthy_candidates.append((score, port, cfg))

  if not healthy_candidates:
    return None, None

  min_score = min(item[0] for item in healthy_candidates)
  best_pool = [item for item in healthy_candidates if item[0] == min_score]
  chosen = random.choice(best_pool)
  return chosen[1], chosen[2]


def strip_markdown(code: str) -> str:
  code_content = code.strip()
  if code_content.startswith("```"):
    lines = code_content.split("\n")
    if lines[0].startswith("```"):
      lines = lines[1:]
    if lines and lines[-1].strip() == "```":
      lines = lines[:-1]
    return "\n".join(lines).strip()
  return code_content


def submit_job(action: str, code: str, timeout: int, port=8000):
  """Submit job to TPU server running on specified local port."""
  endpoint = f"http://127.0.0.1:{port}/submit"
  if action == "autotune":
    try:
      payload_dict = json.loads(code)
      if "timeout" not in payload_dict:
        payload_dict["timeout"] = 120
      if "total_timeout" not in payload_dict:
        payload_dict["total_timeout"] = max(timeout, 1800)
      autotune_req = payload_dict
    except json.JSONDecodeError:
      autotune_req = {
        "code_template": code,
        "search_space": {},
        "timeout": 120,
        "total_timeout": max(timeout, 1800),
      }
    submission = {"action": action, "autotune_request": autotune_req}
  else:
    code_req = {"code": code, "timeout": max(timeout, 180)}
    submission = {"action": action, "code_request": code_req}

  payload = json.dumps(submission).encode("utf-8")
  req = urllib.request.Request(
    endpoint,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
  )

  try:
    with urllib.request.urlopen(req, timeout=30) as response:
      res = json.loads(response.read().decode())
      if isinstance(res, dict) and "job_id" in res:
        orig_job_id = res["job_id"]
        res["job_id"] = f"job_p{port}_{orig_job_id}"
      return res
  except urllib.error.HTTPError as e:
    error_msg = e.read().decode()
    logging.error("HTTPError submitting job to port %s: %s", port, error_msg)
    return {"error": f"HTTPError {e.code}: {error_msg}", "status": "failed"}
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Error submitting job to port %s: %s", port, e)
    return {"error": f"Error submitting job: {str(e)}", "status": "failed"}


def check_job_status(job_id: str, tpu_configs=None):
  """Check status of a job ID across TPU servers."""
  match = re.match(r"^job_p(\d+)_(.+)$", job_id)
  target_ports = []
  raw_job_id = job_id
  if match:
    target_ports.append(int(match.group(1)))
    raw_job_id = match.group(2)
  else:
    configs = tpu_configs if tpu_configs is not None else get_tpu_config()
    if configs:
      target_ports = [cfg.get("local_port", 8000) for cfg in configs]
    else:
      target_ports = [8000]

  last_error = None
  for port in target_ports:
    endpoint = f"http://127.0.0.1:{port}/job/{raw_job_id}"
    req = urllib.request.Request(endpoint, method="GET")
    try:
      with urllib.request.urlopen(req, timeout=10) as response:
        res = json.loads(response.read().decode())
        if isinstance(res, dict):
          res["job_id"] = job_id
        return res
    except urllib.error.HTTPError as e:
      if e.code == 404:
        endpoint_orig = f"http://127.0.0.1:{port}/job/{job_id}"
        try:
          req_orig = urllib.request.Request(endpoint_orig, method="GET")
          with urllib.request.urlopen(req_orig, timeout=10) as resp2:
            res = json.loads(resp2.read().decode())
            if isinstance(res, dict):
              res["job_id"] = job_id
            return res
        except Exception:  # pylint: disable=broad-exception-caught
          pass
        last_error = {
          "job_id": job_id,
          "status": "not_found",
          "error": (
            f"Job '{job_id}' not found on server port {port} (HTTP 404)"
          ),
          "http_code": 404,
        }
      else:
        last_error = {
          "job_id": job_id,
          "status": "http_error",
          "error": f"HTTP Error {e.code}",
          "http_code": e.code,
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
      last_error = {
        "job_id": job_id,
        "status": "network_error",
        "error": f"Network error: {str(e)}",
      }

  return last_error or {
    "job_id": job_id,
    "status": "not_found",
    "error": "Job not found",
  }


def shard_search_space(
  search_space: dict[str, Any], num_shards: int
) -> list[dict[str, Any]]:
  """Recursively split search_space into up to num_shards non-overlapping sub-search-spaces."""
  if num_shards <= 1 or not search_space:
    return [search_space]

  def count_combos(ss):
    combos = 1
    for v in ss.values():
      combos *= len(v) if isinstance(v, list) and len(v) > 0 else 1
    return combos

  sub_spaces = [copy.deepcopy(search_space)]

  while len(sub_spaces) < num_shards:
    best_idx = -1
    best_combos = -1
    best_key = None

    for idx, ss in enumerate(sub_spaces):
      c = count_combos(ss)
      if c > 1 and c > best_combos:
        splittable_keys = [
          k for k, v in ss.items() if isinstance(v, list) and len(v) > 1
        ]
        if splittable_keys:
          key_to_split = max(
            splittable_keys, key=lambda k, current_ss=ss: len(current_ss[k])
          )
          best_idx = idx
          best_combos = c
          best_key = key_to_split

    if best_idx == -1 or not best_key:
      break

    target_ss = sub_spaces.pop(best_idx)
    vals = target_ss[best_key]
    mid = len(vals) // 2

    ss1 = copy.deepcopy(target_ss)
    ss2 = copy.deepcopy(target_ss)
    ss1[best_key] = vals[:mid]
    ss2[best_key] = vals[mid:]

    sub_spaces.insert(best_idx, ss1)
    sub_spaces.insert(best_idx + 1, ss2)

  return sub_spaces


def main():
  parser = argparse.ArgumentParser(
    description="TPU Execution Client wrapper with Async Job Queue support."
  )
  parser.add_argument(
    "--mode",
    choices=["local", "remote"],
    default=None,
    help=(
      "Target execution environment: 'local' (agent running on TPU VM) or"
      " 'remote' (agent accessing TPU VM remotely). Defaults to mode in"
      " tpu_config.json."
    ),
  )
  parser.add_argument(
    "--tpu_config",
    default=None,
    help="Path to tpu_config.json file.",
  )
  parser.add_argument(
    "--run_dir",
    default=None,
    help="Run directory containing state.json / tpu_config.json.",
  )
  parser.add_argument(
    "--action",
    choices=[
      "compilation_test",
      "correctness_test",
      "performance_test",
      "profile",
      "autotune",
    ],
    help="Endpoint to call on server.",
  )
  parser.add_argument(
    "--code_file", help="Path to Python code or JSON payload to execute."
  )
  parser.add_argument(
    "--timeout",
    type=int,
    default=600,
    help="Timeout in seconds for execution (default 600s).",
  )
  parser.add_argument(
    "--poll_interval",
    type=float,
    default=2.0,
    help="Interval in seconds to poll job status when queued.",
  )
  parser.add_argument(
    "--submit_only",
    action="store_true",
    help="Submit job to queue and exit immediately with job_id.",
  )
  parser.add_argument("--check_job", help="Check status of a specific job_id.")
  parser.add_argument(
    "--queue",
    action="store_true",
    help="Show current queue status of TPU server.",
  )
  parser.add_argument(
    "--cancel_job",
    nargs="?",
    const="__ALL__",
    default=None,
    help=(
      "Cancel a hanging TPU job. Pass a job_id to cancel one job, or use"
      " the bare flag to cancel every queued/running job on all"
      " configured servers."
    ),
  )
  parser.add_argument(
    "--add_tpu",
    help=(
      "JSON string or file path containing TPU entry (tpu_name, zone,"
      " project) to safely add to tpu_config.json."
    ),
  )
  parser.add_argument(
    "--output_dir",
    default=None,
    help=(
      "Directory to save extracted trace artifacts (defaults to directory"
      " of code_file or run_dir)."
    ),
  )
  args = parser.parse_args()

  if args.add_tpu:
    tpu_input = args.add_tpu
    if os.path.exists(args.add_tpu):
      with open(args.add_tpu, "r") as f:
        tpu_input = f.read()
    if add_tpu_to_config(
      tpu_input, config_path_override=args.tpu_config, run_dir=args.run_dir
    ):
      print("TPU machine successfully added and configured.")
      sys.exit(0)
    else:
      logging.error("Failed to add TPU machine to config.")
      sys.exit(1)

  tpu_configs = get_tpu_config(
    config_path_override=args.tpu_config, run_dir=args.run_dir
  )

  if not start_server_idempotent(mode=args.mode, tpu_configs=tpu_configs):
    logging.error("Cannot proceed: No TPU server is reachable.")
    sys.exit(1)

  if args.cancel_job:
    # Client-facing job ids are namespaced as "job_p<port>_<raw_job_id>" (see
    # check_job_status) so they stay unique across a multi-TPU pool, but the
    # server only knows the raw id. Strip the prefix and target just that
    # server; a bare --cancel_job fans out to every configured port.
    target = None
    ports = [cfg.get("local_port", 8000) for cfg in tpu_configs]
    if args.cancel_job != "__ALL__":
      match = re.match(r"^job_p(\d+)_(.+)$", args.cancel_job)
      if match:
        ports = [int(match.group(1))]
        target = match.group(2)
      else:
        target = args.cancel_job
    total = 0
    for port in ports:
      info = cancel_jobs(port=port, job_id=target)
      count = info.get("cancelled_count", 0)
      total += count
      if info.get("error"):
        print(f"Port {port}: {info['error']}")
      else:
        print(f"Port {port}: cancelled {count} job(s).")
        for j in info.get("cancelled", []):
          print(f"  - {j['job_id']} ({j['action']})")
    print(f"Total jobs cancelled: {total}")
    sys.exit(0)

  if args.queue:
    for cfg in tpu_configs:
      port = cfg.get("local_port", 8000)
      q_info = get_queue_info(port=port)
      print("\n" + "=" * 40)
      print(f"TPU Server Queue Status (Port {port}):")
      print("=" * 40)
      if q_info and "error" not in q_info:
        print(f"Total Jobs: {q_info.get('total_jobs', 0)}")
        print(f"Queued Count: {q_info.get('queued_count', 0)}")
        print(f"Running Count: {q_info.get('running_count', 0)}")
        if q_info.get("running_jobs"):
          print(f"Running Jobs: {q_info['running_jobs']}")
        if q_info.get("queued_jobs"):
          print(f"Queued Jobs: {q_info['queued_jobs']}")
      else:
        print(f"Failed to retrieve queue info for port {port}.")
    sys.exit(0)

  if args.check_job:
    job_info = check_job_status(args.check_job, tpu_configs=tpu_configs)
    if not job_info or job_info.get("status") in (
      "not_found",
      "http_error",
      "network_error",
    ):
      err = (
        job_info.get("error") if isinstance(job_info, dict) else "Unknown error"
      )
      logging.error("Job '%s' query failed: %s", args.check_job, err)
      sys.exit(1)

    status = job_info.get("status")
    pos = job_info.get("queue_position", 0)
    res = job_info.get("result") or {}

    print("\n" + "=" * 40)
    print("Job Status Report:")
    print("=" * 40)
    print(f"Job ID: {job_info.get('job_id')}")
    print(f"Action: {job_info.get('action')}")
    print(f"Status: {status}")
    if status == "queued":
      print(f"Queue Position: {pos}")
    if job_info.get("created_at"):
      print(f"Created At: {time.ctime(job_info['created_at'])}")
    if job_info.get("started_at"):
      print(f"Started At: {time.ctime(job_info['started_at'])}")
    if job_info.get("completed_at"):
      print(f"Completed At: {time.ctime(job_info['completed_at'])}")
    if job_info.get("started_at") and job_info.get("completed_at"):
      run_duration = job_info["completed_at"] - job_info["started_at"]
      print(f"Execution Duration (excluding queue): {run_duration:.2f}s")
    if job_info.get("created_at") and job_info.get("started_at"):
      queue_duration = job_info["started_at"] - job_info["created_at"]
      print(f"Queue Wait Duration: {queue_duration:.2f}s")

    if res:
      print(f"Exit Code: {res.get('exit_code', 1)}")
      if res.get("error"):
        err_str = res["error"]
        if len(err_str) > 100000:
          err_str = (
            err_str[:100000]
            + f"\n... [STDERR truncated from {len(res['error'])} chars]"
          )
        print(f"\nSTDERR:\n{err_str}")
      if res.get("output"):
        out_str = res["output"]
        if len(out_str) > 100000:
          out_str = (
            out_str[:100000]
            + f"\n... [STDOUT truncated from {len(res['output'])} chars]"
          )
        print(f"\nSTDOUT:\n{out_str}")
      extract_trace_artifacts(
        res, output_dir=args.output_dir, run_dir=args.run_dir
      )
    sys.exit(0 if (status == "completed" and res.get("exit_code") == 0) else 1)

  if not args.action or not args.code_file:
    parser.error(
      "Both --action and --code_file are required unless using --check_job or"
      " --queue."
    )

  if not os.path.exists(args.code_file):
    logging.error("File not found: %s", args.code_file)
    sys.exit(1)

  with open(args.code_file, "r") as f:
    code = strip_markdown(f.read())

  effective_timeout = (
    max(args.timeout, 1800) if args.action == "autotune" else args.timeout
  )

  # Check if autotune sharding across multiple healthy TPUs is applicable
  healthy_configs = [
    cfg for cfg in tpu_configs if check_health(port=cfg.get("local_port", 8000))
  ]

  if (
    args.action == "autotune"
    and len(healthy_configs) > 1
    and not args.submit_only
  ):
    try:
      payload_dict = json.loads(code)
      search_space = payload_dict.get("search_space", {})
      shards = shard_search_space(search_space, len(healthy_configs))
      if len(shards) > 1:
        logging.info(
          "Sharding autotune sweep across %d healthy TPU servers...",
          len(shards),
        )
        sharded_jobs = []
        for i, shard in enumerate(shards):
          shard_payload = copy.deepcopy(payload_dict)
          shard_payload["search_space"] = shard
          target_port = healthy_configs[i]["local_port"]
          sub_res = submit_job(
            "autotune",
            json.dumps(shard_payload),
            effective_timeout,
            port=target_port,
          )
          if sub_res and "job_id" in sub_res:
            sharded_jobs.append((sub_res["job_id"], target_port))

        if sharded_jobs:
          logging.info(
            "Submitted %d sharded autotune sub-jobs. Polling for completion...",
            len(sharded_jobs),
          )
          all_shard_results = []
          success = True
          for sub_job_id, target_port in sharded_jobs:
            start_t = time.time()
            net_err_count = 0
            while True:
              j_info = check_job_status(sub_job_id, tpu_configs=tpu_configs)
              cur_status = (
                j_info.get("status") if isinstance(j_info, dict) else None
              )

              if cur_status == "network_error":
                net_err_count += 1
                err_detail = (
                  j_info.get("error", "") if isinstance(j_info, dict) else ""
                )
                if net_err_count < 3:
                  logging.info(
                    "Transient status check timeout on port %s (%s). TPU"
                    " server busy or tunnel latency; retrying (%d/5)...",
                    target_port,
                    err_detail,
                    net_err_count,
                  )
                else:
                  logging.warning(
                    "Persistent status check delay on port %s (%s) (retry"
                    " %d/5)...",
                    target_port,
                    err_detail,
                    net_err_count,
                  )
                if net_err_count >= 3:
                  for cfg in healthy_configs:
                    if cfg.get("local_port") == target_port:
                      ensure_ssh_tunnel(cfg)
                if net_err_count >= 5:
                  success = False
                  break
                time.sleep(args.poll_interval)
                continue
              else:
                if net_err_count > 0:
                  logging.info(
                    "Status check connection restored on port %s. Continuing"
                    " autotune polling...",
                    target_port,
                  )
                net_err_count = 0

              if cur_status in ("completed", "failed", "cancelled"):
                r = (j_info.get("result") or {}) if j_info else {}
                if cur_status == "completed" and r.get("exit_code") == 0:
                  try:
                    shard_data = json.loads(r.get("output", "{}"))
                    all_shard_results.extend(shard_data.get("all_results", []))
                  except Exception:  # pylint: disable=broad-exception-caught
                    pass
                else:
                  success = False
                break
              if time.time() - start_t > (effective_timeout + 1200):
                success = False
                break
              time.sleep(args.poll_interval)

          print("\n" + "=" * 40)
          print("Execution Result (Multi-TPU Sharded Autotune):")
          print("=" * 40)
          print(
            f"Total Evaluated Configurations Across Shards:"
            f" {len(all_shard_results)}"
          )
          print(f"Status: {'completed' if success else 'failed'}")
          print(f"Exit Code: {0 if success else 1}")
          print(f"\nSTDOUT:\n{json.dumps({'all_results': all_shard_results})}")
          sys.exit(0 if success else 1)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
        "Failed to setup sharded autotune: %s. Falling back to single server"
        " routing.",
        e,
      )

  # Select best TPU server by queue length
  best_port, selected_config = select_best_tpu_server(tpu_configs)
  if not best_port:
    best_port = 8000

  logging.info(
    "Submitting '%s' request to TPU server on port %s (timeout=%ds)...",
    args.action,
    best_port,
    effective_timeout,
  )
  submit_res = submit_job(args.action, code, effective_timeout, port=best_port)

  if (
    not submit_res
    or "job_id" not in submit_res
    or submit_res.get("status") == "failed"
  ):
    err = (
      submit_res.get("error")
      if isinstance(submit_res, dict)
      else "Submission failed"
    )
    logging.error("Failed to submit job to TPU server: %s", err)
    sys.exit(1)

  job_id = submit_res["job_id"]
  status = submit_res.get("status", "queued")
  pos = submit_res.get("queue_position", 0)

  if args.submit_only:
    print("\n" + "=" * 40)
    print("Job Submitted (Async):")
    print("=" * 40)
    print(f"Job ID: {job_id}")
    print(f"Status: {status}")
    print(f"Queue Position: {pos}")
    print(
      "AGENT INSTRUCTION: Your request is enqueued. Use '--check_job"
      f" {job_id}' to poll results later."
    )
    sys.exit(0)

  logging.info(
    "Job '%s' submitted. Status: %s (Queue position: %s)",
    job_id,
    status,
    pos,
  )
  if status == "queued":
    logging.info(
      "TPU is currently busy. Job is waiting in queue. Polling for"
      " completion..."
    )

  start_time = time.time()
  last_status = None
  last_pos = None
  max_wait = effective_timeout + 1200
  net_error_count = 0

  while True:
    job_info = check_job_status(job_id, tpu_configs=tpu_configs)
    if isinstance(job_info, dict):
      current_status = job_info.get("status")

      if current_status == "not_found":
        logging.error(
          "ABORTING: Job '%s' record was lost (TPU server likely restarted)."
          " Ending polling to prevent infinite 404 loop.",
          job_id,
        )
        sys.exit(1)

      if current_status == "network_error":
        net_error_count += 1
        err_detail = (
          job_info.get("error", "") if isinstance(job_info, dict) else ""
        )
        if net_error_count < 3:
          logging.info(
            "Transient status check timeout for job '%s' (%s). TPU server"
            " may be busy compiling or experiencing transient SSH latency;"
            " retrying in background (attempt %d/5)...",
            job_id,
            err_detail,
            net_error_count,
          )
        else:
          logging.warning(
            "Persistent status check delay for job '%s' (%s) (attempt %d/5)...",
            job_id,
            err_detail,
            net_error_count,
          )
        if net_error_count >= 3 and selected_config:
          logging.info("Attempting auto-recovery of SSH tunnel...")
          ensure_ssh_tunnel(selected_config)
        if net_error_count >= 5:
          logging.error(
            "Unrecoverable network failure while polling TPU server."
          )
          sys.exit(1)
        time.sleep(args.poll_interval)
        continue
      else:
        if net_error_count > 0:
          logging.info(
            "Status polling connection re-established for job '%s'."
            " Continuing...",
            job_id,
          )
        net_error_count = 0

      current_pos = job_info.get("queue_position", 0)

      if current_status != last_status or current_pos != last_pos:
        if current_status == "queued":
          logging.info(
            "Job '%s' is WAITING in queue (Position: %s)...",
            job_id,
            current_pos,
          )
        elif current_status == "running":
          logging.info("Job '%s' is now RUNNING on TPU...", job_id)
        last_status = current_status
        last_pos = current_pos

      if current_status in ("completed", "failed", "cancelled"):
        res = job_info.get("result") or {}
        print("\n" + "=" * 40)
        print("Execution Result:")
        print("=" * 40)
        print(f"Job ID: {job_id}")
        print(f"Status: {current_status}")
        if job_info.get("started_at") and job_info.get("completed_at"):
          run_duration = job_info["completed_at"] - job_info["started_at"]
          print(f"Execution Duration (excluding queue): {run_duration:.2f}s")
        if job_info.get("created_at") and job_info.get("started_at"):
          queue_duration = job_info["started_at"] - job_info["created_at"]
          print(f"Queue Wait Duration: {queue_duration:.2f}s")
        print(f"Exit Code: {res.get('exit_code', 1)}")
        if res.get("error"):
          err_str = res["error"]
          if len(err_str) > 100000:
            err_str = (
              err_str[:100000]
              + f"\n... [STDERR truncated from {len(res['error'])} chars]"
            )
          print(f"\nSTDERR:\n{err_str}")
        if res.get("output"):
          out_str = res["output"]
          if len(out_str) > 100000:
            out_str = (
              out_str[:100000]
              + f"\n... [STDOUT truncated from {len(res['output'])} chars]"
            )
          print(f"\nSTDOUT:\n{out_str}")

        extract_trace_artifacts(
          res,
          output_dir=args.output_dir,
          code_file=args.code_file,
          run_dir=args.run_dir,
        )

        sys.exit(
          0
          if (current_status == "completed" and res.get("exit_code") == 0)
          else 1
        )

    if time.time() - start_time > max_wait:
      logging.error(
        "Client timed out after waiting for job '%s' to complete in queue.",
        job_id,
      )
      sys.exit(1)

    time.sleep(args.poll_interval)


if __name__ == "__main__":
  main()
