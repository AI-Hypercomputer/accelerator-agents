"""Shared lightweight configuration and port resolution utilities for server_utils."""

import logging
import os
import socket
from typing import Any, Optional

import yaml

from auto_agent.constants import (
  CPU_SERVER_PORT,
  EVAL_SERVER_PORT,
  TPU_SERVER_PORT,
)


def get_local_ip() -> str:
  """Returns the local IP address of the machine."""
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      s.connect(("8.8.8.8", 80))
      return s.getsockname()[0]
  except Exception:
    return "127.0.0.1"


def _resolve_config_path(cfg_path: str) -> Optional[str]:
  """Resolves configuration path in cwd or relative to this script's directory."""
  if os.path.exists(cfg_path):
    return cfg_path
  script_dir = os.path.dirname(os.path.abspath(__file__))
  alt_path = os.path.join(script_dir, cfg_path)
  if os.path.exists(alt_path):
    return alt_path
  return None


def get_local_tpu_port(cfg_path: str = "eval_config.yaml") -> Optional[int]:
  """Checks eval_config.yaml and returns the port if a local TPU server is needed."""
  resolved_path = _resolve_config_path(cfg_path)
  if not resolved_path:
    return None

  try:
    with open(resolved_path, "r") as file:
      config = yaml.safe_load(file) or {}
  except Exception as e:
    logging.error(f"Config file {resolved_path} error: {e}")
    return None

  if not isinstance(config, dict):
    raise ValueError(
      f"Invalid configuration format in {resolved_path}: "
      "Expected a YAML dictionary at the root level."
    )

  backends = config.get("backends", [])
  if not isinstance(backends, list):
    raise ValueError(
      f"Invalid configuration format in {resolved_path}: "
      "'backends' must be a list."
    )

  local_ip = get_local_ip()

  # Find all backends that are local TPUs
  local_tpu_backends = [
    b
    for b in backends
    if isinstance(b, dict)
    and b.get("type") == "tpu"
    and b.get("ip") in ["127.0.0.1", "localhost", local_ip]
    and "tpu_vm" not in b
  ]

  if not local_tpu_backends:
    return None

  return local_tpu_backends[0].get("port")


def get_local_cpu_port(cfg_path: str = "eval_config.yaml") -> Optional[int]:
  """Checks eval_config.yaml and returns the port if a local CPU server is needed."""
  resolved_path = _resolve_config_path(cfg_path)
  if not resolved_path:
    return None

  try:
    with open(resolved_path, "r") as file:
      config = yaml.safe_load(file) or {}
  except Exception as e:
    logging.error(f"Config file {resolved_path} error: {e}")
    return None

  if not isinstance(config, dict):
    raise ValueError(
      f"Invalid configuration format in {resolved_path}: "
      "Expected a YAML dictionary at the root level."
    )

  backends = config.get("backends", [])
  if not isinstance(backends, list):
    raise ValueError(
      f"Invalid configuration format in {resolved_path}: "
      "'backends' must be a list."
    )

  local_ip = get_local_ip()

  # Find all backends that are local CPUs
  local_cpu_backends = [
    b
    for b in backends
    if isinstance(b, dict)
    and b.get("type") == "cpu"
    and b.get("ip") in ["127.0.0.1", "localhost", local_ip]
  ]

  if not local_cpu_backends:
    return None

  return local_cpu_backends[0].get("port")


def get_bastion_config(
  cfg_path: str = "gke_config.yaml", default_eval_port: int = EVAL_SERVER_PORT
) -> dict[str, Any]:
  """Checks gke_config.yaml and returns bastion configuration dict if configured."""
  resolved_path = _resolve_config_path(cfg_path)
  if not resolved_path:
    return {}

  try:
    with open(resolved_path, "r") as file:
      cfg = yaml.safe_load(file) or {}

    if not isinstance(cfg, dict):
      raise ValueError(
        f"Invalid configuration format in {resolved_path}: "
        "Expected a YAML dictionary at the root level."
      )

    b = cfg.get("bastion_vm", {})
    if isinstance(b, dict) and b:
      return {
        "name": b.get("name") or "",
        "zone": b.get("zone") or "",
        "project": b.get("project") or "",
        "local_port": b.get("local_port") or b.get("port") or default_eval_port,
        "remote_port": b.get("remote_port") or b.get("port") or default_eval_port,
      }
  except Exception as e:
    logging.error(f"Config file {resolved_path} error: {e}")
  return {}


if __name__ == "__main__":
  tpu_p = get_local_tpu_port()
  cpu_p = get_local_cpu_port()
  b = get_bastion_config()

  print(f"EVAL_PORT={EVAL_SERVER_PORT}")
  print(f"TPU_PORT={tpu_p if tpu_p is not None else TPU_SERVER_PORT}")
  print(f"CPU_PORT={cpu_p if cpu_p is not None else CPU_SERVER_PORT}")
  print(f"LOCAL_TPU_PORT={tpu_p if tpu_p is not None else ''}")
  print(f"LOCAL_CPU_PORT={cpu_p if cpu_p is not None else ''}")
  print(f"BASTION_NAME='{b.get('name', '')}'")
  print(f"BASTION_ZONE='{b.get('zone', '')}'")
  print(f"BASTION_PROJECT='{b.get('project', '')}'")
  print(f"BASTION_LOCAL_PORT={b.get('local_port', EVAL_SERVER_PORT)}")
  print(f"BASTION_REMOTE_PORT={b.get('remote_port', EVAL_SERVER_PORT)}")
