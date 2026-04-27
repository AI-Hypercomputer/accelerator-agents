"""Standalone tool for auto-tuning Pallas kernels using grid search on remote servers."""

import json
import logging
from typing import Any
from google.adk import tools
import requests

from hitl_agent.constants import CPU_SERVER_PORT, TPU_SERVER_PORT


def autotune_kernel(
    kernel_name: str,
    code_template: str,
    search_space: dict[str, list[Any]],
    backend: str = "tpu",
    server_addr: str = "http://localhost",
) -> dict:
  """Runs a grid search to auto-tune a Pallas kernel on a remote server.

  Args:
      kernel_name: Name of the kernel.
      code_template: Python code containing placeholders for parameters to be
        tuned. It should produce a line like "RESULT_TIME: <float>" in its
        output to indicate performance.
      search_space: A dictionary mapping placeholder names to lists of feasible
        values.
      backend: 'tpu' or 'cpu'.
      server_addr: Address of the server (default: http://localhost).

  Returns:
      A dictionary containing the status, optimal parameters, and a summary of
      results.
  """
  logging.info(
      f"Starting remote autotuning for kernel: {kernel_name} on {backend}"
  )

  if backend == "tpu":
    port = TPU_SERVER_PORT
  elif backend == "cpu":
    port = CPU_SERVER_PORT
  else:
    return {"status": "error", "message": f"Invalid backend: {backend}"}

  url = f"{server_addr}:{port}/autotune"

  try:
    response = requests.post(
        url,
        json={
            "code_template": code_template,
            "search_space": search_space,
            "timeout": 300,
        },
        timeout=3600,  # 1 hour timeout for the whole autotune request
    )

    if response.status_code == 200:
      result = response.json()
      if result["exit_code"] == 0:
        try:
          output_data = json.loads(result["output"])
          logging.info(
              f"Autotuning completed. Best config: {output_data['best_cfg']} with time {output_data['best_time']} ms"
          )
          return {
              "status": "success",
              "message": "Autotuning completed",
              "best_config": output_data["best_cfg"],
              "best_time_ms": output_data["best_time"],
              "best_output": output_data["best_output"],
              "all_results": output_data.get("all_results", []),
          }
        except json.JSONDecodeError:
          logging.warning("Failed to decode JSON from server output.")
          return {
              "status": "success",
              "message": "Autotuning completed (raw output)",
              "raw_output": result["output"],
          }
      else:
        try:
            output_data = json.loads(result["output"])
            return {
                "status": "failed",
                "message": result["error"] or "Autotune failed on server",
                "all_results": output_data.get("all_results", []),
            }
        except Exception:
            return {
                "status": "failed",
                "message": result["error"] or "Autotune failed on server",
                "server_output": result["output"],
            }
    else:
      return {
          "status": "error",
          "message": f"Server returned status code {response.status_code}: {response.text}",
      }

  except requests.exceptions.ConnectionError:
    return {
        "status": "error",
        "message": f"Could not connect to server at {url}. Make sure it is running.",
    }
  except Exception as e:
    return {"status": "error", "message": str(e)}


# Wrap the function with FunctionTool for compatibility with ADK agents
autotune_tool = tools.FunctionTool(autotune_kernel)
