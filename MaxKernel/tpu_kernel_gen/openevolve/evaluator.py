import logging

import requests

from tpu_kernel_gen.openevolve.constants import REQUEST_TIMEOUT, TPU_TIMEOUT


def check_compilation(code: str, tpu_ip: str) -> dict:
  try:
    response = requests.post(
      f"http://{tpu_ip}:5463/test_compilation",
      json={"code": code, "timeout": TPU_TIMEOUT},
      timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()
  except Exception as e:
    logging.error(f"Compilation check request failed: {e}")
    return {"exit_code": -1, "output": "", "error": str(e)}


def check_correctness(code: str, tpu_ip: str) -> dict:
  try:
    response = requests.post(
      f"http://{tpu_ip}:5463/test_correctness",
      json={"code": code, "timeout": TPU_TIMEOUT},
      timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()
  except Exception as e:
    logging.error(f"Correctness check request failed: {e}")
    return {"exit_code": -1, "output": "", "error": str(e)}


def check_performance(code: str, tpu_ip: str) -> dict:
  try:
    response = requests.post(
      f"http://{tpu_ip}:5463/test_performance",
      json={"code": code, "timeout": TPU_TIMEOUT},
      timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()
  except Exception as e:
    logging.error(f"Performance check request failed: {e}")
    return {
      "exit_code": -1,
      "output": "",
      "error": str(e),
      "simple_time": None,
      "optimized_time": None,
    }


def evaluate(program_path: str, tpu_ip: str) -> dict:
  """
  Evaluate the program and return metrics as a dictionary.

  CRITICAL: Must return a dictionary, not an EvaluationResult object.
  """

  logging.info(f"Evaluating program at {program_path} using TPU {tpu_ip}")
  try:
    with open(program_path, "r") as f:
      code = f.read()
      logging.info(f"File content: {code}")
  except FileNotFoundError:
    logging.error(f"File not found: {program_path}")
    return {
      "combined_score": -1000.0,
      "output_message": "",
      "error_message": f"File not found: {program_path}",
    }
  except Exception as e:
    logging.error(f"Error reading file: {e}")
    return {
      "combined_score": -1000.0,
      "output_message": "",
      "error_message": str(e),
    }

  try:
    compilation_result = check_compilation(code, tpu_ip)
    logging.info(f"Compilation result: {compilation_result}")
    if compilation_result["exit_code"] != 0:
      return {
        "combined_score": -1000.0,
        "output_message": compilation_result["output"],
        "error_message": compilation_result["error"],
      }
  except Exception as e:
    logging.error(f"Error during compilation check: {e}")
    return {
      "combined_score": -1000.0,
      "output_message": "",
      "error_message": str(e),
    }

  try:
    correctness_result = check_correctness(code, tpu_ip)
    if correctness_result["exit_code"] != 0:
      return {
        "combined_score": -1000.0,
        "output_message": correctness_result["output"],
        "error_message": correctness_result["error"],
      }
  except Exception as e:
    logging.error(f"Error during correctness check: {e}")
    return {
      "combined_score": -1000.0,
      "output_message": "",
      "error_message": str(e),
    }

  try:
    performance_result = check_performance(code, tpu_ip)
    if performance_result["simple_time"] is None or performance_result["optimized_time"] is None:
      return {
        "combined_score": 0.0,
        "output_message": performance_result["output"],
        "error_message": performance_result["error"],
      }

    speedup = performance_result["simple_time"] / performance_result["optimized_time"]
    return {
      "combined_score": speedup,
      "output_message": f"Performance improved by {speedup:.2f}x",
      "error_message": None,
    }
  except Exception as e:
    logging.error(f"Error during performance check: {e}")
    return {
      "combined_score": 0.0,
      "output_message": "Error during performance check.",
      "error_message": str(e),
    }
