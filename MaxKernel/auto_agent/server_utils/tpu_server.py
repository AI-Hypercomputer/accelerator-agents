import asyncio
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from auto_agent.constants import TPU_SERVER_PORT
from auto_agent.tools.analyze_profile import analyze_trace

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(title="TPU Code Execution Server", version="1.0.0")

# Semaphore to limit concurrent compilation requests
compilation_semaphore = asyncio.Semaphore(1)
correctness_semaphore = asyncio.Semaphore(1)
performance_semaphore = asyncio.Semaphore(1)
profile_semaphore = asyncio.Semaphore(1)
autotune_semaphore = asyncio.Semaphore(1)


class CodeRequest(BaseModel):
  code: str
  timeout: Optional[int] = 30
  dependencies: Optional[dict] = None


class CodeResponse(BaseModel):
  output: str
  error: Optional[str] = None
  exit_code: int


class AutotuneRequest(BaseModel):
  code_template: str
  search_space: dict[str, list]
  timeout: Optional[int] = 300
  total_timeout: Optional[int] = None
  dependencies: Optional[dict] = None


class GetTpuVersionResponse(BaseModel):
  tpu_version: str


def _extract_code(code: str) -> str:
  """Extracts code from markdown blocks if present."""
  code_content = code.strip()
  if code_content.startswith("```python") and code_content.endswith("```"):
    lines = code_content.split("\n")
    if lines[0].strip() == "```python":
      lines = lines[1:]
    if lines[-1].strip() == "```":
      lines = lines[:-1]
    return "\n".join(lines)
  elif code_content.startswith("```") and code_content.endswith("```"):
    lines = code_content.split("\n")
    if lines[0].strip().startswith("```"):
      lines = lines[1:]
    if lines[-1].strip() == "```":
      lines = lines[:-1]
    return "\n".join(lines)
  return code_content


async def _execute_code(
  request: CodeRequest,
  semaphore: asyncio.Semaphore,
  test_name: str,
  cleanup: bool = True,
) -> tuple[CodeResponse, Optional[str]]:
  """Helper function to execute code in a subprocess with a semaphore."""
  logging.info(f"Starting {test_name}")
  temp_dir = None
  success = False  # Track success to handle cleanup on failure
  async with semaphore:
    try:
      request.code = _extract_code(request.code)
      temp_dir = tempfile.mkdtemp()
      temp_file_path = os.path.join(temp_dir, "run_code.py")

      if request.dependencies:
        for filename, content in request.dependencies.items():
          file_path = os.path.abspath(os.path.join(temp_dir, filename))
          if not file_path.startswith(os.path.abspath(temp_dir) + os.path.sep):
            raise HTTPException(
              status_code=400, detail=f"Invalid filename: {filename}"
            )
          os.makedirs(os.path.dirname(file_path), exist_ok=True)
          with open(file_path, "w") as f:
            f.write(content)

      with open(temp_file_path, "w") as f:
        f.write(request.code)

      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=temp_dir,
      )

      try:
        stdout, stderr = await asyncio.wait_for(
          process.communicate(), timeout=request.timeout
        )

        output = stdout.decode("utf-8") if stdout else ""
        error = stderr.decode("utf-8") if stderr else None
        exit_code = process.returncode

        logging.info(
          f"{test_name} completed successfully with exit_code: {exit_code}"
        )
        success = True  # Mark success before returning
        return CodeResponse(
          output=output, error=error, exit_code=exit_code
        ), temp_dir

      except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logging.error(f"{test_name} timed out after {request.timeout}s")
        raise HTTPException(status_code=408, detail="Code execution timed out")

    except HTTPException:
      raise
    except Exception as e:
      logging.error(f"{test_name} failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

    finally:
      if "process" in locals() and process.returncode is None:
        logging.warning("Process still running in finally block, killing it.")
        try:
          process.kill()
          await process.wait()
        except Exception as e:
          logging.error(f"Failed to kill process: {e}")
      # Clean up if requested or if we failed and caller expects the directory
      if temp_dir and (cleanup or not success):
        try:
          shutil.rmtree(temp_dir)
          logging.info(f"Cleaned up {temp_dir}")
        except Exception:
          pass
      logging.info(f"{test_name} finished")


@app.get("/health")
async def health_check():
  return {"status": "healthy"}


@app.post("/compilation_test", response_model=CodeResponse)
async def compilation_test(request: CodeRequest):
  """Try to execute kernel safely in a subprocess and return the output."""
  resp, _ = await _execute_code(
    request, compilation_semaphore, "Compilation test"
  )
  return resp


@app.post("/correctness_test", response_model=CodeResponse)
async def correctness_test(request: CodeRequest):
  """Test the correctness of the kernel code by executing it and comparing the output."""
  resp, _ = await _execute_code(
    request, correctness_semaphore, "Correctness test"
  )
  return resp


@app.post("/performance_test", response_model=CodeResponse)
async def performance_test(request: CodeRequest):
  """Test the performance of the kernel code by executing it and measuring the execution time."""
  resp, _ = await _execute_code(
    request, performance_semaphore, "Performance test"
  )
  return resp


@app.post("/unified_test", response_model=CodeResponse)
async def unified_test(request: CodeRequest):
  """Test the correctness and performance of the kernel code."""
  resp, _ = await _execute_code(request, performance_semaphore, "Unified test")
  return resp


@app.post("/profile", response_model=CodeResponse)
async def profile(request: CodeRequest):
  logging.info("Starting profile")
  response, temp_dir = await _execute_code(
    request, profile_semaphore, "Profile", cleanup=False
  )

  if response.exit_code != 0 or not temp_dir:
    if temp_dir:
      try:
        shutil.rmtree(temp_dir)
      except Exception:
        pass
    return response

  # Analyze trace
  try:
    output_msg = ""
    logging.info("Profile code executed, now analyzing trace.")
    # Recursively search for .xplane.pb file under temp_dir
    xplane_pb_file = None
    for root, _, files in os.walk(temp_dir):
      for fname in files:
        if fname.endswith(".xplane.pb"):
          xplane_pb_file = os.path.join(root, fname)
          break
      if xplane_pb_file:
        break

    if xplane_pb_file is None:
      # List all files in temp_dir for debugging
      all_files = []
      for root, _, files in os.walk(temp_dir):
        for fname in files:
          all_files.append(os.path.join(root, fname))

      error_msg = (
        "No .xplane.pb trace file found after profiling. Files in"
        f" temp_dir: {all_files[:10]}"
      )
      logging.error(error_msg)

      # Return the execution output/error to help diagnose
      if response.error:
        error_msg += f"\n\nStderr from profiling script:\n{response.error}"
      if response.output:
        output_msg += f"\n\nStdout from profiling script:\n{response.output}"

      return CodeResponse(output=output_msg, error=error_msg, exit_code=-1)

    logging.info("Found xplane file at: " + str(xplane_pb_file))
    try:
      ratio = analyze_trace(xplane_pb_file)
      logging.info("Profile analysis completed successfully")
    except Exception as e:
      error_msg = f"Failed to analyze trace: {str(e)}" + error_msg
      logging.error(error_msg)
      return CodeResponse(output=output_msg, error=error_msg, exit_code=-1)

    return CodeResponse(
      output=json.dumps({"ratio": ratio, "xplane_path": xplane_pb_file}),
      error=response.error,
      exit_code=response.exit_code,
    )
  finally:
    pass
  #   try:
  #     shutil.rmtree(temp_dir)
  #     logging.info(f"Cleaned up {temp_dir} after profiling analysis.")
  #   except Exception as e:
  #     logging.error(f"Failed to clean up {temp_dir}: {e}")


@app.post("/autotune", response_model=CodeResponse)
async def autotune(request: AutotuneRequest):
  logging.info("Starting autotune")
  async with performance_semaphore:
    temp_dir = None
    try:
      # Create unique temporary directory for this autotune request
      temp_dir = tempfile.mkdtemp()
      if request.dependencies:
        for filename, content in request.dependencies.items():
          file_path = os.path.join(temp_dir, filename)
          os.makedirs(os.path.dirname(file_path), exist_ok=True)
          with open(file_path, "w") as f:
            f.write(content)

      # Generate all combinations
      keys = list(request.search_space.keys())
      values = list(request.search_space.values())
      combinations = list(itertools.product(*values))

      best_time = float("inf")
      best_cfg = None
      best_output = ""
      all_results = []

      start_time = time.time()
      for combo in combinations:
        if (
          request.total_timeout
          and (time.time() - start_time) > request.total_timeout
        ):
          logging.warning(
            f"Total timeout of {request.total_timeout}s reached in autotune"
          )
          break
        cfg = dict(zip(keys, combo))
        try:
          code_content = request.code_template
          for k, v in cfg.items():
            code_content = code_content.replace(f"{{{k}}}", str(v))
        except Exception as e:
          logging.error(f"Error during template formatting: {e}. Config: {cfg}")
          continue

        # Execute the code
        temp_file_path = os.path.join(temp_dir, "run_code.py")
        with open(temp_file_path, "w") as temp_file:
          temp_file.write(code_content)

        process = None
        try:
          process = await asyncio.create_subprocess_exec(
            sys.executable,
            temp_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=temp_dir,
          )

          stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=request.timeout
          )

          output = stdout.decode("utf-8") if stdout else ""
          error = stderr.decode("utf-8") if stderr else ""
          exit_code = process.returncode

          if exit_code == 0:
            correctness_match = re.search(
              r"CORRECTNESS:\s*(true|false)", output, re.IGNORECASE
            )
            time_match = re.search(
              r"RESULT_TIME:\s*([0-9.]+)\s*ms", output, re.IGNORECASE
            )

            correctness_passed = False
            if correctness_match:
              correctness_passed = correctness_match.group(1).lower() == "true"

            time_taken = float(time_match.group(1)) if time_match else None

            if not correctness_passed:
              logging.warning(
                f"Correctness check failed or unknown for config {cfg}"
              )
              all_results.append(
                {
                  "cfg": cfg,
                  "status": "correctness_failed_or_unknown",
                  "output": output,
                }
              )
            elif time_taken is None:
              logging.warning(
                f"No RESULT_TIME found in output for config {cfg}"
              )
              all_results.append(
                {"cfg": cfg, "status": "no_result_time", "output": output}
              )
            else:
              all_results.append(
                {"cfg": cfg, "time": time_taken, "status": "success"}
              )
              if time_taken < best_time:
                best_time = time_taken
                best_cfg = cfg
                best_output = output
          else:
            logging.warning(
              f"Config {cfg} failed with exit code {exit_code}. Stderr: {error}"
            )
            all_results.append(
              {
                "cfg": cfg,
                "status": "failed",
                "error": error,
                "exit_code": exit_code,
              }
            )

        except asyncio.TimeoutError:
          logging.warning(f"Config {cfg} timed out")
          if process:
            try:
              process.kill()
              await process.wait()
            except Exception as e:
              logging.error(f"Failed to kill process: {e}")
          all_results.append({"cfg": cfg, "status": "timeout"})
        except Exception as e:
          logging.error(f"Error running config {cfg}: {e}")
          all_results.append(
            {"cfg": cfg, "status": "exception", "error": str(e)}
          )
        finally:
          if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            try:
              os.unlink(temp_file_path)
            except OSError:
              pass

      return CodeResponse(
        output=json.dumps(
          {
            "best_cfg": best_cfg,
            "best_time": best_time,
            "best_output": best_output,
            "all_results": all_results,
          }
        ),
        exit_code=0 if best_cfg is not None else 1,
      )

    except Exception as e:
      logging.error(f"Autotune failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Autotune error: {str(e)}")
    finally:
      if temp_dir and os.path.exists(temp_dir):
        try:
          shutil.rmtree(temp_dir)
        except Exception as e:
          logging.error(
            f"Failed to clean up autotune temp directory {temp_dir}: {e}"
          )


@app.post("/get_tpu_version", response_model=GetTpuVersionResponse)
def get_tpu_version() -> str:
  """Attempts to determine the TPU version by trying three methods.

  Prioritizes non-JAX methods to avoid TPU resource conflicts.

  1.  **`tpu-info` Tool**: Runs the `tpu-info` command-line tool and parses its
  output.
  2.  **TF Profiler**: (Colab only) Uses the TF profiler client.
  3.  **JAX API**: Checks jax.devices() for 'device_kind' (last resort).

  Returns:
      A string like "TPU v4", "TPU v3", etc., or "TPU version not found".
  """

  # --- Method 1: Try running the `tpu-info` command-line tool (No resource conflicts) ---
  try:
    # Run the tpu-info command
    result = subprocess.run(
      ["tpu-info"], capture_output=True, text=True, check=True
    )

    # Regex to find a pattern like "TPU v4", "TPU v5e", "TPU v3 chip", etc.
    # We search the entire output
    match = re.search(r"(TPU v\d[\w-]*)", result.stdout)
    if match:
      # Return the first match, e.g., "TPU v4"
      return match.group(1)

  except (FileNotFoundError, subprocess.CalledProcessError):
    # FileNotFoundError: tpu-info not installed or not in PATH
    # CalledProcessError: tpu-info command failed
    pass
  except (ImportError, KeyError, Exception):
    # ImportError: TensorFlow not installed
    # KeyError: COLAB_TPU_ADDR not set
    # Exception: Profiler connection failed
    pass

  return "TPU version not found"


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=TPU_SERVER_PORT)
