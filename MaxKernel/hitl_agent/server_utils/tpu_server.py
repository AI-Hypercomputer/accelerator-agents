import asyncio
import itertools
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Optional

import jax
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hitl_agent.constants import TPU_SERVER_PORT
from hitl_agent.tools.analyze_profile import analyze_trace

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


class CodeRequest(BaseModel):
  code: str
  timeout: Optional[int] = 30


class CodeResponse(BaseModel):
  output: str
  error: Optional[str] = None
  exit_code: int


class AutotuneRequest(BaseModel):
  code_template: str
  search_space: dict[str, list]
  timeout: Optional[int] = 300
  profile: bool = True
  kernel_name_pattern: Optional[str] = None


class GetTpuVersionResponse(BaseModel):
  tpu_version: str


@app.get("/health")
async def health_check():
  return {"status": "healthy"}


@app.post("/compilation_test", response_model=CodeResponse)
async def compilation_test(request: CodeRequest):
  """
  Try to execute kernel safely in a subprocess and return the output.
  """
  logging.info("Starting compilation test")
  async with compilation_semaphore:
    try:
      # Extract code from markdown format if present
      code_content = request.code.strip()
      if code_content.startswith("```python") and code_content.endswith("```"):
        # Remove the markdown code block markers
        lines = code_content.split("\n")
        if lines[0].strip() == "```python":
          lines = lines[1:]
        if lines[-1].strip() == "```":
          lines = lines[:-1]
        code_content = "\n".join(lines)
      elif code_content.startswith("```") and code_content.endswith("```"):
        # Handle generic code blocks
        lines = code_content.split("\n")
        if lines[0].strip().startswith("```"):
          lines = lines[1:]
        if lines[-1].strip() == "```":
          lines = lines[:-1]
        code_content = "\n".join(lines)

      request.code = code_content
      # Create a temporary file to store the code
      with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="hitl_eval_", delete=False
      ) as temp_file:
        temp_file.write(request.code)
        temp_file_path = temp_file.name

      # Execute the code in a subprocess
      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tempfile.gettempdir(),
      )

      try:
        stdout, stderr = await asyncio.wait_for(
          process.communicate(), timeout=request.timeout
        )

        output = stdout.decode("utf-8") if stdout else ""
        error = stderr.decode("utf-8") if stderr else None
        exit_code = process.returncode

        logging.info(
          f"Compilation test completed successfully with exit_code: {exit_code}"
        )
        return CodeResponse(output=output, error=error, exit_code=exit_code)

      except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logging.error(f"Compilation test timed out after {request.timeout}s")
        raise HTTPException(status_code=408, detail="Code execution timed out")

    except HTTPException:
      # Re-raise HTTPExceptions to avoid logging them twice
      raise

    except Exception as e:
      logging.error(f"Compilation test failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

    finally:
      # Clean up the temporary file
      if "temp_file_path" in locals():
        try:
          os.unlink(temp_file_path)
        except OSError:
          pass
      logging.info("Compilation test finished")


@app.post("/correctness_test", response_model=CodeResponse)
async def correctness_test(request: CodeRequest):
  """
  Test the correctness of the kernel code by executing it and comparing the output.
  """
  logging.info("Starting correctness test")
  async with correctness_semaphore:
    try:
      # Extract code from markdown format if present
      code_content = request.code.strip()
      if code_content.startswith("```python") and code_content.endswith("```"):
        # Remove the markdown code block markers
        lines = code_content.split("\n")
        if lines[0].strip() == "```python":
          lines = lines[1:]
        if lines[-1].strip() == "```":
          lines = lines[:-1]
        code_content = "\n".join(lines)
      elif code_content.startswith("```") and code_content.endswith("```"):
        # Handle generic code blocks
        lines = code_content.split("\n")
        if lines[0].strip().startswith("```"):
          lines = lines[1:]
        if lines[-1].strip() == "```":
          lines = lines[:-1]
        code_content = "\n".join(lines)

      request.code = code_content
      # Create a temporary file to store the code
      with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="hitl_eval_", delete=False
      ) as temp_file:
        temp_file.write(request.code)
        temp_file_path = temp_file.name

      # Execute the code in a subprocess
      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tempfile.gettempdir(),
      )

      try:
        stdout, stderr = await asyncio.wait_for(
          process.communicate(), timeout=request.timeout
        )

        output = stdout.decode("utf-8") if stdout else ""
        error = stderr.decode("utf-8") if stderr else None
        exit_code = process.returncode

        logging.info(
          f"Correctness test completed successfully with exit_code: {exit_code}"
        )
        return CodeResponse(output=output, error=error, exit_code=exit_code)

      except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logging.error(f"Correctness test timed out after {request.timeout}s")
        raise HTTPException(status_code=408, detail="Code execution timed out")

    except HTTPException:
      # Re-raise HTTPExceptions to avoid logging them twice
      raise
    except Exception as e:
      logging.error(f"Correctness test failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

    finally:
      # Clean up the temporary file
      if "temp_file_path" in locals():
        try:
          os.unlink(temp_file_path)
        except OSError:
          pass
      logging.info("Correctness test finished")


@app.post("/performance_test", response_model=CodeResponse)
async def performance_test(request: CodeRequest):
  """
  Test the performance of the kernel code by executing it and measuring the execution time.
  """
  logging.info("Starting performance test")
  async with performance_semaphore:
    try:
      # Extract code from markdown format if present
      code_content = request.code.strip()
      if code_content.startswith("```python") and code_content.endswith("```"):
        # Remove the markdown code block markers
        lines = code_content.split("\n")
        if lines[0].strip() == "```python":
          lines = lines[1:]
        if lines[-1].strip() == "```":
          lines = lines[:-1]
        code_content = "\n".join(lines)
      elif code_content.startswith("```") and code_content.endswith("```"):
        # Handle generic code blocks
        lines = code_content.split("\n")
        if lines[0].strip().startswith("```"):
          lines = lines[1:]
        if lines[-1].strip() == "```":
          lines = lines[:-1]
        code_content = "\n".join(lines)

      request.code = code_content
      # Create a temporary file to store the code
      with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="hitl_eval_", delete=False
      ) as temp_file:
        temp_file.write(request.code)
        temp_file_path = temp_file.name

      # Execute the code in a subprocess
      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tempfile.gettempdir(),
      )

      try:
        stdout, stderr = await asyncio.wait_for(
          process.communicate(), timeout=request.timeout
        )

        output = stdout.decode("utf-8") if stdout else ""
        error = stderr.decode("utf-8") if stderr else None
        exit_code = process.returncode

        logging.info(
          f"Performance test completed successfully with exit_code: {exit_code}"
        )
        return CodeResponse(output=output, error=error, exit_code=exit_code)

      except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logging.error(f"Performance test timed out after {request.timeout}s")
        raise HTTPException(status_code=408, detail="Code execution timed out")

    except HTTPException:
      # Re-raise HTTPExceptions to avoid logging them twice
      raise
    except Exception as e:
      logging.error(f"Performance test failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

    finally:
      # Clean up the temporary file
      if "temp_file_path" in locals():
        try:
          os.unlink(temp_file_path)
        except OSError:
          pass
      logging.info("Performance test finished")


@app.post("/autotune", response_model=CodeResponse)
async def autotune(request: AutotuneRequest):
  logging.info("Starting autotune")
  async with performance_semaphore:
    try:
      # Generate all combinations
      keys = list(request.search_space.keys())
      values = list(request.search_space.values())
      combinations = list(itertools.product(*values))

      best_time = float("inf")
      best_cfg = None
      best_output = ""
      all_results = []

      for combo in combinations:
        cfg = dict(zip(keys, combo))
        profile_dir = None
        try:
          code_content = request.code_template
          for k, v in cfg.items():
            code_content = code_content.replace(f"{{{k}}}", str(v))
        except Exception as e:
          logging.error(f"Error during template formatting: {e}. Config: {cfg}")
          continue

        if request.profile:
          profile_dir = tempfile.mkdtemp(
            prefix="hitl_eval_profile_", dir=tempfile.gettempdir()
          )
          indented_code = "\n".join(
            ["    " + line for line in code_content.split("\n")]
          )
          code_content = f"""
import jax.profiler
import os

with jax.profiler.trace("{profile_dir}"):
{indented_code}
"""

        # Execute the code
        with tempfile.NamedTemporaryFile(
          mode="w", suffix=".py", prefix="hitl_eval_", delete=False
        ) as temp_file:
          temp_file.write(code_content)
          temp_file_path = temp_file.name

        process = None
        try:
          process = await asyncio.create_subprocess_exec(
            sys.executable,
            temp_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tempfile.gettempdir(),
          )

          stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=request.timeout
          )

          output = stdout.decode("utf-8") if stdout else ""
          error = stderr.decode("utf-8") if stderr else ""
          exit_code = process.returncode

          if exit_code == 0:
            time_taken = None
            xplane_path = None

            if request.profile and profile_dir:
              try:
                profile_paths = list(
                  pathlib.Path(profile_dir).glob("**/*.xplane.pb")
                )
                if profile_paths:
                  xplane_path = str(profile_paths[0])
                  logging.info("Found xplane.pb at %s", xplane_path)

                  # Extract time from xplane
                  profile_data_obj = (
                    jax.profiler.ProfileData.from_serialized_xspace(
                      profile_paths[0].read_bytes()
                    )
                  )

                  from collections import defaultdict

                  event_durations = defaultdict(int)
                  for xplane in profile_data_obj.planes:
                    if xplane.name.startswith("/device:"):
                      for xline in xplane.lines:
                        for e in xline.events:
                          try:
                            name = e.name
                            duration = e.duration_ns
                            event_durations[name] += duration
                            logging.info("event name: ", name)
                          except AttributeError:
                            pass

                  if request.kernel_name_pattern:
                    # Use pattern override
                    matched_events = []
                    for name, duration in event_durations.items():
                      if request.kernel_name_pattern in name:
                        matched_events.append(duration)
                    if matched_events:
                      time_taken = sum(matched_events) / 1e6  # Convert ns to ms
                      logging.info(
                        "Found kernel matching %s with total time %s ms",
                        request.kernel_name_pattern,
                        time_taken,
                      )
                    else:
                      logging.warning(
                        "No kernel matching %s found in profile",
                        request.kernel_name_pattern,
                      )

                  if time_taken is None:
                    if event_durations:
                      # First, try to find "jit_computation" or "jitted_computation"
                      jitted_events = []
                      for name, duration in event_durations.items():
                        if (
                          "jit_computation" in name
                          or "jitted_computation" in name
                        ):
                          jitted_events.append(duration)

                      if jitted_events:
                        time_taken = sum(jitted_events) / 1e6
                        logging.info(
                          "Found default kernel 'jit_computation'/'jitted_computation' with total time %s ms",
                          time_taken,
                        )
                      else:
                        # Use heuristic: find event with largest total duration
                        best_kernel_name = max(
                          event_durations, key=event_durations.get
                        )
                        time_taken = (
                          event_durations[best_kernel_name] / 1e6
                        )  # Convert ns to ms
                        logging.info(
                          "Automatically identified kernel by duration: %s with total time %s ms",
                          best_kernel_name,
                          time_taken,
                        )
                    else:
                      logging.warning("No events found in device planes")
                else:
                  logging.warning("No xplane.pb found in %s", profile_dir)
              except Exception as e:
                logging.error("Failed to extract time from profile: %s", e)

            if time_taken is None:
              # Fallback to wall time from output
              match = re.search(r"RESULT_TIME:\s*([0-9.]+)", output)
              if match:
                time_taken = float(match.group(1))
              else:
                time_taken = float("inf")
                logging.warning(
                  "Failed to get time from profile and no RESULT_TIME found."
                )

            result_entry = {"cfg": cfg, "time": time_taken, "status": "success"}
            if xplane_path:
              result_entry["xplane_path"] = xplane_path

            all_results.append(result_entry)

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
            process.kill()
            await process.wait()
          all_results.append({"cfg": cfg, "status": "timeout"})
        except Exception as e:
          logging.error(f"Error running config {cfg}: {e}")
          all_results.append(
            {"cfg": cfg, "status": "exception", "error": str(e)}
          )
        finally:
          if "temp_file_path" in locals():
            try:
              os.unlink(temp_file_path)
            except OSError:
              pass
          await asyncio.sleep(2)

      if best_cfg is None:
        return CodeResponse(
          output=json.dumps({"all_results": all_results}),
          error="No successful configuration found during autotune.",
          exit_code=-1,
        )

      output_data = {
        "best_cfg": best_cfg,
        "best_time": best_time,
        "best_output": best_output,
        "all_results": all_results,
      }
      logging.info("Autotune finished")
      return CodeResponse(
        output=json.dumps(output_data), error=None, exit_code=0
      )

    except Exception as e:
      logging.error(f"Autotune failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Autotune error: {str(e)}")


@app.post("/profile", response_model=CodeResponse)
async def profile(request: CodeRequest):
  logging.info("Starting profile")
  async with profile_semaphore:
    try:
      # Extract code from markdown format if present
      code_content = request.code.strip()
      if code_content.startswith("```python") and code_content.endswith("```"):
        # Remove the markdown code block markers
        lines = code_content.split("\n")
        if lines[0].strip() == "```python":
          lines = lines[1:]
        if lines[-1].strip() == "```":
          lines = lines[:-1]
        code_content = "\n".join(lines)
      elif code_content.startswith("```") and code_content.endswith("```"):
        # Handle generic code blocks
        lines = code_content.split("\n")
        if lines[0].strip().startswith("```"):
          lines = lines[1:]
        if lines[-1].strip() == "```":
          lines = lines[:-1]
        code_content = "\n".join(lines)

      request.code = code_content
      # Create a temporary directory to store the code and any generated files

      temp_dir = tempfile.mkdtemp(prefix="hitl_eval_")
      logging.info("temp_dir: " + str(temp_dir))

      # Create a temporary file to store the code within temp_dir
      temp_file_path = os.path.join(temp_dir, "profile_code.py")
      with open(temp_file_path, "w") as temp_file:
        temp_file.write(request.code)

      # Execute the code in a subprocess
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

        logging.info("Profile code executed, now analyzing trace.")
        # Recursively search for .xplane.pb file under temp_file_path directory
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

          error_msg = f"No .xplane.pb trace file found after profiling. Files in temp_dir: {all_files[:10]}"
          logging.error(error_msg)

          # Return the execution output/error to help diagnose
          if error:
            error_msg += f"\n\nStderr from profiling script:\n{error}"
          if output:
            error_msg += f"\n\nStdout from profiling script:\n{output}"

          return CodeResponse(output="", error=error_msg, exit_code=-1)

        logging.info("Found xplane file at: " + str(xplane_pb_file))

        ratio = analyze_trace(xplane_pb_file)

        logging.info(
          f"Profile analysis completed successfully with exit_code: {exit_code}"
        )

        return CodeResponse(
          output=json.dumps({"ratio": ratio, "xplane_path": xplane_pb_file}),
          error=error,
          exit_code=exit_code,
        )

      except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logging.error(f"Profile analysis timed out after {request.timeout}s")
        raise HTTPException(status_code=408, detail="Code execution timed out")

    except HTTPException:
      # Re-raise HTTPExceptions to avoid logging them twice
      raise
    except Exception as e:
      logging.error(f"Profile analysis failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

    finally:
      # Clean up the temporary directory
      # try:
      #     shutil.rmtree(temp_dir)
      # except Exception:
      #     pass
      logging.info("Profile analysis finished")


@app.post("/get_tpu_version", response_model=GetTpuVersionResponse)
async def get_tpu_version() -> str:
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
