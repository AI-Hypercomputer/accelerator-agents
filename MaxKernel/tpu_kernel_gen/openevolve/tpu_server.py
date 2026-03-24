import asyncio
import logging
import os
import re
import subprocess
import tempfile
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from tpu_kernel_gen.openevolve.constants import TPU_TIMEOUT

app = FastAPI(title="OpenEvolve TPU Servers", version="1.0.0")

# Semaphore to limit concurrent compilation requests
compilation_semaphore = asyncio.Semaphore(1)
correctness_semaphore = asyncio.Semaphore(1)


class CodeRequest(BaseModel):
  code: str
  timeout: Optional[int] = 30


class CodeResponse(BaseModel):
  output: str
  error: Optional[str] = None
  exit_code: int


class PerformanceResponse(BaseModel):
  simple_time: Optional[float] = None
  optimized_time: Optional[float] = None
  output: str
  error: Optional[str] = None
  exit_code: int


@app.get("/health")
async def health_check():
  return {"status": "healthy"}


@app.post("/test_compilation", response_model=CodeResponse)
async def test_compilation(request: CodeRequest):
  """
  Try to execute kernel safely in a subprocess and return the output.
  """
  logging.info("Starting compilation test")
  try:
    # Write the code to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
      tmp_file.write(request.code)
      program_path = tmp_file.name

    os.chmod(program_path, 0o755)

    result = subprocess.run(
      ["python3", program_path, "--compilation"],
      capture_output=True,
      text=True,
      timeout=TPU_TIMEOUT,
    )

    return CodeResponse(
      output=result.stdout,
      error=result.stderr if result.stderr else None,
      exit_code=result.returncode,
    )
  except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
    return CodeResponse(output="", error=str(e), exit_code=-1)


@app.post("/test_correctness", response_model=CodeResponse)
async def test_correctness(request: CodeRequest):
  """
  Test the correctness of the kernel code by executing it and comparing the output.
  """
  logging.info("Starting correctness test")

  try:
    # Write the code to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
      tmp_file.write(request.code)
      program_path = tmp_file.name

    # Make the program executable
    os.chmod(program_path, 0o755)

    result = subprocess.run(
      ["python3", program_path, "--correctness"],
      capture_output=True,
      text=True,
      timeout=TPU_TIMEOUT,
    )
    return CodeResponse(
      output=result.stdout,
      error=result.stderr if result.stderr else None,
      exit_code=result.returncode,
    )
  except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
    return CodeResponse(output="", error=str(e), exit_code=-1)


@app.post("/test_performance", response_model=PerformanceResponse)
async def test_performance(request: CodeRequest):
  """
  Test the performance of the kernel code by executing it and measuring the time taken.
  """
  logging.info("Starting performance test")
  try:
    # Write the code to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
      tmp_file.write(request.code)
      program_path = tmp_file.name

    # Make the program executable
    os.chmod(program_path, 0o755)

    result = subprocess.run(
      ["python3", program_path, "--performance"],
      capture_output=True,
      text=True,
      timeout=TPU_TIMEOUT,
    )

    if result.returncode != 0:
      logging.error(f"Performance check failed with return code {result.returncode}")
      return PerformanceResponse(
        simple_time=None,
        optimized_time=None,
        output=result.stdout,
        error=result.stderr,
        exit_code=result.returncode,
      )

    output = result.stdout

    # Parse simple compute time
    simple_match = re.search(r"Simple compute average time:\s+([\d.]+)\s+s", output)
    optimized_match = re.search(r"Optimized compute average time:\s+([\d.]+)\s+s", output)

    if simple_match and optimized_match:
      simple_time = float(simple_match.group(1))
      optimized_time = float(optimized_match.group(1))
      return PerformanceResponse(
        simple_time=simple_time,
        optimized_time=optimized_time,
        output=output,
        error=None,
        exit_code=0,
      )
    else:
      logging.error(f"Could not parse performance output: {output}")
      return PerformanceResponse(
        simple_time=None,
        optimized_time=None,
        output=output,
        error="Could not parse performance output",
        exit_code=-1,
      )

  except (
    subprocess.TimeoutExpired,
    FileNotFoundError,
    PermissionError,
    ValueError,
  ) as e:
    logging.error(f"Performance check failed: {e}")
    return PerformanceResponse(
      simple_time=None,
      optimized_time=None,
      output="",
      error=str(e),
      exit_code=-1,
    )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=5463)
