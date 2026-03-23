import asyncio
import json
import logging
import os
import sys
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tpu_kernel_gen.agents.kernel_gen_agent.analyze_profile import analyze_trace
from tpu_kernel_gen.agents.kernel_gen_agent.constants import CPU_SERVER_PORT

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(title="CPU Code Execution Server", version="1.0.0")

# Semaphore to limit concurrent compilation requests
compilation_semaphore = asyncio.Semaphore(1)
correctness_semaphore = asyncio.Semaphore(1)
performance_semaphore = asyncio.Semaphore(1)
profile_semaphore = asyncio.Semaphore(1)


class CodeRequest(BaseModel):
  code: str
  timeout: Optional[int] = 30
  files: Optional[dict[str, str]] = None


class CodeResponse(BaseModel):
  output: str
  error: Optional[str] = None
  exit_code: int


class GetBackendVersionResponse(BaseModel):
  backend_version: str


def get_cpu_env():
  """
  Returns environment variables that force JAX to use CPU backend.
  """
  env = os.environ.copy()
  env["JAX_PLATFORMS"] = "cpu"
  env["JAX_PLATFORM_NAME"] = "cpu"
  # Disable GPU visibility to ensure CPU-only execution
  env["CUDA_VISIBLE_DEVICES"] = ""
  return env


@app.get("/health")
async def health_check():
  return {"status": "healthy", "backend": "cpu"}


@app.post("/compilation_test", response_model=CodeResponse)
async def compilation_test(request: CodeRequest):
  """
  Try to execute kernel safely in a subprocess with CPU backend and return the output.
  """
  logging.info("Starting compilation test on CPU backend")
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
      with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(request.code)
        temp_file_path = temp_file.name

      # Execute the code in a subprocess with CPU-only environment
      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tempfile.gettempdir(),
        env=get_cpu_env(),  # Force CPU backend
      )

      try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=request.timeout)

        output = stdout.decode("utf-8") if stdout else ""
        error = stderr.decode("utf-8") if stderr else None
        exit_code = process.returncode

        logging.info(f"Compilation test completed successfully on CPU with exit_code: {exit_code}")
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
  Test the correctness of the kernel code by executing it on CPU and comparing the output.
  """
  logging.info("Starting correctness test on CPU backend")
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
      with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(request.code)
        temp_file_path = temp_file.name

      # Execute the code in a subprocess with CPU-only environment
      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tempfile.gettempdir(),
        env=get_cpu_env(),  # Force CPU backend
      )

      try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=request.timeout)

        output = stdout.decode("utf-8") if stdout else ""
        error = stderr.decode("utf-8") if stderr else None
        exit_code = process.returncode

        logging.info(f"Correctness test completed successfully on CPU with exit_code: {exit_code}")
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
  Test the performance of the kernel code by executing it on CPU and measuring the execution time.
  """
  logging.info("Starting performance test on CPU backend")
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
      with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(request.code)
        temp_file_path = temp_file.name

      # Execute the code in a subprocess with CPU-only environment
      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tempfile.gettempdir(),
        env=get_cpu_env(),  # Force CPU backend
      )

      try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=request.timeout)

        output = stdout.decode("utf-8") if stdout else ""
        error = stderr.decode("utf-8") if stderr else None
        exit_code = process.returncode

        logging.info(f"Performance test completed successfully on CPU with exit_code: {exit_code}")
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


@app.post("/profile", response_model=CodeResponse)
async def profile(request: CodeRequest):
  logging.info("Starting profile on CPU backend")
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

      temp_dir = tempfile.mkdtemp()
      logging.info("temp_dir: " + str(temp_dir))

      # Create a temporary file to store the code within temp_dir
      temp_file_path = os.path.join(temp_dir, "profile_code.py")
      with open(temp_file_path, "w") as temp_file:
        temp_file.write(request.code)

      # Write any additional files provided in the request
      if request.files:
        for filename, content in request.files.items():
          file_path = os.path.join(temp_dir, filename)
          # Ensure subdirectories exist if filename contains them
          os.makedirs(os.path.dirname(file_path), exist_ok=True)
          with open(file_path, "w") as f:
            f.write(content)
        logging.info(f"Wrote {len(request.files)} additional files to {temp_dir}")

      # Execute the code in a subprocess with CPU-only environment
      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=temp_dir,
        env=get_cpu_env(),  # Force CPU backend
      )

      try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=request.timeout)

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

        logging.info("Found xplane file at: " + str(xplane_pb_file))

        ratio = analyze_trace(xplane_pb_file)

        logging.info(f"Profile analysis completed successfully on CPU with exit_code: {exit_code}")

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


@app.post("/get_backend_version", response_model=GetBackendVersionResponse)
async def get_backend_version() -> str:
  """
  Returns the backend version for CPU execution.

  Returns:
      A string indicating CPU backend.
  """
  return GetBackendVersionResponse(backend_version="CPU")


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=CPU_SERVER_PORT)
