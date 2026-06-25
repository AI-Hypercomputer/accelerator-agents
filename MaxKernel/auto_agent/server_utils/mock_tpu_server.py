import asyncio
import itertools
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, HTTPException

from auto_agent.constants import TPU_SERVER_PORT
from auto_agent.server_utils.common_models import (
    CodeRequest,
    CodeResponse,
    AutotuneRequest,
    GetTpuVersionResponse,
    extract_code,
)
from evaluation.fake_kernel_evaluator import FakeKernelEvaluator


logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(title="Mock TPU Code Execution Server", version="1.0.0")

# Semaphore to limit concurrent compilation requests
compilation_semaphore = asyncio.Semaphore(1)
correctness_semaphore = asyncio.Semaphore(1)
performance_semaphore = asyncio.Semaphore(1)
profile_semaphore = asyncio.Semaphore(1)
autotune_semaphore = asyncio.Semaphore(1)


@app.get("/health")
async def health_check():
  return {"status": "healthy", "mode": "mock"}


@app.post("/compilation_test", response_model=CodeResponse)
async def compilation_test(request: CodeRequest):
  """Try to execute kernel safely in a subprocess and return the output."""
  logging.info("Starting compilation test (mock server)")
  async with compilation_semaphore:
    try:
      request.code = extract_code(request.code)
      temp_dir = tempfile.mkdtemp()
      temp_file_path = os.path.join(temp_dir, "run_code.py")

      if request.dependencies:
        for filename, content in request.dependencies.items():
          file_path = os.path.join(temp_dir, filename)
          os.makedirs(os.path.dirname(file_path), exist_ok=True)
          with open(file_path, "w") as f:
            f.write(content)

      # Write sitecustomize.py to force Pallas interpret mode on CPU fallback
      sitecustomize_path = os.path.join(temp_dir, "sitecustomize.py")
      with open(sitecustomize_path, "w") as f:
        f.write("""
import sys
try:
  import jax.experimental.pallas as pl
  original_pallas_call = pl.pallas_call
  def mocked_pallas_call(*args, **kwargs):
    kwargs['interpret'] = True
    return original_pallas_call(*args, **kwargs)
  pl.pallas_call = mocked_pallas_call
except Exception:
  pass
""")

      with open(temp_file_path, "w") as f:
        f.write(request.code)

      env = os.environ.copy()
      current_pythonpath = env.get("PYTHONPATH", "")
      env["PYTHONPATH"] = f"{temp_dir}:{current_pythonpath}" if current_pythonpath else temp_dir

      process = await asyncio.create_subprocess_exec(
        sys.executable,
        temp_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=temp_dir,
        env=env,
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
      raise
    except Exception as e:
      logging.error(f"Compilation test failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
    finally:
      if "process" in locals() and process.returncode is None:
        try:
          process.kill()
          await process.wait()
        except Exception as e:
          logging.error(f"Failed to kill process: {e}")
      if "temp_dir" in locals():
        try:
          shutil.rmtree(temp_dir)
        except Exception:
          pass


@app.post("/correctness_test", response_model=CodeResponse)
async def correctness_test(request: CodeRequest):
  """Test correctness by running local subprocess."""
  logging.info("Starting correctness test (mock server)")
  async with correctness_semaphore:
    try:
      request.code = extract_code(request.code)
      temp_dir = tempfile.mkdtemp()
      temp_file_path = os.path.join(temp_dir, "run_code.py")

      if request.dependencies:
        for filename, content in request.dependencies.items():
          file_path = os.path.join(temp_dir, filename)
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
          f"Correctness test completed successfully with exit_code: {exit_code}"
        )
        return CodeResponse(output=output, error=error, exit_code=exit_code)

      except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logging.error(f"Correctness test timed out after {request.timeout}s")
        raise HTTPException(status_code=408, detail="Code execution timed out")

    except HTTPException:
      raise
    except Exception as e:
      logging.error(f"Correctness test failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
    finally:
      if "process" in locals() and process.returncode is None:
        try:
          process.kill()
          await process.wait()
        except Exception as e:
          logging.error(f"Failed to kill process: {e}")
      if "temp_dir" in locals():
        try:
          shutil.rmtree(temp_dir)
        except Exception:
          pass


@app.post("/performance_test", response_model=CodeResponse)
async def performance_test(request: CodeRequest):
  """Test performance by running local subprocess."""
  logging.info("Starting performance test (mock server)")
  async with performance_semaphore:
    try:
      request.code = extract_code(request.code)
      temp_dir = tempfile.mkdtemp()
      temp_file_path = os.path.join(temp_dir, "run_code.py")

      if request.dependencies:
        for filename, content in request.dependencies.items():
          file_path = os.path.join(temp_dir, filename)
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
          f"Performance test completed successfully with exit_code: {exit_code}"
        )
        return CodeResponse(output=output, error=error, exit_code=exit_code)

      except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logging.error(f"Performance test timed out after {request.timeout}s")
        raise HTTPException(status_code=408, detail="Code execution timed out")

    except HTTPException:
      raise
    except Exception as e:
      logging.error(f"Performance test failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
    finally:
      if "process" in locals() and process.returncode is None:
        try:
          process.kill()
          await process.wait()
        except Exception as e:
          logging.error(f"Failed to kill process: {e}")
      if "temp_dir" in locals():
        try:
          shutil.rmtree(temp_dir)
        except Exception:
          pass


@app.post("/unified_test", response_model=CodeResponse)
async def unified_test(request: CodeRequest):
  """Simulate correctness and performance with FakeKernelEvaluator."""
  logging.info("Starting mock unified test")
  async with performance_semaphore:
    try:
      request.code = extract_code(request.code)
      temp_dir = tempfile.mkdtemp()
      temp_file_path = os.path.join(temp_dir, "run_code.py")

      if request.dependencies:
        for filename, content in request.dependencies.items():
          file_path = os.path.join(temp_dir, filename)
          os.makedirs(os.path.dirname(file_path), exist_ok=True)
          with open(file_path, "w") as f:
            f.write(content)

      with open(temp_file_path, "w") as f:
        f.write(request.code)

      ref_path = None
      opt_path = None
      if request.dependencies:
        for name in request.dependencies.keys():
          if "optimized" in name:
            opt_path = os.path.join(temp_dir, name)
          elif name == "kernel.py" or name.endswith(".py"):
            ref_path = os.path.join(temp_dir, name)

      if not ref_path and request.dependencies:
        for name in request.dependencies.keys():
          if name.endswith(".py") and "optimized" not in name:
            ref_path = os.path.join(temp_dir, name)
            break

      if not ref_path:
        ref_path = temp_file_path
      if not opt_path:
        opt_path = temp_file_path

      evaluator = FakeKernelEvaluator()
      eval_result = evaluator.evaluate(
          reference_code_path=ref_path,
          optimized_code_path=opt_path,
      )

      output = (
          f"CORRECTNESS: {str(eval_result.numerically_correct).upper()}\n"
          f"RESULT_TIME: {eval_result.optimized_time_ms} ms\n"
      )
      exit_code = (
          0
          if eval_result.compiled_successfully and eval_result.numerically_correct
          else 1
      )
      error = (
          None
          if exit_code == 0
          else "Simulated compilation or correctness check failed."
      )
      logging.info(
          f"Mock unified test completed with exit_code: {exit_code}"
      )
      return CodeResponse(output=output, error=error, exit_code=exit_code)

    except Exception as e:
      logging.error(f"Unified test failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
    finally:
      if "temp_dir" in locals():
        try:
          shutil.rmtree(temp_dir)
        except Exception:
          pass


@app.post("/profile", response_model=CodeResponse)
async def profile(request: CodeRequest):
  """Simulate profile runs and output mock xplane.pb file."""
  logging.info("Starting mock profile")
  async with profile_semaphore:
    try:
      request.code = extract_code(request.code)
      temp_dir = tempfile.mkdtemp()
      logging.info(f"temp_dir: {temp_dir}")

      if request.dependencies:
        for filename, content in request.dependencies.items():
          file_path = os.path.join(temp_dir, filename)
          os.makedirs(os.path.dirname(file_path), exist_ok=True)
          with open(file_path, "w") as f:
            f.write(content)

      temp_file_path = os.path.join(temp_dir, "profile_code.py")
      with open(temp_file_path, "w") as temp_file:
        temp_file.write(request.code)

      ref_path = None
      opt_path = None
      if request.dependencies:
        for name in request.dependencies.keys():
          if "optimized" in name:
            opt_path = os.path.join(temp_dir, name)
          elif name == "kernel.py" or name.endswith(".py"):
            ref_path = os.path.join(temp_dir, name)

      if not ref_path and request.dependencies:
        for name in request.dependencies.keys():
          if name.endswith(".py") and "optimized" not in name:
            ref_path = os.path.join(temp_dir, name)
            break

      if not ref_path:
        ref_path = temp_file_path
      if not opt_path:
        opt_path = temp_file_path

      evaluator = FakeKernelEvaluator()
      eval_result = evaluator.evaluate(
          reference_code_path=ref_path,
          optimized_code_path=opt_path,
      )

      metrics_path = os.path.join(temp_dir, "evaluation_metrics.json")
      hbm_util = 45.0
      compute_util = 15.0
      if os.path.exists(metrics_path):
        try:
          with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
            hbm_util = float(metrics_data.get("estimated_hbm_utilization", 45.0))
            compute_util = float(
                metrics_data.get("estimated_compute_utilization", 15.0)
            )
        except Exception as e:
          logging.warning(f"Failed to read evaluation_metrics.json: {e}")

      total_util = hbm_util + compute_util
      ratio = hbm_util / total_util if total_util > 0 else 0.75

      mock_xplane_path = os.path.join(temp_dir, "mock_profile.xplane.pb")
      with open(mock_xplane_path, "w") as f:
        f.write("MOCK_TRACE_DATA")

      logging.info(
          f"Mock profile completed. Ratio: {ratio}, Path: {mock_xplane_path}"
      )
      return CodeResponse(
          output=json.dumps({"ratio": ratio, "xplane_path": mock_xplane_path}),
          error=None,
          exit_code=0,
      )

    except Exception as e:
      logging.error(f"Profile failed with error: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
    finally:
      # Keep temp_dir alive for parent profile analysis reads of xplane file
      pass


@app.post("/autotune", response_model=CodeResponse)
async def autotune(request: AutotuneRequest):
  """Simulate autotune sweeps using Gemini or fallbacks."""
  logging.info("Starting mock autotune")
  async with performance_semaphore:
    temp_dir = None
    try:
      temp_dir = tempfile.mkdtemp()
      if request.dependencies:
        for filename, content in request.dependencies.items():
          file_path = os.path.join(temp_dir, filename)
          os.makedirs(os.path.dirname(file_path), exist_ok=True)
          with open(file_path, "w") as f:
            f.write(content)

      keys = list(request.search_space.keys())
      values = list(request.search_space.values())
      combinations = list(itertools.product(*values))

      best_time = float("inf")
      best_cfg = None
      best_output = ""
      all_results = []

      configs_list = []
      candidates_formatted = []
      for idx, combo in enumerate(combinations):
        cfg = dict(zip(keys, combo))
        configs_list.append(cfg)
        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
        candidates_formatted.append(f"Candidate {idx}: {cfg_str}")
      candidates_list_str = "\n".join(candidates_formatted)

      ref_code = ""
      if request.dependencies:
        for name, content in request.dependencies.items():
          if name == "kernel.py" or name.endswith(".py"):
            ref_code = content
            break

      prompt = f"""You are an expert TPU performance engineer.
Analyze the baseline JAX kernel and the optimized JAX kernel template.
Estimate the execution latency (in milliseconds) on Cloud TPU v4 for each configuration candidate.

Baseline Reference Code:
{ref_code}

Optimized Kernel Code Template:
{request.code_template}

Given:
- The reference baseline latency is 12.0 ms.
- The hardware target is a Cloud TPU v4.

Predict metrics for each of the following candidate configurations:
{candidates_list_str}

For each candidate, predict:
1. `latency_ms`: Execution latency in milliseconds. If it triggers compiler errors, shape mismatches, or JAX/Pallas errors, return 999.0.
2. `status`: "success" or "failed".
3. `error`: A brief string of the compiler error if failed, otherwise empty.

Respond with a single raw JSON matching this schema:
{{
  "results": [
    {{
      "cfg_index": int,
      "latency_ms": float,
      "status": "string",
      "error": "string"
    }},
    ...
  ]
}}
Do NOT wrap response in markdown blocks. Return only JSON.
"""
      gemini_success = False
      try:
        from google.genai import Client
        from google.genai import types

        client = Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        response_text = response.text.strip()
        data = json.loads(response_text)
        results = data.get("results", [])
        for res in results:
          idx = res.get("cfg_index")
          if idx is not None and 0 <= idx < len(configs_list):
            cfg = configs_list[idx]
            latency = float(res.get("latency_ms", 999.0))
            status = res.get("status", "success")
            err_msg = res.get("error", "")

            if status == "success" and latency < 999.0:
              all_results.append(
                  {"cfg": cfg, "time": latency, "status": "success"}
              )
              if latency < best_time:
                best_time = latency
                best_cfg = cfg
                best_output = f"CORRECTNESS: TRUE\nRESULT_TIME: {latency} ms"
            else:
              all_results.append(
                  {
                      "cfg": cfg,
                      "status": "failed",
                      "error": err_msg or "Simulated compiler error",
                      "exit_code": 1,
                  }
              )
        gemini_success = len(all_results) > 0
      except Exception as e:
        logging.warning(
            f"Failed Gemini autotune sweep: {e}. Falling back to rule-based timing."
        )

      if not gemini_success:
        all_results = []
        for idx, cfg in enumerate(configs_list):
          penalty = 0.0
          for val in cfg.values():
            if isinstance(val, int):
              if val < 16:
                penalty += 2.0
              elif (val & (val - 1)) != 0:
                penalty += 1.5
          h = hash(frozenset(cfg.items()))
          diversity = (h % 100) / 100.0 * 0.5
          simulated_time = 6.0 + penalty + diversity
          all_results.append(
              {"cfg": cfg, "time": simulated_time, "status": "success"}
          )
          if simulated_time < best_time:
            best_time = simulated_time
            best_cfg = cfg
            best_output = f"CORRECTNESS: TRUE\nRESULT_TIME: {simulated_time} ms"

      logging.info(
          f"Mock autotune sweep completed. Best cfg: {best_cfg}, Best time: {best_time} ms"
      )
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
def get_tpu_version() -> GetTpuVersionResponse:
  return GetTpuVersionResponse(tpu_version="TPU v4")


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=TPU_SERVER_PORT)
