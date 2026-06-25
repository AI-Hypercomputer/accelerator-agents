"""Mock evaluator simulating kernel execution on hardware accelerators using Gemini prediction."""

import json
import logging
import os
import re
from typing import List, Optional
from dotenv import load_dotenv

from google.genai import Client
from google.genai import types
from evaluation.custom_types.evaluation_result import EvaluationResult
from evaluation.evaluation_utils import load_kernel_task_from_yaml

logger = logging.getLogger(__name__)

# Try to load environment variables from the autocomp project env
autocomp_env = "/usr/local/google/home/ligh/github/autocomp/.env"
if os.path.exists(autocomp_env):
  load_dotenv(autocomp_env)


class FakeKernelEvaluator:
  """Mock evaluator that uses LLM reasoning to predict kernel performance metrics."""

  def __init__(self, *args, **kwargs):
    self.reference_time_ms = 12.0
    try:
      self.client = Client()
      logger.info("[FakeKernelEvaluator] GenAI Client initialized successfully.")
    except Exception as e:
      logger.warning(
        f"Failed to initialize GenAI client for FakeKernelEvaluator: {e}. "
        "Falling back to rule-based decay."
      )
      self.client = None
    self.fallback_count = 0

  def evaluate(
    self,
    reference_code_path: str,
    optimized_code_path: str,
    task_yaml_path: Optional[str] = None,
    adapt: Optional[List[str]] = None,
    timeout_seconds: int = 300,
    cleanup: bool = True,
    atol: float = 1e-3,
    rtol: float = 1e-3,
  ) -> EvaluationResult:
    """Simulates evaluation using Gemini analysis or rule-based decay."""
    task_id = "fake_task"
    if task_yaml_path:
      try:
        task = load_kernel_task_from_yaml(task_yaml_path)
        task_id = task.task_id
      except Exception as e:
        logger.warning(f"Failed to load task yaml: {e}")

    # Read reference and optimized code
    ref_code = ""
    if os.path.exists(reference_code_path):
      try:
        with open(reference_code_path, "r") as f:
          ref_code = f.read()
      except Exception as e:
        logger.error(f"Failed to read reference code: {e}")

    opt_code = ""
    if os.path.exists(optimized_code_path):
      try:
        with open(optimized_code_path, "r") as f:
          opt_code = f.read()
      except Exception as e:
        logger.error(f"Failed to read optimized code: {e}")

    # Check if the code contains multiple inlined solution functions
    inlined_candidates = re.findall(r"def solution_(\d+)\(", opt_code)
    is_grouped = len(inlined_candidates) > 0

    # Default metrics
    latency = self.reference_time_ms
    best_latency = self.reference_time_ms
    metrics_to_save = {}

    if self.client and ref_code and opt_code:
      try:
        if is_grouped:
          # Prompt for evaluating multiple grouped candidates
          prompt = f"""You are an expert TPU performance engineer.
Analyze the following baseline JAX kernel and the optimized file containing multiple candidate implementations (solution_0, solution_1, etc.) inlined.
Estimate the execution performance metrics of each candidate on a Cloud TPU v4.

Baseline Reference Code:
```python
{ref_code}
```

Optimized Candidates Code (Grouped):
```python
{opt_code}
```

Given:
- The reference baseline latency is {self.reference_time_ms} ms.
- The hardware target is a Cloud TPU v4.

Predict the following metrics for EACH candidate (solution_0, solution_1, etc.):
1. `latency_ms`: Predicted execution latency of the candidate in milliseconds.
2. `estimated_hbm_utilization`: Estimated memory bandwidth utilization (percentage, 0.0 to 100.0).
3. `estimated_compute_utilization`: Estimated compute FLOPs utilization (percentage, 0.0 to 100.0).
4. `estimated_computation_density`: Estimated operational intensity of the kernel (FLOPs per byte).
5. `analysis`: A brief explanation of your prediction for this candidate.

Respond with a single raw JSON object matching this schema:
{{
  "candidates": [
    {{
      "candidate_id": int,
      "latency_ms": float,
      "estimated_hbm_utilization": float,
      "estimated_compute_utilization": float,
      "estimated_computation_density": float,
      "analysis": "string"
    }},
    ...
  ]
}}
Do NOT wrap response in markdown blocks. Return only JSON.
"""
          response = self.client.models.generate_content(
              model="gemini-2.5-flash",
              contents=prompt,
              config=types.GenerateContentConfig(
                  response_mime_type="application/json",
                  temperature=0.1,
              )
          )
          data = json.loads(response.text.strip())
          candidates_list = data.get("candidates", [])
          metrics_to_save = {"candidates": candidates_list}
          
          # The returned latency is the best (lowest) candidate's latency
          if candidates_list:
            best_latency = min(c["latency_ms"] for c in candidates_list)
          else:
            best_latency = self.reference_time_ms

        else:
          # Prompt for a single candidate
          prompt = f"""You are an expert TPU performance engineer.
Analyze the following baseline JAX kernel and the optimized candidate kernel.
Estimate the execution performance metrics of the optimized kernel on a Cloud TPU v4.

Baseline Reference Code:
```python
{ref_code}
```

Optimized Candidate Code:
```python
{opt_code}
```

Given:
- The reference baseline latency is {self.reference_time_ms} ms.
- The hardware target is a Cloud TPU v4.

Predict the following metrics:
1. `latency_ms`: The predicted execution latency of the optimized kernel in milliseconds.
   - If the optimized code is empty or holds syntactic bugs, return {self.reference_time_ms * 5} ms.
   - If it implements valid tiling or vectorization optimization, it should be faster (e.g. 4.0 to 11.5 ms).
2. `estimated_hbm_utilization`: Estimated memory bandwidth utilization (percentage, 0.0 to 100.0).
3. `estimated_compute_utilization`: Estimated compute FLOPs utilization (percentage, 0.0 to 100.0).
4. `estimated_computation_density`: Estimated operational intensity of the kernel (FLOPs per byte).
5. `analysis`: A brief explanation of your prediction.

Respond with a single raw JSON object matching this schema:
{{
  "latency_ms": float,
  "estimated_hbm_utilization": float,
  "estimated_compute_utilization": float,
  "estimated_computation_density": float,
  "analysis": "string"
}}
Do NOT wrap response in markdown blocks. Return only JSON.
"""
          response = self.client.models.generate_content(
              model="gemini-2.5-flash",
              contents=prompt,
              config=types.GenerateContentConfig(
                  response_mime_type="application/json",
                  temperature=0.1,
              )
          )
          data = json.loads(response.text.strip())
          latency = float(data.get("latency_ms", self.reference_time_ms))
          hbm_util = float(data.get("estimated_hbm_utilization", 0.0))
          comp_util = float(data.get("estimated_compute_utilization", 0.0))
          comp_density = float(data.get("estimated_computation_density", 0.0))
          analysis = data.get("analysis", "")
          
          best_latency = latency
          metrics_to_save = {
              "estimated_hbm_utilization": hbm_util,
              "estimated_compute_utilization": comp_util,
              "estimated_computation_density_flops_per_byte": comp_density,
              "performance_analysis": analysis
          }

      except Exception as e:
        logger.warning(f"Gemini evaluation prediction failed: {e}. Using fallback.")
        self.fallback_count += 1
        best_latency = self.reference_time_ms * (0.95 ** self.fallback_count)
        metrics_to_save = {
            "estimated_hbm_utilization": 10.0,
            "estimated_compute_utilization": 2.0,
            "estimated_computation_density_flops_per_byte": 0.5,
            "performance_analysis": f"Fallback decay latency: {best_latency:.3f} ms"
        }

    # Write metrics to evaluation_metrics.json in the optimized code directory (skip if evaluating the reference code itself)
    if os.path.abspath(optimized_code_path) != os.path.abspath(reference_code_path):
      opt_dir = os.path.dirname(optimized_code_path)
      metrics_path = os.path.join(opt_dir, "evaluation_metrics.json")
      try:
        with open(metrics_path, "w") as f:
          json.dump(metrics_to_save, f, indent=2)
        logger.info(f"[FakeKernelEvaluator] Wrote metrics to {metrics_path}")
      except Exception as e:
        logger.error(f"Failed to write metrics to {metrics_path}: {e}")
    else:
      logger.info("[FakeKernelEvaluator] Skipping metrics JSON write for reference code evaluation.")

    logger.info(
      f"[FakeKernelEvaluator] Predicted latency for {task_id}: {best_latency:.3f} ms"
    )

    return EvaluationResult(
      task_id=task_id,
      compiled_successfully=True,
      numerically_correct=True,
      max_abs_diff=0.0,
      max_rel_diff=0.0,
      reference_time_ms=self.reference_time_ms,
      optimized_time_ms=round(best_latency, 3),
      xprof_reference_time_ms=self.reference_time_ms,
      xprof_optimized_time_ms=round(best_latency, 3),
    )
