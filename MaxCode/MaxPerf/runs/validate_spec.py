#!/usr/bin/env python3
# pylint: skip-file
"""validate_spec.py — Validation and artifact generation utility for optimization_spec.json.

Validates the standardization schema and provides helper extraction for:
- session.json
- roofline_inputs (exec_config.json, hardware.json, model_evidence_links.txt)
- maxshard_recipes (declarative recipes for CDK, XManager, and UBench drivers)
"""

import argparse
import json
import os
from pathlib import Path
import sys

VALID_ACCELERATORS = {
    "TPU v7x",
    "GhostFish",
    "tpu7x",
    "TPU v6e",
    "TPU v5p",
    "TPU v5e",
    "TPU v4",
    "GB200",
    "H100",
}
VALID_PRECISIONS = {
    "fp32",
    "tf32",
    "fp16",
    "bf16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp4",
    "int8",
    "int4",
}
VALID_REGIMES = {
    "serving_inference",
    "prefill_only",
    "decode_only",
    "training",
    "disaggregated_prefill",
    "disaggregated_decode",
}
VALID_BACKENDS = {"cdk", "xmanager", "ubench", "xpk", "direct_tpu_vm"}


def validate_spec(spec_data: dict) -> list[str]:
  """Validate specification schema and return list of error messages."""
  errors = []

  # 1. Target block
  target = spec_data.get("target")
  if not isinstance(target, dict):
    errors.append("Missing or invalid 'target' block (must be an object).")
  else:
    if not target.get("model_name"):
      errors.append("Target block missing 'model_name'.")

  # 2. Hardware block
  hardware = spec_data.get("hardware")
  if not isinstance(hardware, dict):
    errors.append("Missing or invalid 'hardware' block (must be an object).")
  else:
    accel = hardware.get("accelerator")
    if not accel:
      errors.append("Hardware block missing 'accelerator'.")
    elif accel not in VALID_ACCELERATORS:
      errors.append(
          f"Hardware accelerator '{accel}' is unrecognized. Valid options:"
          f" {VALID_ACCELERATORS}"
      )

    chip_count = hardware.get("chip_count")
    if not isinstance(chip_count, int) or chip_count <= 0:
      errors.append("Hardware block 'chip_count' must be a positive integer.")

  # 3. Execution block (optional, defaults to direct_tpu_vm if omitted)
  execution = spec_data.get("execution", {})
  if not isinstance(execution, dict):
    errors.append("Execution block must be an object.")
  else:
    backend = execution.get("backend", "direct_tpu_vm")
    if backend not in VALID_BACKENDS:
      errors.append(
          f"Execution backend '{backend}' is invalid. Valid options:"
          f" {VALID_BACKENDS}"
      )

  # 4. Workload block
  workload = spec_data.get("workload")
  if not isinstance(workload, dict):
    errors.append("Missing or invalid 'workload' block (must be an object).")
  else:
    regime = workload.get("regime")
    if not regime or regime not in VALID_REGIMES:
      errors.append(
          f"Workload regime '{regime}' invalid. Valid options: {VALID_REGIMES}"
      )

    precisions = workload.get("target_precision", {})
    if not isinstance(precisions, dict):
      errors.append("Workload 'target_precision' must be an object.")
    else:
      for k, dtype in precisions.items():
        if dtype not in VALID_PRECISIONS:
          errors.append(
              f"Invalid precision datatype '{dtype}' specified for '{k}'."
              f" Valid: {VALID_PRECISIONS}"
          )

  # 5. Objective block
  objective = spec_data.get("objective")
  if not isinstance(objective, dict):
    errors.append("Missing or invalid 'objective' block (must be an object).")
  else:
    if not objective.get("primary_metric"):
      errors.append("Objective block missing 'primary_metric'.")

  return errors


def export_roofline_inputs(spec: dict, output_dir: Path):
  """Export exec_config.json, hardware.json, and model_evidence_links.txt for roofline skill."""
  output_dir.mkdir(parents=True, exist_ok=True)

  target = spec.get("target", {})
  hardware = spec.get("hardware", {})
  workload = spec.get("workload", {})

  exec_config = {
      "regime": workload.get("regime", "serving_inference"),
      "batch_size": workload.get("batch_size", 1),
      "prompt_length": workload.get("prompt_length_tokens", 2048),
      "output_length": workload.get("output_length_tokens", 512),
      "precision": workload.get("target_precision", {}),
      "mesh_shape": hardware.get("mesh_shape", [hardware.get("chip_count", 1)]),
  }

  hardware_config = {
      "accelerator": hardware.get("accelerator", "TPU v6e"),
      "chip_count": hardware.get("chip_count", 1),
      "mesh_shape": hardware.get("mesh_shape", [hardware.get("chip_count", 1)]),
  }

  with open(output_dir / "exec_config.json", "w") as f:
    json.dump(exec_config, f, indent=2)

  with open(output_dir / "hardware.json", "w") as f:
    json.dump(hardware_config, f, indent=2)

  evidence_url = target.get("model_evidence_url", "")
  with open(output_dir / "model_evidence_links.txt", "w") as f:
    f.write(evidence_url + "\n" if evidence_url else "")

  print(f"Successfully generated Roofline inputs in: {output_dir}")


def export_session_json(spec: dict, session_file_path: Path):
  """Populate or update session.json from the specification."""
  target = spec.get("target", {})
  hardware = spec.get("hardware", {})
  workload = spec.get("workload", {})
  execution = spec.get("execution", {})

  session = {
      "session_id": (
          f"session-{target.get('model_name', 'model').split('/')[-1].lower()}"
      ),
      "model_name": target.get("model_name"),
      "tpu_vm": hardware.get("tpu_vm_hostname", "localhost"),
      "execution_backend": execution.get("backend", "direct_tpu_vm"),
      "chip_count": hardware.get("chip_count", 1),
      "mesh_shape": hardware.get("mesh_shape", []),
      "kv_cache_dtype": (
          workload.get("target_precision", {}).get("kv_cache", "fp8")
      ),
      "primary_metric": spec.get("objective", {}).get("primary_metric"),
      "baseline_throughput": None,
      "status": "active",
  }

  with open(session_file_path, "w") as f:
    json.dump(session, f, indent=2)

  print(f"Successfully generated/updated: {session_file_path}")


def export_maxshard_recipe(spec: dict, branch: str, output_recipe_path: Path):
  """Generate a declarative MaxShard execution recipe for target driver."""
  output_recipe_path.parent.mkdir(parents=True, exist_ok=True)

  target = spec.get("target", {})
  hardware = spec.get("hardware", {})
  execution = spec.get("execution", {})
  workload = spec.get("workload", {})

  backend = execution.get("backend", "cdk")

  if backend == "cdk":
    # Generate Kubernetes JobSet YAML recipe for CDK/GKE
    recipe_content = f"""# MaxShard CDK Recipe for GKE Kubernetes JobSet
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: maxperf-{target.get('model_name', 'model').split('/')[-1].lower()}-{branch.replace('/', '-')[:20]}
  namespace: {execution.get('namespace', 'default')}
spec:
  replicatedJobs:
    - name: workers
      replicas: 1
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-tpu-accelerator: {hardware.get('accelerator', 'tpu-v6e-slice').lower().replace(' ', '-')}
            cloud.google.com/gke-tpu-topology: "{'x'.join(map(str, hardware.get('mesh_shape', [hardware.get('chip_count', 8)])))}"
          containers:
            - name: maxperf-benchmark
              image: us-docker.pkg.dev/cloud-tpu-images/vllm/vllm-tpu:latest
              env:
                - name: GIT_BRANCH
                  value: "{branch}"
                - name: MODEL_NAME
                  value: "{target.get('model_name')}"
                - name: TP_SIZE
                  value: "{hardware.get('chip_count', 8)}"
                - name: BATCH_SIZE
                  value: "{workload.get('batch_size', 64)}"
                - name: PROMPT_LEN
                  value: "{workload.get('prompt_length_tokens', 4096)}"
                - name: OUTPUT_LEN
                  value: "{workload.get('output_length_tokens', 1024)}"
                - name: KV_CACHE_DTYPE
                  value: "{workload.get('target_precision', {}).get('kv_cache', 'fp8')}"
              command: ["/bin/bash", "-c"]
              args:
                - |
                  git checkout $GIT_BRANCH
                  bash runs/run_level0.sh
"""
  elif backend == "xmanager":
    # Generate XManager/Borg launch recipe
    recipe_content = f"""# MaxShard XManager Launch Recipe
experiment_name: "maxperf_{target.get('model_name', 'model').split('/')[-1].lower()}_{branch.replace('/', '_')}"
target_branch: "{branch}"
hardware:
  accelerator: "{hardware.get('accelerator', 'TPU v6e')}"
  chip_count: {hardware.get('chip_count', 8)}
  mesh_shape: {hardware.get('mesh_shape', [hardware.get('chip_count', 8)])}
workload:
  model: "{target.get('model_name')}"
  batch_size: {workload.get('batch_size', 64)}
  prompt_length: {workload.get('prompt_length_tokens', 4096)}
  output_length: {workload.get('output_length_tokens', 1024)}
  kv_cache_dtype: "{workload.get('target_precision', {}).get('kv_cache', 'fp8')}"
entrypoint: "bash runs/run_level0.sh"
"""
  else:
    # Default shell/UBench recipe
    recipe_content = f"""#!/usr/bin/env bash
# MaxShard Standardized Driver Recipe
export GIT_BRANCH="{branch}"
export MODEL_NAME="{target.get('model_name')}"
export TP_SIZE={hardware.get('chip_count', 8)}
export BATCH_SIZE={workload.get('batch_size', 64)}
export PROMPT_LEN={workload.get('prompt_length_tokens', 4096)}
export OUTPUT_LEN={workload.get('output_length_tokens', 1024)}
export KV_CACHE_DTYPE="{workload.get('target_precision', {}).get('kv_cache', 'fp8')}"

echo "=== Executing MaxShard Workload Benchmark ==="
bash runs/run_level0.sh
"""

  with open(output_recipe_path, "w") as f:
    f.write(recipe_content)

  print(
      f"Successfully generated MaxShard ({backend}) recipe in:"
      f" {output_recipe_path}"
  )


def main():
  parser = argparse.ArgumentParser(
      description="Validate optimization_spec.json and export agent inputs."
  )
  parser.add_argument(
      "spec_path",
      nargs="?",
      default="optimization_spec.json",
      help="Path to optimization_spec.json",
  )
  parser.add_argument(
      "--export-roofline-dir",
      type=str,
      default=None,
      help="Directory to export roofline input files",
  )
  parser.add_argument(
      "--export-session",
      type=str,
      default=None,
      help="File path to write session.json",
  )
  parser.add_argument(
      "--export-maxshard-recipe",
      type=str,
      default=None,
      help="File path to write MaxShard execution recipe",
  )
  parser.add_argument(
      "--branch",
      type=str,
      default="main",
      help="Target git branch for the recipe",
  )

  args = parser.parse_args()
  spec_file = Path(args.spec_path)

  if not spec_file.exists():
    print(
        f"Error: Specification file '{spec_file}' not found.", file=sys.stderr
    )
    sys.exit(1)

  try:
    with open(spec_file, "r") as f:
      data = json.load(f)
  except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON syntax in '{spec_file}': {e}", file=sys.stderr)
    sys.exit(1)

  errors = validate_spec(data)
  if errors:
    print("Validation FAILED with the following errors:", file=sys.stderr)
    for err in errors:
      print(f"  - {err}", file=sys.stderr)
    sys.exit(1)

  print(f"Optimization spec '{spec_file}' is VALID.")

  if args.export_roofline_dir:
    export_roofline_inputs(data, Path(args.export_roofline_dir))

  if args.export_session:
    export_session_json(data, Path(args.export_session))

  if args.export_maxshard_recipe:
    export_maxshard_recipe(data, args.branch, Path(args.export_maxshard_recipe))


if __name__ == "__main__":
  main()
