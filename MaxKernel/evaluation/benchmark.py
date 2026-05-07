import argparse
import json
import logging
import os
from dataclasses import asdict
from typing import List, Optional

from evaluation.custom_types.evaluation_result import EvaluationResult
from evaluation.evaluation_utils import summarize_results, visualize_speed_up
from evaluation.jax_kernel_evaluator import JAXKernelEvaluator

logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def benchmark(
  local: bool = False,
  tpu_name: Optional[str] = None,
  zone: Optional[str] = None,
  project: Optional[str] = None,
  venv_path: Optional[str] = None,
  dataset_dir: str = "",
  reference_file_name: str = "reference.py",
  optimized_file_name: str = "optimized.py",
  task_file_name: str = "kernel_task.yaml",
  adapt: Optional[List[str]] = None,
  atol: float = 1e-3,
  rtol: float = 1e-3,
):
  """
  Evaluates JAX kernels across a dataset of problems on a remote TPU VM.

  Iterates through a dataset directory containing reference and optimized JAX scripts,
  runs them on the specified TPU using JAXKernelEvaluator, and saves the evaluation
  results to a JSON file. Automatically resumes from previous runs if an output file exists.

  Args:
      local: If True, runs evaluation locally instead of on a remote TPU VM.
      tpu_name: The name of the target TPU VM.
      zone: The Google Cloud zone where the TPU is located.
      project: The Google Cloud project ID.
      venv_path: The absolute path to the Python virtual environment on the remote TPU.
      dataset_dir: The local directory containing the benchmark dataset (problem folders).
      adapt: Optional list specifying which components to adapt via LLM before evaluation.
             Can contain 'reference_code', 'optimized_code', 'kernel_task'.
  """
  evaluator = JAXKernelEvaluator(
    local=local,
    tpu_name=tpu_name,
    zone=zone,
    project=project,
    venv_path=venv_path,
  )

  dataset_dir = dataset_dir
  if not os.path.exists(dataset_dir):
    logger.error(f"Directory {dataset_dir} does not exist.")
    return

  # Add a file handler to save the log to the dataset directory
  file_handler = logging.FileHandler(os.path.join(dataset_dir, "benchmark.log"))
  file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
  )
  logging.getLogger().addHandler(file_handler)

  # Get all subdirectories (problem folders)
  problem_dirs = sorted(
    [
      d
      for d in os.listdir(dataset_dir)
      if os.path.isdir(os.path.join(dataset_dir, d))
    ]
  )
  if local:
    output_file_name = "evaluation_results_local.json"
  else:
    output_file_name = f"evaluation_results_{evaluator.client.tpu_name}.json"
  output_file = os.path.join(dataset_dir, output_file_name)
  results = []
  evaluated_tasks = set()
  skipped_problems = []

  # Load existing results and populate the set of evaluated tasks.
  if os.path.exists(output_file):
    try:
      with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
      if not isinstance(results, list):
        logger.warning(
          f"{output_file} does not contain a list. Starting fresh."
        )
        results = []
      else:
        for res in results:
          if isinstance(res, dict) and "task_id" in res:
            evaluated_tasks.add(res["task_id"])
        logger.info(
          f"Loaded {len(evaluated_tasks)} existing results from {output_file}."
        )
    except (json.JSONDecodeError, IOError) as e:
      logger.warning(
        f"Could not read or parse {output_file}: {e}. Starting fresh."
      )
      results = []

  for problem_name in problem_dirs:
    if problem_name in evaluated_tasks:
      logger.info(
        f"Skipping {problem_name}: Result already exists in {output_file}."
      )
      skipped_problems.append(f"{problem_name} (already evaluated)")
      continue
    problem_path = os.path.join(dataset_dir, problem_name)
    yaml_path = os.path.join(problem_path, task_file_name)
    reference_code_path = os.path.join(problem_path, reference_file_name)
    optimized_code_path = os.path.join(problem_path, optimized_file_name)

    if not os.path.exists(yaml_path) and not (adapt and "kernel_task" in adapt):
      logger.warning(
        f"{problem_name}: {task_file_name} not found and 'kernel_task' not in adapt, skipping."
      )
      skipped_problems.append(f"{problem_name} (missing task yaml)")
      result = EvaluationResult(task_id=problem_name)
    elif not os.path.exists(reference_code_path):
      logger.warning(f"{problem_name}: reference script not found, skipping.")
      skipped_problems.append(f"{problem_name} (missing reference script)")
      result = EvaluationResult(task_id=problem_name)
    elif not os.path.exists(optimized_code_path):
      logger.warning(f"{problem_name}: optimized script not found, skipping.")
      skipped_problems.append(f"{problem_name} (missing optimized script)")
      result = EvaluationResult(task_id=problem_name)
    else:
      logger.info(f"Evaluating {problem_name}...")
      try:
        result = evaluator.evaluate(
          task_yaml_path=yaml_path,
          reference_code_path=reference_code_path,
          optimized_code_path=optimized_code_path,
          adapt=adapt,
          atol=atol,
          rtol=rtol,
        )
      except Exception as e:
        logger.error(f"Error evaluating {problem_name}: {e}")
        result = EvaluationResult(
          task_id=problem_name,
          error_trace=f"Error encountered during evaluation: {str(e)}",
        )

    # Convert the new result to a dictionary
    res_dict = asdict(result)
    res_dict["speedup"] = result.speedup if result.speedup is not None else None

    # Append to the main list and save immediately
    results.append(res_dict)
    with open(output_file, "w") as f:
      json.dump(results, f, indent=4)
    logger.info(
      f"Saved result for {problem_name}. Total results: {len(results)}."
    )

  # Calculate and print statistics
  if results:
    summarize_results(results, speedup_threshold=1.05, output_dir=dataset_dir)

    # Generate visualization
    visualize_speed_up(results, output_dir=dataset_dir)

  if skipped_problems:
    logger.info("\n" + "=" * 40)
    logger.info("SKIPPED PROBLEMS SUMMARY")
    logger.info("=" * 40)
    for sp in skipped_problems:
      logger.info(f"  - {sp}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Evaluate JAX kernels on a remote TPU VM."
  )
  parser.add_argument(
    "--local",
    action="store_true",
    help="Run evaluation locally instead of on a remote TPU.",
  )
  parser.add_argument(
    "--tpu_name",
    type=str,
    help="The name of the target TPU VM.",
  )
  parser.add_argument(
    "--zone",
    type=str,
    help="The Google Cloud zone where the TPU is located.",
  )
  parser.add_argument(
    "--project",
    type=str,
    help="The Google Cloud project ID.",
  )
  parser.add_argument(
    "--venv_path",
    type=str,
    help="The absolute path to the Python virtual environment on the remote TPU.",
  )
  parser.add_argument(
    "--dataset_dir",
    type=str,
    help="The local directory containing the benchmark dataset (problem folders).",
  )
  parser.add_argument(
    "--adapt",
    type=str,
    nargs="*",
    default=None,
    help="Optional list specifying which components to adapt via LLM before evaluation. Can contain 'reference_code', 'optimized_code', 'kernel_task'.",
  )
  parser.add_argument(
    "--reference_file_name",
    type=str,
    default="reference.py",
    help="The filename for the reference code.",
  )
  parser.add_argument(
    "--optimized_file_name",
    type=str,
    default="optimized.py",
    help="The filename for the optimized code.",
  )
  parser.add_argument(
    "--task_file_name",
    type=str,
    default="kernel_task.yaml",
    help="The filename for the kernel task yaml.",
  )
  parser.add_argument(
    "--atol",
    type=float,
    default=1e-3,
    help="Absolute tolerance for comparison.",
  )
  parser.add_argument(
    "--rtol",
    type=float,
    default=1e-3,
    help="Relative tolerance for comparison.",
  )
  args = parser.parse_args()

  benchmark(
    local=args.local,
    tpu_name=args.tpu_name,
    zone=args.zone,
    project=args.project,
    venv_path=args.venv_path,
    dataset_dir=args.dataset_dir,
    reference_file_name=args.reference_file_name,
    optimized_file_name=args.optimized_file_name,
    task_file_name=args.task_file_name,
    adapt=args.adapt,
    atol=args.atol,
    rtol=args.rtol,
  )
