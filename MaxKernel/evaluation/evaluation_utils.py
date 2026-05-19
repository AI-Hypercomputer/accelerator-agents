import json
import logging
import math
import os
from dataclasses import asdict
from typing import Optional

import matplotlib.pyplot as plt
import yaml

from evaluation.custom_types.evaluation_result import EvaluationResult
from evaluation.custom_types.kernel_task import KernelTask

logger = logging.getLogger(__name__)


class _LiteralString(str):
  pass


def _literal_presenter(dumper, data):
  return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_LiteralString, _literal_presenter, Dumper=yaml.SafeDumper)


def write_kernel_task_to_yaml(task: KernelTask, yaml_path: str) -> None:
  """
  Writes a KernelTask instance to a YAML file.

  Args:
      task: The KernelTask instance to save.
      yaml_path: Path to the output YAML file.
  """
  task_dict = asdict(task)

  # Ensure multiline code string is formatted cleanly as a literal block
  if "input_gen_code" in task_dict and isinstance(
    task_dict["input_gen_code"], str
  ):
    task_dict["input_gen_code"] = _LiteralString(task_dict["input_gen_code"])

  with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(task_dict, f, sort_keys=False, width=1000)


def load_kernel_task_from_yaml(yaml_path: str) -> KernelTask:
  """
  Reads a YAML file and converts it into a KernelTask instance.

  Args:
      yaml_path: Path to the kernel_task.yaml file.

  Returns:
      An instance of KernelTask.
  """
  if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"The file {yaml_path} does not exist.")

  with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  return KernelTask(
    task_id=data.get("task_id"),
    description=data.get("description"),
    input_gen_code=data.get("input_gen_code"),
  )


def print_eval_result(result: EvaluationResult):
  """Formats and prints the evaluation results.

  Args:
      result: An EvaluationResult object.
  """
  print("\n" + "=" * 40)
  print("JAX KERNEL EVALUATION REPORT")
  print("=" * 40)
  if result.error_trace:
    print(f"Error:          {result.error_trace}")
  else:
    print(
      f"Correctness:    [{'PASS' if result.numerically_correct else 'FAIL'}]"
    )
    if result.max_abs_diff is not None:
      print(f"Max Absolute Difference: {result.max_abs_diff:.6e}")
    if result.max_rel_diff is not None:
      print(f"Max Relative Difference: {result.max_rel_diff:.6e}")
    print("-" * 40)
    print(f"Reference time:       {result.reference_time_ms:.3f} ms")
    print(f"Optimized time:       {result.optimized_time_ms:.3f} ms")
    speedup = result.speedup
    if speedup is not None:
      print(f"Speedup:        {speedup:.2f}x")
    else:
      print("Speedup:        N/A")
  print("=" * 40 + "\n")


def summarize_results(
  results: list,
  speedup_threshold: float,
  output_dir: Optional[str] = None,
) -> None:
  """
  Calculates and prints summary statistics for a list of evaluation results.

  Args:
      results: A list of dictionaries, where each dictionary is an evaluation result.
      speedup_threshold: The minimum speedup factor to consider an improvement.
      output_dir: Optional directory path to save the summary report and stats.
  """
  total_attempted = len(results)
  if not total_attempted:
    logger.info("No results found to summarize and no tasks discovered.")
    return

  compiled_tasks = [r for r in results if r.get("compiled_successfully")]
  num_compiled = len(compiled_tasks)

  correct_tasks = [r for r in compiled_tasks if r.get("numerically_correct")]
  num_correct = len(correct_tasks)

  # Speedup calculations should only be on tasks that are numerically correct.
  speedups = [
    r["speedup"] for r in correct_tasks if r.get("speedup") is not None
  ]

  improvements = [s for s in speedups if s > speedup_threshold]
  num_improved = len(improvements)

  slowdowns = [s for s in speedups if s < 1]
  num_slowdowns = len(slowdowns)

  # --- Ratios ---
  compile_ratio = (num_compiled / total_attempted) * 100
  correct_ratio = (num_correct / total_attempted) * 100
  improvement_ratio = (num_improved / total_attempted) * 100
  slowdown_ratio = (num_slowdowns / total_attempted) * 100

  # --- Speedup Statistics ---
  speedups_cliped = [max(speed, 1) for speed in speedups]
  max_speedup = max(speedups_cliped) if speedups_cliped else 1

  arithmetic_mean_speedup = (
    sum(speedups_cliped) / len(speedups_cliped) if speedups_cliped else 1
  )

  # Geometric mean is more robust for averaging ratios like speedup.
  geo_mean_speedup = (
    math.exp(sum(math.log(s) for s in speedups_cliped) / len(speedups_cliped))
    if speedups_cliped
    else 1
  )

  summary_lines = [
    "",
    "=" * 40,
    "EVALUATION SUMMARY",
    "=" * 40,
    "Task Scope:",
    f"  - Attempted / Evaluated:     {total_attempted}",
    "-" * 40,
    "Compilation & Correctness (of attempted tasks):",
    f"  - Compiled Successfully: {num_compiled} ({compile_ratio:.2f}%)",
    f"  - Failed Compilation/Runtime: {total_attempted - num_compiled}",
    f"  - Numerically Correct: {num_correct} ({correct_ratio:.2f}%)",
    "-" * 40,
    "Performance Summary (for passed tasks):",
    f"  - Improved (Speedup > 1x): {num_improved} ({improvement_ratio:.2f}%)",
    f"  - Slower (Speedup < 1x):   {num_slowdowns} ({slowdown_ratio:.2f}%)",
    f"  - Max Speedup:             {max_speedup:.2f}x",
    f"  - Average Speedup (Arithmetic): {arithmetic_mean_speedup:.2f}x",
    f"  - Average Speedup (Geometric):  {geo_mean_speedup:.2f}x",
    "=" * 40,
    "",
  ]

  for line in summary_lines:
    logger.info(line)

  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

    with open(
      os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8"
    ) as f:
      f.write("\n".join(summary_lines))

    stats = {
      "total_attempted": total_attempted,
      "compiled_successfully": num_compiled,
      "failed_compilation": total_attempted - num_compiled,
      "numerically_correct": num_correct,
      "improved": num_improved,
      "slower": num_slowdowns,
      "max_speedup": max_speedup,
      "arithmetic_mean_speedup": arithmetic_mean_speedup,
      "geometric_mean_speedup": geo_mean_speedup,
      "compile_ratio_pct": compile_ratio,
      "correct_ratio_pct": correct_ratio,
      "improvement_ratio_pct": improvement_ratio,
      "slowdown_ratio_pct": slowdown_ratio,
    }
    with open(
      os.path.join(output_dir, "summary.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(stats, f, indent=4)

    logger.info(f"Saved evaluation summary and stats to {output_dir}")


def visualize_speed_up(results: list, output_dir: str) -> None:
  """
  Visualizes the evaluation results.

  Args:
      results: A list of dictionaries, where each dictionary is an evaluation result.
      output_dir: Directory path to save the output PNG files.
                  Will generate speedup_distribution.png and
                  speedup_barplot.png in this directory.
  """
  os.makedirs(output_dir, exist_ok=True)

  def set_log_ticks(ax, log_values):
    min_log = min(math.floor(min(log_values)) if log_values else 0, 0)
    max_log = max(math.ceil(max(log_values)) if log_values else 0, 0)
    ticks = list(range(min_log, max_log + 1))
    labels = [f"{int(2**t)}x" if 2**t >= 1 else f"{2**t:.2f}x" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

  # Prepare data
  plot_data = []
  for r in results:
    is_valid = r.get("compiled_successfully") and r.get("numerically_correct")
    s = r.get("speedup") if is_valid else None
    log_s = math.log2(s) if s and s > 0 else -10.0
    plot_data.append((r["task_id"], log_s, not is_valid))

  plot_data.sort(key=lambda x: x[1])
  sorted_task_ids, sorted_log_speedups, sorted_is_invalid = zip(*plot_data)

  # Derive paths for the two separate PNGs
  dist_path = os.path.join(output_dir, "speedup_distribution.png")
  lollipop_path = os.path.join(output_dir, "speedup_lollipop.png")

  # 1. Speedup distribution
  fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
  valid_logs = [s for s, invalid in zip(sorted_log_speedups, sorted_is_invalid) if not invalid]
  
  if not valid_logs:
    logger.warning("No valid speedup results found for distribution plot.")
    ax1.text(0.5, 0.5, "No valid speedup data", ha="center", va="center")
  else:
    ax1.hist(valid_logs, bins=50, color="skyblue", edgecolor="black")
    ax1.set_title("Speedup Distribution (Log Scale)")
    set_log_ticks(ax1, valid_logs)
    ax1.set_xlabel("Speedup")
    ax1.set_ylabel("Frequency")
    ax1.axvline(x=0.0, color="red", linestyle="--", label="Baseline (1.0x)")
    ax1.legend()

  plt.tight_layout()
  fig1.savefig(dist_path)
  plt.close(fig1)
  logger.info(f"Saved distribution plot to {dist_path}")

  # 2. Lollipop chart for each problem
  fig2, ax2 = plt.subplots(1, 1, figsize=(10, 12))
  y_pos = range(len(sorted_task_ids))
  
  # Separate valid and invalid data for vectorized plotting
  valid_idx = [i for i, invalid in enumerate(sorted_is_invalid) if not invalid]
  invalid_idx = [i for i, invalid in enumerate(sorted_is_invalid) if invalid]
  
  y_valid = [y_pos[i] for i in valid_idx]
  x_valid = [sorted_log_speedups[i] for i in valid_idx]
  y_invalid = [y_pos[i] for i in invalid_idx]
  
  # Plot lollipops for valid tasks
  ax2.hlines(y=y_valid, xmin=0, xmax=x_valid, color="skyblue", linewidth=2)
  ax2.plot(x_valid, y_valid, "o", color="blue", markersize=6)
  
  # Plot 'x' for invalid tasks at baseline
  ax2.plot([0] * len(y_invalid), y_invalid, marker="x", color="red", linestyle="None", markersize=8)

  ax2.set_yticks(y_pos)
  ax2.set_yticklabels(sorted_task_ids, fontsize=8)
  ax2.set_title("Log2 Speedup for Each Problem")
      
  set_log_ticks(ax2, x_valid)
  
  ax2.set_xlabel("Speedup")
  ax2.axvline(x=0.0, color="red", linestyle="--")

  plt.tight_layout()
  fig2.savefig(lollipop_path)
  plt.close(fig2)
  logger.info(f"Saved lollipop plot to {lollipop_path}")
