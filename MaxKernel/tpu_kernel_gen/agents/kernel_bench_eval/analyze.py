#!/usr/bin/env python3
"""
Analysis script for TPU kernel generation evaluation results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_eval_data(file_path: str) -> Dict[str, Any]:
  """Load evaluation data from JSON file."""
  try:
    with open(file_path, "r") as f:
      return json.load(f)
  except FileNotFoundError:
    print(f"Error: File {file_path} not found")
    sys.exit(1)
  except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in {file_path}: {e}")
    sys.exit(1)


def analyze_compilation_results(data: Dict[str, Any]) -> Dict[str, int]:
  """Analyze compilation success/failure rates."""
  stats = {
    "base_kernel_success": 0,
    "base_kernel_failure": 0,
    "tiled_kernel_success": 0,
    "tiled_kernel_failure": 0,
    "total_kernels": len(data),
  }

  for kernel_id, results in data.items():
    # Base kernel compilation
    if results.get("base_kernel_compilation_result") == "Success":
      stats["base_kernel_success"] += 1
    else:
      stats["base_kernel_failure"] += 1

    # Tiled kernel compilation
    tiled_result = results.get("tiled_kernel_compilation_result")
    if tiled_result == "Success":
      stats["tiled_kernel_success"] += 1
    elif tiled_result and tiled_result != "Success":
      stats["tiled_kernel_failure"] += 1

  return stats


def analyze_correctness_results(data: Dict[str, Any]) -> Dict[str, int]:
  """Analyze correctness test results."""
  stats = {
    "base_kernel_correct": 0,
    "base_kernel_incorrect": 0,
    "tiled_kernel_correct": 0,
    "tiled_kernel_incorrect": 0,
  }

  for kernel_id, results in data.items():
    # Base kernel correctness
    if results.get("base_kernel_correctness_result") == "Success":
      stats["base_kernel_correct"] += 1
    elif results.get("base_kernel_correctness_result"):
      stats["base_kernel_incorrect"] += 1

    # Tiled kernel correctness
    if results.get("tiled_kernel_correctness_result") == "Success":
      stats["tiled_kernel_correct"] += 1
    elif results.get("tiled_kernel_correctness_result"):
      stats["tiled_kernel_incorrect"] += 1

  return stats


def analyze_speedup_results(data: Dict[str, Any]) -> Dict[str, Any]:
  """Analyze speedup and performance optimization results."""
  speedups = []
  base_times = []
  optimized_times = []
  failed_optimizations = 0
  successful_optimizations = 0

  for kernel_id, results in data.items():
    opt_result = results.get("kernel_tiling_optimization_result")

    if isinstance(opt_result, dict) and "speedup" in opt_result:
      speedup = opt_result["speedup"]
      base_time = opt_result.get("base_time")
      opt_time = opt_result.get("optimized_time")

      if speedup > 0 and base_time and opt_time:
        speedups.append(speedup)
        base_times.append(base_time)
        optimized_times.append(opt_time)
        successful_optimizations += 1
      else:
        failed_optimizations += 1
    else:
      failed_optimizations += 1

  stats = {
    "successful_optimizations": successful_optimizations,
    "failed_optimizations": failed_optimizations,
    "total_attempts": successful_optimizations + failed_optimizations,
  }

  if speedups:
    stats.update(
      {
        "avg_speedup": sum(speedups) / len(speedups),
        "max_speedup": max(speedups),
        "min_speedup": min(speedups),
        "speedups_above_1": len([s for s in speedups if s > 1.0]),
        "speedups_below_1": len([s for s in speedups if s < 1.0]),
        "avg_base_time": sum(base_times) / len(base_times),
        "avg_optimized_time": sum(optimized_times) / len(optimized_times),
      }
    )

  return stats


def analyze_iteration_counts(data: Dict[str, Any]) -> Dict[str, Any]:
  """Analyze the number of iterations required for each step."""
  jax_iters = []
  base_iters = []
  tiled_iters = []

  for kernel_id, results in data.items():
    jax_iters.append(results.get("jax_conversion_loop_iter", 0))
    base_iters.append(results.get("fix_base_kernel_loop_iter", 0))
    tiled_iters.append(results.get("fix_tiled_kernel_loop_iter", 0))

  return {
    "avg_jax_conversion_iters": sum(jax_iters) / len(jax_iters) if jax_iters else 0,
    "max_jax_conversion_iters": max(jax_iters) if jax_iters else 0,
    "avg_base_kernel_fix_iters": sum(base_iters) / len(base_iters) if base_iters else 0,
    "max_base_kernel_fix_iters": max(base_iters) if base_iters else 0,
    "avg_tiled_kernel_fix_iters": sum(tiled_iters) / len(tiled_iters) if tiled_iters else 0,
    "max_tiled_kernel_fix_iters": max(tiled_iters) if tiled_iters else 0,
  }


def find_best_performing_kernels(data: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, float]]:
  """Find the kernels with the highest speedups."""
  kernel_speedups = []

  for kernel_id, results in data.items():
    opt_result = results.get("kernel_tiling_optimization_result")
    if isinstance(opt_result, dict) and "speedup" in opt_result:
      speedup = opt_result["speedup"]
      if speedup > 0:
        kernel_speedups.append((kernel_id, speedup))

  return sorted(kernel_speedups, key=lambda x: x[1], reverse=True)[:top_n]


def find_problematic_kernels(data: Dict[str, Any]) -> Dict[str, List[str]]:
  """Find kernels that failed at various stages."""
  problems = {"compilation_failures": [], "correctness_failures": [], "optimization_failures": []}

  for kernel_id, results in data.items():
    # Compilation failures
    if results.get("base_kernel_compilation_result") != "Success" or (
      results.get("tiled_kernel_compilation_result") and results.get("tiled_kernel_compilation_result") != "Success"
    ):
      problems["compilation_failures"].append(kernel_id)

    # Correctness failures
    if results.get("base_kernel_correctness_result") != "Success" or (
      results.get("tiled_kernel_correctness_result") and results.get("tiled_kernel_correctness_result") != "Success"
    ):
      problems["correctness_failures"].append(kernel_id)

    # Optimization failures
    opt_result = results.get("kernel_tiling_optimization_result")
    if not isinstance(opt_result, dict) or opt_result.get("speedup", 0) <= 0:
      problems["optimization_failures"].append(kernel_id)

  return problems


def print_analysis_report(data: Dict[str, Any]) -> None:
  """Print comprehensive analysis report."""
  print("=" * 80)
  print("TPU KERNEL EVALUATION ANALYSIS REPORT")
  print("=" * 80)

  # Basic statistics
  total_kernels = len(data)
  print(f"\nTOTAL KERNELS EVALUATED: {total_kernels}")

  # Compilation analysis
  print("\n" + "=" * 50)
  print("COMPILATION ANALYSIS")
  print("=" * 50)
  comp_stats = analyze_compilation_results(data)
  print(
    f"Base Kernel Compilation Success Rate: {comp_stats['base_kernel_success']}/{total_kernels} ({comp_stats['base_kernel_success'] / total_kernels * 100:.1f}%)"
  )
  print(
    f"Tiled Kernel Compilation Success Rate: {comp_stats['tiled_kernel_success']}/{total_kernels} ({comp_stats['tiled_kernel_success'] / total_kernels * 100:.1f}%)"
  )

  # Correctness analysis
  print("\n" + "=" * 50)
  print("CORRECTNESS ANALYSIS")
  print("=" * 50)
  corr_stats = analyze_correctness_results(data)
  print(
    f"Base Kernel Correctness Rate: {corr_stats['base_kernel_correct']}/{total_kernels} ({corr_stats['base_kernel_correct'] / total_kernels * 100:.1f}%)"
  )
  print(
    f"Tiled Kernel Correctness Rate: {corr_stats['tiled_kernel_correct']}/{total_kernels} ({corr_stats['tiled_kernel_correct'] / total_kernels * 100:.1f}%)"
  )

  # Performance analysis
  print("\n" + "=" * 50)
  print("PERFORMANCE OPTIMIZATION ANALYSIS")
  print("=" * 50)
  perf_stats = analyze_speedup_results(data)
  print(
    f"Successful Optimizations: {perf_stats['successful_optimizations']}/{perf_stats['total_attempts']} ({perf_stats['successful_optimizations'] / perf_stats['total_attempts'] * 100:.1f}%)"
  )

  if perf_stats["successful_optimizations"] > 0:
    print(f"Average Speedup: {perf_stats['avg_speedup']:.3f}x")
    print(f"Maximum Speedup: {perf_stats['max_speedup']:.3f}x")
    print(f"Minimum Speedup: {perf_stats['min_speedup']:.3f}x")
    print(f"Kernels with Speedup > 1.0x: {perf_stats['speedups_above_1']}")
    print(f"Kernels with Speedup < 1.0x: {perf_stats['speedups_below_1']}")
    print(f"Average Base Time: {perf_stats['avg_base_time']:.6f}s")
    print(f"Average Optimized Time: {perf_stats['avg_optimized_time']:.6f}s")

  # Iteration analysis
  print("\n" + "=" * 50)
  print("ITERATION ANALYSIS")
  print("=" * 50)
  iter_stats = analyze_iteration_counts(data)
  print(
    f"Average JAX Conversion Iterations: {iter_stats['avg_jax_conversion_iters']:.2f} (max: {iter_stats['max_jax_conversion_iters']})"
  )
  print(
    f"Average Base Kernel Fix Iterations: {iter_stats['avg_base_kernel_fix_iters']:.2f} (max: {iter_stats['max_base_kernel_fix_iters']})"
  )
  print(
    f"Average Tiled Kernel Fix Iterations: {iter_stats['avg_tiled_kernel_fix_iters']:.2f} (max: {iter_stats['max_tiled_kernel_fix_iters']})"
  )

  # Best performing kernels
  print("\n" + "=" * 50)
  print("TOP PERFORMING KERNELS")
  print("=" * 50)
  best_kernels = find_best_performing_kernels(data)
  for i, (kernel_id, speedup) in enumerate(best_kernels, 1):
    print(f"{i}. Kernel {kernel_id}: {speedup:.3f}x speedup")

  # Problematic kernels
  print("\n" + "=" * 50)
  print("PROBLEMATIC KERNELS")
  print("=" * 50)
  problems = find_problematic_kernels(data)
  print(f"Compilation Failures: {len(problems['compilation_failures'])} kernels")
  if problems["compilation_failures"]:
    print(
      f"  Kernels: {', '.join(problems['compilation_failures'][:10])}{'...' if len(problems['compilation_failures']) > 10 else ''}"
    )

  print(f"Correctness Failures: {len(problems['correctness_failures'])} kernels")
  if problems["correctness_failures"]:
    print(
      f"  Kernels: {', '.join(problems['correctness_failures'][:10])}{'...' if len(problems['correctness_failures']) > 10 else ''}"
    )

  print(f"Optimization Failures: {len(problems['optimization_failures'])} kernels")
  if problems["optimization_failures"]:
    print(
      f"  Kernels: {', '.join(problems['optimization_failures'][:10])}{'...' if len(problems['optimization_failures']) > 10 else ''}"
    )


def main():
  """Main function to run the analysis."""
  parser = argparse.ArgumentParser(description="Analyze TPU kernel evaluation results")
  parser.add_argument("eval_json", help="Path to the eval.json file")
  parser.add_argument("--detailed", action="store_true", help="Show detailed analysis including error messages")

  args = parser.parse_args()

  # Validate file path
  if not Path(args.eval_json).exists():
    print(f"Error: File {args.eval_json} does not exist")
    sys.exit(1)

  # Load and analyze data
  data = load_eval_data(args.eval_json)

  # Print analysis report
  print_analysis_report(data)

  # Show detailed error messages if requested
  if args.detailed:
    print("\n" + "=" * 50)
    print("DETAILED ERROR ANALYSIS")
    print("=" * 50)

    for kernel_id, results in data.items():
      errors = []

      # Check for compilation errors
      base_comp = results.get("base_kernel_compilation_result")
      if base_comp and base_comp != "Success":
        errors.append(f"Base compilation error: {base_comp[:200]}...")

      tiled_comp = results.get("tiled_kernel_compilation_result")
      if tiled_comp and tiled_comp != "Success":
        errors.append(f"Tiled compilation error: {tiled_comp[:200]}...")

      # Check for optimization errors
      opt_result = results.get("kernel_tiling_optimization_result")
      if isinstance(opt_result, str) and "error" in opt_result.lower():
        errors.append(f"Optimization error: {opt_result}")

      if errors:
        print(f"\nKernel {kernel_id}:")
        for error in errors:
          print(f"  - {error}")


if __name__ == "__main__":
  main()
