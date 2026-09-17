import argparse
import json
import os
import statistics


def analyze_distribution(target_dir: str) -> str:
  """Parses search_graph.json in the target directory and returns a markdown summary."""
  graph_json_path = os.path.join(target_dir, "search_graph.json")
  if not os.path.exists(graph_json_path):
    return f"Error: Could not find {graph_json_path}"
  try:
    with open(graph_json_path, "r", encoding="utf-8") as f:
      graph_data = json.load(f)
  except (json.JSONDecodeError, OSError) as e:
    return f"Error loading {graph_json_path}: {e}"

  nodes = graph_data.get("nodes", {})
  # Only count actual LLM attempts (ignore base node_000)
  total_nodes = len(
    [
      n
      for k, n in nodes.items()
      if str(k) != "node_000" and n.get("depth") != 0
    ]
  )

  framework_failures = 0
  compilation_failures = 0
  execution_failures = 0
  correctness_failures = 0
  valid_speedups = []

  token_counts = []
  call_counts = []
  corrupted_token_nodes = []

  lines = []
  lines.append("--- MaxKernel Search Distribution Report ---")
  lines.append(f"Problem ID: {graph_data.get('problem_id')}")
  lines.append(f"Total LLM Candidates Generated: {total_nodes}\n")

  for node_id, node in nodes.items():
    # Skip the baseline reference code
    if str(node_id) == "node_000" or node.get("depth") == 0:
      continue

    eval_result = node.get("evaluation")

    # 1. Catch ADK Framework Failures (Agent crashed before it could test code)
    if not eval_result:
      framework_failures += 1
      continue

    # 2. Count execution/compilation outcomes (decoupled)
    if not eval_result.get("compiled", False):
      compilation_failures += 1
    elif not eval_result.get("execution_status", True):
      execution_failures += 1
    elif not eval_result.get("correct", False):
      correctness_failures += 1
    else:
      speedup = eval_result.get("speedup")
      if speedup is not None:
        valid_speedups.append(speedup)

    # 3. Extract Token metrics using the relocated nodes/ directory
    session_dir = node.get("session_dir", "")
    node_folder_name = (
      os.path.basename(session_dir.rstrip("/")) if session_dir else node_id
    )
    token_metrics_path = os.path.join(
      target_dir, "nodes", node_folder_name, "token_metrics.json"
    )

    if os.path.exists(token_metrics_path):
      try:
        with open(token_metrics_path, "r", encoding="utf-8") as sf:
          tm_data = json.load(sf)

          node_tokens = 0
          node_calls = 0
          for iter_data in tm_data.get("iterations", {}).values():
            for agent_name, metrics in iter_data.get("agents", {}).items():
              if isinstance(metrics, dict):
                node_tokens += metrics.get("total_tokens", 0)
                node_calls += metrics.get("calls", 0)

          if node_calls > 0:
            token_counts.append(node_tokens)
            call_counts.append(node_calls)
      except (json.JSONDecodeError, OSError) as e:
        print(
          f"WARNING: Skipping corrupted token metrics for node '{node_id}'. Cost will be undercounted! Error: {e}"
        )
        corrupted_token_nodes.append(node_id)

  # 4. Calculate Results
  valid_count = len(valid_speedups)
  success_rate = (valid_count / total_nodes) * 100 if total_nodes > 0 else 0

  lines.append("--- Pipeline Reliability ---")
  lines.append(f"ADK/Framework Failures: {framework_failures} / {total_nodes}")
  lines.append(f"Compile Failures: {compilation_failures} / {total_nodes}")
  lines.append(f"Execution Failures: {execution_failures} / {total_nodes}")
  lines.append(
    f"Correctness/Test Failures: {correctness_failures} / {total_nodes}"
  )
  lines.append(f"Successful Candidates: {valid_count} ({success_rate:.1f}%)\n")

  lines.append("--- Performance Distribution (Speedups) ---")
  if valid_speedups:
    lines.append(f"Best Speedup:  {max(valid_speedups):.3f}x")
    lines.append(f"Mean Speedup:  {statistics.mean(valid_speedups):.3f}x")
    lines.append(f"Median Speedup:{statistics.median(valid_speedups):.3f}x")
    lines.append(f"Worst Speedup: {min(valid_speedups):.3f}x\n")
  else:
    lines.append(
      "Since no candidates compiled or succeeded at all, no speed-up was found.\n"
    )

  lines.append("--- Cost / Overhead ---")

  if corrupted_token_nodes:
    lines.append(
      f"WARNING: Token computations are undercounted! The following nodes had corrupted metric files: {', '.join(corrupted_token_nodes)}\n"
    )

  if token_counts:
    lines.append("Tokens:")
    lines.append(f"  Min:    {min(token_counts):,}")
    lines.append(f"  Median: {statistics.median(token_counts):,.0f}")
    lines.append(f"  Mean:   {statistics.mean(token_counts):,.0f}")
    lines.append(f"  Max:    {max(token_counts):,}")
    lines.append(f"  SUM:    {sum(token_counts):,}\n")

  if call_counts:
    lines.append("LLM API Calls:")
    lines.append(f"  Min:    {min(call_counts):,}")
    lines.append(f"  Median: {statistics.median(call_counts):,.0f}")
    lines.append(f"  Mean:   {statistics.mean(call_counts):,.0f}")
    lines.append(f"  Max:    {max(call_counts):,}")
    lines.append(f"  SUM:    {sum(call_counts):,}\n")

  md_output = "\n".join(lines)

  # 5. Save exports natively inside the run directory before returning
  md_out = os.path.join(target_dir, "search_distribution_summary.md")
  with open(md_out, "w", encoding="utf-8") as mdf:
    mdf.write(md_output)

  pipeline_data = {
    "problem_id": graph_data.get("problem_id"),
    "total_llm_candidates": total_nodes,
    "pipeline_reliability": {
      "framework_failures": framework_failures,
      "compile_failures": compilation_failures,
      "execution_failures": execution_failures,
      "correctness_failures": correctness_failures,
      "successful_candidates": valid_count,
      "success_rate": success_rate,
    },
    "performance_speedups": {
      "best": max(valid_speedups) if valid_speedups else None,
      "mean": statistics.mean(valid_speedups) if valid_speedups else None,
      "median": statistics.median(valid_speedups) if valid_speedups else None,
      "worst": min(valid_speedups) if valid_speedups else None,
    },
    "overhead": {
      "tokens": {
        "min": min(token_counts) if token_counts else None,
        "median": statistics.median(token_counts) if token_counts else None,
        "mean": statistics.mean(token_counts) if token_counts else None,
        "max": max(token_counts) if token_counts else None,
        "sum": sum(token_counts) if token_counts else 0,
      },
      "calls": {
        "min": min(call_counts) if call_counts else None,
        "median": statistics.median(call_counts) if call_counts else None,
        "mean": statistics.mean(call_counts) if call_counts else None,
        "max": max(call_counts) if call_counts else None,
        "sum": sum(call_counts) if call_counts else 0,
      },
    },
  }

  json_out = os.path.join(target_dir, "search_distribution_metrics.json")
  with open(json_out, "w", encoding="utf-8") as jf:
    json.dump(pipeline_data, jf, indent=2)

  return md_output


def parse_search_results_print(target_dir: str):
  print(analyze_distribution(target_dir))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Parse MaxKernel Auto-Search distributions."
  )
  parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="Path to the output directory containing search_graph.json",
  )
  args = parser.parse_args()
  parse_search_results_print(args.target)
