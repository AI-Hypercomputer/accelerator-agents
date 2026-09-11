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
  total_nodes = len(nodes)

  compilation_failures = 0
  correctness_failures = 0
  valid_speedups = []
  total_tokens = 0
  lines = []
  lines.append("--- MaxKernel Search Distribution Report ---")
  lines.append(f"Problem ID: {graph_data.get('problem_id')}")
  lines.append(f"Total Candidates Explored: {total_nodes}\n")
  for node_id, node in nodes.items():
    eval_result = node.get("evaluation", {})
    session_dir = node.get("session_dir")

    # 1. Tally execution/compilation outcomes
    if not eval_result.get("compiled", False):
      compilation_failures += 1
    elif not eval_result.get("correct", False):
      correctness_failures += 1
    else:
      speedup = eval_result.get("speedup")
      if speedup is not None:
        valid_speedups.append(speedup)
    # 2. Extract Token metrics from individual session.json files
    if session_dir and os.path.exists(
      os.path.join(session_dir, "session.json")
    ):
      try:
        with open(
          os.path.join(session_dir, "session.json"), "r", encoding="utf-8"
        ) as sf:
          session_data = json.load(sf)
          usage = session_data.get("usage_metadata", {})
          total_tokens += usage.get("total_token_count", 0)
      except (json.JSONDecodeError, OSError):
        pass
  # 3. Calculate Distribution
  valid_count = len(valid_speedups)
  success_rate = (valid_count / total_nodes) * 100 if total_nodes > 0 else 0

  lines.append("--- Pipeline Reliability ---")
  lines.append(f"Compile Failures: {compilation_failures} / {total_nodes}")
  lines.append(
    f"Correctness/Test Failures: {correctness_failures} / {total_nodes}"
  )
  lines.append(f"Successful Candidates: {valid_count} ({success_rate:.1f}%)\n")
  if valid_speedups:
    lines.append("--- Performance Distribution (Speedups) ---")
    lines.append(f"Best Speedup:  {max(valid_speedups):.3f}x")
    lines.append(f"Mean Speedup:  {statistics.mean(valid_speedups):.3f}x")
    lines.append(f"Median Speedup:{statistics.median(valid_speedups):.3f}x")
    lines.append(f"Worst Speedup: {min(valid_speedups):.3f}x\n")
  lines.append("--- Cost / Overhead ---")
  lines.append(f"Total Tokens across all candidates: {total_tokens:,}")
  return "\n".join(lines)


def parse_search_results_print(target_dir: str):
  """Legacy printing for CLI usage."""
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
