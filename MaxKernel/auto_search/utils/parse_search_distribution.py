import argparse
import json
import os
import statistics


def parse_search_results(graph_json_path: str):
  if not os.path.exists(graph_json_path):
    print(f"Error: Could not find {graph_json_path}")
    return

  graph_dir = os.path.dirname(os.path.abspath(graph_json_path))

  with open(graph_json_path, "r") as f:
    graph_data = json.load(f)

  nodes = graph_data.get("nodes", {})

  compilation_failures = 0
  correctness_failures = 0
  valid_speedups = []
  generated_candidates = 0

  tokens_per_candidate = []
  llm_calls_per_candidate = []
  time_per_candidate = []

  # Data structure for JSON pipeline export
  pipeline_data = {
    "problem_id": graph_data.get("problem_id", "unknown"),
    "candidates": [],
    "summary": {},
  }

  output_lines = []

  def log(msg=""):
    print(msg)
    output_lines.append(msg)

  log("--- MaxKernel Search Distribution Report ---")
  log(f"Problem ID: {pipeline_data['problem_id']}\n")

  for node_id, node in nodes.items():
    if node.get("parent_id") is None:
      continue

    generated_candidates += 1
    eval_result = node.get("evaluation", {})
    session_dir = node.get("session_dir")

    status = "failed_compile"
    speedup = None
    if not eval_result.get("compiled", False):
      compilation_failures += 1
    elif not eval_result.get("correct", False):
      correctness_failures += 1
      status = "failed_correctness"
    else:
      speedup = eval_result.get("speedup")
      if speedup is not None:
        valid_speedups.append(speedup)
        status = "success"

    candidate_tokens = 0
    candidate_llm_calls = 0
    candidate_time = 0.0

    if session_dir:
      if not os.path.isabs(session_dir):
        session_dir = os.path.join(graph_dir, session_dir)

      token_json_path = os.path.join(session_dir, "token_metrics.json")
      if os.path.exists(token_json_path):
        with open(token_json_path, "r") as tf:
          token_data = json.load(tf)
          iterations = token_data.get("iterations", {})
          for iter_key, iter_data in iterations.items():
            candidate_llm_calls += len(iter_data.get("llm_calls", []))
            agents = iter_data.get("agents", {})
            for agent_name, agent_data in agents.items():
              candidate_tokens += agent_data.get("total_tokens", 0)

      timing_json_path = os.path.join(session_dir, "timing_metrics.json")
      if os.path.exists(timing_json_path):
        try:
          with open(timing_json_path, "r") as tmf:
            timing_data = json.load(tmf)
            for it, it_data in timing_data.get("iterations", {}).items():
              events = it_data.get("events", [])
              if events:
                earliest = min(
                  (
                    e.get("start_time", float("inf"))
                    for e in events
                    if "start_time" in e
                  ),
                  default=0,
                )
                latest = max(
                  (e.get("end_time", 0) for e in events if "end_time" in e),
                  default=0,
                )
                if latest > earliest:
                  candidate_time += latest - earliest
        except Exception:
          pass

    tokens_per_candidate.append(candidate_tokens)
    llm_calls_per_candidate.append(candidate_llm_calls)
    if candidate_time > 0:
      time_per_candidate.append(candidate_time)

    pipeline_data["candidates"].append(
      {
        "node_id": node_id,
        "status": status,
        "speedup": speedup,
        "tokens": candidate_tokens,
        "llm_calls": candidate_llm_calls,
        "wall_time_s": candidate_time,
      }
    )

  success_rate = (
    (len(valid_speedups) / generated_candidates) * 100
    if generated_candidates > 0
    else 0
  )

  log(f"Total LLM Candidates Generated: {generated_candidates}")
  log("--- Pipeline Reliability ---")
  log(f"Compile Failures: {compilation_failures} / {generated_candidates}")
  log(
    f"Correctness/Test Failures: {correctness_failures} / {generated_candidates}"
  )
  log(f"Successful Candidates: {len(valid_speedups)} ({success_rate:.1f}%)\n")

  pipeline_data["summary"]["reliability"] = {
    "generated": generated_candidates,
    "compile_failures": compilation_failures,
    "correctness_failures": correctness_failures,
    "success_rate": success_rate,
  }

  if valid_speedups:
    log("--- Performance Distribution (Speedups) ---")
    log(f"Best Speedup:  {max(valid_speedups):.3f}x")
    log(f"Mean Speedup:  {statistics.mean(valid_speedups):.3f}x")
    log(f"Median Speedup:{statistics.median(valid_speedups):.3f}x")
    log(f"Worst Speedup: {min(valid_speedups):.3f}x\n")

    pipeline_data["summary"]["performance"] = {
      "best": max(valid_speedups),
      "mean": statistics.mean(valid_speedups),
      "median": statistics.median(valid_speedups),
      "worst": min(valid_speedups),
    }

  log("--- Cost / Overhead ---")

  def log_stats(name, data_list, format_string, json_key):
    if not data_list:
      return
    stats = {
      "min": min(data_list),
      "median": statistics.median(data_list),
      "mean": statistics.mean(data_list),
      "max": max(data_list),
      "sum": sum(data_list),
    }
    pipeline_data["summary"][json_key] = stats

    log(f"{name}:")
    log(f"  Min:    {format_string.format(stats['min'])}")
    log(f"  Median: {format_string.format(stats['median'])}")
    log(f"  Mean:   {format_string.format(stats['mean'])}")
    log(f"  Max:    {format_string.format(stats['max'])}")
    log(f"  SUM:    {format_string.format(stats['sum'])}\n")

  log_stats("Tokens", tokens_per_candidate, "{:,.0f}", "tokens")
  log_stats("LLM API Calls", llm_calls_per_candidate, "{:,.0f}", "llm_calls")
  log_stats(
    "Wall Search Time (Seconds)", time_per_candidate, "{:,.1f}s", "wall_time"
  )

  # 4. Save exports natively inside the run directory!
  md_out = os.path.join(graph_dir, "search_distribution_summary.md")
  json_out = os.path.join(graph_dir, "search_distribution_metrics.json")

  with open(md_out, "w") as mdf:
    mdf.write("```text\n" + "\n".join(output_lines) + "\n```\n")

  with open(json_out, "w") as jf:
    json.dump(pipeline_data, jf, indent=2)

  print(f"\n[+] Saved Markdown Report: {md_out}")
  print(f"[+] Saved JSON Pipeline Data: {json_out}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Parse MaxKernel Auto-Search distributions."
  )
  parser.add_argument(
    "--graph_path",
    type=str,
    required=True,
    help="Path to the graph UI JSON dump.",
  )
  args = parser.parse_args()

  parse_search_results(args.graph_path)
