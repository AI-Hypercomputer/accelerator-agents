import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def generate_padded_table(headers, rows):
  # Calculate max widths
  widths = [len(h) for h in headers]
  for row in rows:
    for i, col in enumerate(row):
      widths[i] = max(widths[i], len(str(col)))

  # Format header
  header_str = (
    "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
  )
  # Format separator
  sep_str = "|" + "|".join("-" * (w + 2) for w in widths) + "|"

  # Format rows
  row_strs = []
  for row in rows:
    row_str = (
      "| "
      + " | ".join(f"{str(col):<{w}}" for col, w in zip(row, widths))
      + " |"
    )
    row_strs.append(row_str)

  return [header_str, sep_str] + row_strs


def analyze_path(target_path: str) -> str:
  path = Path(target_path)
  if not path.exists():
    return f"Target path {target_path} does not exist."

  if path.is_file():
    if path.name == "token_metrics.json":
      files_to_process = [path]
    else:
      return "Please provide a token_metrics.json file or a directory containing them."
  else:
    files_to_process = list(path.glob("**/token_metrics.json"))

  if not files_to_process:
    return ""

  all_markdown = []

  # Global Aggregators
  global_agents = {}
  global_total_calls = 0
  total_nodes = 0

  for file_path in files_to_process:
    with open(file_path, "r", encoding="utf-8") as f:
      try:
        data = json.load(f)
        node_md, node_agents, node_calls = generate_node_markdown(
          file_path, data
        )
        all_markdown.append(node_md)

        # Accumulate macros
        total_nodes += 1
        global_total_calls += node_calls
        for ag_name, ag_data in node_agents.items():
          if ag_name not in global_agents:
            global_agents[ag_name] = {
              "calls": 0,
              "prompt_tokens": 0,
              "completion_tokens": 0,
              "total_tokens": 0,
            }
          global_agents[ag_name]["calls"] += ag_data.get("calls", 0)
          global_agents[ag_name]["prompt_tokens"] += ag_data.get(
            "prompt_tokens", 0
          )
          global_agents[ag_name]["completion_tokens"] += ag_data.get(
            "completion_tokens", 0
          )
          global_agents[ag_name]["total_tokens"] += ag_data.get(
            "total_tokens", 0
          )

      except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")

  # Build Macro Summary
  if total_nodes > 0:
    macro_md = [
      "============================================================",
      "      MACRO TOKEN SUMMARY (ACROSS ALL DISCOVERED NODES)     ",
      "============================================================",
      f"Total Nodes/Attempts Analyzed : {total_nodes}",
      "",
    ]

    headers = [
      "Agent",
      "LLM Calls",
      "Prompt Tokens",
      "Completion Tokens",
      "Total Tokens",
    ]
    rows = []
    sorted_global = sorted(
      global_agents.items(),
      key=lambda x: x[1].get("total_tokens", 0),
      reverse=True,
    )

    grand_prompt = 0
    grand_comp = 0
    grand_total = 0

    for agent_name, agent_data in sorted_global:
      p = agent_data["prompt_tokens"]
      c = agent_data["completion_tokens"]
      t = agent_data["total_tokens"]
      calls = agent_data["calls"]
      grand_prompt += p
      grand_comp += c
      grand_total += t
      rows.append([agent_name, str(calls), f"{p:,}", f"{c:,}", f"{t:,}"])

    macro_md.append("### Global Agent Aggregate Tokens")
    macro_md.extend(generate_padded_table(headers, rows))
    macro_md.append("")
    macro_md.append(
      f"*(Grand Total: {global_total_calls} LLM calls | {grand_prompt:,} prompt | {grand_comp:,} completion | {grand_total:,} total)*"
    )
    macro_md.append(
      "============================================================"
    )

    all_markdown.append("\n".join(macro_md))

  return "\n\n---\n\n".join(all_markdown)


def generate_node_markdown(
  file_path: Path, metrics: Dict[str, Any]
) -> tuple[str, dict, int]:
  md = [f"# Token Metrics: {file_path.parent.name}"]

  iterations = metrics.get("iterations", {})
  if not iterations:
    md.append("No iterations tracked.")
    return "\n".join(md), {}, 0

  node_agents = {}
  node_total_calls = 0

  for iter_key, iter_data in sorted(
    iterations.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]
  ):
    md.append(f"## Iteration {iter_key}")

    agents = iter_data.get("agents", {})
    if agents:
      md.append("### Agent Aggregate Tokens")
      headers = [
        "Agent",
        "LLM Calls",
        "Prompt Tokens",
        "Completion Tokens",
        "Total Tokens",
      ]
      rows = []

      # Sort agents by total tokens descending
      sorted_agents = sorted(
        agents.items(), key=lambda x: x[1].get("total_tokens", 0), reverse=True
      )
      for agent_name, agent_data in sorted_agents:
        prompt = agent_data.get("prompt_tokens", 0)
        completion = agent_data.get("completion_tokens", 0)
        total = agent_data.get("total_tokens", 0)
        calls = agent_data.get("calls", 0)
        rows.append(
          [
            agent_name,
            str(calls),
            f"{prompt:,}",
            f"{completion:,}",
            f"{total:,}",
          ]
        )

        # Accumulate for node return
        if agent_name not in node_agents:
          node_agents[agent_name] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
          }
        node_agents[agent_name]["calls"] += calls
        node_agents[agent_name]["prompt_tokens"] += prompt
        node_agents[agent_name]["completion_tokens"] += completion
        node_agents[agent_name]["total_tokens"] += total

      md.extend(generate_padded_table(headers, rows))

    md.append("")
    llm_calls = iter_data.get("llm_calls", [])
    if llm_calls:
      calls_len = len(llm_calls)
      node_total_calls += calls_len
      md.append(f"*(Total {calls_len} LLM calls in this iteration)*")

    md.append("")

  return "\n".join(md), node_agents, node_total_calls


def main():
  parser = argparse.ArgumentParser(description="Analyze ADK token metrics")
  parser.add_argument(
    "target",
    help="Path to token_metrics.json OR a root directory",
  )
  parser.add_argument(
    "--write",
    action="store_true",
    help="Write token.md in the target directory",
  )
  args = parser.parse_args()

  md_content = analyze_path(args.target)

  if args.write:
    target_path = Path(args.target)
    out_dir = target_path.parent if target_path.is_file() else target_path
    out_file = out_dir / "token.md"
    with open(out_file, "w", encoding="utf-8") as f:
      f.write(md_content)
    print(f"Wrote {out_file}")
  else:
    print(md_content)


if __name__ == "__main__":
  main()
