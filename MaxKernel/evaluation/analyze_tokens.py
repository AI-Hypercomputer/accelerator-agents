"""
Token Analysis Tool
Analyzes an ADK session JSON file to extract token usage, identify compactions,
generate a markdown report, and plot a visualization of token usage over time.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def load_session(file_path: str) -> Dict[str, Any]:
  """Loads a session JSON file."""
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

  with open(file_path, "r", encoding="utf-8") as f:
    return json.load(f)


def analyze_token(file_path: str) -> None:
  """Analyzes token usage from a session file and generates reports."""
  print(f"Analyzing {file_path}...\n")

  try:
    session = load_session(file_path)
  except Exception as e:
    print(f"Error loading JSON: {e}")
    return

  events = session.get("events", [])
  if not events:
    print("No events found in the session file.")
    return

  # Configure output paths
  base_name = os.path.splitext(os.path.basename(file_path))[0]
  output_dir = os.path.dirname(os.path.abspath(file_path))
  md_output_file = os.path.join(output_dir, f"{base_name}_token_analysis.md")
  img_output_file = os.path.join(
    output_dir, f"{base_name}_token_visualization.png"
  )

  # Analyze events and generate report
  plot_data, compaction_indices = generate_token_report(
    events, file_path, md_output_file, img_output_file
  )

  # Generate visualization
  generate_visualization(
    plot_data, compaction_indices, file_path, img_output_file
  )

  print(f"\n✅ Visualization successfully saved to: {img_output_file}")
  print(f"✅ Markdown log successfully saved to: {md_output_file}")


def generate_token_report(
  events: List[Dict[str, Any]],
  file_path: str,
  md_output_file: str,
  img_output_file: str,
) -> Tuple[Tuple[List[int], List[int]], List[int]]:
  """Processes events, calculates token statistics, and writes the markdown report."""
  total_tokens_sum = 0
  prompt_tokens_sum = 0
  candidates_tokens_sum = 0
  thoughts_tokens_sum = 0

  plot_indices = []
  plot_tokens = []
  compaction_indices = []

  with open(md_output_file, "w", encoding="utf-8") as md_f:
    # Write markdown header
    md_f.write(f"# Token Analysis for `{os.path.basename(file_path)}`\n\n")
    md_f.write(
      f"![Token Visualization]({os.path.basename(img_output_file)})\n\n"
    )
    md_f.write("| Index | Author | Tokens | Notes |\n")
    md_f.write("|-------|--------|--------------|-------|\n")

    print(f"{'Index':<6} | {'Author':<32} | {'Tokens':<12} | {'Notes'}")
    print("-" * 75)

    for i, event in enumerate(events):
      author = event.get("author", "unknown")
      usage = event.get("usageMetadata", {})

      prompt_tokens = usage.get("promptTokenCount", 0)
      candidates_tokens = usage.get("candidatesTokenCount", 0)
      thoughts_tokens = usage.get("thoughtsTokenCount", 0)
      total_tokens = usage.get("totalTokenCount", 0)

      total_tokens_sum += total_tokens
      prompt_tokens_sum += prompt_tokens
      candidates_tokens_sum += candidates_tokens
      thoughts_tokens_sum += thoughts_tokens

      token_str = f"{total_tokens:,}" if total_tokens else "N/A"

      actions = event.get("actions", {})
      is_compaction = "compaction" in actions

      notes = ""
      if is_compaction:
        notes = "COMPACTION HAPPENED"
        compaction_indices.append(i)
        print("-" * 75)

      # Log to console and file
      print(f"{i:<6} | {author:<32} | {token_str:<12} | {notes}")
      md_f.write(f"| {i} | {author} | {token_str} | {notes} |\n")

      if is_compaction:
        print("-" * 75)

      if total_tokens > 0:
        plot_indices.append(i)
        plot_tokens.append(total_tokens)

    # Write summary to console and file
    write_summary(
      total_tokens_sum,
      prompt_tokens_sum,
      candidates_tokens_sum,
      thoughts_tokens_sum,
      md_f,
    )

  return (plot_indices, plot_tokens), compaction_indices


def write_summary(
  total: int, prompt: int, candidates: int, thoughts: int, md_f
) -> None:
  """Prints the token summary to console and writes it to the markdown file."""
  summary_text = (
    f"\n{'=' * 40}\n"
    f"Token Count: {total:,}\n"
    f"Prompt Token Count: {prompt:,}\n"
    f"Candidates Token Count: {candidates:,}\n"
    f"Thoughts Token Count: {thoughts:,}\n"
    f"{'=' * 40}\n"
  )
  print(summary_text)

  md_f.write("\n### Summary\n")
  md_f.write("```text\n")
  md_f.write(f"Token Count: {total:,}\n")
  md_f.write(f"Prompt Token Count: {prompt:,}\n")
  md_f.write(f"Candidates Token Count: {candidates:,}\n")
  md_f.write(f"Thoughts Token Count: {thoughts:,}\n")
  md_f.write("```\n")


def generate_visualization(
  plot_data: Tuple[List[int], List[int]],
  compaction_indices: List[int],
  file_path: str,
  img_output_file: str,
) -> None:
  """Generates and saves a line plot of token usage over time."""
  plot_indices, plot_tokens = plot_data

  plt.figure(figsize=(14, 7))
  plt.plot(
    plot_indices,
    plot_tokens,
    marker="o",
    linestyle="-",
    color="b",
    markersize=3,
    alpha=0.8,
    label="Tokens",
  )

  for idx, c_idx in enumerate(compaction_indices):
    label = "Compaction Triggered" if idx == 0 else ""
    plt.axvline(x=c_idx, color="r", linestyle="--", alpha=0.6, label=label)

  plt.title(f"Token Count Over Time ({os.path.basename(file_path)})")
  plt.xlabel("Event Index")
  plt.ylabel("Token Count at Each Event")
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend()

  plt.tight_layout()
  plt.savefig(img_output_file, dpi=150)
  plt.close()


def main():
  parser = argparse.ArgumentParser(
    description="Analyze token usage from an ADK session JSON file."
  )
  parser.add_argument("file_path", help="Path to the session JSON file.")
  args = parser.parse_args()

  analyze_token(args.file_path)


if __name__ == "__main__":
  main()
