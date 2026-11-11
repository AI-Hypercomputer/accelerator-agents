#!/usr/bin/env python3
"""Export run metrics from a MaxKernel run directory and Claude stream.

Reads <run_dir>/state.json and optional claude_stream.json to emit a
clean, structured JSON summary of kernel optimization performance and LLM token usage.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Any, Optional


@dataclasses.dataclass
class RunMetricsSummary:
  """Structured metrics summary for a MaxKernel optimization run."""

  run_id: Optional[str] = None
  iteration_count: int = 0
  best_speedup: Optional[float] = None
  base_time_ms: Optional[float] = None
  best_optimized_time: Optional[float] = None
  best_code_path: Optional[str] = None
  stopping_recommendations: Optional[Any] = None
  input_tokens: Optional[int] = None
  output_tokens: Optional[int] = None
  cache_read_tokens: Optional[int] = None
  total_tokens: Optional[int] = None

  @property
  def stopping_recommendation(self) -> Optional[Any]:
    return self.stopping_recommendations

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def parse_state_file(state_path: str) -> dict[str, Any]:
  """Parses state.json; raises FileNotFoundError or ValueError fail-closed."""
  if not os.path.exists(state_path):
    raise FileNotFoundError(f"State file not found: {state_path}")
  try:
    with open(state_path, "r", encoding="utf-8") as f:
      content = f.read().strip()
      if not content:
        raise ValueError(f"State file is empty: {state_path}")
      data = json.loads(content)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Invalid JSON in state file {state_path}: {exc}") from exc

  if not isinstance(data, dict):
    raise ValueError(
      f"State file {state_path} must be a JSON object, got {type(data).__name__}"
    )
  return data


def parse_stream_tokens(stream_path: str) -> dict[str, Optional[int]]:
  """Parses claude_stream.json (JSONL or JSON) for token usage."""
  if not os.path.exists(stream_path):
    return {
      "input_tokens": None,
      "output_tokens": None,
      "cache_read_tokens": None,
      "total_tokens": None,
    }

  input_tokens = 0
  output_tokens = 0
  cache_read_tokens = 0
  found_tokens = False
  result_usage: Optional[dict[str, Any]] = None

  try:
    with open(stream_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          event = json.loads(line)
        except json.JSONDecodeError:
          continue

        if not isinstance(event, dict):
          continue

        # Check for final result event with total usage
        if event.get("type") == "result" and isinstance(event.get("usage"), dict):
          result_usage = event["usage"]
          continue

        # Parse message_start usage
        msg = event.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("usage"), dict):
          u = msg["usage"]
          if "input_tokens" in u and u.get("input_tokens") is not None:
            input_tokens += int(u["input_tokens"])
            found_tokens = True
          if "cache_read_input_tokens" in u or "cache_read_tokens" in u:
            val = u.get("cache_read_input_tokens") or u.get("cache_read_tokens")
            if val is not None:
              cache_read_tokens += int(val)
              found_tokens = True
          if "output_tokens" in u and u.get("output_tokens") is not None:
            output_tokens += int(u["output_tokens"])
            found_tokens = True
        # Parse message_delta usage or direct usage
        elif isinstance(event.get("usage"), dict):
          u = event["usage"]
          if "input_tokens" in u and u.get("input_tokens") is not None:
            input_tokens += int(u["input_tokens"])
            found_tokens = True
          if "cache_read_input_tokens" in u or "cache_read_tokens" in u:
            val = u.get("cache_read_input_tokens") or u.get("cache_read_tokens")
            if val is not None:
              cache_read_tokens += int(val)
              found_tokens = True
          if "output_tokens" in u and u.get("output_tokens") is not None:
            output_tokens += int(u["output_tokens"])
            found_tokens = True


  except Exception:  # pylint: disable=broad-exception-caught
    return {
      "input_tokens": None,
      "output_tokens": None,
      "cache_read_tokens": None,
      "total_tokens": None,
    }

  # If a final result event with total usage was observed, it takes precedence
  if result_usage is not None:
    in_tok = result_usage.get("input_tokens")
    out_tok = result_usage.get("output_tokens")
    cache_tok = result_usage.get("cache_read_input_tokens") or result_usage.get("cache_read_tokens")
    tot_tok = result_usage.get("total_tokens")
    if tot_tok is None and (in_tok is not None or out_tok is not None):
      tot_tok = (in_tok or 0) + (out_tok or 0) + (cache_tok or 0)
    return {
      "input_tokens": int(in_tok) if in_tok is not None else None,
      "output_tokens": int(out_tok) if out_tok is not None else None,
      "cache_read_tokens": int(cache_tok) if cache_tok is not None else None,
      "total_tokens": int(tot_tok) if tot_tok is not None else None,
    }

  if not found_tokens:
    return {
      "input_tokens": None,
      "output_tokens": None,
      "cache_read_tokens": None,
      "total_tokens": None,
    }

  total = input_tokens + output_tokens + cache_read_tokens
  return {
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "cache_read_tokens": cache_read_tokens,
    "total_tokens": total,
  }


def extract_run_metrics(
  run_dir: str, stream_file: Optional[str] = None
) -> RunMetricsSummary:
  """Extracts metrics from state.json and optional stream file."""
  state_path = os.path.join(run_dir, "state.json")
  state = parse_state_file(state_path)

  # Run ID
  run_id = state.get("run_id") or os.path.basename(os.path.abspath(run_dir))

  # Iteration count
  raw_iter = state.get("iteration")
  if raw_iter is None:
    raw_history = state.get("history")
    iteration_count = len(raw_history) if isinstance(raw_history, list) else 0
  else:
    try:
      iteration_count = int(raw_iter)
    except (ValueError, TypeError):
      iteration_count = 0

  # Best speedup
  raw_speedup = state.get("best_speedup")
  best_speedup: Optional[float] = None
  if raw_speedup is not None:
    try:
      best_speedup = float(raw_speedup)
    except (ValueError, TypeError):
      best_speedup = None

  # Base time
  raw_base = state.get("base_time_ms")
  base_time_ms: Optional[float] = None
  if raw_base is not None:
    try:
      base_time_ms = float(raw_base)
    except (ValueError, TypeError):
      base_time_ms = None

  # Best optimized time
  raw_opt = state.get("best_optimized_time")
  best_optimized_time: Optional[float] = None
  if raw_opt is not None:
    try:
      best_optimized_time = float(raw_opt)
    except (ValueError, TypeError):
      best_optimized_time = None

  # Best code path
  best_code_path = state.get("best_code_path")
  if best_code_path is not None:
    best_code_path = str(best_code_path)

  # Stopping recommendations
  stopping = (
    state.get("stopping_recommendations")
    if "stopping_recommendations" in state
    else state.get("stopping")
    if "stopping" in state
    else state.get("stopping_recommendation")
  )

  # Stream tokens
  if not stream_file:
    candidate_stream = os.path.join(run_dir, "claude_stream.json")
    if os.path.exists(candidate_stream):
      stream_file = candidate_stream

  token_metrics = parse_stream_tokens(stream_file) if stream_file else {
    "input_tokens": None,
    "output_tokens": None,
    "cache_read_tokens": None,
    "total_tokens": None,
  }

  return RunMetricsSummary(
    run_id=run_id,
    iteration_count=iteration_count,
    best_speedup=best_speedup,
    base_time_ms=base_time_ms,
    best_optimized_time=best_optimized_time,
    best_code_path=best_code_path,
    stopping_recommendations=stopping,
    input_tokens=token_metrics["input_tokens"],
    output_tokens=token_metrics["output_tokens"],
    cache_read_tokens=token_metrics["cache_read_tokens"],
    total_tokens=token_metrics["total_tokens"],
  )


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export metrics summary from a MaxKernel run directory."
  )
  parser.add_argument(
    "--run_dir",
    "--run-dir",
    dest="run_dir",
    required=True,
    help="Path to run directory containing state.json",
  )
  parser.add_argument(
    "--output",
    "-o",
    dest="output",
    default=None,
    help="Path to write summary_metrics.json (defaults to stdout)",
  )
  parser.add_argument(
    "--stream_file",
    "--stream-file",
    dest="stream_file",
    default=None,
    help="Path to claude_stream.json file (defaults to <run_dir>/claude_stream.json)",
  )
  args = parser.parse_args()

  try:
    summary = extract_run_metrics(args.run_dir, stream_file=args.stream_file)
  except Exception as exc:
    sys.stderr.write(f"Error exporting metrics: {exc}\n")
    return 1

  output_json = json.dumps(summary.to_dict(), indent=2) + "\n"
  if args.output:
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
      f.write(output_json)
  else:
    sys.stdout.write(output_json)

  return 0


if __name__ == "__main__":
  sys.exit(main())
