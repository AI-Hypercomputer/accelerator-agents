#!/usr/bin/env python3
"""Analyzes .xplane.pb trace for DMA/SyncWait ratio vs compute ratio."""

import argparse
import json
import sys

from xprof.convert import raw_to_tool_data


def analyze_trace(path: str):
  """Computes DMA/sync-wait ratio vs compute ratio from xplane.pb trace."""
  try:
    tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data(
        [path], "trace_viewer", {}
    )
    trace_data = json.loads(tool_data_result)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error loading trace from {path}: {e}", file=sys.stderr)
    return None

  events = trace_data.get("traceEvents", [])
  pid = None
  events_for_tpu_0 = []
  jit_computation_events = []

  for event in events:
    if "args" in event and event["args"].get("name", None) == "/device:TPU:0":
      pid = event.get("pid", -1)
    if pid is not None and event.get("pid", -1) == pid:
      events_for_tpu_0.append(event)
      if "jit_computation" in event.get("name", ""):
        jit_computation_events.append(event)

  if len(jit_computation_events) < 2:
    print(
        f"Error: Found {len(jit_computation_events)} 'jit_computation'"
        " events on TPU:0 (requires at least 2). Trace may be a single-step or"
        " short trace.",
        file=sys.stderr,
    )
    return None

  start_last = (
      jit_computation_events[-2]["ts"] + jit_computation_events[-2]["dur"]
  )
  end_last = (
      jit_computation_events[-1]["ts"] + jit_computation_events[-1]["dur"]
  )

  sync_wait_total = 0
  for event in events_for_tpu_0:
    if "dur" in event:
      if event["ts"] >= start_last and (event["ts"] + event["dur"]) <= end_last:
        if "SyncWait" in event.get("name", ""):
          sync_wait_total += event["dur"]

  total_computation_time = end_last - start_last
  if total_computation_time > 0:
    ratio = sync_wait_total / total_computation_time
    print(
        f"We see that kernel spends {ratio * 100:.4f}% waiting for"
        f" synchronization and {(1 - ratio) * 100:.4f}% computing."
    )
    print(f"DMA_AND_MEMORY_TRANSFERS_RATIO: {ratio:.6f}")
    print(f"COMPUTE_RATIO: {1 - ratio:.6f}")
    return ratio
  else:
    print("Error: Total computation time is non-positive.", file=sys.stderr)
    return None


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Analyze a .xplane.pb trace and report the fraction of the last"
          " computation's time spent in SyncWait vs. compute."
      )
  )
  parser.add_argument(
      "xplane_path",
      help="Path to the .xplane.pb file to analyze.",
  )

  argv = sys.argv[1:]
  if argv and argv[0] == "--":
    argv = argv[1:]
  args = parser.parse_args(argv)

  res = analyze_trace(args.xplane_path)
  if res is None:
    sys.exit(1)


if __name__ == "__main__":
  main()
