#!/usr/bin/env python3
"""Extracts high-level overview metrics from an xplane.pb file.

This delegates the real metrics to xprof's own `overview_page` converter
(the same computation that backs the XProf UI's Overview page), rather than
re-deriving them from raw proto fields. That matters because the hand-rolled
version could only ever produce a crude "sum of event durations / wall time"
duty-cycle estimate, and had no access at all to the metrics the profiling
guide actually asks for -- MXU utilization, memory-bandwidth utilization and
step time. Those come from xprof's op-level analysis, not from event spans.

Plane counts and wall-clock duration are still derived directly from the
trace via `xplane_loader`, since `overview_page` does not report them.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import xplane_loader  # pylint: disable=g-import-not-at-top
from xprof.convert import (
  raw_to_tool_data,  # pylint: disable=g-import-not-at-top
)

# Substrings identifying a device (accelerator) plane. Host planes are
# everything else. Matches the naming XPlane uses: "/device:TPU:0" etc.
_DEVICE_PLANE_MARKERS = ("device", "tpu", "gpu")


def _overview_page_metrics(xplane_path: str) -> dict:
  """Returns xprof's own overview-page metrics, or {} if unavailable."""
  data, _ = raw_to_tool_data.xspace_to_tool_data(
    [xplane_path], "overview_page", {}
  )
  if isinstance(data, bytes):
    data = data.decode("utf-8")
  parsed = json.loads(data)

  # overview_page returns a list of sections; the performance-summary and
  # run-environment values live in each section's "p" (properties) dict.
  merged = {}
  if isinstance(parsed, list):
    for section in parsed:
      if isinstance(section, dict) and isinstance(section.get("p"), dict):
        merged.update(section["p"])
  elif isinstance(parsed, dict) and isinstance(parsed.get("p"), dict):
    merged.update(parsed["p"])
  return merged


def get_overview_page_metrics(xplane_path: str) -> str:
  """Returns metrics and metadata from the overview page for an Xprof session.

  Args:
      xplane_path: Path to the .xplane.pb file (or a directory containing one).

  Returns:
      A JSON string containing metrics and metadata.
  """
  try:
    resolved = xplane_loader.resolve_xplane_path(xplane_path)

    metrics = {}

    device_planes = 0
    host_planes = 0
    max_end_ns = 0
    min_start_ns = float("inf")
    found_events = False

    for plane in xplane_loader.iter_planes(resolved):
      lowered = plane.name.lower()
      if any(m in lowered for m in _DEVICE_PLANE_MARKERS):
        device_planes += 1
      else:
        host_planes += 1
      for line in plane.lines:
        for event in line.events:
          found_events = True
          start = event.start_ns
          end = start + event.duration_ns
          min_start_ns = min(min_start_ns, start)
          max_end_ns = max(max_end_ns, end)

    # xprof's real analysis first: MXU %, memory BW %, duty cycle, step time.
    overview = _overview_page_metrics(resolved)
    if overview:
      metrics.update(overview)
    else:
      metrics["overview_page"] = "unavailable for this trace"

    # Trace-derived values last, under names that cannot collide with
    # overview_page's own keys. `overview_page` already publishes a
    # `host_count` (number of profiled hosts) and `device_core_count`, which
    # mean something different from the number of XPlanes -- merging these in
    # under the same names would silently overwrite one with the other.
    metrics["device_plane_count"] = device_planes
    metrics["host_plane_count"] = host_planes

    if found_events:
      total_ns = max_end_ns - min_start_ns
      metrics["trace_duration_ms"] = total_ns / 1e6
      metrics["trace_duration_ns"] = total_ns
    else:
      metrics["trace_duration_ms"] = 0

    metrics["xplane_path"] = resolved

    return json.dumps(metrics, indent=2)

  except Exception as e:  # pylint: disable=broad-except
    return f"Error generating overview metrics: {e}"


def main():
  parser = argparse.ArgumentParser(
    description="Extract overview metrics from an XProf xplane.pb file."
  )
  parser.add_argument("xplane_path", help="Path to the .xplane.pb file.")

  argv = sys.argv[1:]
  if argv and argv[0] == "--":
    argv = argv[1:]
  args = parser.parse_args(argv)

  result = get_overview_page_metrics(args.xplane_path)
  print(result)


if __name__ == "__main__":
  main()
