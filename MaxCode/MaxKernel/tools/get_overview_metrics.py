#!/usr/bin/env python3
"""Extracts high-level overview metrics directly from an xplane.pb file."""

import argparse
import gzip
import json
import sys

from tensorflow.tsl.profiler.protobuf import xplane_pb2


def get_overview_page_metrics(xplane_path: str) -> str:
  """Returns metrics and metadata from overview page for a given Xprof session.

  Mimics the behavior of overview_page_tool.get_overview_page_metrics by
  extracting high-level metrics from the xplane.pb file directly.

  Args:
      xplane_path: Path to the .xplane.pb file.

  Returns:
      A JSON string containing metrics and metadata.
  """
  try:
    open_func = gzip.open if xplane_path.endswith(".gz") else open
    with open_func(xplane_path, "rb") as f:
      xspace = xplane_pb2.XSpace()
      xspace.ParseFromString(f.read())

    metrics = {}

    # 1. Host/Device Identification
    host_planes = []
    device_planes = []

    for plane in xspace.planes:
      if (
          "device" in plane.name.lower()
          or "tpu" in plane.name.lower()
          or "gpu" in plane.name.lower()
      ):
        device_planes.append(plane)
      else:
        host_planes.append(plane)

    metrics["device_count"] = len(device_planes)
    metrics["host_count"] = len(host_planes)

    # 2. Total Duration (from all planes)
    min_start_ps = float("inf")
    max_end_ps = 0

    all_planes = host_planes + device_planes
    found_events = False
    total_duration_ps = 0

    for plane in all_planes:
      for line in plane.lines:
        for event in line.events:
          found_events = True
          start = event.offset_ps
          end = start + event.duration_ps
          if start < min_start_ps:
            min_start_ps = start
          if end > max_end_ps:
            max_end_ps = end

    if found_events:
      total_duration_ps = max_end_ps - min_start_ps
      metrics["total_duration_ms"] = total_duration_ps / 1e9
      metrics["total_duration_ns"] = total_duration_ps / 1000
    else:
      metrics["total_duration_ms"] = 0

    # 3. Device Duty Cycle (Approximate)
    if device_planes and found_events and total_duration_ps > 0:
      total_device_busy_ps = 0
      for plane in device_planes:
        for line in plane.lines:
          for event in line.events:
            total_device_busy_ps += event.duration_ps

      potential_ps = len(device_planes) * total_duration_ps
      if potential_ps > 0:
        metrics["device_duty_cycle_percent"] = (
            total_device_busy_ps / potential_ps
        ) * 100
      else:
        metrics["device_duty_cycle_percent"] = 0

    # 4. Step Time (if steps are annotated)
    step_count = 0
    step_durations = []

    for plane in host_planes + device_planes:
      for line in plane.lines:
        if "steps" in line.name.lower():
          for event in line.events:
            step_count += 1
            step_durations.append(event.duration_ps)

    if step_count > 0:
      avg_step_ps = sum(step_durations) / step_count
      metrics["average_step_time_ms"] = avg_step_ps / 1e9
      metrics["step_count"] = step_count

    metrics["build_target"] = "N/A (Offline)"
    metrics["xid"] = "N/A (Offline)"

    return json.dumps(metrics, indent=2)

  except Exception as e:  # pylint: disable=broad-exception-caught
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
