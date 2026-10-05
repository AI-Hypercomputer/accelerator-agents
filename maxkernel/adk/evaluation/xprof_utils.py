import glob
import logging
import os
import re


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
  """Merges overlapping (start_ps, end_ps) intervals."""
  if not intervals:
    return []
  intervals.sort()
  merged = []
  curr_start, curr_end = intervals[0]
  for next_start, next_end in intervals[1:]:
    if next_start < curr_end:
      curr_end = max(curr_end, next_end)
    else:
      merged.append((curr_start, curr_end))
      curr_start, curr_end = next_start, next_end
  merged.append((curr_start, curr_end))
  return merged


def extract_xprof_time(
  trace_dir: str,
  event_name: str,
  num_runs: int = 20,
  line_name: str = None,
  ignore_event_regex: str = r"(?i)(^%?copy\b|barrier)",
) -> float:
  """Extracts execution time for a named event from xprof trace.

  Args:
    trace_dir: Directory containing the trace files.
    event_name: Name of the event to search for (e.g., 'bench_kernel').
    num_runs: Number of benchmark runs to average over.
    line_name: Optional XLine name ('XLA Modules' or 'XLA Ops').
    ignore_event_regex: Regex for XLA Ops events (e.g., defensive HBM copies or
      barriers) whose overlapping duration should be subtracted from XLA Modules
      events.

  Returns:
    Average execution time per run in milliseconds, or 0.0 if failed.
  """
  logging.info(
    f"Attempting to extract xprof time for {event_name} from {trace_dir}"
  )

  try:
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
  except ImportError:
    logging.warning(
      "TensorFlow not available. Cannot parse xprof trace programmatically."
    )
    return 0.0

  # Find .xplane.pb files
  xplane_files = glob.glob(
    os.path.join(trace_dir, "**/*.xplane.pb"), recursive=True
  )
  if not xplane_files:
    logging.warning(f"No .xplane.pb files found in {trace_dir}")
    return 0.0

  ignore_pattern = (
    re.compile(ignore_event_regex) if ignore_event_regex else None
  )

  total_duration_ps = 0
  count = 0

  xla_module_events = []
  xla_op_events = []
  for file_path in xplane_files:
    try:
      with open(file_path, "rb") as f:
        xspace = xplane_pb2.XSpace()
        xspace.ParseFromString(f.read())

        for plane in xspace.planes:
          if "/device:TPU:0" not in plane.name:
            continue
          plane_module_events = []
          plane_ignore_events = []
          for line in plane.lines:
            if line.name not in ("XLA Modules", "XLA Ops"):
              continue
            line_base_ps = line.timestamp_ns * 1000
            for event in line.events:
              name = ""
              if event.metadata_id in plane.event_metadata:
                name = plane.event_metadata[event.metadata_id].name

              start_ps = line_base_ps + event.offset_ps
              end_ps = start_ps + event.duration_ps

              if line.name == "XLA Modules":
                if event_name in name:
                  plane_module_events.append(
                    {
                      "file": file_path,
                      "plane": plane.name,
                      "name": name,
                      "start_ps": start_ps,
                      "end_ps": end_ps,
                      "raw_duration_ps": event.duration_ps,
                      "ignored_duration_ps": 0,
                      "duration_ps": event.duration_ps,
                    }
                  )
              elif line.name == "XLA Ops":
                if ignore_pattern and name and ignore_pattern.search(name):
                  plane_ignore_events.append((start_ps, end_ps))
                if event_name in name and name.startswith("%benchmark_func"):
                  xla_op_events.append(
                    {
                      "file": file_path,
                      "plane": plane.name,
                      "name": name,
                      "duration_ps": event.duration_ps,
                    }
                  )

          if plane_ignore_events and plane_module_events:
            plane_ignore_events.sort()
            for mod_ev in plane_module_events:
              m_start = mod_ev["start_ps"]
              m_end = mod_ev["end_ps"]
              overlapping = []
              for i_start, i_end in plane_ignore_events:
                if i_end <= m_start:
                  continue
                if i_start >= m_end:
                  break
                c_start = max(m_start, i_start)
                c_end = min(m_end, i_end)
                if c_start < c_end:
                  overlapping.append((c_start, c_end))
              if overlapping:
                ignored_ps = sum(
                  e - s for s, e in _merge_intervals(overlapping)
                )
                mod_ev["ignored_duration_ps"] = ignored_ps
                mod_ev["duration_ps"] = max(
                  0, mod_ev["raw_duration_ps"] - ignored_ps
                )

          xla_module_events.extend(plane_module_events)
    except Exception as e:
      logging.warning(f"Failed to parse {file_path}: {e}")

  if line_name == "XLA Ops":
    if xla_op_events:
      logging.info(f"Using XLA Ops events. Found {len(xla_op_events)} events.")
      target_events = xla_op_events
    else:
      logging.info(
        "XLA Ops events requested but not found. Falling back to XLA Modules events."
      )
      target_events = xla_module_events
  else:
    # Default behavior or explicit "XLA Modules"
    logging.info(
      f"Using XLA Modules events. Found {len(xla_module_events)} events."
    )
    target_events = xla_module_events

  total_duration_ps = sum(ev["duration_ps"] for ev in target_events)
  count = len(target_events)

  print(f"\n--- Used Events for {event_name} in {trace_dir} ---")
  for ev in target_events:
    ignored_ps = ev.get("ignored_duration_ps", 0)
    if ignored_ps > 0:
      print(
        f"File: {ev['file']}, Plane: {ev['plane']}, Name: {ev['name']}, "
        f"Duration: {ev['duration_ps'] / 1e9} ms "
        f"(raw: {ev['raw_duration_ps'] / 1e9} ms, ignored copy/barrier: {ignored_ps / 1e9} ms)"
      )
    else:
      print(
        f"File: {ev['file']}, Plane: {ev['plane']}, Name: {ev['name']}, Duration: {ev['duration_ps'] / 1e9} ms"
      )
  print(f"Total used events: {count}")
  print("---------------------------------------------------\n")

  if count == 0:
    logging.warning(f"No events matching {event_name} found in trace.")
    return 0.0

  # Convert picoseconds to milliseconds and divide by num_runs
  avg_duration_ms = (total_duration_ps / num_runs) / 1e9
  logging.info(
    f"Extracted xprof time: {avg_duration_ms} ms (based on {count} events, averaged over {num_runs} runs)"
  )
  return avg_duration_ms
