import glob
import logging
import os


def extract_xprof_time(
  trace_dir: str, event_name: str, num_runs: int = 20
) -> float:
  """Extracts execution time for a named event from xprof trace.

  Args:
    trace_dir: Directory containing the trace files.
    event_name: Name of the event to search for (e.g., 'bench_kernel').
    num_runs: Number of benchmark runs to average over.

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
          for line in plane.lines:
            if line.name not in ("XLA Modules", "XLA Ops"):
              continue
            for event in line.events:
              name = ""
              if event.metadata_id in plane.event_metadata:
                name = plane.event_metadata[event.metadata_id].name

              if event_name in name:
                if line.name == "XLA Modules":
                  xla_module_events.append(
                    {
                      "file": file_path,
                      "plane": plane.name,
                      "name": name,
                      "duration_ps": event.duration_ps,
                    }
                  )
                elif name.startswith("%benchmark_func"):
                  xla_op_events.append(
                    {
                      "file": file_path,
                      "plane": plane.name,
                      "name": name,
                      "duration_ps": event.duration_ps,
                    }
                  )
    except Exception as e:
      logging.warning(f"Failed to parse {file_path}: {e}")

  if xla_op_events:
    logging.info("Found kernel events starting with %. Using them.")
    target_events = xla_op_events
  else:
    logging.info(
      "No xla_op_events events found. Falling back to xla_module_events matched events."
    )
    target_events = xla_module_events

  total_duration_ps = sum(ev["duration_ps"] for ev in target_events)
  count = len(target_events)

  print(f"\n--- Used Events for {event_name} in {trace_dir} ---")
  for ev in target_events:
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
