"""Standalone XProf tools for analyzing xplane.pb files without external services."""

import gzip
import json
import sqlite3
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.tsl.profiler.protobuf import xplane_pb2


def _get_xplane_path(profiling_results: str) -> str:
  """Extracts xplane path from profiling results string or context."""

  return profiling_results.strip()


def load_xplane_and_query(xplane_path: str, sql_query: str) -> str:
  """Loads an xplane.pb file into an in-memory SQLite DB and runs a SQL query.

  The database schema is:
  - planes (id, name)
  - lines (id, plane_id, display_id, name, timestamp_ns)
  - events (plane_id, line_id, name, offset_ps, duration_ps, start_ps, end_ps)

  Args:
      xplane_path: Path to the .xplane.pb file.
      sql_query: The SQL query to execute against the loaded data.

  Returns:
      A markdown-formatted table of the query results.
  """
  try:
    # Open file (handle gz if needed)
    open_func = gzip.open if xplane_path.endswith(".gz") else open
    with open_func(xplane_path, "rb") as f:
      xspace = xplane_pb2.XSpace()
      xspace.ParseFromString(f.read())

    conn = sqlite3.connect(":memory:")
    c = conn.cursor()

    # Create Tables
    c.executescript("""
            CREATE TABLE planes (id INTEGER, name TEXT);
            CREATE TABLE lines (id INTEGER, plane_id INTEGER, display_id INTEGER, name TEXT, timestamp_ns INTEGER);
            CREATE TABLE events (
                plane_id INTEGER, line_id INTEGER,
                name TEXT, offset_ps INTEGER, duration_ps INTEGER,
                start_ps INTEGER, end_ps INTEGER
            );
        """)

    # Populate
    for plane in xspace.planes:
      c.execute("INSERT INTO planes VALUES (?, ?)", (plane.id, plane.name))

      # Metadata map lookup helper
      def get_meta_name(meta_map, mid):
        return meta_map[mid].name if mid in meta_map else str(mid)

      for line in plane.lines:
        c.execute(
          "INSERT INTO lines VALUES (?, ?, ?, ?, ?)",
          (line.id, plane.id, line.display_id, line.name, line.timestamp_ns),
        )

        for event in line.events:
          name = get_meta_name(plane.event_metadata, event.metadata_id)
          start_ps = event.offset_ps
          end_ps = start_ps + event.duration_ps
          c.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
              plane.id,
              line.id,
              name,
              event.offset_ps,
              event.duration_ps,
              start_ps,
              end_ps,
            ),
          )

    conn.commit()

    # Run Query
    df = pd.read_sql_query(sql_query, conn)
    conn.close()

    return df.to_markdown(index=False)

  except Exception as e:
    return f"Error executing query: {e}"


def get_hlo_dump(
  xplane_path: str, hlo_module_name: Optional[str] = None
) -> str:
  """Extracts HLO proto from xplane.pb if available.

  Args:
      xplane_path: Path to .xplane.pb file.
      hlo_module_name: Optional name filter.

  Returns:
      Status string indicating where HLO was saved or if not found.
  """
  try:
    open_func = gzip.open if xplane_path.endswith(".gz") else open
    with open_func(xplane_path, "rb") as f:
      xspace = xplane_pb2.XSpace()
      xspace.ParseFromString(f.read())

    saved_files = []
    for plane in xspace.planes:
      for stat in plane.stats:
        # Check known XStat metadata names for HLO
        # This is heuristic-based; often HLO protos are embedded as bytes
        # in stats with specific names like 'hlo_proto' or similar.
        # Since we don't have the exact meta name map handy, we might need to search metadata.
        # Simplification: In XPlane, HLOs are often in a dedicated plane or attached to 'device' plane stats.
        pass
        # For now, returning a placeholder as true extraction requires inspecting specific metadata IDs

    return (
      "HLO extraction not fully implemented in this standalone version yet"
      " (requires metadata ID mapping). Please use `load_xplane_and_query` to"
      " explore 'hlo' related events."
    )

  except Exception as e:
    return f"Error extracting HLO: {e}"


def create_chart_from_xplane(
  xplane_path: str,
  sql_query: str,
  chart_type: str = "bar",
  x_col: str = "name",
  y_col: str = "value",
  title: str = "",
) -> str:
  """Generates a chart from xplane data using SQL query.

  Args:
      xplane_path: Path to .xplane.pb
      sql_query: SQL query to get data.
      chart_type: 'bar' or 'pie'.
      x_col: Column for X axis (bar).
      y_col: Column for Y axis (bar) or values (pie).
      title: Chart title.
  """
  try:
    # Re-use loading logic (inefficient but stateless)
    # TODO: we might want to cache the DB connection or pass it around.
    # For simplicity here, we reload.
    open_func = gzip.open if xplane_path.endswith(".gz") else open
    with open_func(xplane_path, "rb") as f:
      xspace = xplane_pb2.XSpace()
      xspace.ParseFromString(f.read())

    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.executescript("""
            CREATE TABLE planes (id INTEGER, name TEXT);
            CREATE TABLE lines (id INTEGER, plane_id INTEGER, display_id INTEGER, name TEXT, timestamp_ns INTEGER);
            CREATE TABLE events (
                plane_id INTEGER, line_id INTEGER,
                name TEXT, offset_ps INTEGER, duration_ps INTEGER,
                start_ps INTEGER, end_ps INTEGER
            );
        """)
    for plane in xspace.planes:
      c.execute("INSERT INTO planes VALUES (?, ?)", (plane.id, plane.name))

      def get_meta_name(meta_map, mid):
        return meta_map[mid].name if mid in meta_map else str(mid)

      for line in plane.lines:
        c.execute(
          "INSERT INTO lines VALUES (?, ?, ?, ?, ?)",
          (line.id, plane.id, line.display_id, line.name, line.timestamp_ns),
        )
        for event in line.events:
          name = get_meta_name(plane.event_metadata, event.metadata_id)
          start_ps = event.offset_ps
          c.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
              plane.id,
              line.id,
              name,
              event.offset_ps,
              event.duration_ps,
              start_ps,
              start_ps + event.duration_ps,
            ),
          )
    conn.commit()

    df = pd.read_sql_query(sql_query, conn)
    conn.close()

    if df.empty:
      return "Query returned no data, cannot plot."

    plt.figure(figsize=(10, 6))
    if chart_type == "bar":
      plt.bar(df[x_col], df[y_col])
      plt.xlabel(x_col)
      plt.ylabel(y_col)
      plt.xticks(rotation=45)
    elif chart_type == "pie":
      plt.pie(df[y_col], labels=df[x_col], autopct="%1.1f%%")

    if title:
      plt.title(title)

    output_filename = f"{xplane_path}.png"
    plt.savefig(output_filename)
    plt.close()

    return f"Chart saved to {output_filename}"

  except Exception as e:
    return f"Error creating chart: {e}"


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
    # Sum of all event durations on device planes / (Total Duration * Device Count)
    # This is very rough; meaningful duty cycle usually requires specific "idle" vs "active" ops.
    # But as a placeholder, we can sum specific "compute" ops if we knew them.
    # For now, let's just sum all event durations on device planes.

    if device_planes and found_events and total_duration_ps > 0:
      total_device_busy_ps = 0
      for plane in device_planes:
        for line in plane.lines:
          for event in line.events:
            total_device_busy_ps += event.duration_ps

      # Divide by device count, or just raw sum?
      # Usually users care about "Average Device Duty Cycle".
      # Assume full parallelism potential = device_count * total_duration
      potential_ps = len(device_planes) * total_duration_ps
      if potential_ps > 0:
        metrics["device_duty_cycle_percent"] = (
          total_device_busy_ps / potential_ps
        ) * 100
      else:
        metrics["device_duty_cycle_percent"] = 0

    # 4. Step Time (if steps are annotated)
    # Often steps are in a specific plane or line.
    # We can look for "Step" events?
    step_count = 0
    step_durations = []

    for plane in host_planes + device_planes:
      for line in plane.lines:
        # Heuristic: line name often is "Steps"
        if "steps" in line.name.lower():
          for event in line.events:
            step_count += 1
            step_durations.append(event.duration_ps)

    if step_count > 0:
      avg_step_ps = sum(step_durations) / step_count
      metrics["average_step_time_ms"] = avg_step_ps / 1e9
      metrics["step_count"] = step_count

    # Add some metadata placeholders if keys are expected
    metrics["build_target"] = "N/A (Offline)"
    metrics["xid"] = "N/A (Offline)"

    return json.dumps(metrics, indent=2)

  except Exception as e:
    return f"Error generating overview metrics: {e}"
