#!/usr/bin/env python3
"""Generates charts from xplane.pb profiling data using SQL queries."""

import argparse
import gzip
import sqlite3
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.tsl.profiler.protobuf import xplane_pb2


def create_chart_from_xplane(
    xplane_path: str,
    sql_query: str,
    chart_type: str = "bar",
    x_col: str = "name",
    y_col: str = "value",
    title: str = "",
    output_path: Optional[str] = None,
) -> str:
  """Generates a chart from xplane data using SQL query.

  Args:
      xplane_path: Path to .xplane.pb
      sql_query: SQL query to get data.
      chart_type: 'bar' or 'pie'.
      x_col: Column for X axis (bar).
      y_col: Column for Y axis (bar) or values (pie).
      title: Chart title.
      output_path: Optional output file path for the generated chart PNG.

  Returns:
      Status string indicating where the chart was saved.
  """
  try:
    # Re-use loading logic (inefficient but stateless)
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
      plt.xticks(rotation=45, ha="right")
    elif chart_type == "pie":
      plt.pie(df[y_col], labels=df[x_col], autopct="%1.1f%%")

    if title:
      plt.title(title)

    plt.tight_layout()
    output_filename = output_path or f"{xplane_path}.png"
    plt.savefig(output_filename)
    plt.close()

    return f"Chart saved to {output_filename}"

  except Exception as e:  # pylint: disable=broad-exception-caught
    return f"Error creating chart: {e}"


def main():
  parser = argparse.ArgumentParser(
      description="Generate charts from an XProf xplane.pb file using SQL."
  )
  parser.add_argument("xplane_path", help="Path to the .xplane.pb file.")
  parser.add_argument("sql_query", help="SQL query to retrieve chart data.")
  parser.add_argument(
      "--chart-type",
      "--chart_type",
      dest="chart_type",
      default="bar",
      choices=["bar", "pie"],
      help="Chart type (bar or pie).",
  )
  parser.add_argument(
      "--x-col",
      "--x_col",
      dest="x_col",
      default="name",
      help="Column for X axis.",
  )
  parser.add_argument(
      "--y-col",
      "--y_col",
      dest="y_col",
      default="value",
      help="Column for Y axis.",
  )
  parser.add_argument("--title", default="", help="Chart title.")
  parser.add_argument(
      "--output-path",
      "--output_path",
      "-o",
      dest="output_path",
      default=None,
      help="Output PNG path (defaults to <xplane_path>.png).",
  )

  argv = sys.argv[1:]
  if argv and argv[0] == "--":
    argv = argv[1:]
  args = parser.parse_args(argv)

  result = create_chart_from_xplane(
      args.xplane_path,
      args.sql_query,
      chart_type=args.chart_type,
      x_col=args.x_col,
      y_col=args.y_col,
      title=args.title,
      output_path=args.output_path,
  )
  print(result)


if __name__ == "__main__":
  main()
