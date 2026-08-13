#!/usr/bin/env python3
"""Loads an xplane.pb file into an in-memory SQLite DB and executes SQL queries."""

import argparse
import gzip
import sqlite3
import sys

import pandas as pd
from tensorflow.tsl.profiler.protobuf import xplane_pb2


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
      A markdown-formatted table (or plain text table) of the query results.
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

    try:
      return df.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
      return df.to_string(index=False)

  except Exception as e:  # pylint: disable=broad-exception-caught
    return f"Error executing query: {e}"


def main():
  parser = argparse.ArgumentParser(
      description="Query an XProf xplane.pb file using SQL."
  )
  parser.add_argument("xplane_path", help="Path to the .xplane.pb file.")
  parser.add_argument("sql_query", help="SQL query to execute.")

  argv = sys.argv[1:]
  if argv and argv[0] == "--":
    argv = argv[1:]
  args = parser.parse_args(argv)

  result = load_xplane_and_query(args.xplane_path, args.sql_query)
  print(result)


if __name__ == "__main__":
  main()
