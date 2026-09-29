#!/usr/bin/env python3
"""Loads an xplane.pb file into an in-memory SQLite DB and executes SQL queries."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd

import xplane_loader


def load_xplane_and_query(xplane_path: str, sql_query: str) -> str:
  """Loads an xplane.pb file into an in-memory SQLite DB and runs a SQL query.

  The database schema is:
  - planes (id, name)
  - lines (id, plane_id, display_id, name, timestamp_ns)
  - events (plane_id, line_id, name, offset_ps, duration_ps, start_ps, end_ps)

  See `xplane_loader` for which of these columns are surrogate values rather
  than the profiler's original ids.

  Args:
      xplane_path: Path to the .xplane.pb file (or a directory containing one).
      sql_query: The SQL query to execute against the loaded data.

  Returns:
      A markdown-formatted table (or plain text table) of the query results.
  """
  try:
    conn = xplane_loader.load_into_sqlite(xplane_path)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()

    try:
      return df.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
      return df.to_string(index=False)

  except Exception as e:  # pylint: disable=broad-except
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
