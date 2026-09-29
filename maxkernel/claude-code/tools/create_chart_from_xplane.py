#!/usr/bin/env python3
"""Generates charts from xplane.pb profiling data using SQL queries."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Headless: TPU VMs/agents have no display.
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import pandas as pd  # pylint: disable=g-import-not-at-top

import xplane_loader  # pylint: disable=g-import-not-at-top


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
      xplane_path: Path to .xplane.pb (or a directory containing one).
      sql_query: SQL query to get data (same schema as query_xplane).
      chart_type: 'bar' or 'pie'.
      x_col: Column for X axis (bar) / labels (pie).
      y_col: Column for Y axis (bar) or values (pie).
      title: Chart title.
      output_path: Optional output file path for the generated chart PNG.

  Returns:
      Status string indicating where the chart was saved.
  """
  try:
    conn = xplane_loader.load_into_sqlite(xplane_path)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()

    if df.empty:
      return "Query returned no data, cannot plot."

    # Fail with an actionable message rather than a bare KeyError from
    # matplotlib when the query's column names don't match --x-col/--y-col.
    missing = [c for c in (x_col, y_col) if c not in df.columns]
    if missing:
      return (
          f"Column(s) {missing} not in query result. Available columns:"
          f" {list(df.columns)}. Pass --x-col/--y-col to match your SELECT"
          " aliases."
      )

    # Op names in a trace are routinely 100+ chars; untruncated they make
    # tight_layout fail to fit the axes and emit a warning.
    labels = df[x_col].astype(str).map(
        lambda s: s if len(s) <= 40 else s[:37] + "..."
    )

    plt.figure(figsize=(10, 6))
    if chart_type == "bar":
      plt.bar(labels, df[y_col])
      plt.xlabel(x_col)
      plt.ylabel(y_col)
      plt.xticks(rotation=45, ha="right")
    elif chart_type == "pie":
      plt.pie(df[y_col], labels=labels, autopct="%1.1f%%")

    if title:
      plt.title(title)

    plt.tight_layout()
    output_filename = output_path or f"{xplane_path}.png"
    plt.savefig(output_filename)
    plt.close()

    return f"Chart saved to {output_filename}"

  except Exception as e:  # pylint: disable=broad-except
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
      help="Output file path for the chart PNG (defaults to <xplane_path>.png).",
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
