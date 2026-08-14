#!/usr/bin/env python3
"""Extracts HLO proto from xplane.pb files if available."""

import argparse
import gzip
import sys
from typing import Optional

from tensorflow.tsl.profiler.protobuf import xplane_pb2


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
  del hlo_module_name  # Unused in current standalone version.
  try:
    open_func = gzip.open if xplane_path.endswith(".gz") else open
    with open_func(xplane_path, "rb") as f:
      xspace = xplane_pb2.XSpace()
      xspace.ParseFromString(f.read())

    # XSpace parsed successfully. Full HLO extraction requires metadata ID
    # mapping.
    del xspace

    return (
        "HLO extraction not fully implemented in this standalone version yet"
        " (requires metadata ID mapping). Please use `load_xplane_and_query`"
        " to explore 'hlo' related events."
    )

  except Exception as e:  # pylint: disable=broad-exception-caught
    return f"Error extracting HLO: {e}"


def main():
  parser = argparse.ArgumentParser(
      description="Extract HLO dump from an XProf xplane.pb file."
  )
  parser.add_argument("xplane_path", help="Path to the .xplane.pb file.")
  parser.add_argument(
      "--module-name",
      "--module_name",
      dest="module_name",
      default=None,
      help="Optional HLO module name filter.",
  )

  argv = sys.argv[1:]
  if argv and argv[0] == "--":
    argv = argv[1:]
  args = parser.parse_args(argv)

  result = get_hlo_dump(args.xplane_path, args.module_name)
  print(result)


if __name__ == "__main__":
  main()
