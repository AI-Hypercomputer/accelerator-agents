#!/usr/bin/env python3
"""Extracts HLO module text from an xplane.pb trace.

The previous version of this file was a stub: it parsed the XSpace, ran an
empty `for stat in plane.stats: pass` loop, and unconditionally returned
"HLO extraction not fully implemented in this standalone version yet". No
caller could get HLO out of it.

HLO *is* recoverable offline. xprof's converter writes
`<module_name>.hlo_proto.pb` files next to the trace as a side effect of
`xspace_to_tool_names()`, and its `graph_viewer` tool renders any of those
modules to text -- this is exactly the path xprof's own CLI uses in
`xprof/cli/internal/oss/hlo_tools.py`, minus the server round-trip.
"""

import argparse
import pathlib
import sys
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from xprof.convert import raw_to_tool_data  # pylint: disable=g-import-not-at-top

import xplane_loader  # pylint: disable=g-import-not-at-top

_HLO_PROTO_SUFFIX = ".hlo_proto.pb"


def list_hlo_modules(xplane_path: str) -> list[str]:
  """Returns the HLO module names recorded in the trace.

  `xspace_to_tool_names` writes `<module>.hlo_proto.pb` beside the trace as a
  side effect; that is what makes the modules addressable by name below.
  """
  resolved = xplane_loader.resolve_xplane_path(xplane_path)
  raw_to_tool_data.xspace_to_tool_names([resolved])
  trace_dir = pathlib.Path(resolved).parent
  return sorted(
      f.name.removesuffix(_HLO_PROTO_SUFFIX)
      for f in trace_dir.glob(f"*{_HLO_PROTO_SUFFIX}")
  )


def get_hlo_dump(
    xplane_path: str,
    hlo_module_name: Optional[str] = None,
    print_metadata: bool = False,
    output_path: Optional[str] = None,
) -> str:
  """Extracts HLO module text from an xplane.pb trace.

  Args:
      xplane_path: Path to .xplane.pb file (or a directory containing one).
      hlo_module_name: Optional module name; defaults to the first module.
      print_metadata: If True, emit the long form (with metadata) instead of
        the short form.
      output_path: If set, write the HLO text here instead of returning it
        inline.

  Returns:
      The HLO module text, or a status string describing where it was saved.
  """
  try:
    resolved = xplane_loader.resolve_xplane_path(xplane_path)
    modules = list_hlo_modules(resolved)
    if not modules:
      return (
          f"No HLO modules found in {resolved}. The trace may not contain XLA"
          " program data (e.g. a host-only or metadata-only capture)."
      )

    if hlo_module_name:
      # Accept an exact name or an unambiguous substring: real module names
      # carry a program-id suffix, e.g. "jit_computation(2377016034403575603)",
      # which a caller is unlikely to know or type.
      if hlo_module_name in modules:
        target = hlo_module_name
      else:
        matches = [m for m in modules if hlo_module_name in m]
        if not matches:
          return (
              f"Module {hlo_module_name!r} not found. Available modules:"
              f" {', '.join(modules)}"
          )
        if len(matches) > 1:
          return (
              f"Module {hlo_module_name!r} is ambiguous, matching:"
              f" {', '.join(matches)}. Pass a full module name."
          )
        target = matches[0]
    else:
      target = modules[0]

    data, _ = raw_to_tool_data.xspace_to_tool_data(
        [resolved],
        "graph_viewer",
        {
            "graph_viewer_options": {
                "type": "long_txt" if print_metadata else "short_txt",
                "module_name": target,
            }
        },
    )
    text = data.decode("utf-8") if isinstance(data, bytes) else data

    if output_path:
      pathlib.Path(output_path).write_text(text)
      return f"HLO for module {target!r} saved to {output_path}"

    header = f"# HLO module: {target}\n# Available modules: {', '.join(modules)}\n"
    return header + text

  except Exception as e:  # pylint: disable=broad-except
    return f"Error extracting HLO: {e}"


def main():
  parser = argparse.ArgumentParser(
      description="Extract HLO module text from an XProf xplane.pb file."
  )
  parser.add_argument("xplane_path", help="Path to the .xplane.pb file.")
  parser.add_argument(
      "--module-name",
      "--module_name",
      dest="module_name",
      default=None,
      help="HLO module name (exact or unambiguous substring).",
  )
  parser.add_argument(
      "--list",
      action="store_true",
      help="List available HLO module names and exit.",
  )
  parser.add_argument(
      "--print-metadata",
      "--print_metadata",
      dest="print_metadata",
      action="store_true",
      help="Emit the long form, including instruction metadata.",
  )
  parser.add_argument(
      "--output-path",
      "--output_path",
      "-o",
      dest="output_path",
      default=None,
      help="Write the HLO text to this file instead of stdout.",
  )

  argv = sys.argv[1:]
  if argv and argv[0] == "--":
    argv = argv[1:]
  args = parser.parse_args(argv)

  if args.list:
    try:
      modules = list_hlo_modules(args.xplane_path)
    except Exception as e:  # pylint: disable=broad-except
      print(f"Error listing HLO modules: {e}")
      return
    print("\n".join(modules) if modules else "No HLO modules found.")
    return

  print(
      get_hlo_dump(
          args.xplane_path,
          args.module_name,
          print_metadata=args.print_metadata,
          output_path=args.output_path,
      )
  )


if __name__ == "__main__":
  main()
