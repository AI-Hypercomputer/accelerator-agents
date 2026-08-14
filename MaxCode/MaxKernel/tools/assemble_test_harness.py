#!/usr/bin/env python3
"""Deterministically assembles the shared test harness (run ONCE per run).

Replaces asking an LLM to hand-copy `test_harness_template.py` and hand-rename
the base kernel's entry point. Two problems with that: (1) an LLM told to
"copy this verbatim" is still generating tokens, not guaranteeing byte-for-byte
fidelity -- the same class of risk the upstream PR eliminated by moving this
substitution into a real tool function (`write_test_file_tool_fn`) instead of
an LLM-authored pytest file; (2) if the base kernel's `computation` is simply
renamed to `base_computation` while its internal `kernel` helper keeps its
original name, and the optimized kernel is later concatenated into the same
file with its own module-level `kernel` (mandated by `implement_kernel_agent.md`
for every kernel), the second `kernel` definition silently overwrites the
first in the shared module namespace -- so `base_computation()` would end up
calling the *optimized* kernel's inner logic at test time, invalidating every
correctness check.

This script avoids both by:
  - copying `test_harness_template.py` byte-for-byte (real file I/O).
  - embedding the base kernel's source as a string constant and running it
    through `exec()` into its OWN dict namespace, exactly like Python's
    `import` does for separate modules -- so `base_computation` and the later
    `opt_computation` (bound the same way by `assemble_test_run.py`) can never
    collide on an internal helper name, no matter what either kernel calls its
    helpers.
  - only ever touching ATOL/RTOL via a regex substitution on those two known
    constant-assignment lines -- never rewriting correctness/benchmark logic.
"""

import argparse
import pathlib
import re
import sys

TOOLS_DIR = pathlib.Path(__file__).parent
TEMPLATE_PATH = TOOLS_DIR / "test_harness_template.py"


def assemble(
    base_kernel_path: str,
    get_inputs_path: str,
    output_path: str,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
  """Assembles the shared test harness deterministically.

  Args:
    base_kernel_path: Path to the base kernel python file.
    get_inputs_path: Path to the get_inputs python file.
    output_path: Output file path.
    atol: Absolute tolerance.
    rtol: Relative tolerance.
  """
  base_kernel_src = pathlib.Path(base_kernel_path).read_text()
  if "def computation" not in base_kernel_src:
    raise ValueError(
        f"{base_kernel_path} has no module-level `computation` function -- "
        "cannot bind it as base_computation."
    )

  get_inputs_src = pathlib.Path(get_inputs_path).read_text()
  if "def get_inputs" not in get_inputs_src:
    raise ValueError(f"{get_inputs_path} has no `get_inputs()` function.")

  template_src = TEMPLATE_PATH.read_text()
  template_src = re.sub(
      r"^ATOL = .*$", f"ATOL = {atol!r}", template_src, count=1, flags=re.M
  )
  template_src = re.sub(
      r"^RTOL = .*$", f"RTOL = {rtol!r}", template_src, count=1, flags=re.M
  )

  base_kernel_literal = repr(base_kernel_src)

  parts = [
      template_src,
      "",
      "# =====================================================================",
      f"# Base kernel, embedded verbatim from {base_kernel_path}.",
      (
          "# Executed into its own namespace (like a separate module import)"
          " so its"
      ),
      "# internal helpers (e.g. `kernel`) can never collide with the optimized",
      (
          "# kernel's helpers of the same name -- see this script's module"
          " docstring."
      ),
      "# =====================================================================",
      f"_BASE_KERNEL_SRC = {base_kernel_literal}",
      "_base_ns = {}",
      "exec(compile(_BASE_KERNEL_SRC, "
      + repr(str(base_kernel_path))
      + ", 'exec'), _base_ns)",
      "base_computation = _base_ns['computation']",
      "",
      "# =====================================================================",
      (
          "# get_inputs(), authored by generate_test_file_agent from"
          f" {get_inputs_path}."
      ),
      "# This is the only LLM-authored part of this file.",
      "# =====================================================================",
      get_inputs_src,
  ]

  pathlib.Path(output_path).write_text("\n".join(parts))
  print(f"Assembled shared test harness at {output_path}")


def main():
  parser = argparse.ArgumentParser(
      description="Assemble the shared test harness (run once per run)."
  )
  parser.add_argument("base_kernel_path")
  parser.add_argument("get_inputs_path")
  parser.add_argument("output_path")
  parser.add_argument("--atol", type=float, default=1e-2)
  parser.add_argument("--rtol", type=float, default=1e-2)
  args = parser.parse_args()

  try:
    assemble(
        args.base_kernel_path,
        args.get_inputs_path,
        args.output_path,
        atol=args.atol,
        rtol=args.rtol,
    )
  except Exception as e:  # pylint: disable=broad-except
    print(f"Failed to assemble test harness: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
