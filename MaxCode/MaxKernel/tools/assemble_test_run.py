#!/usr/bin/env python3
"""Deterministically binds an optimized kernel into the shared harness.

Run once per iteration (Phase 3, against `{optimized_kernel_path}`) and once
per autotune trial (Phase 4, against each placeholder-substituted candidate).
Takes the already-assembled shared harness (`assemble_test_harness.py`'s
output -- unchanged across the whole run) and the optimized kernel's current
source, and produces a runnable script.

Like `assemble_test_harness.py`, this embeds the optimized kernel's source as
a string and `exec()`s it into its own namespace rather than concatenating
raw source into one shared module -- so its `kernel` helper (or any other
module-level name) can never collide with the base kernel's helper of the
same name, which a naive text-paste would risk.
"""

import argparse
import pathlib
import sys


def assemble(
    harness_path: str, optimized_kernel_path: str, output_path: str
) -> None:
  """Binds an optimized kernel into the shared harness.

  Args:
    harness_path: Path to the shared harness python file.
    optimized_kernel_path: Path to the optimized kernel python file.
    output_path: Path to write the runnable test script.
  """
  harness_src = pathlib.Path(harness_path).read_text()
  if "def main" not in harness_src:
    raise ValueError(
        f"{harness_path} does not look like an assembled harness (no main())."
    )
  if "opt_computation = " in harness_src:
    raise ValueError(
        f"{harness_path} already binds opt_computation -- pass the harness "
        "output from assemble_test_harness.py, not an already-assembled run."
    )

  opt_kernel_src = pathlib.Path(optimized_kernel_path).read_text()
  if "def computation" not in opt_kernel_src:
    raise ValueError(
        f"{optimized_kernel_path} has no module-level `computation` function --"
        " cannot bind it as opt_computation."
    )

  parts = [
      harness_src,
      "",
      "# =====================================================================",
      f"# Optimized kernel, embedded verbatim from {optimized_kernel_path}.",
      (
          "# Executed into its own namespace -- see this script's module"
          " docstring."
      ),
      "# =====================================================================",
      f"_OPT_KERNEL_SRC = {opt_kernel_src!r}",
      "_opt_ns = {}",
      "exec(compile(_OPT_KERNEL_SRC, "
      + repr(str(optimized_kernel_path))
      + ", 'exec'), _opt_ns)",
      "opt_computation = _opt_ns['computation']",
      "",
      'if __name__ == "__main__":',
      "    main()",
      "",
  ]

  pathlib.Path(output_path).write_text("\n".join(parts))
  print(f"Assembled runnable test script at {output_path}")


def main():
  parser = argparse.ArgumentParser(
      description="Bind an optimized kernel into the shared test harness."
  )
  parser.add_argument(
      "harness_path", help="Path to the shared harness (test_file_path)."
  )
  parser.add_argument(
      "optimized_kernel_path", help="Path to the current candidate kernel."
  )
  parser.add_argument(
      "output_path", help="Path to write the runnable script to."
  )
  args = parser.parse_args()

  try:
    assemble(args.harness_path, args.optimized_kernel_path, args.output_path)
  except Exception as e:  # pylint: disable=broad-except
    print(f"Failed to assemble test run: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
