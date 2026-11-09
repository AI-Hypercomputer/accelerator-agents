# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entry point for the TPU probe suite.

    python run_probes.py --group memory --group precision
    python run_probes.py --pick vmem_scratch_bisect
    python run_probes.py --list

Selection happens after import, so adding a probe never changes what an
unrelated selection runs. Destructive probes are excluded unless named: see
``harness.probe`` for why they cannot share an interpreter with the rest.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys

import harness


def _import_all_check_modules() -> list[str]:
  """Imports every module under ``checks/`` so its probes register."""
  import checks

  names = []
  for module in pkgutil.iter_modules(checks.__path__):
    importlib.import_module(f"checks.{module.name}")
    names.append(module.name)
  return sorted(names)


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--group",
    action="append",
    default=[],
    help="Run only these groups. Repeatable.",
  )
  parser.add_argument(
    "--pick",
    action="append",
    default=[],
    help="Run only probes with these exact names. Repeatable. This is the "
    "only way to run a destructive probe.",
  )
  parser.add_argument(
    "--include-destructive",
    action="store_true",
    help="Include destructive probes in a group selection. They can halt "
    "the Mosaic core and poison every later probe in this process.",
  )
  parser.add_argument(
    "--list",
    action="store_true",
    help="Print the registry and exit without touching the device.",
  )
  parser.add_argument(
    "--traceback",
    action="store_true",
    help="Print a full traceback for every FAIL.",
  )
  return parser.parse_args(argv)


def main(argv: list[str]) -> int:
  args = _parse_args(argv)
  modules = _import_all_check_modules()
  probes = harness.registered_probes()

  if args.list:
    for target in probes:
      flag = " [destructive]" if target.destructive else ""
      print(f"{target.group:<14} {target.name}{flag}")
      print(f"    {target.question}")
    return 0

  selected = []
  for target in probes:
    if args.pick:
      if target.name in args.pick:
        selected.append(target)
      continue
    if args.group and target.group not in args.group:
      continue
    if target.destructive and not args.include_destructive:
      continue
    selected.append(target)

  print("=" * 72)
  print(f"probe suite: {len(selected)} of {len(probes)} probes")
  print(f"modules: {', '.join(modules)}")
  print("=" * 72)

  results = []
  for target in selected:
    result = harness.run_probe(target, show_traceback=args.traceback)
    harness.report(result)
    results.append(result)
  return harness.report_summary(results)


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
