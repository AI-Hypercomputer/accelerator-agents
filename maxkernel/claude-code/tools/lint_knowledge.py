"""Keeps the TPU kernel knowledge base free of pins, past numbers and evidence.

The files under `reference/design/`, `reference/strategies/`, the probe suite
and the method/catalog/hardware-facts references are read by agents as rules.
A measured number copied into them is a number from one machine and one
toolchain, quoted as if it held everywhere; a toolchain version pinned in them
goes stale on the next upgrade. Numbers come from running a probe on the
target (`tools/run_probe.py`), never from these files.

This lint fails on:

  * a jax / jaxlib / libtpu version (`jax 0.11.0`, `libtpu 0.0.47`, ...)
  * a past job id (`j-0123abcd...`)
  * `[measured: ...]` / `[verified: ...]` evidence tags
  * `cdk` job tooling (`cdk job`, `tpu-fanout`, `sync-outputs`)

Exit codes: 0 clean, 1 at least one violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
  "reference/design",
  "reference/strategies",
  "reference/tpu_hardware_facts.md",
  "reference/kernel_design_method.md",
  "reference/probe_catalog.md",
  "reference/pallas_code_style.md",
  "probes",
]

RULES = [
  (
    "toolchain version pin",
    re.compile(r"\b(?:jax|jaxlib|libtpu)[ =/@-]+v?\d+\.\d+(?:\.\d+)*", re.I),
  ),
  ("past job id", re.compile(r"\bj-[0-9a-f]{8}\b")),
  ("evidence tag", re.compile(r"\[(?:measured|verified):", re.I)),
  ("cdk tooling", re.compile(r"\bcdk\b|tpu-fanout|sync-outputs")),
]


def _files() -> list[Path]:
  found = []
  for target in TARGETS:
    path = ROOT / target
    if path.is_file():
      found.append(path)
    elif path.is_dir():
      found += sorted(
        p for p in path.rglob("*") if p.suffix in (".md", ".py") and p.is_file()
      )
  return found


def main() -> int:
  violations = []
  for path in _files():
    if path.resolve() == Path(__file__).resolve():
      continue
    for number, line in enumerate(path.read_text().splitlines(), 1):
      for label, pattern in RULES:
        if pattern.search(line):
          rel = path.relative_to(ROOT)
          violations.append(f"{rel}:{number}: {label}: {line.strip()[:120]}")
  for violation in violations:
    print(violation)
  if violations:
    print(f"\n{len(violations)} violation(s).")
    return 1
  print(f"lint_knowledge: {len(_files())} files clean.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
