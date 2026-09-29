#!/usr/bin/env python3
"""Classifies every user-supplied input file and proposes slot assignments.

MaxKernel accepts two *kinds* of input, and they carry different authority:

  * the **primary** -- the thing being converted. It defines the semantics,
    the baseline `base.py`, and therefore every number the run reports.
  * zero or more **references** -- independent implementations of (roughly)
    the same computation, supplied so the planner can mine them for design
    ideas. A reference is never executed, never measured, and never compared
    against.

This tool is the deterministic front half of that decision. It classifies
each file by marker matching -- never by judgement -- and reports which files
are *candidates* for which slot. It deliberately does NOT make the final
assignment when the answer is ambiguous: picking the wrong primary silently
benchmarks the wrong computation for the whole run, so an ambiguous set exits
non-zero and the orchestrator asks the user.

Replaces `detect_input_language.py`, whose single-slot verdict
(`jax` | `cuda` | `pytorch`) cannot express "this CUDA file is a reference for
that PyTorch file".

Exit codes:
  0  a single unambiguous primary candidate was found
  3  zero or multiple primary candidates -- the orchestrator must ask the user
  4  no readable input files at all
"""

import argparse
import json
from pathlib import Path
import re
import sys

# Directories and files that are never user source.
SKIP_DIRS = {
    "__pycache__", ".git", ".ipynb_checkpoints", "build", "dist", ".mypy_cache"
}
SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp"}

# ---------------------------------------------------------------------------
# Markers. Each entry is (compiled_regex, human_readable_evidence).
# These are matched against file text; the first match of each marker is
# recorded with its line number so a verdict can always be justified.
# ---------------------------------------------------------------------------

CUDA_MARKERS = [
    (re.compile(r"\b__global__\b"), "__global__ kernel entry point"),
    (re.compile(r"\b__device__\b"), "__device__ function"),
    (re.compile(r"\b(?:threadIdx|blockIdx|blockDim|gridDim)\b"), "SIMT index arithmetic"),
    (re.compile(r"\b__syncthreads\s*\("), "__syncthreads barrier"),
    (re.compile(r"\b__shared__\b"), "__shared__ memory declaration"),
    (re.compile(r"\bcuda(?:Malloc|Memcpy|Free|DeviceSynchronize)\b"), "CUDA runtime API call"),
    (re.compile(r"<<<[^>]*>>>"), "kernel launch configuration"),
    (re.compile(r"\b__shfl(?:_down|_up|_xor)?_sync\b"), "warp shuffle"),
    (re.compile(r"\bwmma\s*::"), "tensor-core wmma fragment"),
]

# CUDA carried inside a Python file for torch.utils.cpp_extension.load_inline
# -- the KernelBench layout.
INLINE_CUDA_MARKERS = [
    (re.compile(r"\bload_inline\s*\("), "torch.utils.cpp_extension.load_inline"),
    (re.compile(r"\bcuda_sources\s*="), "cuda_sources string"),
    (re.compile(r"\bCUDAExtension\b"), "CUDAExtension build"),
]

TORCH_MARKERS = [
    (re.compile(r"^\s*import\s+torch\b", re.M), "import torch"),
    (re.compile(r"^\s*from\s+torch\b", re.M), "from torch import ..."),
    (re.compile(r"\bnn\.Module\b"), "nn.Module subclass"),
    (re.compile(r"\btorch\.(?:nn|Tensor|randn|zeros|empty|cat|matmul)\b"), "torch API call"),
]

JAX_MARKERS = [
    (re.compile(r"^\s*import\s+jax\b", re.M), "import jax"),
    (re.compile(r"^\s*from\s+jax\b", re.M), "from jax import ..."),
    (re.compile(r"\bjax\.numpy\b|\bjnp\."), "jax.numpy usage"),
    (re.compile(r"\bjax\.lax\b|\blax\."), "jax.lax usage"),
]

PALLAS_MARKERS = [
    (re.compile(r"\bpallas_call\b"), "pl.pallas_call"),
    (re.compile(r"from\s+jax\.experimental\s+import\s+pallas"), "pallas import"),
    (re.compile(r"\bpltpu\."), "pallas TPU primitives"),
]

# A file is a plausible *primary* only if it exposes something callable that
# the loop can bind. Ordered most- to least-specific.
PY_ENTRY_MARKERS = [
    (re.compile(r"^\s*def\s+computation\s*\(", re.M), "def computation(...)"),
    (re.compile(r"^\s*class\s+Model\s*\(", re.M), "class Model(...)  [KernelBench]"),
    (re.compile(r"^\s*def\s+get_inputs\s*\(", re.M), "def get_inputs()  [KernelBench]"),
    (re.compile(r"^\s*(?:class\s+\w+\s*\(\s*(?:nn\.)?Module)", re.M), "nn.Module subclass"),
    (re.compile(r"^\s{0,4}def\s+forward\s*\(", re.M), "def forward(...)"),
]

CUDA_ENTRY_MARKERS = [
    (re.compile(r"\b__global__\b"), "__global__ kernel"),
]


def _scan(text, markers):
  """Returns [{marker, line}] for every marker that fires at least once."""
  hits = []
  for pattern, description in markers:
    match = pattern.search(text)
    if match:
      line = text.count("\n", 0, match.start()) + 1
      hits.append({"marker": description, "line": line})
  return hits


def classify_text(text, suffix):
  """Classifies one file's contents. Returns (language, evidence_dict).

  Language is one of:
    cuda                      -- CUDA C++ device code
    pytorch                   -- a torch reference
    pytorch_with_inline_cuda  -- a torch module that also carries CUDA source
                                 for load_inline (the KernelBench layout).
                                 This single file yields BOTH a pytorch
                                 primary candidate and a cuda reference.
    jax                       -- JAX, with `has_pallas` saying whether it is
                                 already a Pallas kernel
    unknown                   -- no marker of any known framework
  """
  cuda_hits = _scan(text, CUDA_MARKERS)
  inline_hits = _scan(text, INLINE_CUDA_MARKERS)
  torch_hits = _scan(text, TORCH_MARKERS)
  jax_hits = _scan(text, JAX_MARKERS)
  pallas_hits = _scan(text, PALLAS_MARKERS)

  evidence = {
      "cuda": cuda_hits,
      "inline_cuda": inline_hits,
      "torch": torch_hits,
      "jax": jax_hits,
      "pallas": pallas_hits,
  }

  # A .cu/.cuh file is CUDA by extension even if it is mostly host code.
  if suffix in (".cu", ".cuh"):
    return "cuda", evidence

  # A C/C++ file carrying device-code markers is CUDA too.
  if suffix in (".cc", ".cpp", ".h", ".hpp") and cuda_hits:
    return "cuda", evidence

  if suffix == ".py":
    # Order matters. A KernelBench task imports torch AND embeds CUDA; it is
    # both, and collapsing it to one language throws away a slot.
    if torch_hits and (inline_hits or cuda_hits):
      return "pytorch_with_inline_cuda", evidence
    if torch_hits:
      return "pytorch", evidence
    if jax_hits:
      return "jax", evidence
    # CUDA embedded in Python with no torch at all -- still a CUDA reference.
    if cuda_hits or inline_hits:
      return "cuda", evidence
    return "unknown", evidence

  if cuda_hits:
    return "cuda", evidence
  return "unknown", evidence


def has_pallas(evidence):
  return bool(evidence.get("pallas"))


def find_entry_point(text, language):
  """Returns (bool, evidence_list) for whether this file exposes an entry point."""
  markers = CUDA_ENTRY_MARKERS if language == "cuda" else PY_ENTRY_MARKERS
  hits = _scan(text, markers)
  return bool(hits), hits


def slot_candidacy(language, entry_ok):
  """Which slot(s) this file could fill.

  A primary must be something the loop can port and measure: JAX or PyTorch
  with an entry point. CUDA can be a primary too -- that is today's behaviour,
  used when the user supplies CUDA and nothing else -- but it is only ever
  *preferred* as a reference when a non-CUDA candidate exists.
  """
  if language == "pytorch_with_inline_cuda":
    return ["primary", "reference"] if entry_ok else ["reference"]
  if language in ("pytorch", "jax"):
    return ["primary"] if entry_ok else []
  if language == "cuda":
    return ["reference", "primary"] if entry_ok else ["reference"]
  return []


def collect_files(paths):
  """Expands the given paths into a sorted list of candidate source files."""
  out = []
  for raw in paths:
    p = Path(raw)
    if p.is_dir():
      for child in sorted(p.rglob("*")):
        if not child.is_file():
          continue
        if any(part in SKIP_DIRS for part in child.parts):
          continue
        if child.suffix in SOURCE_SUFFIXES:
          out.append(child)
    elif p.is_file():
      out.append(p)
  # De-duplicate while preserving order.
  seen = set()
  unique = []
  for p in out:
    key = str(p.resolve())
    if key not in seen:
      seen.add(key)
      unique.append(p)
  return unique


def classify_paths(paths):
  """Classifies every file and proposes slot assignments."""
  files = collect_files(paths)
  records = []
  for path in files:
    try:
      text = path.read_text(errors="replace")
    except OSError as e:
      records.append({
          "path": str(path.resolve()),
          "language": "unreadable",
          "error": str(e),
          "has_entry_point": False,
          "slot_candidacy": [],
          "evidence": {},
      })
      continue

    language, evidence = classify_text(text, path.suffix)
    entry_ok, entry_evidence = find_entry_point(text, language)
    records.append({
        "path": str(path.resolve()),
        "language": language,
        "has_pallas": has_pallas(evidence),
        "has_entry_point": entry_ok,
        "entry_point_evidence": entry_evidence,
        "slot_candidacy": slot_candidacy(language, entry_ok),
        "evidence": {k: v for k, v in evidence.items() if v},
        "bytes": len(text),
    })

  return records


def propose_slots(records):
  """Applies the assignment rules. Returns (primary, references, ambiguity)."""
  primaries = [r for r in records if "primary" in r["slot_candidacy"]]

  # Prefer a non-CUDA primary whenever one exists: if the user supplied both
  # a torch module and a .cu file, the torch module is what is being
  # converted and the .cu is the reference. This is the whole point of the
  # two-slot model.
  non_cuda = [r for r in primaries if r["language"] != "cuda"]
  if non_cuda:
    primaries = non_cuda

  ambiguity = None
  primary = None
  if len(primaries) == 1:
    primary = primaries[0]
  elif not primaries:
    ambiguity = (
        "No file exposes an entry point the loop can bind. Expected a JAX or "
        "PyTorch module with a `computation`, `forward`, `Model` or "
        "`get_inputs` definition, or a CUDA file with a `__global__` kernel."
    )
  else:
    names = ", ".join(Path(r["path"]).name for r in primaries)
    ambiguity = (
        f"{len(primaries)} files could each be the primary source ({names}). "
        "Ask the user which one is being converted -- guessing here silently "
        "benchmarks the wrong computation for the entire run."
    )

  references = []
  for r in records:
    if primary is not None and r["path"] == primary["path"]:
      # The KernelBench layout is its own reference: the same file holds the
      # torch module (primary) and the CUDA it was written against.
      if r["language"] == "pytorch_with_inline_cuda":
        references.append({
            "kind": "cuda",
            "path": r["path"],
            "embedded": True,
            "note": "CUDA extracted from the primary file's inline sources",
        })
      continue
    if "reference" in r["slot_candidacy"]:
      references.append({
          "kind": "cuda" if r["language"] in ("cuda", "pytorch_with_inline_cuda") else r["language"],
          "path": r["path"],
          "embedded": r["language"] == "pytorch_with_inline_cuda",
      })

  return primary, references, ambiguity


def main():
  parser = argparse.ArgumentParser(
      description="Classify MaxKernel input files and propose slot assignments."
  )
  parser.add_argument("paths", nargs="+", help="files and/or directories")
  parser.add_argument("--json", help="write the full report to this path")
  args = parser.parse_args()

  records = classify_paths(args.paths)
  if not records:
    print("No readable source files found.", file=sys.stderr)
    sys.exit(4)

  primary, references, ambiguity = propose_slots(records)

  report = {
      "files": records,
      "primary": primary,
      "references": references,
      "ambiguity": ambiguity,
  }

  text = json.dumps(report, indent=2)
  if args.json:
    Path(args.json).write_text(text + "\n")
  print(text)

  if ambiguity:
    print(f"\nAMBIGUOUS: {ambiguity}", file=sys.stderr)
    sys.exit(3)

  print(
      f"\nPRIMARY: {primary['language']} {primary['path']}\n"
      f"REFERENCES: {len(references)}",
      file=sys.stderr,
  )
  sys.exit(0)


if __name__ == "__main__":
  main()
