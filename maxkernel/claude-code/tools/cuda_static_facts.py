#!/usr/bin/env python3
"""Extracts hard facts from CUDA source by parsing, not by reading comprehension.

`maxkernel-analyze-cuda-reference` writes a prose brief about a reference
kernel. Prose is the right medium for intent and for the CUDA->TPU translation
argument, but it is the wrong medium for *numbers*: an LLM that paraphrases
`dim3 grid((n + 255) / 256)` as "one block per 256 elements" has already lost
the ceiling division, and every inference built on top of that -- occupancy,
working-set size, what the author believed fit in shared memory -- inherits
the error.

So the numbers are extracted mechanically first, and the analyzer is handed
them as ground truth. Grid and block dimensions, shared-memory byte counts,
tile constants and wmma fragment shapes are evidence about the decomposition;
they are not the analyzer's to restate.

This is a regex extractor, not a C++ parser. It is deliberately conservative:
it reports what it matched together with the source line, so a human or an
agent can check it, and it never guesses at a value it could not find. A
field it could not determine is `null`, never a plausible-looking default.

Handles both standalone `.cu`/`.cuh` files and CUDA carried inside a Python
string for `torch.utils.cpp_extension.load_inline` (the KernelBench layout).

Exit codes:
  0  parsed at least one translation unit
  3  no CUDA source found in the given paths
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Comment stripping. Done first so that commented-out launches and dead kernels
# are never reported as facts about the live code.
# ---------------------------------------------------------------------------

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")


def strip_comments(text):
  """Removes comments while preserving line numbering."""

  def _blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))

  text = _BLOCK_COMMENT.sub(_blank, text)
  text = _LINE_COMMENT.sub(_blank, text)
  return text


def line_of(text, index):
  return text.count("\n", 0, index) + 1


# ---------------------------------------------------------------------------
# Kernel entry points
# ---------------------------------------------------------------------------

_GLOBAL_FN = re.compile(
  r"(?:extern\s+\"C\"\s+)?"
  r"__global__\s+"
  r"(?:__launch_bounds__\s*\([^)]*\)\s*)?"
  r"(?:void|[\w:<>,\s\*&]+?)\s+"
  r"(?P<name>[A-Za-z_]\w*)\s*"
  r"(?:<[^>(]*>\s*)?"
  r"\((?P<params>[^;{]*?)\)\s*\{",
  re.S,
)

_LAUNCH_BOUNDS = re.compile(r"__launch_bounds__\s*\(\s*(?P<args>[^)]*)\)")


def split_params(param_text):
  """Splits a C++ parameter list on top-level commas."""
  params, depth, current = [], 0, []
  for ch in param_text:
    if ch in "<([":
      depth += 1
    elif ch in ">)]":
      depth -= 1
    if ch == "," and depth == 0:
      params.append("".join(current).strip())
      current = []
    else:
      current.append(ch)
  tail = "".join(current).strip()
  if tail:
    params.append(tail)
  return [p for p in params if p]


def parse_param(param):
  """Best-effort (type, name, qualifiers) for one parameter."""
  qualifiers = []
  for q in ("__restrict__", "const", "volatile"):
    if re.search(r"\b" + re.escape(q) + r"\b", param):
      qualifiers.append(q)
  cleaned = re.sub(r"\b(?:__restrict__|const|volatile)\b", " ", param).strip()
  cleaned = re.sub(r"\s+", " ", cleaned)
  match = re.match(
    r"^(?P<type>.*?[\s\*&])(?P<name>[A-Za-z_]\w*)\s*(?:\[\s*\])?$", cleaned
  )
  if match:
    return {
      "type": match.group("type").strip(),
      "name": match.group("name"),
      "qualifiers": qualifiers,
      "raw": param.strip(),
    }
  return {
    "type": cleaned,
    "name": None,
    "qualifiers": qualifiers,
    "raw": param.strip(),
  }


def find_kernels(text):
  kernels = []
  for match in _GLOBAL_FN.finditer(text):
    bounds = None
    window = text[max(0, match.start() - 120) : match.start() + 200]
    lb = _LAUNCH_BOUNDS.search(window)
    if lb:
      bounds = lb.group("args").strip()
    kernels.append(
      {
        "name": match.group("name"),
        "line": line_of(text, match.start()),
        "params": [parse_param(p) for p in split_params(match.group("params"))],
        "launch_bounds": bounds,
      }
    )
  return kernels


# ---------------------------------------------------------------------------
# Launch configuration:  kernel<<<grid, block, shmem, stream>>>(args)
# ---------------------------------------------------------------------------

_LAUNCH = re.compile(
  r"(?P<name>[A-Za-z_]\w*)\s*"
  r"(?:<[^<>]*>\s*)?"
  r"<<<(?P<config>.*?)>>>\s*\(",
  re.S,
)


_DIM3_DECL = re.compile(
  r"\bdim3\s+(?P<name>\w+)\s*(?:\((?P<call>[^)]*)\)|=\s*dim3\s*\((?P<assign>[^)]*)\))"
)
_INT_DECL = re.compile(
  r"\b(?:int|unsigned|size_t|unsigned\s+int|const\s+int)\s+(?P<name>\w+)\s*=\s*(?P<value>[^;,]+)[;,]"
)


def resolve_dims(text):
  """Maps launch-config variable names to the expression they were built from.

  `kernel<<<grid, block>>>` tells us nothing on its own; `dim3 grid(R)` two
  lines above is the fact. Without this pass every launch reports the
  placeholder identifier instead of the decomposition, which is exactly the
  detail the analyzer must not have to guess at.
  """
  table = {}
  for match in _DIM3_DECL.finditer(text):
    args = match.group("call") or match.group("assign") or ""
    table[match.group("name")] = {
      "expr": f"dim3({args.strip()})",
      "components": [a.strip() for a in split_params(args)],
      "line": line_of(text, match.start()),
    }
  for match in _INT_DECL.finditer(text):
    name = match.group("name")
    if name not in table:
      table[name] = {
        "expr": match.group("value").strip(),
        "components": [match.group("value").strip()],
        "line": line_of(text, match.start()),
      }
  return table


def _resolve(value, table):
  """Annotates one launch-config slot with its declaration, when there is one."""
  if value is None:
    return None
  key = value.strip()
  entry = table.get(key)
  if entry is None:
    return {"raw": key, "resolved": None, "components": None}
  return {
    "raw": key,
    "resolved": entry["expr"],
    "components": entry["components"],
    "declared_line": entry["line"],
  }


def find_launches(text):
  table = resolve_dims(text)
  launches = []
  for match in _LAUNCH.finditer(text):
    config = split_params(match.group("config"))
    launches.append(
      {
        "kernel": match.group("name"),
        "line": line_of(text, match.start()),
        "grid": _resolve(config[0] if len(config) > 0 else None, table),
        "block": _resolve(config[1] if len(config) > 1 else None, table),
        "dynamic_shared_mem": _resolve(
          config[2] if len(config) > 2 else "0", table
        ),
        "stream": config[3].strip() if len(config) > 3 else None,
        "raw": "<<<" + match.group("config").strip() + ">>>",
      }
    )
  return launches


# ---------------------------------------------------------------------------
# Compile-time constants -- the tile shapes the author chose.
# ---------------------------------------------------------------------------

_DEFINE = re.compile(
  r"^[ \t]*#[ \t]*define[ \t]+(?P<name>[A-Z_][A-Z0-9_]*)[ \t]+(?P<value>[^\n\\]+)",
  re.M,
)
_CONSTEXPR = re.compile(
  r"\b(?:static\s+)?(?:constexpr|const)\s+(?:int|unsigned|size_t|unsigned\s+int|long)\s+"
  r"(?P<name>\w+)\s*=\s*(?P<value>[^;]+);"
)
_TEMPLATE_PARAM = re.compile(r"template\s*<(?P<params>[^>]*)>")


def find_constants(text):
  constants = []
  for match in _DEFINE.finditer(text):
    constants.append(
      {
        "kind": "define",
        "name": match.group("name"),
        "value": match.group("value").strip(),
        "line": line_of(text, match.start()),
      }
    )
  for match in _CONSTEXPR.finditer(text):
    constants.append(
      {
        "kind": "constexpr",
        "name": match.group("name"),
        "value": match.group("value").strip(),
        "line": line_of(text, match.start()),
      }
    )
  template_params = []
  for match in _TEMPLATE_PARAM.finditer(text):
    for p in split_params(match.group("params")):
      template_params.append(
        {"decl": p.strip(), "line": line_of(text, match.start())}
      )
  return constants, template_params


# ---------------------------------------------------------------------------
# Shared memory
# ---------------------------------------------------------------------------

_SHARED = re.compile(
  r"(?:extern\s+)?__shared__\s+(?P<type>[\w:<>\s\*]+?)\s+(?P<name>\w+)\s*(?P<dims>(?:\[[^\]]*\])*)\s*;"
)
_EXTERN_SHARED = re.compile(r"extern\s+__shared__")


def find_shared(text):
  entries = []
  for match in _SHARED.finditer(text):
    dims = re.findall(r"\[([^\]]*)\]", match.group("dims") or "")
    entries.append(
      {
        "type": match.group("type").strip(),
        "name": match.group("name"),
        "dims": [d.strip() for d in dims],
        "dynamic": bool(
          _EXTERN_SHARED.search(text[max(0, match.start() - 12) : match.end()])
        ),
        "line": line_of(text, match.start()),
      }
    )
  return entries


# ---------------------------------------------------------------------------
# Tensor cores
# ---------------------------------------------------------------------------

_WMMA_FRAGMENT = re.compile(
  r"wmma\s*::\s*fragment\s*<\s*(?P<args>[^>]*(?:<[^>]*>)?[^>]*)>\s*(?P<name>\w+)?"
)
_MMA_PTX = re.compile(r"mma\.sync\.aligned\.(?P<shape>m\d+n\d+k\d+)")


def find_tensor_core(text):
  fragments = []
  for match in _WMMA_FRAGMENT.finditer(text):
    args = [a.strip() for a in split_params(match.group("args"))]
    fragments.append(
      {
        "args": args,
        "name": match.group("name"),
        "line": line_of(text, match.start()),
      }
    )
  ptx = [
    {"shape": m.group("shape"), "line": line_of(text, m.start())}
    for m in _MMA_PTX.finditer(text)
  ]
  return {"wmma_fragments": fragments, "mma_ptx": ptx}


# ---------------------------------------------------------------------------
# Mechanism census -- what SIMT machinery this kernel actually uses.
#
# The reconciler turns this into the NON_PORTABLE bucket of the ideas ledger:
# every mechanism here has either a named TPU counterpart or none at all, and
# naming the ones with none is what stops a planner transliterating them.
# ---------------------------------------------------------------------------

MECHANISMS = {
  "syncthreads": r"\b__syncthreads\s*\(",
  "syncwarp": r"\b__syncwarp\s*\(",
  "warp_shuffle": r"\b__shfl(?:_down|_up|_xor)?_sync\b",
  "warp_vote": r"\b__(?:ballot|any|all)_sync\b",
  "atomics": r"\batomic(?:Add|Sub|Max|Min|CAS|Exch|And|Or|Xor)\b",
  "threadfence": r"\b__threadfence(?:_block|_system)?\s*\(",
  "cooperative_groups": r"\bcooperative_groups\s*::|\bcg\s*::",
  "ldg": r"\b__ldg\s*\(",
  "restrict": r"\b__restrict__\b",
  "async_copy": r"\bcp\.async\b|\bmemcpy_async\b|\bpipeline\b",
  "vectorized_load": r"\b(?:float4|float2|int4|half2|uint4)\b",
  "fast_math_intrinsics": r"\b__(?:expf|logf|sinf|cosf|powf|fdividef)\b|\brsqrtf\s*\(",
  "grid_stride_loop": r"for\s*\([^;]*;[^;]*;[^)]*\+=\s*(?:blockDim\.x\s*\*\s*gridDim\.x|gridDim\.x\s*\*\s*blockDim\.x)",
  "tensor_cores": r"\bwmma\s*::|\bmma\.sync\b",
  "shared_memory": r"\b__shared__\b",
  "dynamic_parallelism": r"<<<[^>]*>>>\s*\([^)]*\)\s*;[^}]*__global__",
}

# The counterpart each mechanism has on TPU. `None` means it genuinely has
# none -- the strongest possible signal to the planner not to port it.
TPU_COUNTERPART = {
  "syncthreads": None,
  "syncwarp": None,
  "warp_shuffle": "a plain jnp reduction over the axis",
  "warp_vote": "a jnp boolean reduction",
  "atomics": "accumulate across the grid into an output block, or input_output_aliases",
  "threadfence": None,
  "cooperative_groups": None,
  "ldg": None,
  "restrict": None,
  "async_copy": "Pallas already double-buffers BlockSpec DMAs",
  "vectorized_load": None,
  "fast_math_intrinsics": "jnp equivalents; note the precision difference",
  "grid_stride_loop": "the Pallas grid itself",
  "tensor_cores": "MXU; contracting dims want multiples of 128",
  "shared_memory": "the VMEM block Pallas stages via BlockSpec",
  "dynamic_parallelism": None,
}


def census(text):
  found = {}
  for name, pattern in MECHANISMS.items():
    matches = list(re.finditer(pattern, text))
    if matches:
      found[name] = {
        "count": len(matches),
        "first_line": line_of(text, matches[0].start()),
        "tpu_counterpart": TPU_COUNTERPART.get(name),
      }
  return found


# ---------------------------------------------------------------------------
# Inline CUDA carried in a Python file
# ---------------------------------------------------------------------------

_PY_STRING_ASSIGN = re.compile(
  r"(?P<name>\w*(?:cuda|kernel|source)\w*)\s*=\s*(?P<quote>\"\"\"|''')(?P<body>.*?)(?P=quote)",
  re.S | re.I,
)


def extract_inline_cuda(text):
  """Pulls CUDA out of triple-quoted Python strings (the KernelBench layout)."""
  chunks = []
  for match in _PY_STRING_ASSIGN.finditer(text):
    body = match.group("body")
    if re.search(r"__global__|__device__|threadIdx|<<<", body):
      chunks.append(
        {
          "variable": match.group("name"),
          "start_line": line_of(text, match.start("body")),
          "source": body,
        }
      )
  return chunks


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze_source(source, origin):
  """Runs every extractor over one translation unit."""
  clean = strip_comments(source)
  constants, template_params = find_constants(clean)
  return {
    "origin": origin,
    "lines": source.count("\n") + 1,
    "kernels": find_kernels(clean),
    "launches": find_launches(clean),
    "constants": constants,
    "template_params": template_params,
    "shared_memory": find_shared(clean),
    "tensor_core": find_tensor_core(clean),
    "mechanisms": census(clean),
  }


def analyze_path(path):
  """Returns a list of unit records for one file (a .py may hold several)."""
  text = Path(path).read_text(errors="replace")
  if path.suffix == ".py":
    units = []
    for chunk in extract_inline_cuda(text):
      units.append(
        analyze_source(
          chunk["source"],
          f"{path}:{chunk['variable']}@L{chunk['start_line']}",
        )
      )
    return units
  return [analyze_source(text, str(path))]


def main():
  parser = argparse.ArgumentParser(
    description="Extract structural facts from CUDA source deterministically."
  )
  parser.add_argument("paths", nargs="+")
  parser.add_argument("--json", help="write the report to this path")
  args = parser.parse_args()

  units = []
  for raw in args.paths:
    path = Path(raw)
    if not path.is_file():
      print(f"Not a file, skipping: {path}", file=sys.stderr)
      continue
    try:
      units.extend(analyze_path(path))
    except OSError as e:
      print(f"Could not read {path}: {e}", file=sys.stderr)

  units = [u for u in units if u["kernels"] or u["launches"]]
  if not units:
    print("No CUDA kernels found in the given paths.", file=sys.stderr)
    sys.exit(3)

  report = {
    "units": units,
    "summary": {
      "kernel_count": sum(len(u["kernels"]) for u in units),
      "launch_count": sum(len(u["launches"]) for u in units),
      "mechanisms_present": sorted({m for u in units for m in u["mechanisms"]}),
      "mechanisms_without_tpu_counterpart": sorted(
        {
          m
          for u in units
          for m, info in u["mechanisms"].items()
          if info["tpu_counterpart"] is None
        }
      ),
    },
  }

  text = json.dumps(report, indent=2)
  if args.json:
    Path(args.json).write_text(text + "\n")
    print(f"Wrote {args.json}", file=sys.stderr)
  else:
    print(text)

  s = report["summary"]
  print(
    f"kernels={s['kernel_count']} launches={s['launch_count']} "
    f"mechanisms={len(s['mechanisms_present'])} "
    f"non-portable={len(s['mechanisms_without_tpu_counterpart'])}",
    file=sys.stderr,
  )
  sys.exit(0)


if __name__ == "__main__":
  main()
