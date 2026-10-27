#!/usr/bin/env python3
"""Extracts hard facts from GPU kernel source by parsing, not by reading comprehension.

`maxkernel-analyze-gpu-reference` writes a prose brief about a reference
kernel. Prose is the right medium for intent and for the GPU->TPU translation
argument, but it is the wrong medium for *numbers*: an LLM that paraphrases
`dim3 grid((n + 255) / 256)` as "one block per 256 elements" has already lost
the ceiling division, and every inference built on top of that -- occupancy,
working-set size, what the author believed fit in shared memory -- inherits
the error.

So the numbers are extracted mechanically first, and the analyzer is handed
them as ground truth. Grid and block dimensions, shared-memory byte counts,
tile constants and wmma fragment shapes are evidence about the decomposition;
they are not the analyzer's to restate.

It is deliberately conservative: it reports what it matched together with the
source line, so a human or an agent can check it, and it never guesses at a
value it could not find. A field it could not determine is `null`, never a
plausible-looking default.

Dialects (`--dialect`, default `auto`, which detects per unit):

  cuda    A regex extractor, not a C++ parser. Handles standalone `.cu`/`.cuh`
          files and CUDA carried inside a Python string for
          `torch.utils.cpp_extension.load_inline` (the KernelBench layout).
  triton  Python `ast` over `@triton.jit` kernels, `@triton.autotune` configs,
          `@triton.heuristics` and `kernel[grid](...)` launch sites.
  cutile  Python `ast` over NVIDIA cuTile (`import cuda.tile as ct`):
          `@ct.kernel` entry points, `ct.load`/`ct.store` tile shapes and
          `ct.launch(stream, grid, kernel, args)` sites.

The Python dialects are parsed, never imported or executed: neither triton nor
cuda.tile needs to be installed, and nothing in the reference ever runs.

Exit codes:
  0  parsed at least one unit with a kernel or a launch
  3  no GPU kernel source found in the given paths, in any dialect
"""

import argparse
import ast
import io
import tokenize
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
  r"\bdim3\s+(?P<name>\w+)\s*(?:\((?P<call>[^)]*)\)|=\s*dim3\s*\((?P<assign>[^)]*)\)|=\s*\{(?P<brace>[^}]*)\)|\{(?P<direct_brace>[^}]*)\})"
)
_INT_DECL = re.compile(
  r"\b(?:const\s+)?(?:int|unsigned(?:\s+int)?|size_t|u?int(?:8|16|32|64)_t)\s+(?P<name>\w+)\s*=\s*(?P<value>[^;,]+)[;,]"
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
# Python tile dialects: Triton and cuTile
#
# Both are Python, so they are parsed with `ast` rather than matched with
# regexes: decorators, keyword arguments and launch subscripts survive line
# breaks and formatting, and every value is reported as the source text the
# author wrote (`ast.unparse`), never evaluated.
# ---------------------------------------------------------------------------

# TPU counterparts for the tile-dialect mechanisms. A mechanism that is the
# same thing as a CUDA mechanism above reuses that entry's text verbatim, so
# the dialects never disagree about one TPU fact.
TRITON_MECHANISMS = {
  "program_id": r"\b{tl}\.program_id\s*\(",
  "tensor_cores": r"\b{tl}\.dot(?:_scaled)?\s*\(",
  "masked_memory_access": r"\b{tl}\.(?:load|store)\s*\([^()]*(?:\([^()]*\)[^()]*)*\bmask\s*=",
  "block_pointers": r"\b{tl}\.make_block_ptr\s*\(|\bmake_tensor_descriptor\s*\(|\bTensorDescriptor\b",
  "atomics": r"\b{tl}\.atomic_(?:add|cas|xchg|max|min|and|or|xor)\s*\(",
  "reductions": r"\b{tl}\.(?:sum|max|min|reduce|argmax|argmin|xor_sum)\s*\(",
  "select": r"\b{tl}\.where\s*\(",
  "pipeline_stages": r"\bnum_stages\s*=",
  "warp_count": r"\bnum_warps\s*=",
  "barrier": r"\b{tl}\.debug_barrier\s*\(",
  "inline_asm": r"\binline_asm_elementwise\s*\(",
  "fast_math_intrinsics": r"\blibdevice\.|\b{tl}\.extra\.|\bfast_(?:expf|dividef)\s*\(",
  "alignment_hints": r"\b{tl}\.(?:multiple_of|max_contiguous|assume)\s*\(",
}

TRITON_COUNTERPART = {
  "program_id": TPU_COUNTERPART["grid_stride_loop"],
  "tensor_cores": TPU_COUNTERPART["tensor_cores"],
  "masked_memory_access": "in-kernel masking of the edge block",
  "block_pointers": TPU_COUNTERPART["shared_memory"],
  "atomics": TPU_COUNTERPART["atomics"],
  "reductions": TPU_COUNTERPART["warp_shuffle"],
  "select": "jnp.where",
  "pipeline_stages": TPU_COUNTERPART["async_copy"],
  "warp_count": None,
  "barrier": TPU_COUNTERPART["syncthreads"],
  "inline_asm": None,
  "fast_math_intrinsics": TPU_COUNTERPART["fast_math_intrinsics"],
  "alignment_hints": TPU_COUNTERPART["restrict"],
}

CUTILE_MECHANISMS = {
  "program_id": r"\b{ct}\.(?:bid|num_blocks)\s*\(",
  "tensor_cores": r"\b{ct}\.(?:mma|mma_scaled|matmul)\s*\(",
  "tile_memory_access": r"\b{ct}\.(?:load|store)\s*\(",
  "gather_scatter": r"\b{ct}\.(?:gather|scatter|load_advanced_indexing|store_advanced_indexing)\s*\(",
  "atomics": r"\b{ct}\.atomic_(?:add|cas|xchg|max|min|and|or|xor)\s*\(",
  "reductions": r"\b{ct}\.(?:sum|max|min|prod|argmax|argmin|reduce)\s*\(",
  "scans": r"\b{ct}\.(?:cumsum|cumprod|scan)\s*\(",
  "select": r"\b{ct}\.where\s*\(",
  "occupancy_hints": r"\b(?:occupancy|num_ctas|num_worker_warps)\s*=",
  "alignment_hints": r"\b{ct}\.assume_divisible_by\s*\(",
  "grid_dependency_control": r"\b{ct}\.grid_dependency_control_\w+\s*\(",
}

CUTILE_COUNTERPART = {
  "program_id": TPU_COUNTERPART["grid_stride_loop"],
  "tensor_cores": TPU_COUNTERPART["tensor_cores"],
  "tile_memory_access": TPU_COUNTERPART["shared_memory"],
  "gather_scatter": "data-dependent indexing; record the purpose, the planner picks the TPU form",
  "atomics": TPU_COUNTERPART["atomics"],
  "reductions": TPU_COUNTERPART["warp_shuffle"],
  "scans": "a jnp scan over the axis",
  "select": "jnp.where",
  "occupancy_hints": None,
  "alignment_hints": TPU_COUNTERPART["restrict"],
  "grid_dependency_control": TPU_COUNTERPART["dynamic_parallelism"],
}


def strip_py_comments(text):
  """Blanks `#` comments while preserving line numbering and strings."""
  try:
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
  except (tokenize.TokenError, IndentationError, SyntaxError):
    return text
  lines = text.splitlines(keepends=True)
  for tok in tokens:
    if tok.type == tokenize.COMMENT:
      row, col = tok.start
      line = lines[row - 1]
      end = col + len(tok.string)
      lines[row - 1] = line[:col] + " " * (end - col) + line[end:]
  return "".join(lines)


def module_aliases(tree, module):
  """Names bound to `module` by imports (`import triton.language as tl`)."""
  aliases = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for a in node.names:
        if a.name == module:
          aliases.add(a.asname or module)
    elif isinstance(node, ast.ImportFrom) and node.module:
      parent, _, leaf = module.rpartition(".")
      for a in node.names:
        if parent and node.module == parent and a.name == leaf:
          aliases.add(a.asname or leaf)
  return aliases


def imported_names(tree, module):
  """Names imported *from* `module` (`from triton import jit` -> {"jit"})."""
  names = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module == module:
      for a in node.names:
        names[a.asname or a.name] = a.name
  return names


def _src(node):
  return ast.unparse(node) if node is not None else None


def _call_name(node):
  """Dotted name of a call or decorator target: `triton.autotune(...)` -> 'triton.autotune'."""
  target = node.func if isinstance(node, ast.Call) else node
  return _src(target)


def _kwargs(call):
  return {k.arg: _src(k.value) for k in call.keywords if k.arg}


def _param_records(fn, constant_marker, tree):
  """Parameters, flagging compile-time constants.

  An annotation may be a module-level alias of the marker type
  (`ConstInt = ct.Constant[int]`, as NVIDIA's cuTile samples write it), so
  the alias's value is checked too.
  """
  aliases = {
    t.id: _src(n.value)
    for n in tree.body
    if isinstance(n, ast.Assign)
    for t in n.targets
    if isinstance(t, ast.Name)
  }
  args = fn.args
  positional = args.posonlyargs + args.args
  defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
  params = []
  for arg, default in zip(positional, defaults):
    annotation = _src(arg.annotation)
    params.append(
      {
        "name": arg.arg,
        "annotation": annotation,
        "compile_time_constant": bool(
          annotation
          and constant_marker in annotation + aliases.get(annotation, "")
        ),
        "default": _src(default),
      }
    )
  return params


def _assignments(tree):
  """name -> [(line, value_source)] for every simple assignment."""
  table = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
      for target in node.targets:
        if isinstance(target, ast.Name):
          table.setdefault(target.id, []).append((node.lineno, _src(node.value)))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      if node.value is not None:
        table.setdefault(node.target.id, []).append(
          (node.lineno, _src(node.value))
        )
  return table


def _resolve_py(node, assignments):
  """Annotates a launch grid with the assignment it was built from.

  `kernel[grid](...)` says nothing on its own; `grid = lambda META: (...)`
  three lines above is the fact. The nearest assignment at or before the
  launch wins; a name assigned only after it, or never, resolves to `null`.
  """
  raw = _src(node)
  if not isinstance(node, ast.Name):
    return {"raw": raw, "resolved": raw}
  candidates = [
    (line, value)
    for line, value in assignments.get(node.id, [])
    if line <= node.lineno
  ]
  if not candidates:
    return {"raw": raw, "resolved": None}
  line, value = max(candidates)
  return {"raw": raw, "resolved": value, "declared_line": line}


def find_py_constants(tree):
  """Module-level UPPER_CASE assignments -- the tile sizes the author fixed."""
  constants = []
  for node in tree.body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      value = node.value
      for target in targets:
        if (
          isinstance(target, ast.Name)
          and re.fullmatch(r"[A-Z_][A-Z0-9_]*", target.id)
          and value is not None
        ):
          constants.append(
            {
              "kind": "module_constant",
              "name": target.id,
              "value": _src(value),
              "line": node.lineno,
            }
          )
  return constants


def py_census(text, mechanisms, counterparts, aliases):
  """Runs one dialect's mechanism regexes, once per import alias."""
  found = {}
  for name, template in mechanisms.items():
    # Keyed by offset: an alias-free pattern (`num_warps=`) matches the same
    # text once per alias, and must be counted once.
    starts = set()
    for alias in aliases:
      pattern = template.format(tl=re.escape(alias), ct=re.escape(alias))
      starts.update(m.start() for m in re.finditer(pattern, text, re.S))
    if starts:
      found[name] = {
        "count": len(starts),
        "first_line": line_of(text, min(starts)),
        "tpu_counterpart": counterparts.get(name),
      }
  return found


def _axis_calls(tree, aliases, names):
  calls = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
      if (
        node.func.attr in names
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in aliases
      ):
        axis = _src(node.args[0]) if node.args else _kwargs(node).get("axis")
        calls.append(
          {"call": node.func.attr, "axis": axis, "line": node.lineno}
        )
  return calls


def analyze_triton(source, origin):
  clean = strip_py_comments(source)
  tree = ast.parse(clean)
  triton_aliases = module_aliases(tree, "triton") or {"triton"}
  tl_aliases = module_aliases(tree, "triton.language") or {"tl"}
  from_triton = imported_names(tree, "triton")

  def is_triton(dec, leaf):
    name = _call_name(dec)
    if name in {f"{a}.{leaf}" for a in triton_aliases}:
      return True
    return from_triton.get(name) == leaf

  kernels, autotune, heuristics = [], [], []
  for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef):
      continue
    if not any(is_triton(d, "jit") for d in node.decorator_list):
      continue
    kernels.append(
      {
        "name": node.name,
        "line": node.lineno,
        "params": _param_records(node, "constexpr", tree),
        "decorators": [_src(d) for d in node.decorator_list],
      }
    )
    for dec in node.decorator_list:
      if not isinstance(dec, ast.Call):
        continue
      if is_triton(dec, "autotune"):
        kw = {k.arg: k.value for k in dec.keywords if k.arg}
        configs_node = kw.get("configs", dec.args[0] if dec.args else None)
        configs = None
        if isinstance(configs_node, (ast.List, ast.Tuple)):
          configs = []
          for c in configs_node.elts:
            record = {"raw": _src(c), "line": c.lineno}
            if isinstance(c, ast.Call):
              meta = c.args[0] if c.args else None
              if isinstance(meta, ast.Dict):
                record["meta"] = {
                  k.value
                  if isinstance(k, ast.Constant) and isinstance(k.value, str)
                  else _src(k): _src(v)
                  for k, v in zip(meta.keys, meta.values)
                }
              record.update(_kwargs(c))
            configs.append(record)
        autotune.append(
          {
            "kernel": node.name,
            "line": dec.lineno,
            "configs": configs,
            "configs_raw": None if configs is not None else _src(configs_node),
            "key": _src(kw.get("key")),
            "other": {
              k: _src(v) for k, v in kw.items() if k not in ("configs", "key")
            },
          }
        )
      elif is_triton(dec, "heuristics"):
        heuristics.append(
          {
            "kernel": node.name,
            "line": dec.lineno,
            "raw": _src(dec.args[0]) if dec.args else _src(dec),
          }
        )

  kernel_names = {k["name"] for k in kernels}
  assignments = _assignments(tree)
  launches = []
  for node in ast.walk(tree):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Subscript)
      and isinstance(node.func.value, ast.Name)
      and node.func.value.id in kernel_names
    ):
      kw = _kwargs(node)
      launches.append(
        {
          "kernel": node.func.value.id,
          "line": node.lineno,
          "grid": _resolve_py(node.func.slice, assignments),
          "args": [_src(a) for a in node.args],
          "launch_kwargs": kw,
          "raw": _src(node.func) + "(...)",
        }
      )

  tile_shapes = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
      owner = node.func.value
      if not (isinstance(owner, ast.Name) and owner.id in tl_aliases):
        continue
      if node.func.attr == "arange" and len(node.args) == 2:
        start, end = node.args
        zero = isinstance(start, ast.Constant) and start.value == 0
        tile_shapes.append(
          {
            "call": "arange",
            "extent": _src(end) if zero else f"{_src(end)} - {_src(start)}",
            "raw": _src(node),
            "line": node.lineno,
          }
        )
      elif node.func.attr == "make_block_ptr":
        tile_shapes.append(
          {
            "call": "make_block_ptr",
            "block_shape": _kwargs(node).get("block_shape"),
            "order": _kwargs(node).get("order"),
            "line": node.lineno,
          }
        )

  return {
    "dialect": "triton",
    "origin": origin,
    "lines": source.count("\n") + 1,
    "kernels": kernels,
    "launches": launches,
    "constants": find_py_constants(tree),
    "autotune_configs": autotune,
    "heuristics": heuristics,
    "program_ids": _axis_calls(tree, tl_aliases, {"program_id", "num_programs"}),
    "tile_shapes": sorted(tile_shapes, key=lambda t: t["line"]),
    "mechanisms": py_census(
      clean, TRITON_MECHANISMS, TRITON_COUNTERPART, tl_aliases
    ),
  }


def analyze_cutile(source, origin):
  clean = strip_py_comments(source)
  tree = ast.parse(clean)
  ct_aliases = module_aliases(tree, "cuda.tile") or {"ct"}

  def is_ct(node, leaf):
    return _call_name(node) in {f"{a}.{leaf}" for a in ct_aliases}

  kernels = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef):
      continue
    decs = [d for d in node.decorator_list if is_ct(d, "kernel")]
    if not decs:
      continue
    kernels.append(
      {
        "name": node.name,
        "line": node.lineno,
        "params": _param_records(node, "Constant", tree),
        "decorator_options": _kwargs(decs[0])
        if isinstance(decs[0], ast.Call)
        else {},
      }
    )

  assignments = _assignments(tree)
  launches, tile_shapes = [], []
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    if is_ct(node, "launch"):
      args = node.args
      kernel = args[2] if len(args) > 2 else None
      launches.append(
        {
          "kernel": _src(kernel),
          "line": node.lineno,
          "grid": _resolve_py(args[1], assignments) if len(args) > 1 else None,
          "stream": _src(args[0]) if args else None,
          "args": _src(args[3]) if len(args) > 3 else None,
          "raw": _src(node),
        }
      )
    elif any(is_ct(node, op) for op in ("load", "store", "gather", "scatter")):
      kw = _kwargs(node)
      tile_shapes.append(
        {
          "call": node.func.attr,
          "shape": kw.pop("shape", None),
          "index": kw.pop("index", None),
          "options": {k: v for k, v in kw.items() if k not in ("tile",)},
          "line": node.lineno,
        }
      )

  return {
    "dialect": "cutile",
    "origin": origin,
    "lines": source.count("\n") + 1,
    "kernels": kernels,
    "launches": launches,
    "constants": find_py_constants(tree),
    "program_ids": _axis_calls(tree, ct_aliases, {"bid", "num_blocks"}),
    "tile_shapes": sorted(tile_shapes, key=lambda t: t["line"]),
    "mechanisms": py_census(
      clean, CUTILE_MECHANISMS, CUTILE_COUNTERPART, ct_aliases
    ),
  }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

DIALECTS = ("cuda", "triton", "cutile")

_TRITON_DETECT = re.compile(
  r"^\s*(?:import\s+triton\b|from\s+triton\b)|@\s*(?:triton\.)?jit\b", re.M
)
_CUTILE_DETECT = re.compile(
  r"^\s*(?:import\s+cuda\.tile\b|from\s+cuda\.tile\b|from\s+cuda\s+import\s+tile\b)",
  re.M,
)


def analyze_source(source, origin):
  """Runs every CUDA extractor over one translation unit."""
  clean = strip_comments(source)
  constants, template_params = find_constants(clean)
  return {
    "dialect": "cuda",
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


def detect_dialects(path, text):
  """Which dialects a file carries. One file can hold several."""
  if path.suffix != ".py":
    return ["cuda"]
  found = []
  if _TRITON_DETECT.search(text):
    found.append("triton")
  if _CUTILE_DETECT.search(text):
    found.append("cutile")
  if extract_inline_cuda(text):
    found.append("cuda")
  return found


def analyze_path(path, dialect="auto"):
  """Returns a list of unit records for one file (a .py may hold several)."""
  text = Path(path).read_text(errors="replace")
  dialects = detect_dialects(path, text) if dialect == "auto" else [dialect]
  units = []
  for d in dialects:
    if d == "cuda":
      if path.suffix == ".py":
        for chunk in extract_inline_cuda(text):
          units.append(
            analyze_source(
              chunk["source"],
              f"{path}:{chunk['variable']}@L{chunk['start_line']}",
            )
          )
      else:
        units.append(analyze_source(text, str(path)))
      continue
    analyzer = analyze_triton if d == "triton" else analyze_cutile
    try:
      units.append(analyzer(text, str(path)))
    except SyntaxError as e:
      print(f"Could not parse {path} as {d}: {e}", file=sys.stderr)
  return units


def main():
  parser = argparse.ArgumentParser(
    description="Extract structural facts from GPU kernel source deterministically."
  )
  parser.add_argument("paths", nargs="+")
  parser.add_argument(
    "--dialect",
    choices=("auto",) + DIALECTS,
    default="auto",
    help="force one dialect instead of detecting it per file",
  )
  parser.add_argument("--json", help="write the report to this path")
  args = parser.parse_args()

  units = []
  for raw in args.paths:
    path = Path(raw)
    if not path.is_file():
      print(f"Not a file, skipping: {path}", file=sys.stderr)
      continue
    try:
      units.extend(analyze_path(path, args.dialect))
    except OSError as e:
      print(f"Could not read {path}: {e}", file=sys.stderr)

  units = [u for u in units if u["kernels"] or u["launches"]]
  if not units:
    print(
      "No GPU kernels found in the given paths "
      f"(dialects tried: {args.dialect}).",
      file=sys.stderr,
    )
    sys.exit(3)

  report = {
    "units": units,
    "summary": {
      "dialects": sorted({u["dialect"] for u in units}),
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
    f"dialects={','.join(s['dialects'])} "
    f"kernels={s['kernel_count']} launches={s['launch_count']} "
    f"mechanisms={len(s['mechanisms_present'])} "
    f"non-portable={len(s['mechanisms_without_tpu_counterpart'])}",
    file=sys.stderr,
  )
  sys.exit(0)


if __name__ == "__main__":
  main()
