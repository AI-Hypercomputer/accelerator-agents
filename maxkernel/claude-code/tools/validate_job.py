#!/usr/bin/env python3
"""Validates a MaxKernel job file and reports the plan it implies.

A job file is the single declaration that starts a run. It replaces inference:
`tools/classify_inputs.py` has to guess which of four `.py` files in a directory
is the thing being converted, and refuses (exit 3) when it cannot tell. A job
file says so outright, and this tool checks that what it says is coherent
*before* a TPU is touched.

The four supported shapes:

    JAX           -> Pallas
    PyTorch       -> Pallas
    PyTorch       -> Pallas,  with a CUDA kernel as an advisory reference
    CUDA          -> Pallas
    specification -> Pallas,  when there is no source file at all and the
                              baseline has to be synthesized from a description

Any of them can target the backward pass instead by adding a `backward`
object: either the input *is* a backward implementation (`backward.from:
"source"`), or it is the forward and Phase 0.85 derives the backward from its
port with `jax.vjp` (`backward.from: "autodiff"`; both passes are optimized
unless `backward.optimize_forward` is false).

Code from the wild -- a package with relative imports, GPU-only dependencies
and no `get_inputs()` -- declares `input.extraction`. Phase 0.15 then has an
agent write a standalone PyTorch `source.py` from it, built around the shapes in
`input.configs`, and stops for the user to approve it before anything else
runs. A PyTorch `input.path` that is a directory requires it: the capture tools
load one file.

Two rules govern the whole schema:

  * **Redundant fields are checksums, not duplication.** `has_reference` repeats
    what `len(references)` already says. That is deliberate: a job that declares
    a reference and forgets the path is a mistake worth catching here rather
    than discovering as a silently reference-less run four phases later.
  * **Every contradiction fails loudly.** A CUDA input cannot be converted by
    torchax; a run with no JAX conversion has no `base.py` and therefore cannot
    use the internal timing harness. Both are legal intentions and illegal
    accidents, so each must be declared explicitly rather than inferred.

Exit codes:
  0  valid; the implied plan is printed
  1  invalid: contradictions or missing required fields
  2  valid shape, but a referenced path does not exist
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from input_spec import validate_input_spec  # noqa: E402

SCHEMA = "maxkernel.job/1"

# GPU kernel dialects; "gpu" is any other (CUTLASS/CuTe, ...). As an input
# they are ported by an agent, never traced by torchax, and only Triton has an
# oracle (its CPU interpreter). As a reference the type becomes
# `references[].kind` in state.json.
GPU_TYPES = {"cuda", "triton", "cutile", "cutedsl", "gpu"}
INPUT_TYPES = GPU_TYPES | {"jax", "pytorch", "specification"}
SYNTHESIS_TARGETS = {"pytorch", "jax"}
REFERENCE_TYPES = GPU_TYPES | {"pallas", "jax", "pytorch"}
CONVERSION_METHODS = {"torchax", "llm_port", "none"}
RELATIONSHIPS = {"same_operation", "same_family", "unknown"}
BACKWARD_FROM = {"source", "autodiff"}
BACKWARD_KEYS = {"from", "wrt", "optimize_forward", "forward_context"}
EXTRACTION_KEYS = {"entry_point", "notes", "user_confirms_extraction"}


class JobError(Exception):
  pass


def resolve_extraction_entry(input_path, entry_point):
  """Resolves `input.extraction.entry_point` against `input.path`.

  The file part is relative to `input.path` itself -- not to its parent, not
  to the repository root. For a single-file `input.path` it is that file's
  name, or omitted (`"Name"`). Returns (relpath, name, error): `relpath` is
  what the orchestrator joins onto the copy of `input.path` under
  `<run_dir>/original/`, so the extracting agent never joins paths itself.
  """
  file_part, sep, name = entry_point.rpartition(":")
  if not sep:
    file_part, name = "", entry_point
  if not name:
    return None, None, (
      f"input.extraction.entry_point {entry_point!r} names no class or "
      "function; the form is 'path/relative/to/input.path.py:Name'"
    )

  if input_path.is_file():
    if file_part not in ("", input_path.name):
      return None, None, (
        f"input.extraction.entry_point {entry_point!r}: input.path is the "
        f"single file {input_path.name}, so the file part must be "
        f"{input_path.name!r} or omitted ('{name}')"
      )
    relpath, target = "", input_path
  else:
    if not file_part:
      return None, None, (
        f"input.extraction.entry_point {entry_point!r} has no file part, but "
        f"input.path {input_path} is a directory; use "
        "'path/relative/to/input.path.py:Name'"
      )
    relpath, target = file_part, input_path / file_part
    if not target.is_file():
      hint = ""
      parts = Path(file_part).parts
      if len(parts) > 1 and parts[0] == input_path.name and (
        input_path / Path(*parts[1:])
      ).is_file():
        hint = (
          f" The path is relative to input.path, which already ends in "
          f"{input_path.name}/ -- use '{Path(*parts[1:]).as_posix()}:{name}'."
        )
      elif (input_path.parent / file_part).is_file():
        hint = (
          " That path exists relative to input.path's parent; the file "
          "part is relative to input.path itself."
        )
      else:
        found = sorted(input_path.rglob(Path(file_part).name))
        if len(found) == 1:
          rel = found[0].relative_to(input_path).as_posix()
          hint = f" Did you mean '{rel}:{name}'?"
      return None, None, (
        f"input.extraction.entry_point {entry_point!r}: {target} does not "
        f"exist.{hint}"
      )

  top = name.split(".")[0]
  src = target.read_text(errors="replace")
  if not re.search(rf"^\s*(?:class|def)\s+{re.escape(top)}\b", src, re.M):
    return None, None, (
      f"input.extraction.entry_point {entry_point!r}: {target} defines no "
      f"class or function named {top!r}"
    )
  return relpath, name, None


def _get(d, path, default=None):
  cur = d
  for part in path.split("."):
    if not isinstance(cur, dict) or part not in cur:
      return default
    cur = cur[part]
  return cur


def validate(job, job_path):
  """Returns (errors, warnings, plan). Errors make the job unusable."""
  errors, warnings = [], []
  base = Path(job_path).resolve().parent

  # ---- schema ---------------------------------------------------------
  if job.get("schema") != SCHEMA:
    warnings.append(
      f"schema is {job.get('schema')!r}, expected {SCHEMA!r}; "
      "validating as v1 anyway"
    )

  # ---- input ----------------------------------------------------------
  inp = job.get("input")
  if not isinstance(inp, dict):
    errors.append("`input` is required and must be an object")
    return errors, warnings, None

  itype = inp.get("type")
  if itype not in INPUT_TYPES:
    errors.append(
      f"input.type must be one of {sorted(INPUT_TYPES)}, got {itype!r}"
    )

  spec = inp.get("specification")
  ipath = inp.get("path")

  if itype == "specification":
    # No source file: the baseline is synthesized from a description. This
    # mode is structurally weaker than every other one and the schema makes
    # that impossible to enter by accident.
    if ipath:
      errors.append(
        "input.type is 'specification' but input.path is set. A "
        "specification job has no source file -- if you have one, give "
        "its real type instead."
      )
    if not isinstance(spec, dict):
      errors.append(
        "input.type is 'specification' requires an input.specification "
        "object describing what to compute."
      )
    else:
      if not spec.get("description"):
        errors.append(
          "input.specification.description is required: one "
          "or two sentences naming the computation."
        )
      tensors = spec.get("tensors")
      if not isinstance(tensors, list) or not tensors:
        errors.append(
          "input.specification.tensors is required and must be a "
          "non-empty list. Shapes and dtypes cannot be invented: they "
          "define the problem, and general_rules #6 forbids "
          "substituting them."
        )
      else:
        for i, t in enumerate(tensors):
          if not isinstance(t, dict):
            errors.append(f"input.specification.tensors[{i}] must be an object")
            continue
          for field in ("name", "shape", "dtype", "role"):
            if not t.get(field):
              errors.append(
                f"input.specification.tensors[{i}] is missing "
                f"{field!r} (role is 'input', 'output' or 'parameter')"
              )
      target = spec.get("synthesize_as", "pytorch")
      if target not in SYNTHESIS_TARGETS:
        errors.append(
          f"input.specification.synthesize_as must be one of "
          f"{sorted(SYNTHESIS_TARGETS)}, got {target!r}"
        )
      if spec.get("user_confirms_baseline") is not True:
        errors.append(
          "input.specification.user_confirms_baseline must be true. "
          "In this mode the agent writes the baseline it will then be "
          "scored against, so the run pauses for you to read and "
          "approve that baseline before any optimization begins. "
          "Setting this field is how you accept that step exists; the "
          "run will still stop and wait for the actual approval."
        )
  elif not ipath:
    errors.append("input.path is required")

  # ---- declared entry point and inputs ---------------------------------
  for field in ("entry_point", "inputs_fn", "init_inputs_fn"):
    v = inp.get(field)
    if v is not None and not (isinstance(v, str) and v):
      errors.append(f"input.{field} must be a non-empty string, got {v!r}")
  errors.extend(validate_input_spec(inp, where="input"))
  has_configs = bool(inp.get("configs"))
  if has_configs and inp.get("inputs_fn"):
    errors.append(
      "input.configs and input.inputs_fn are both set. They are two sources "
      "for the same forward arguments; keep the one you mean."
    )
  if inp.get("init_inputs_fn") and (
    "init_args" in inp or "init_kwargs" in inp
  ):
    errors.append(
      "input.init_inputs_fn and input.init_args/init_kwargs are both set. "
      "They are two sources for the same constructor arguments; keep one."
    )

  extraction = inp.get("extraction")
  if extraction is not None:
    if not isinstance(extraction, dict):
      errors.append("input.extraction must be an object")
      extraction = {}
    unknown = sorted(set(extraction) - EXTRACTION_KEYS)
    if unknown:
      errors.append(
        f"input.extraction has unknown keys {unknown}; allowed: "
        f"{sorted(EXTRACTION_KEYS)}"
      )
    if itype != "pytorch":
      errors.append(
        f"input.extraction is set for a {itype!r} input. Extraction writes a "
        "standalone PyTorch module from a PyTorch codebase; it only applies "
        "to input.type 'pytorch'."
      )
    if not (
      isinstance(extraction.get("entry_point"), str)
      and extraction["entry_point"]
    ):
      errors.append(
        "input.extraction.entry_point is required: the class or function in "
        "the original code being converted, as 'file.py:Name' with the file "
        "path relative to input.path, e.g. 'sonicmoe/moe.py:MoE' for a "
        "repository root. It is what the extracting agent reproduces, so it "
        "cannot be guessed."
      )
    if not has_configs:
      errors.append(
        "input.extraction requires input.configs. The extracted source's "
        "get_inputs() is built from the declared shapes; an agent may not "
        "choose the problem dimensions (general_rules #6)."
      )
    for field in ("entry_point", "inputs_fn", "init_inputs_fn"):
      if inp.get(field):
        errors.append(
          f"input.{field} is set alongside input.extraction. The extracted "
          "source always exposes `Model` and is driven by input.configs / "
          "init_args / init_kwargs; name the original entry point in "
          "input.extraction.entry_point instead."
        )
    if extraction.get("user_confirms_extraction") is not True:
      errors.append(
        "input.extraction.user_confirms_extraction must be true. An agent "
        "writes the source this run is scored against, so the run pauses for "
        "you to read and approve it before anything else happens. Setting "
        "this field is how you accept that step exists; the run still stops "
        "for the actual approval."
      )

  # ---- references -----------------------------------------------------
  refs = job.get("references", [])
  has_ref = job.get("has_reference")
  if has_ref is None:
    errors.append("has_reference is required (true or false)")
  if not isinstance(refs, list):
    errors.append("references must be a list")
    refs = []

  # The checksum. A declared reference with no path is the mistake this
  # redundancy exists to catch.
  if has_ref is True and not refs:
    errors.append(
      "has_reference is true but references is empty. Either give the "
      "reference path, or set has_reference to false — otherwise the run "
      "would silently proceed with no reference at all."
    )
  if has_ref is False and refs:
    errors.append(
      f"has_reference is false but {len(refs)} reference(s) are listed. "
      "The references would be ignored; say which you meant."
    )
  if len(refs) > 1:
    errors.append(
      f"{len(refs)} references are listed; a run takes at most one. Keep the "
      "one the planner should learn from."
    )

  for i, r in enumerate(refs):
    if not isinstance(r, dict):
      errors.append(f"references[{i}] must be an object")
      continue
    if r.get("type") not in REFERENCE_TYPES:
      errors.append(
        f"references[{i}].type must be one of {sorted(REFERENCE_TYPES)}"
      )
    if not r.get("path"):
      errors.append(f"references[{i}].path is required")
    rel = r.get("relationship", "unknown")
    if rel not in RELATIONSHIPS:
      errors.append(
        f"references[{i}].relationship must be one of {sorted(RELATIONSHIPS)}"
      )

  # ---- jax conversion -------------------------------------------------
  needs = job.get("needs_jax_conversion")
  if needs is None:
    errors.append("needs_jax_conversion is required (true or false)")
  method = _get(
    job, "jax_conversion.method", "none" if needs is False else None
  )
  emit = _get(job, "jax_conversion.emit_readable_reference", True)

  if needs is True:
    if method not in CONVERSION_METHODS - {"none"}:
      errors.append(
        "needs_jax_conversion is true, so jax_conversion.method must be "
        "'torchax' or 'llm_port'"
      )
    if itype == "jax":
      errors.append(
        "input.type is 'jax' and needs_jax_conversion is true. A JAX "
        "input is already the reference; there is nothing to convert."
      )
    if (
      itype == "specification"
      and _get(job, "input.specification.synthesize_as", "pytorch") == "jax"
    ):
      errors.append(
        "input.specification.synthesize_as is 'jax' and "
        "needs_jax_conversion is true. A synthesized JAX baseline IS "
        "base.py; there is nothing further to convert."
      )
    if extraction is not None and method == "torchax":
      warnings.append(
        "input.extraction with jax_conversion.method 'torchax': torchax "
        "cannot trace data-dependent shapes (MoE routing, .tolist(), "
        "nonzero, boolean indexing), which code from the wild often has. If "
        "step 0 cannot run, the run falls back to llm_port; choose "
        "'llm_port' now if you know the source has them."
      )
    if itype in GPU_TYPES and method == "torchax":
      errors.append(
        f"jax_conversion.method 'torchax' cannot convert a {itype} input. "
        "torchax traces PyTorch modules; GPU kernels have to be ported by "
        "an agent — use 'llm_port'."
      )
  elif needs is False:
    if method not in (None, "none"):
      errors.append(
        f"needs_jax_conversion is false but jax_conversion.method is "
        f"{method!r}; set it to 'none' or omit it"
      )
    if (
      itype == "specification"
      and _get(job, "input.specification.synthesize_as", "pytorch") == "pytorch"
    ):
      errors.append(
        "synthesize_as is 'pytorch' but needs_jax_conversion is false. "
        "A synthesized PyTorch baseline still has to become base.py "
        "like any other PyTorch input -- set needs_jax_conversion true "
        "with a method, or synthesize_as 'jax' to skip the step."
      )
    elif itype not in ("jax", "specification"):
      # This is the constraint that otherwise fails four phases later.
      errors.append(
        f"input.type is {itype!r} with needs_jax_conversion false, so "
        "no base.py is produced. tools/assemble_test_harness.py binds "
        "`base_computation` from a file defining `computation`, so the "
        "internal timing harness cannot run. Either set "
        "needs_jax_conversion true, or set "
        "evaluation.compare_against_source true to measure externally "
        "with evaluation/compare_kernel.py."
      )

  if needs is True and method == "torchax" and emit is False:
    msg = (
      "jax_conversion.method is 'torchax' with emit_readable_reference "
      "false: torchax produces the oracle but no base.py, so the "
      "internal timing harness cannot run."
    )
    if _get(job, "evaluation.compare_against_source") is True:
      warnings.append(
        msg + " evaluation.compare_against_source is set, so "
        "performance will be measured externally."
      )
    else:
      errors.append(
        msg + " Set emit_readable_reference true, or set "
        "evaluation.compare_against_source true."
      )

  if (
    itype == "specification"
    and _get(job, "input.specification.synthesize_as", "pytorch") == "jax"
    and _get(job, "evaluation.compare_against_source") is True
  ):
    errors.append(
      "evaluation.compare_against_source is true, but synthesize_as is "
      "'jax', so there is no PyTorch source to compare against. Use "
      "synthesize_as 'pytorch' if you want that measurement."
    )

  if (
    itype in GPU_TYPES
    and _get(job, "evaluation.compare_against_source") is True
  ):
    errors.append(
      f"evaluation.compare_against_source is true for a {itype} input. A TPU "
      "host cannot execute GPU kernels at speed, so there is nothing to "
      "compare against. Every number for a GPU input is measured against "
      "the JAX port."
    )

  # ---- correctness / target ------------------------------------------
  atol = _get(job, "correctness.atol", 1e-2)
  rtol = _get(job, "correctness.rtol", 1e-2)
  for name, v in (("atol", atol), ("rtol", rtol)):
    if not isinstance(v, (int, float)) or v <= 0:
      errors.append(f"correctness.{name} must be a positive number, got {v!r}")
    elif v > 0.5:
      warnings.append(
        f"correctness.{name} is {v}; a tolerance that loose accepts "
        "almost any output and will not catch a real error"
      )

  iters = _get(job, "loop.max_iterations", 5)
  if not isinstance(iters, int) or iters < 1:
    errors.append(
      f"loop.max_iterations must be a positive integer, got {iters!r}"
    )

  # ---- backward --------------------------------------------------------
  # A `backward` object makes this a backward-pass run; without one the run
  # optimizes the input as written.
  bwd = job.get("backward")
  bfrom = None
  if bwd is not None:
    if not isinstance(bwd, dict):
      errors.append(
        "backward must be an object with `from`: 'source' (the input is a "
        "backward implementation) or 'autodiff' (the input is the forward; "
        "Phase 0.85 wraps it in jax.vjp)."
      )
    else:
      bfrom = bwd.get("from")
      wrt = bwd.get("wrt")
      unknown = sorted(set(bwd) - BACKWARD_KEYS)
      if unknown:
        errors.append(
          f"backward has unknown keys {unknown}; allowed: "
          f"{sorted(BACKWARD_KEYS)}"
        )
      if bfrom not in BACKWARD_FROM:
        errors.append(
          f"backward.from must be one of {sorted(BACKWARD_FROM)}, got {bfrom!r}"
        )
      if wrt is not None and bfrom != "autodiff":
        errors.append(
          "backward.wrt is only meaningful with backward.from 'autodiff'. "
          "A backward source already decides which gradients it returns."
        )
      if wrt is not None and (
        not isinstance(wrt, list)
        or not wrt
        or not all(isinstance(w, str) and w for w in wrt)
      ):
        errors.append(
          "backward.wrt must be a non-empty list of argument names; omit it "
          "to differentiate with respect to every floating-point input"
        )
      optfwd = bwd.get("optimize_forward")
      if optfwd is not None and not isinstance(optfwd, bool):
        errors.append(
          f"backward.optimize_forward must be true or false, got {optfwd!r}"
        )
      if optfwd is not None and bfrom != "autodiff":
        errors.append(
          "backward.optimize_forward is only meaningful with backward.from "
          "'autodiff'. A backward source has no forward to optimize."
        )
      fctx = bwd.get("forward_context")
      if fctx is not None and not (isinstance(fctx, str) and fctx):
        errors.append(
          f"backward.forward_context must be a path string, got {fctx!r}"
        )
      if fctx is not None and bfrom != "source":
        errors.append(
          "backward.forward_context is only meaningful with backward.from "
          "'source'. With 'autodiff' the input already is the forward."
        )
      if (
        bfrom == "autodiff"
        and needs is True
        and method == "torchax"
        and emit is False
      ):
        errors.append(
          "backward.from is 'autodiff', which differentiates base.py, but "
          "emit_readable_reference is false so no base.py is produced. Set "
          "emit_readable_reference true."
        )
      if (
        bfrom == "autodiff"
        and _get(job, "evaluation.compare_against_source") is True
      ):
        errors.append(
          "backward.from is 'autodiff', so the result computes gradients, "
          "but evaluation.compare_against_source would compare it against "
          "the forward source. Set compare_against_source false."
        )

  if errors:
    return errors, warnings, None

  # ---- path existence (non-fatal for schema, fatal for a run) ---------
  missing = []

  def _resolve(p):
    q = Path(p)
    return q if q.is_absolute() else (base / q)

  # A specification job has no input.path by design; there is nothing to
  # resolve until the baseline has been synthesized.
  if ipath and not _resolve(ipath).exists():
    missing.append(f"input.path: {_resolve(ipath)}")
  elif (
    ipath
    and itype == "pytorch"
    and extraction is None
    and _resolve(ipath).is_dir()
  ):
    errors.append(
      f"input.path {_resolve(ipath)} is a directory. The PyTorch capture "
      "tools execute a single self-contained file, so a package (relative "
      "imports, GPU-only dependencies) cannot be run as is. Either point "
      "input.path at one self-contained file, or declare input.extraction "
      "so Phase 0.15 writes one for you to approve."
    )
    return errors, warnings, None
  extraction_entry = None
  if ipath and extraction is not None and _resolve(ipath).exists():
    relpath, name, err = resolve_extraction_entry(
      _resolve(ipath), extraction["entry_point"]
    )
    if err:
      errors.append(err)
      return errors, warnings, None
    extraction_entry = {"relpath": relpath, "name": name}
  for i, r in enumerate(refs):
    if not _resolve(r["path"]).exists():
      missing.append(f"references[{i}].path: {_resolve(r['path'])}")
  fctx = _get(job, "backward.forward_context")
  if fctx and not _resolve(fctx).exists():
    missing.append(f"backward.forward_context: {_resolve(fctx)}")

  plan = build_plan(job, itype, needs, method, emit, refs, bfrom, extraction)
  if bfrom == "autodiff":
    plan["optimize_forward"] = _get(job, "backward.optimize_forward", True)
  plan["missing_paths"] = missing
  # Resolved against the real input.path. `relpath` is relative to it ("" when
  # input.path is the file itself). Null until input.path exists.
  plan["extraction_entry"] = extraction_entry
  return errors, warnings, plan


def build_plan(
  job, itype, needs, method, emit, refs, bfrom=None, extraction=None
):
  """What the orchestrator will actually do, derived from the declaration."""
  synth = (
    _get(job, "input.specification.synthesize_as", "pytorch")
    if itype == "specification"
    else None
  )
  # After synthesis a specification job behaves exactly like an input of the
  # synthesized kind, so the rest of the plan is derived from that.
  effective = synth if itype == "specification" else itype

  oracle = {
    (
      "pytorch",
      "torchax",
    ): "tools/torchax_oracle.py (mechanical, with step-0 check vs eager)",
    (
      "pytorch",
      "llm_port",
    ): "tools/capture_torch_golden.py (eager PyTorch on CPU)",
    (
      "triton",
      "llm_port",
    ): "tools/capture_torch_golden.py (Triton interpreter on CPU)",
  }.get((effective, method))
  if effective in GPU_TYPES - {"triton"}:
    name = {"cuda": "CUDA", "cutile": "cuTile", "cutedsl": "CuTe DSL"}.get(
      effective, "GPU kernels"
    )
    oracle = f"none — a TPU host cannot execute {name}, so there is no oracle"
  if effective == "jax" and itype != "specification":
    oracle = "none — the input is already the reference"
  if itype == "specification" and synth == "jax":
    oracle = (
      "none — the baseline was synthesized directly as JAX, so "
      "nothing independent exists to check it against"
    )

  if extraction is not None and oracle:
    oracle += (
      " -- run on the EXTRACTED source.py, which the user approved; the "
      "original code is never executed"
    )

  configs = _get(job, "input.configs") or []
  if configs:
    inputs_origin = f"input.configs ({len(configs)} declared)"
  elif itype == "specification":
    inputs_origin = "synthesized source get_inputs(), from specification.tensors"
  else:
    inputs_origin = f"source {_get(job, 'input.inputs_fn') or 'get_inputs'}()"

  if itype == "specification" and synth == "jax":
    base_py = "SYNTHESIZED from the specification, then user-approved"
    producer = "maxkernel-synthesize-baseline"
  elif effective == "jax":
    base_py = "copied from the input, unchanged"
    producer = "orchestrator"
  elif method == "torchax" and emit:
    base_py = "written from the jaxpr in Phase 0.7"
    producer = "maxkernel-write-jnp-reference"
  elif method == "llm_port":
    base_py = "hand-written in Phase 0.4, gated in Phase 0.8"
    producer = (
      "maxkernel-analyze-torch-source"
      if effective == "pytorch"
      else "maxkernel-analyze-source"
    )
  else:
    base_py = "NONE — the internal timing harness cannot run"
    producer = None
  if bfrom == "autodiff" and producer:
    base_py += ", then turned into a training step (or backward) in Phase 0.85"

  phases = ["0 read state"]
  if itype == "specification":
    phases.append(f"0.1 synthesize {synth} baseline + USER APPROVAL GATE")
  if extraction is not None:
    phases.append("0.15 extract standalone PyTorch source + USER APPROVAL GATE")
  if effective in ("pytorch", "triton"):
    phases.append("0.2 oracle")
  if method == "torchax":
    phases.append("0.3 jaxpr + HLO export")
  if method == "llm_port" and effective != "jax":
    phases.append("0.4 analyze primary -> base.py")
  if refs:
    phases += [
      "0.5 analyze each reference",
      "0.6 reconcile -> ideas ledger, seal ref/",
    ]
  if method == "torchax" and emit:
    phases.append("0.7 write jnp reference + jaxpr_compare gate")
  if method == "llm_port" and effective != "jax":
    phases.append("0.8 verify_port gate")
  if bfrom == "autodiff":
    phases.append("0.85 derive the backward with jax.vjp")
  phases += ["0.9 harness", "1-6 plan/implement/compile/test/autotune/profile"]

  return {
    "input_type": itype,
    "synthesize_as": synth,
    "backward_from": bfrom,
    "reference_count": len(refs),
    "advisory_spine": bool(refs),
    "extraction": extraction is not None,
    "entry_point": (
      _get(job, "input.extraction.entry_point") + " (extracted to Model)"
      if extraction is not None
      else _get(job, "input.entry_point")
    ),
    "inputs_origin": inputs_origin,
    "oracle": oracle,
    "base_py": base_py,
    "base_py_producer": producer,
    "internal_harness": base_py
    != "NONE — the internal timing harness cannot run",
    "external_eval": bool(_get(job, "evaluation.compare_against_source")),
    "phases": phases,
  }


def main():
  p = argparse.ArgumentParser(description="Validate a MaxKernel job file.")
  p.add_argument("job_path")
  p.add_argument("--json", help="write the validation report here")
  p.add_argument("--quiet", action="store_true")
  args = p.parse_args()

  path = Path(args.job_path)
  if not path.is_file():
    print(f"JOB_NOT_FOUND: {path}", file=sys.stderr)
    return 1
  try:
    job = json.loads(path.read_text())
  except json.JSONDecodeError as e:
    print(f"JOB_INVALID_JSON: {e}", file=sys.stderr)
    return 1

  errors, warnings, plan = validate(job, path)
  report = {
    "job": str(path.resolve()),
    "valid": not errors,
    "errors": errors,
    "warnings": warnings,
    "plan": plan,
  }
  if args.json:
    Path(args.json).write_text(json.dumps(report, indent=2) + "\n")

  if errors:
    print("JOB INVALID", file=sys.stderr)
    for e in errors:
      print(f"  ERROR: {e}", file=sys.stderr)
    for w in warnings:
      print(f"  warn : {w}", file=sys.stderr)
    return 1

  if not args.quiet:
    print(f"JOB VALID: {job.get('name', path.stem)}")
    for w in warnings:
      print(f"  warn : {w}")
    where = job["input"].get("path") or (
      "synthesize as "
      + (plan.get("synthesize_as") or "?")
      + ": "
      + _get(job, "input.specification.operation", "(unnamed operation)")
    )
    print(f"  input        : {plan['input_type']}  ({where})")
    if plan["entry_point"]:
      print(f"  entry point  : {plan['entry_point']}")
    print(f"  inputs       : {plan['inputs_origin']}")
    if plan["extraction"]:
      print(
        "  extraction   : an agent writes a standalone source.py from the "
        "original; the run STOPS for your approval"
      )
    if plan["backward_from"] == "source":
      print("  backward     : from source (the input is the backward)")
    elif plan["backward_from"] == "autodiff":
      kernel = (
        "forward + backward, optimized together"
        if plan["optimize_forward"]
        else "backward only, the forward stays fixed"
      )
      print(f"  backward     : from autodiff ({kernel})")
    print(
      f"  references   : {plan['reference_count']}"
      f"{'  (advisory spine ON)' if plan['advisory_spine'] else ''}"
    )
    print(f"  oracle       : {plan['oracle']}")
    print(
      f"  base.py      : {plan['base_py']}"
      + (f"  [{plan['base_py_producer']}]" if plan["base_py_producer"] else "")
    )
    print(
      f"  harness      : {'internal (paired timing)' if plan['internal_harness'] else 'EXTERNAL ONLY'}"
    )
    print(f"  external eval: {plan['external_eval']}")
    print(f"  phases       : {' -> '.join(plan['phases'])}")
    if plan["missing_paths"]:
      print("  MISSING PATHS:", file=sys.stderr)
      for m in plan["missing_paths"]:
        print(f"    {m}", file=sys.stderr)

  return 2 if plan and plan["missing_paths"] else 0


if __name__ == "__main__":
  sys.exit(main())
