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
from pathlib import Path
import sys

SCHEMA = "maxkernel.job/1"

INPUT_TYPES = {"jax", "pytorch", "cuda", "specification"}
SYNTHESIS_TARGETS = {"pytorch", "jax"}
REFERENCE_TYPES = {"cuda", "triton", "pallas", "jax", "pytorch"}
CONVERSION_METHODS = {"torchax", "llm_port", "none"}
RELATIONSHIPS = {"same_operation", "same_family", "unknown"}


class JobError(Exception):
    pass


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
            "validating as v1 anyway")

    # ---- input ----------------------------------------------------------
    inp = job.get("input")
    if not isinstance(inp, dict):
        errors.append("`input` is required and must be an object")
        return errors, warnings, None

    itype = inp.get("type")
    if itype not in INPUT_TYPES:
        errors.append(f"input.type must be one of {sorted(INPUT_TYPES)}, got {itype!r}")

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
                "its real type instead.")
        if not isinstance(spec, dict):
            errors.append(
                "input.type is 'specification' requires an input.specification "
                "object describing what to compute.")
        else:
            if not spec.get("description"):
                errors.append("input.specification.description is required: one "
                              "or two sentences naming the computation.")
            tensors = spec.get("tensors")
            if not isinstance(tensors, list) or not tensors:
                errors.append(
                    "input.specification.tensors is required and must be a "
                    "non-empty list. Shapes and dtypes cannot be invented: they "
                    "define the problem, and general_rules #6 forbids "
                    "substituting them.")
            else:
                for i, t in enumerate(tensors):
                    if not isinstance(t, dict):
                        errors.append(f"input.specification.tensors[{i}] must be an object")
                        continue
                    for field in ("name", "shape", "dtype", "role"):
                        if not t.get(field):
                            errors.append(
                                f"input.specification.tensors[{i}] is missing "
                                f"{field!r} (role is 'input', 'output' or 'parameter')")
            target = spec.get("synthesize_as", "pytorch")
            if target not in SYNTHESIS_TARGETS:
                errors.append(
                    f"input.specification.synthesize_as must be one of "
                    f"{sorted(SYNTHESIS_TARGETS)}, got {target!r}")
            if spec.get("user_confirms_baseline") is not True:
                errors.append(
                    "input.specification.user_confirms_baseline must be true. "
                    "In this mode the agent writes the baseline it will then be "
                    "scored against, so the run pauses for you to read and "
                    "approve that baseline before any optimization begins. "
                    "Setting this field is how you accept that step exists; the "
                    "run will still stop and wait for the actual approval.")
    elif not ipath:
        errors.append("input.path is required")

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
            "would silently proceed with no reference at all.")
    if has_ref is False and refs:
        errors.append(
            f"has_reference is false but {len(refs)} reference(s) are listed. "
            "The references would be ignored; say which you meant.")

    for i, r in enumerate(refs):
        if not isinstance(r, dict):
            errors.append(f"references[{i}] must be an object")
            continue
        if r.get("type") not in REFERENCE_TYPES:
            errors.append(f"references[{i}].type must be one of {sorted(REFERENCE_TYPES)}")
        if not r.get("path"):
            errors.append(f"references[{i}].path is required")
        rel = r.get("relationship", "unknown")
        if rel not in RELATIONSHIPS:
            errors.append(
                f"references[{i}].relationship must be one of {sorted(RELATIONSHIPS)}")

    # ---- jax conversion -------------------------------------------------
    needs = job.get("needs_jax_conversion")
    if needs is None:
        errors.append("needs_jax_conversion is required (true or false)")
    method = _get(job, "jax_conversion.method", "none" if needs is False else None)
    emit = _get(job, "jax_conversion.emit_readable_reference", True)

    if needs is True:
        if method not in CONVERSION_METHODS - {"none"}:
            errors.append(
                "needs_jax_conversion is true, so jax_conversion.method must be "
                "'torchax' or 'llm_port'")
        if itype == "jax":
            errors.append(
                "input.type is 'jax' and needs_jax_conversion is true. A JAX "
                "input is already the reference; there is nothing to convert.")
        if itype == "specification" and _get(job, "input.specification.synthesize_as",
                                             "pytorch") == "jax":
            errors.append(
                "input.specification.synthesize_as is 'jax' and "
                "needs_jax_conversion is true. A synthesized JAX baseline IS "
                "base.py; there is nothing further to convert.")
        if itype == "cuda" and method == "torchax":
            errors.append(
                "jax_conversion.method 'torchax' cannot convert a CUDA input. "
                "torchax traces PyTorch modules; CUDA C++ has to be ported by "
                "an agent — use 'llm_port'.")
    elif needs is False:
        if method not in (None, "none"):
            errors.append(
                f"needs_jax_conversion is false but jax_conversion.method is "
                f"{method!r}; set it to 'none' or omit it")
        if itype == "specification" and _get(job, "input.specification.synthesize_as",
                                             "pytorch") == "pytorch":
            errors.append(
                "synthesize_as is 'pytorch' but needs_jax_conversion is false. "
                "A synthesized PyTorch baseline still has to become base.py "
                "like any other PyTorch input -- set needs_jax_conversion true "
                "with a method, or synthesize_as 'jax' to skip the step.")
        elif itype not in ("jax", "specification"):
            # This is the constraint that otherwise fails four phases later.
            errors.append(
                f"input.type is {itype!r} with needs_jax_conversion false, so "
                "no base.py is produced. tools/assemble_test_harness.py binds "
                "`base_computation` from a file defining `computation`, so the "
                "internal timing harness cannot run. Either set "
                "needs_jax_conversion true, or set "
                "evaluation.compare_against_source true to measure externally "
                "with evaluation/compare_kernel.py.")

    if needs is True and method == "torchax" and emit is False:
        msg = ("jax_conversion.method is 'torchax' with emit_readable_reference "
               "false: torchax produces the oracle but no base.py, so the "
               "internal timing harness cannot run.")
        if _get(job, "evaluation.compare_against_source") is True:
            warnings.append(msg + " evaluation.compare_against_source is set, so "
                            "performance will be measured externally.")
        else:
            errors.append(
                msg + " Set emit_readable_reference true, or set "
                "evaluation.compare_against_source true.")

    if (itype == "specification"
            and _get(job, "input.specification.synthesize_as", "pytorch") == "jax"
            and _get(job, "evaluation.compare_against_source") is True):
        errors.append(
            "evaluation.compare_against_source is true, but synthesize_as is "
            "'jax', so there is no PyTorch source to compare against. Use "
            "synthesize_as 'pytorch' if you want that measurement.")

    if itype == "cuda" and _get(job, "evaluation.compare_against_source") is True:
        errors.append(
            "evaluation.compare_against_source is true for a CUDA input. A TPU "
            "host cannot execute CUDA, so there is nothing to compare against. "
            "Every number for a CUDA input is measured against the JAX port.")

    # ---- correctness / target ------------------------------------------
    atol = _get(job, "correctness.atol", 1e-2)
    rtol = _get(job, "correctness.rtol", 1e-2)
    for name, v in (("atol", atol), ("rtol", rtol)):
        if not isinstance(v, (int, float)) or v <= 0:
            errors.append(f"correctness.{name} must be a positive number, got {v!r}")
        elif v > 0.5:
            warnings.append(
                f"correctness.{name} is {v}; a tolerance that loose accepts "
                "almost any output and will not catch a real error")

    iters = _get(job, "loop.max_iterations", 5)
    if not isinstance(iters, int) or iters < 1:
        errors.append(f"loop.max_iterations must be a positive integer, got {iters!r}")

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
    for i, r in enumerate(refs):
        if not _resolve(r["path"]).exists():
            missing.append(f"references[{i}].path: {_resolve(r['path'])}")

    plan = build_plan(job, itype, needs, method, emit, refs)
    plan["missing_paths"] = missing
    return errors, warnings, plan


def build_plan(job, itype, needs, method, emit, refs):
    """What the orchestrator will actually do, derived from the declaration."""
    synth = _get(job, "input.specification.synthesize_as", "pytorch") \
        if itype == "specification" else None
    # After synthesis a specification job behaves exactly like an input of the
    # synthesized kind, so the rest of the plan is derived from that.
    effective = synth if itype == "specification" else itype

    oracle = {
        ("pytorch", "torchax"): "tools/torchax_oracle.py (mechanical, with step-0 check vs eager)",
        ("pytorch", "llm_port"): "tools/capture_torch_golden.py (eager PyTorch on CPU)",
    }.get((effective, method))
    if effective == "cuda":
        oracle = "none — a TPU host cannot execute CUDA, so there is no oracle"
    if effective == "jax" and itype != "specification":
        oracle = "none — the input is already the reference"
    if itype == "specification" and synth == "jax":
        oracle = ("none — the baseline was synthesized directly as JAX, so "
                  "nothing independent exists to check it against")

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
        producer = ("maxkernel-analyze-torch-source" if effective == "pytorch"
                    else "maxkernel-analyze-source")
    else:
        base_py = "NONE — the internal timing harness cannot run"
        producer = None

    phases = ["0 read state"]
    if itype == "specification":
        phases.append(f"0.1 synthesize {synth} baseline + USER APPROVAL GATE")
    if effective == "pytorch":
        phases.append("0.2 oracle")
    if method == "torchax":
        phases.append("0.3 jaxpr + HLO export")
    if method == "llm_port" and effective != "jax":
        phases.append("0.4 analyze primary -> base.py")
    if refs:
        phases += ["0.5 analyze each reference", "0.6 reconcile -> ideas ledger, seal ref/"]
    if method == "torchax" and emit:
        phases.append("0.7 write jnp reference + jaxpr_compare gate")
    if method == "llm_port" and effective != "jax":
        phases.append("0.8 verify_port gate")
    phases += ["0.9 harness", "1-6 plan/implement/compile/test/autotune/profile"]

    return {
        "input_type": itype,
        "synthesize_as": synth,
        "reference_count": len(refs),
        "advisory_spine": bool(refs),
        "oracle": oracle,
        "base_py": base_py,
        "base_py_producer": producer,
        "internal_harness": base_py != "NONE — the internal timing harness cannot run",
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
    report = {"job": str(path.resolve()), "valid": not errors,
              "errors": errors, "warnings": warnings, "plan": plan}
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
            "synthesize as " + (plan.get("synthesize_as") or "?") + ": "
            + _get(job, "input.specification.operation", "(unnamed operation)"))
        print(f"  input        : {plan['input_type']}  ({where})")
        print(f"  references   : {plan['reference_count']}"
              f"{'  (advisory spine ON)' if plan['advisory_spine'] else ''}")
        print(f"  oracle       : {plan['oracle']}")
        print(f"  base.py      : {plan['base_py']}"
              + (f"  [{plan['base_py_producer']}]" if plan['base_py_producer'] else ""))
        print(f"  harness      : {'internal (paired timing)' if plan['internal_harness'] else 'EXTERNAL ONLY'}")
        print(f"  external eval: {plan['external_eval']}")
        print(f"  phases       : {' -> '.join(plan['phases'])}")
        if plan["missing_paths"]:
            print("  MISSING PATHS:", file=sys.stderr)
            for m in plan["missing_paths"]:
                print(f"    {m}", file=sys.stderr)

    return 2 if plan and plan["missing_paths"] else 0


if __name__ == "__main__":
    sys.exit(main())
