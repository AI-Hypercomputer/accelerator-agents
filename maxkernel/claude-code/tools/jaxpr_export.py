#!/usr/bin/env python3
"""Exports the reference material the kernel planner reads: jaxpr + optimized HLO.

When the JAX reference is obtained mechanically through torchax, there is no
readable `base.py` for the planner to study. The jaxpr replaces it -- and is in
several ways better source material, because it is what JAX will actually
compile rather than a person's description of it. On a plain RMSNorm the jaxpr
shows `.mean(-1)` decomposed into `reduce_sum` + `div 2048.0`, the exact
broadcast structure, the reduction axis as `axes=(2,)`, and eps materialized as
`9.999999747378752e-06` rather than the `1e-5` that appears in the source.

TWO IRs, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
------------------------------------------------
  * **jaxpr** is pre-optimization. It says WHAT was asked for: the operations,
    their dtypes, the reduction axes, the broadcast structure.
  * **optimized HLO** is post-XLA. It says WHAT XLA DID: which operations got
    fused into one kernel and, by omission, where the HBM round trips remain.

A Pallas kernel earns its keep by collapsing fusion boundaries XLA could not.
Those boundaries are invisible in the jaxpr -- planning from it alone produces
proposals to fuse things XLA already fused. Export both.

SCALE
-----
A single transformer block lowers to ~76 jaxpr equations, roughly 1,500 tokens.
That is fine for one fusion target and not fine for a whole model, which is why
`--max-eqns` warns rather than silently handing an agent a 50,000-token
listing.

Exit codes:
  0  exported
  2  torchax unavailable or the module will not convert
  3  no discoverable entry point
"""

import argparse
import collections
import json
import os
from pathlib import Path
import re
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torchax_oracle import (  # noqa: E402
    build_model, import_torchax, load_source, to_jax,
)

# Primitives that move or relabel data without changing what is computed. They
# are reported separately because how many appear depends largely on how the
# source happened to be written, so mixing them into the operation census makes
# two equivalent programs look different.
LAYOUT_PRIMITIVES = {
    "broadcast_in_dim", "reshape", "transpose", "squeeze", "expand_dims",
    "convert_element_type", "copy", "device_put", "slice", "concatenate",
}


def jaxpr_facts(closed_jaxpr):
    """Structured summary of a jaxpr, for planning and for comparison."""
    eqns = closed_jaxpr.jaxpr.eqns
    prims = collections.Counter(e.primitive.name for e in eqns)
    compute = {k: v for k, v in prims.items() if k not in LAYOUT_PRIMITIVES}
    layout = {k: v for k, v in prims.items() if k in LAYOUT_PRIMITIVES}

    reductions = []
    for e in eqns:
        if e.primitive.name.startswith("reduce"):
            reductions.append({
                "primitive": e.primitive.name,
                "axes": list(e.params.get("axes", ())),
                "out_shape": [list(getattr(v.aval, "shape", ())) for v in e.outvars],
            })

    contractions = []
    for e in eqns:
        if e.primitive.name == "dot_general":
            dn = e.params.get("dimension_numbers")
            contractions.append({
                "dimension_numbers": str(dn),
                "operand_shapes": [list(getattr(v.aval, "shape", ())) for v in e.invars],
                "out_shape": [list(getattr(v.aval, "shape", ())) for v in e.outvars],
            })

    dtypes = sorted({str(getattr(v.aval, "dtype", "?"))
                     for e in eqns for v in list(e.invars) + list(e.outvars)
                     if hasattr(v, "aval")})

    return {
        "n_eqns": len(eqns),
        "n_constvars": len(closed_jaxpr.jaxpr.constvars),
        "primitives": dict(sorted(prims.items())),
        "compute_primitives": dict(sorted(compute.items())),
        "layout_primitives": dict(sorted(layout.items())),
        "reductions": reductions,
        "contractions": contractions,
        "dtypes": dtypes,
        "invars": [{"shape": list(getattr(v.aval, "shape", ())),
                    "dtype": str(getattr(v.aval, "dtype", "?"))}
                   for v in closed_jaxpr.jaxpr.invars],
        "outvars": [{"shape": list(getattr(v.aval, "shape", ())),
                     "dtype": str(getattr(v.aval, "dtype", "?"))}
                    for v in closed_jaxpr.jaxpr.outvars],
    }


_FUSION_RE = re.compile(r"^\s*%?(?P<name>[\w.\-]+)\s*=\s*(?P<shape>\S+)\s+fusion\(", re.M)
_ALLOC_RE = re.compile(r"allocated_bytes[^\d]*(\d+)")


def hlo_facts(hlo_text):
    """What XLA actually did: fusions, and the traffic they imply."""
    if not hlo_text:
        return {"available": False}
    fusions = [{"name": m.group("name"), "shape": m.group("shape")}
               for m in _FUSION_RE.finditer(hlo_text)]
    kinds = collections.Counter(
        m.group(1) for m in re.finditer(r'kind=(\w+)', hlo_text))
    customs = re.findall(r'custom_call_target="([^"]+)"', hlo_text)
    return {
        "available": True,
        "n_fusions": len(fusions),
        "fusions": fusions[:64],
        "fusion_kinds": dict(kinds),
        "custom_calls": sorted(set(customs)),
        "n_lines": hlo_text.count("\n") + 1,
        "note": (
            "Each fusion is a region XLA already keeps in registers/VMEM. The "
            "HBM round trips a Pallas kernel can remove are the boundaries "
            "BETWEEN fusions, not the fusions themselves."
        ),
    }


def export(source_path, seed, max_eqns):
    import jax

    torchax = import_torchax()
    ns = load_source(source_path)
    model, model_name = build_model(ns, seed)
    if not callable(ns.get("get_inputs")):
        raise LookupError("the source defines no get_inputs()")

    import torch
    torch.manual_seed(seed)
    raw = ns["get_inputs"]()
    raw = [raw] if isinstance(raw, torch.Tensor) else list(raw)
    args = tuple(to_jax(a) if isinstance(a, torch.Tensor) else a for a in raw)

    states, jax_fn = torchax.extract_jax(model)
    closed = jax.make_jaxpr(jax_fn)(states, args)
    jaxpr_text = str(closed)

    hlo_text, hlo_error = None, None
    try:
        lowered = jax.jit(jax_fn).lower(states, args)
        hlo_text = lowered.compile().as_text()
    except Exception as e:  # pylint: disable=broad-except
        hlo_error = f"{type(e).__name__}: {e}"

    facts = jaxpr_facts(closed)
    facts["model_class"] = model_name
    facts["source_path"] = str(Path(source_path).resolve())
    facts["states_order"] = list(states.keys())
    facts["jaxpr_chars"] = len(jaxpr_text)
    facts["jaxpr_estimated_tokens"] = len(jaxpr_text) // 4
    facts["hlo"] = hlo_facts(hlo_text)
    if hlo_error:
        facts["hlo"]["error"] = hlo_error

    warnings = []
    if facts["n_eqns"] > max_eqns:
        warnings.append(
            f"{facts['n_eqns']} equations (> --max-eqns {max_eqns}), about "
            f"{facts['jaxpr_estimated_tokens']} tokens. This is a whole-model "
            "jaxpr, not a kernel-sized one. Scope it to the fusion target "
            "before handing it to a planner."
        )
    if facts["n_constvars"]:
        warnings.append(
            f"{facts['n_constvars']} constvars are inlined in the jaxpr. If any "
            "is a weight tensor its values are printed in full -- check before "
            "passing this to an agent."
        )
    facts["warnings"] = warnings
    return jaxpr_text, hlo_text, facts


def main():
    p = argparse.ArgumentParser(
        description="Export jaxpr and optimized HLO for a PyTorch module via torchax.")
    p.add_argument("source_path")
    p.add_argument("--jaxpr", help="write the jaxpr text here")
    p.add_argument("--hlo", help="write the optimized HLO here")
    p.add_argument("--json", help="write the structured facts here")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--max-eqns", type=int, default=400,
                   help="warn above this; a whole-model jaxpr is not planning material")
    args = p.parse_args()

    try:
        jaxpr_text, hlo_text, facts = export(args.source_path, args.seed, args.max_eqns)
    except LookupError as e:
        print(f"NO_ENTRY_POINT: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # pylint: disable=broad-except
        print(f"DEGRADED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2

    if args.jaxpr:
        Path(args.jaxpr).write_text(jaxpr_text + "\n")
    if args.hlo and hlo_text:
        Path(args.hlo).write_text(hlo_text)
    if args.json:
        Path(args.json).write_text(json.dumps(facts, indent=2, default=str) + "\n")

    print(f"JAXPR_OK model={facts['model_class']}")
    print(f"  equations     : {facts['n_eqns']}  (~{facts['jaxpr_estimated_tokens']} tokens)")
    print(f"  compute prims : {facts['compute_primitives']}")
    print(f"  layout prims  : {facts['layout_primitives']}")
    print(f"  reductions    : {[(r['primitive'], r['axes']) for r in facts['reductions']]}")
    h = facts["hlo"]
    if h.get("available"):
        print(f"  HLO fusions   : {h['n_fusions']}  custom_calls={h['custom_calls']}")
    else:
        print(f"  HLO           : unavailable ({h.get('error', 'not lowered')})")
    for w in facts["warnings"]:
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
