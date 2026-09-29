#!/usr/bin/env python3
"""Structurally compares an agent-written jnp reference against the torchax jaxpr.

WHY VALUE EQUALITY IS NOT ENOUGH
--------------------------------
`allclose` against the oracle is checked on sampled inputs. It can pass while
the computation genuinely differs -- on untested edge cases, in masked regions,
or through a reformulation whose numerics differ only in the tails. Measured on
a plain RMSNorm, an agent writing `x / jnp.sqrt(v + eps)` instead of
`x * jax.lax.rsqrt(v + eps)` passes allclose at 1.9e-06 while computing
something with different behaviour near zero and worse performance.

WHY NAIVE STRUCTURAL EQUALITY IS WORSE
--------------------------------------
It rejects perfect implementations. Also measured: an agent writing
`jnp.mean(jnp.square(x), -1)` produces output that is **bit-identical** to the
torchax reference -- max|diff| exactly 0.0 -- and yet its jaxpr differs,
because torch's `.pow(2)` maps to the `pow` primitive while `jnp.square` emits
`square`. A strict comparison fails it.

So the comparison normalizes first, and reports a graded verdict rather than a
boolean:

    identical   normalized census matches exactly
    equivalent  differs only through known-equivalent spellings
    divergent   a real structural difference: a missing reduction, an extra
                contraction, different reduction axes, a changed dtype ladder

`divergent` is worth a human look. It is NOT automatically an error -- but on a
jnp *reference*, which is supposed to be a faithful restatement, it usually is.

DO NOT RUN THIS AGAINST A PALLAS KERNEL
---------------------------------------
A good Pallas kernel deliberately changes the structure: online softmax,
unnormalized accumulators, a fused epilogue, a different loop order. Structural
divergence there is the goal, not a defect. Mechanically it would not work
either -- `make_jaxpr` on a `pallas_call` yields one opaque primitive with the
body nested inside, so the census is apples to oranges. The Pallas kernel is
checked against the oracle by value, and by nothing else.

Exit codes:
  0  identical or equivalent
  1  divergent (only when --strict; otherwise divergence is reported at 0)
  3  bad input
"""

import argparse
import collections
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jaxpr_export import LAYOUT_PRIMITIVES, jaxpr_facts  # noqa: E402

# Spellings that denote the same mathematical operation. Without this table the
# comparison flags `pow(x, 2)` against `square(x)` -- which is how it would
# reject a bit-identical implementation.
EQUIVALENCE_CLASSES = {
    "square": "SQUARE",
    "integer_pow": "SQUARE",     # refined below using the exponent
    "pow": "POW",                # refined below: exponent 2 becomes SQUARE
    "rsqrt": "RSQRT",
    "sqrt": "SQRT",
    "mul": "MUL",
    "div": "DIV",
    "add": "ADD",
    "sub": "SUB",
    "reduce_sum": "REDUCE_SUM",
    "reduce_max": "REDUCE_MAX",
    "reduce_min": "REDUCE_MIN",
    "reduce_prod": "REDUCE_PROD",
    "dot_general": "CONTRACT",
    "exp": "EXP",
    "log": "LOG",
    "tanh": "TANH",
    "logistic": "LOGISTIC",
    "max": "MAX",
    "min": "MIN",
    "select_n": "SELECT",
    "custom_jvp_call": "CUSTOM",
    "pjit": "NESTED",
}


def normalize_primitive(eqn):
    """Maps one equation to its equivalence class, using params and dataflow.

    Squaring has at least three common spellings and an agent may use any of
    them: `x ** 2` (pow), `jnp.square(x)` (square), and `x * x` (mul with both
    operands the same variable). Collapsing all three is what lets a faithful
    restatement come back `identical` instead of being flagged for a difference
    that exists only in the source text.
    """
    name = eqn.primitive.name

    if name in ("pow", "integer_pow"):
        exp = eqn.params.get("y")
        if exp is None and len(eqn.invars) > 1:
            exp = getattr(eqn.invars[1], "val", None)
        try:
            if exp is not None and float(exp) == 2.0:
                return "SQUARE"
        except (TypeError, ValueError):
            pass
        return "POW"

    if name == "mul" and len(eqn.invars) == 2:
        a, b = eqn.invars
        # Same jaxpr variable on both sides -- `x * x`, i.e. a squaring.
        if a is b or (hasattr(a, "count") and hasattr(b, "count")
                      and getattr(a, "count", None) == getattr(b, "count", object())):
            return "SQUARE"

    return EQUIVALENCE_CLASSES.get(name, name.upper())


def census(closed_jaxpr):
    """Normalized operation census, layout primitives excluded."""
    counts = collections.Counter()
    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive.name in LAYOUT_PRIMITIVES:
            continue
        counts[normalize_primitive(eqn)] += 1
    return counts


def reduction_signature(closed_jaxpr):
    """Which axes are reduced, by which operation. Order-insensitive."""
    sig = []
    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive.name.startswith("reduce"):
            sig.append((normalize_primitive(eqn), tuple(eqn.params.get("axes", ()))))
    return sorted(sig)


# Differences that are a matter of spelling rather than of computation. Each is
# a (reference-only, candidate-only) pair that cancels out.
BENIGN_SWAPS = [
    # x * rsqrt(v)  vs  x / sqrt(v): the reference spends one RSQRT where the
    # candidate spends a SQRT and a DIV.
    ({"RSQRT": 1}, {"SQRT": 1, "DIV": 1}),
    # /N  vs  *(1/N)
    ({"DIV": 1}, {"MUL": 1}),
    ({"MUL": 1}, {"DIV": 1}),
]


def classify(ref_counts, cand_counts):
    """identical | equivalent | divergent, with the surviving difference."""
    only_ref = {k: ref_counts[k] - cand_counts.get(k, 0)
                for k in ref_counts if ref_counts[k] > cand_counts.get(k, 0)}
    only_cand = {k: cand_counts[k] - ref_counts.get(k, 0)
                 for k in cand_counts if cand_counts[k] > ref_counts.get(k, 0)}

    if not only_ref and not only_cand:
        return "identical", {}, {}

    r, c = dict(only_ref), dict(only_cand)
    for swap_ref, swap_cand in BENIGN_SWAPS:
        if all(r.get(k, 0) >= v for k, v in swap_ref.items()) and \
           all(c.get(k, 0) >= v for k, v in swap_cand.items()):
            for k, v in swap_ref.items():
                r[k] -= v
                if r[k] == 0:
                    del r[k]
            for k, v in swap_cand.items():
                c[k] -= v
                if c[k] == 0:
                    del c[k]

    if not r and not c:
        return "equivalent", dict(only_ref), dict(only_cand)
    return "divergent", r, c


def load_candidate(path, entry):
    """Loads the agent's jnp reference and returns its entry-point callable."""
    src = Path(path).read_text()
    ns = {"__name__": "__maxkernel_candidate__", "__file__": str(Path(path).resolve())}
    exec(compile(src, str(path), "exec"), ns)  # pylint: disable=exec-used
    fn = ns.get(entry)
    if not callable(fn):
        raise LookupError(f"{path} defines no callable `{entry}`")
    return fn


def compare(source_path, candidate_path, entry, seed, states_first,
            atol=1e-4, rtol=1e-4):
    import jax
    import torch

    from jaxpr_export import build_model, import_torchax, load_source, to_jax

    torchax = import_torchax()
    ns = load_source(source_path)
    model, _ = build_model(ns, seed)
    torch.manual_seed(seed)
    raw = ns["get_inputs"]()
    raw = [raw] if isinstance(raw, torch.Tensor) else list(raw)
    args = tuple(to_jax(a) if isinstance(a, torch.Tensor) else a for a in raw)

    states, jax_fn = torchax.extract_jax(model)
    ref_jaxpr = jax.make_jaxpr(jax_fn)(states, args)

    # The candidate is a plain function. Feed it the same tensors in whichever
    # order it declares -- the two conventions genuinely differ and guessing is
    # how the argument mapping goes silently wrong.
    flat_states = [states[k] for k in states.keys()]
    cand_args = (flat_states + list(args)) if states_first else (list(args) + flat_states)
    cand_jaxpr = jax.make_jaxpr(fn := load_candidate(candidate_path, entry))(*cand_args)

    ref_counts, cand_counts = census(ref_jaxpr), census(cand_jaxpr)
    verdict, only_ref, only_cand = classify(ref_counts, cand_counts)

    ref_red, cand_red = reduction_signature(ref_jaxpr), reduction_signature(cand_jaxpr)
    reduction_match = ref_red == cand_red
    if not reduction_match and verdict != "divergent":
        verdict = "divergent"

    value = {"checked": False}
    try:
        g = jax.jit(jax_fn)(states, args)
        c = fn(*cand_args)
        gl = [x for x in jax.tree_util.tree_leaves(g)]
        cl = [x for x in jax.tree_util.tree_leaves(c)]
        import numpy as np
        if len(gl) == len(cl):
            worst = max(float(np.max(np.abs(np.asarray(a, np.float64)
                                            - np.asarray(b, np.float64))))
                        for a, b in zip(gl, cl))
            within = all(
                bool(np.allclose(np.asarray(a, np.float64), np.asarray(b, np.float64),
                                 atol=atol, rtol=rtol, equal_nan=True))
                for a, b in zip(gl, cl))
            value = {"checked": True, "max_abs_diff": worst,
                     "bit_identical": worst == 0.0,
                     "within_tolerance": within, "atol": atol, "rtol": rtol}
        else:
            value = {"checked": True, "within_tolerance": False,
                     "error": f"output arity {len(gl)} vs {len(cl)}"}
    except Exception as e:  # pylint: disable=broad-except
        value = {"checked": False, "error": f"{type(e).__name__}: {e}"}

    # The value check is a gate; the structural check is advisory. A candidate
    # that disagrees with the oracle numerically is wrong whatever its census
    # says, so it fails outright rather than being graded on structure.
    value_ok = value.get("within_tolerance", None)
    return {
        "schema": "maxkernel.jaxpr_compare/1",
        "verdict": verdict,
        "value_ok": value_ok,
        "reference": {"census": dict(sorted(ref_counts.items())),
                      "reductions": [[p, list(a)] for p, a in ref_red],
                      "facts": jaxpr_facts(ref_jaxpr)},
        "candidate": {"census": dict(sorted(cand_counts.items())),
                      "reductions": [[p, list(a)] for p, a in cand_red],
                      "facts": jaxpr_facts(cand_jaxpr)},
        "difference": {"reference_only": only_ref, "candidate_only": only_cand,
                       "reduction_signature_match": reduction_match},
        "value_check": value,
        "interpretation": {
            "identical": "the candidate restates the reference operation for operation",
            "equivalent": "differences are known-equivalent spellings only",
            "divergent": ("a real structural difference. On a jnp REFERENCE this "
                          "usually means the restatement is not faithful. On a "
                          "Pallas kernel it would be expected -- do not run this "
                          "tool against one."),
        }[verdict],
    }


def main():
    p = argparse.ArgumentParser(
        description="Structurally compare an agent jnp reference against the torchax jaxpr.")
    p.add_argument("source_path", help="the PyTorch module (the torchax reference)")
    p.add_argument("candidate", help="the agent's jnp .py")
    p.add_argument("--entry", default="computation")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--inputs-first", action="store_true",
                   help="candidate takes forward inputs before states "
                        "(capture_torch_golden order). Default is states first, "
                        "matching torchax's flattened signature.")
    p.add_argument("--out")
    p.add_argument("--atol", type=float, default=1e-4,
                   help="value gate: candidate vs oracle")
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--strict", action="store_true",
                   help="also exit 1 on a divergent STRUCTURAL verdict; the value "
                        "gate fails with exit 1 regardless")
    args = p.parse_args()

    try:
        report = compare(args.source_path, args.candidate, args.entry, args.seed,
                         states_first=not args.inputs_first,
                         atol=args.atol, rtol=args.rtol)
    except LookupError as e:
        print(f"BAD_INPUT: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # pylint: disable=broad-except
        import traceback
        print(f"BAD_INPUT: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 3

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n")

    v = report["verdict"]
    print(f"VERDICT: {v}")
    print(f"  {report['interpretation']}")
    print(f"  reference census: {report['reference']['census']}")
    print(f"  candidate census: {report['candidate']['census']}")
    d = report["difference"]
    if d["reference_only"] or d["candidate_only"]:
        print(f"  surviving diff  : ref-only={d['reference_only']} "
              f"cand-only={d['candidate_only']}")
    print(f"  reductions match: {d['reduction_signature_match']}")
    val = report["value_check"]
    if val.get("checked") and "max_abs_diff" in val:
        tag = " (bit-identical)" if val["bit_identical"] else ""
        gate = "PASS" if val.get("within_tolerance") else "FAIL"
        print(f"  value vs oracle : max|diff|={val['max_abs_diff']:.3e}{tag}  "
              f"[gate {gate} @ atol={args.atol:g} rtol={args.rtol:g}]")
    elif val.get("error"):
        print(f"  value vs oracle : ERROR {val['error']}")

    if report.get("value_ok") is False:
        print("  -> REJECTED: the candidate disagrees with the oracle numerically. "
              "That is decisive regardless of structure.")
        return 1
    if v == "divergent" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
