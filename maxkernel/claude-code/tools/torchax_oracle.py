#!/usr/bin/env python3
"""Builds a correctness oracle for a PyTorch module by running it as JAX, via torchax.

WHY THIS AND NOT capture_torch_golden.py
----------------------------------------
`capture_torch_golden.py` runs the user's module in eager PyTorch on the CPU.
That is the right oracle when the loop is going to hand-port the module to JAX,
because it is independent of the port.

This tool answers a different need. `torchax.extract_jax` converts the module to
a JAX function *mechanically* -- no LLM, no transcription -- so the run gets a
JAX reference with no mistranslation risk at all, at the same abstraction level
as the Pallas kernel that will replace it. The agent never sees this function.
It exists only to say whether the agent's output is right.

THE ORACLE IS ITSELF A TRANSLATION, SO IT IS VERIFIED FIRST
-----------------------------------------------------------
torchax is mechanical but not infallible: op mappings can carry dtype and
reduction-order differences. Measured on a plain RMSNorm, torchax and eager
PyTorch differ by ~1.9e-06 in fp32 -- small, real, and not zero. So before
`jax_fn` becomes ground truth for everything downstream, step 0 compares it
against the actual PyTorch module and records the result. An oracle that was
never checked is an assumption wearing a lab coat.

CALLING CONVENTION -- read this before mapping arguments
--------------------------------------------------------
`torchax.extract_jax(model)` returns `(states, jax_func)` where:

  * `states` is a FLAT DICT of buffers and parameters, keyed by name;
  * `jax_func(states, args, kwargs=None)` -- `args` is a tuple even for one
    argument.

Flattened for tracing, the signature comes out **states first, then forward
inputs**:  `(weight[2048], x[4,256,2048])`.

That is the OPPOSITE of the order `capture_torch_golden.py` records, where
forward inputs come first and parameters follow. Both orders are written into
the manifest explicitly, under `flat_signature` and `golden_compatible_order`,
so whatever consumes this cannot pick the wrong one by accident.

Exit codes:
  0  oracle built and validated
  2  torchax unavailable, or the module will not convert -- DEGRADE, do not abort
  3  no discoverable entry point
  4  STEP 0 FAILED: torchax disagrees with eager PyTorch beyond tolerance.
     The oracle is not trustworthy and must not be used.
  5  the capture would exceed --max-bytes
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))

import numpy as np  # noqa: E402

# Step 0 compares torchax's conversion against eager PyTorch. How closely the
# two can possibly agree is a property of the BACKEND, not of the conversion:
#
#   cpu   ~1.9e-06   fp32 throughout, differing only in reduction order.
#   tpu   ~1.0e-3    XLA's default matmul precision on TPU is bf16, so a
#                    matmul-bearing module diverges by roughly bf16 epsilon.
#                    Measured on a v6e-8 VM, Linear+GELU, torch 2.9 /
#                    torchax 0.0.13.
#
# A single fixed default is therefore wrong on one of the two. At 1e-4 a TPU
# run fails step 0 and falls back to llm_port, reporting "torchax does not
# reproduce eager PyTorch" -- which is false, and sends you hunting a
# conversion bug that is really just bf16.
#
# The accelerator floor is set AT the measured divergence rather than above it,
# so it has no headroom by design: a module that diverges materially more than
# bf16 epsilon should fail step 0 rather than be waved through. The cost is
# that a borderline module may fail and need an explicit --atol; that is the
# intended trade, since a too-loose step 0 admits a broken oracle and every
# later check is measured against it.
DEFAULT_TOLERANCE = {"cpu": 1e-4, "tpu": 1e-3, "gpu": 1e-3}


def resolve_tolerance(atol, rtol):
  """Picks a step-0 tolerance matched to the backend, unless one was given."""
  import jax

  backend = jax.default_backend()
  floor = DEFAULT_TOLERANCE.get(backend, 1e-3)
  return (
    floor if atol is None else atol,
    floor if rtol is None else rtol,
    backend,
  )


def import_torchax():
  """Imports torchax, with a precise message when the pin is wrong.

  torchax builds an autocast policy table at import time keyed on aten
  overloads. Several of those were removed in recent torch, so a mismatched
  pair fails with an AttributeError naming an operator, which reads like a
  missing dependency rather than a version conflict. Say what it actually is.
  """
  try:
    import torchax

    return torchax
  except AttributeError as e:
    raise RuntimeError(
      f"torchax failed to import against this torch version ({e}). This is "
      "a torch/torchax version conflict, not a missing package: torchax "
      "references aten overloads that newer torch removed. Pin a "
      "compatible pair in requirements.txt."
    ) from e
  except ImportError as e:
    raise RuntimeError(f"torchax is not installed: {e}") from e


def load_source(source_path):
  """Executes the user's file in its own namespace."""
  path = Path(source_path)
  ns = {"__name__": "__maxkernel_source__", "__file__": str(path.resolve())}
  sys.path.insert(0, str(path.parent.resolve()))
  try:
    exec(compile(path.read_text(), str(path), "exec"), ns)  # pylint: disable=exec-used
  finally:
    sys.path.pop(0)
  return ns


def normalize_init_inputs(raw):
  """Same rule as capture_torch_golden.py, so the two cannot disagree."""
  if raw is None:
    return [], {}
  if (
    isinstance(raw, tuple)
    and len(raw) == 2
    and isinstance(raw[1], dict)
    and isinstance(raw[0], (list, tuple))
  ):
    return list(raw[0]), dict(raw[1])
  if isinstance(raw, (list, tuple)):
    return list(raw), {}
  if isinstance(raw, dict):
    return [], dict(raw)
  return [raw], {}


def build_model(ns, seed):
  import torch
  import torch.nn as nn

  torch.manual_seed(seed)
  modules = {
    n: o
    for n, o in ns.items()
    if isinstance(o, type)
    and issubclass(o, nn.Module)
    and o is not nn.Module
    and o.__module__ == "__maxkernel_source__"
  }
  if "Model" in modules:
    cls = modules["Model"]
  elif len(modules) == 1:
    cls = next(iter(modules.values()))
  elif not modules:
    raise LookupError("no nn.Module subclass defined in the source")
  else:
    raise LookupError(
      f"{len(modules)} nn.Module subclasses ({', '.join(sorted(modules))}) "
      "and none named `Model`. Ask the user which one is being converted."
    )

  args, kwargs = normalize_init_inputs(
    ns["get_init_inputs"]() if callable(ns.get("get_init_inputs")) else None
  )
  model = cls(*args, **kwargs)
  model.eval()
  return model, cls.__name__


def to_jax(t):
  import jax.numpy as jnp

  return jnp.asarray(t.detach().cpu().numpy())


def flatten_outputs(out):
  import torch

  if isinstance(out, torch.Tensor):
    return [out]
  if isinstance(out, (list, tuple)):
    flat = []
    for o in out:
      flat.extend(flatten_outputs(o))
    return flat
  if isinstance(out, dict):
    flat = []
    for k in sorted(out):
      flat.extend(flatten_outputs(out[k]))
    return flat
  raise TypeError(f"unsupported output type {type(out).__name__}")


def flatten_jax_outputs(out):
  import jax

  return [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]


def validate_oracle(eager_outs, jax_outs, atol, rtol):
  """STEP 0. Does the mechanical translation agree with the real module?"""
  import torch

  report = {"checked": True, "outputs": [], "worst_abs": 0.0, "worst_rel": 0.0}
  if len(eager_outs) != len(jax_outs):
    report["ok"] = False
    report["reason"] = (
      f"output arity differs: eager produced "
      f"{len(eager_outs)}, torchax produced {len(jax_outs)}"
    )
    return report

  ok = True
  for i, (e, j) in enumerate(zip(eager_outs, jax_outs)):
    a = (
      e.detach().cpu().to(torch.float64).numpy()
      if e.is_floating_point()
      else e.detach().cpu().numpy().astype(np.float64)
    )
    b = np.asarray(j, dtype=np.float64)
    if a.shape != b.shape:
      ok = False
      report["outputs"].append(
        {"index": i, "shape_mismatch": [list(a.shape), list(b.shape)]}
      )
      continue
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-30)
    abs_max, rel_max = float(diff.max()), float((diff / denom).max())
    within = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    ok = ok and within
    report["worst_abs"] = max(report["worst_abs"], abs_max)
    report["worst_rel"] = max(report["worst_rel"], rel_max)
    report["outputs"].append(
      {
        "index": i,
        "max_abs_diff": abs_max,
        "max_rel_diff": rel_max,
        "within_tolerance": within,
      }
    )

  report["ok"] = ok
  if not ok:
    report["reason"] = (
      "torchax's translation of this module does not reproduce eager "
      "PyTorch within tolerance. It cannot serve as the correctness "
      "oracle: every later check would be against a computation that is "
      "not the user's."
    )
  return report


def build(
  source_path,
  *,
  seed,
  num_configs,
  atol,
  rtol,
  max_bytes,
  skip_step0,
  backend=None,
):
  import jax
  import torch

  torchax = import_torchax()
  ns = load_source(source_path)
  model, model_name = build_model(ns, seed)

  if not callable(ns.get("get_inputs")):
    raise LookupError(
      "the source defines no get_inputs(); there are no canonical input "
      "shapes to build an oracle over."
    )

  states, jax_fn = torchax.extract_jax(model)
  states_order = list(states.keys())

  jitted = jax.jit(jax_fn)
  arrays, cfgs, total = {}, [], 0

  for idx in range(num_configs):
    torch.manual_seed(seed + idx)
    raw = ns["get_inputs"]()
    raw = [raw] if isinstance(raw, torch.Tensor) else list(raw)
    tensors = [a for a in raw if isinstance(a, torch.Tensor)]
    extras = [a for a in raw if not isinstance(a, torch.Tensor)]

    jax_args = tuple(
      to_jax(a) if isinstance(a, torch.Tensor) else a for a in raw
    )
    jax_out = jitted(states, jax_args)
    jax_flat = flatten_jax_outputs(jax_out)

    step0 = {"checked": False, "reason": "skipped by --skip-eager-check"}
    if not skip_step0:
      with torch.no_grad():
        eager_flat = flatten_outputs(model(*raw))
      step0 = validate_oracle(eager_flat, jax_flat, atol, rtol)
      if not step0["ok"]:
        raise ValueError(json.dumps(step0, indent=2))

    in_meta = []
    for j, a in enumerate(raw):
      if isinstance(a, torch.Tensor):
        arr = a.detach().cpu().numpy()
        key = f"cfg{idx}/in{j}"
        arrays[key] = arr
        total += arr.nbytes
        in_meta.append(
          {
            "key": key,
            "position": j,
            "shape": list(a.shape),
            "dtype": str(a.dtype).replace("torch.", ""),
            "kind": "tensor",
          }
        )
      else:
        in_meta.append(
          {
            "position": j,
            "value": a,
            "python_type": type(a).__name__,
            "kind": "static",
          }
        )

    out_meta = []
    for j, o in enumerate(jax_flat):
      key = f"cfg{idx}/out{j}"
      arrays[key] = o
      total += o.nbytes
      out_meta.append(
        {"key": key, "shape": list(o.shape), "dtype": str(o.dtype)}
      )

    cfgs.append(
      {
        "index": idx,
        "inputs": in_meta,
        "outputs": out_meta,
        "step0": step0,
        "n_tensor_inputs": len(tensors),
        "n_static_inputs": len(extras),
      }
    )

    if total > max_bytes:
      raise MemoryError(
        f"oracle would need {total / 1e6:.1f} MB, over the "
        f"{max_bytes / 1e6:.1f} MB cap; raise --max-bytes or lower --num-configs"
      )

  for name, arr in states.items():
    key = f"states/{name}"
    arrays[key] = np.asarray(arr)
    total += arrays[key].nbytes

  n_tensors = cfgs[0]["n_tensor_inputs"]
  manifest = {
    "schema": "maxkernel.torchax_oracle/1",
    "source_path": str(Path(source_path).resolve()),
    "model_class": model_name,
    "seed": seed,
    "torch_version": torch.__version__,
    "jax_version": jax.__version__,
    "calling_convention": {
      "jax_func": "jax_func(states, args_tuple, kwargs=None)",
      "states": "flat dict of buffers and parameters, keyed by name",
      "states_order": states_order,
      "note": (
        "Flattened for tracing, the signature is STATES FIRST then "
        "forward inputs. capture_torch_golden.py records the opposite "
        "order (forward inputs first, then parameters). Use "
        "flat_signature or golden_compatible_order explicitly -- never "
        "assume."
      ),
      "flat_signature": (
        [{"role": "state", "name": n} for n in states_order]
        + [{"role": "forward_input", "position": i} for i in range(n_tensors)]
      ),
      "golden_compatible_order": (
        [{"role": "forward_input", "position": i} for i in range(n_tensors)]
        + [{"role": "state", "name": n} for n in states_order]
      ),
    },
    "states": {
      n: {
        "key": f"states/{n}",
        "shape": list(np.asarray(states[n]).shape),
        "dtype": str(np.asarray(states[n]).dtype),
      }
      for n in states_order
    },
    "configs": cfgs,
    "total_bytes": total,
    "step0_tolerance": {
      "atol": atol,
      "rtol": rtol,
      "backend": backend or "unknown",
    },
  }
  return arrays, manifest


def main():
  p = argparse.ArgumentParser(
    description="Build a torchax-based correctness oracle for a PyTorch module."
  )
  p.add_argument("source_path")
  p.add_argument("--out", required=True, help="path for the .npz")
  p.add_argument("--meta", required=True, help="path for the .json manifest")
  p.add_argument("--seed", type=int, default=1024)
  p.add_argument("--num-configs", type=int, default=1)
  p.add_argument(
    "--atol",
    type=float,
    default=None,
    help="step-0 tolerance: torchax vs eager PyTorch. Defaults "
    "to the backend's own floor -- see DEFAULT_TOLERANCE.",
  )
  p.add_argument("--rtol", type=float, default=None)
  p.add_argument("--max-bytes", type=int, default=256_000_000)
  p.add_argument(
    "--skip-eager-check",
    action="store_true",
    help="skip step 0. The oracle is then unvalidated -- say so downstream.",
  )
  args = p.parse_args()

  atol, rtol, backend = resolve_tolerance(args.atol, args.rtol)
  if args.atol is None or args.rtol is None:
    print(
      f"step-0 tolerance {atol:g} (backend {backend!r}); override with "
      "--atol/--rtol",
      file=sys.stderr,
    )

  try:
    arrays, manifest = build(
      args.source_path,
      seed=args.seed,
      num_configs=args.num_configs,
      atol=atol,
      rtol=rtol,
      max_bytes=args.max_bytes,
      skip_step0=args.skip_eager_check,
      backend=backend,
    )
  except LookupError as e:
    print(f"NO_ENTRY_POINT: {e}", file=sys.stderr)
    return 3
  except ValueError as e:
    print(
      "STEP0_FAILED: torchax does not reproduce eager PyTorch.", file=sys.stderr
    )
    print(e, file=sys.stderr)
    print(
      f"\nThe comparison ran on backend {backend!r} at atol={atol:g}. On "
      "an accelerator, XLA's default matmul precision is bf16, so a "
      "matmul-bearing module diverges from eager fp32 by roughly bf16 "
      "epsilon (~1e-3) however faithful the conversion is. If the "
      "divergence above is near that scale this is precision, not a "
      "conversion error -- re-run with a matching --atol, or on CPU "
      "where the two agree to ~1e-6.",
      file=sys.stderr,
    )
    return 4
  except MemoryError as e:
    print(f"TOO_LARGE: {e}", file=sys.stderr)
    return 5
  except Exception as e:  # pylint: disable=broad-except
    print(
      f"DEGRADED: could not build a torchax oracle: {type(e).__name__}: {e}",
      file=sys.stderr,
    )
    traceback.print_exc()
    return 2

  np.savez(args.out, **arrays)
  Path(args.meta).write_text(json.dumps(manifest, indent=2, default=str) + "\n")

  c0 = manifest["configs"][0]
  s0 = c0["step0"]
  print(f"ORACLE_OK model={manifest['model_class']}")
  print(
    f"  states        : {len(manifest['states'])} {list(manifest['states'])}"
  )
  print(
    f"  inputs        : {c0['n_tensor_inputs']} tensor, {c0['n_static_inputs']} static"
  )
  print(f"  outputs       : {len(c0['outputs'])}")
  print(f"  total bytes   : {manifest['total_bytes']}")
  if s0.get("checked"):
    print(
      f"  STEP 0        : PASS  torchax vs eager max|diff|={s0['worst_abs']:.3e} "
      f"rel={s0['worst_rel']:.3e}"
    )
  else:
    print(
      f"  STEP 0        : NOT CHECKED ({s0.get('reason')}) -- oracle unvalidated"
    )
  return 0


if __name__ == "__main__":
  sys.exit(main())
