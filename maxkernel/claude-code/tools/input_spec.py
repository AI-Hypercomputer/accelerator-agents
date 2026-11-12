#!/usr/bin/env python3
"""Declared entry points and input configs, for sources that do not ship them.

WHY THIS EXISTS
---------------
`capture_torch_golden.py` and `torchax_oracle.py` were written against the
KernelBench layout: one file, a `Model` class, `get_init_inputs()` and
`get_inputs()`. Code from the wild usually has none of that: a layer in a
GitHub repository takes several constructor arguments, ships no input factory,
and keeps its realistic shapes in a test or benchmark script. general_rules #6 forbids inventing dimensions, so
the only legitimate source of shapes for such code is the user, through the
job file.

This module is the one place that turns that declaration into Python objects.
Both capture tools import it, so the two oracles cannot disagree about what a
declared config means, and `validate_job.py` imports `validate_input_spec` so a
malformed config fails before a TPU is touched. Nothing at module level imports
torch: the validator runs without it.

THE SPEC FILE
-------------
The orchestrator writes `<run_dir>/input_spec.json` from the job's `input`
object:

    {
      "init_args":   [ ... ],              # constructor positional args
      "init_kwargs": { ... },              # constructor keyword args
      "configs": [                         # one entry per captured config
        {"name": "T8192",
         "inputs": [                       # forward() args, in order
           {"name": "hidden_states", "shape": [8192, 768],
            "dtype": "bfloat16", "init": "randn", "scale": 1.0},
           {"name": "top_k", "value": 8}   # a static (non-tensor) argument
         ]}
      ]
    }

Every field is optional. Constructor arguments are JSON values, so a source
whose constructor takes an enum must accept its string value -- an extracted
source (Phase 0.15) is written that way on purpose.

OUTPUTS
-------
`output_leaves` is the one rule for what a forward() may return: tensors,
nested in lists, tuples and dicts (dict keys in sorted order, as
jax.tree_util.tree_leaves orders them). Both capture tools and the smoke check
use it, so a source that passes smoke cannot fail capture on its outputs. A
`None` leaf -- e.g. `(output, None)` from a layer whose auxiliary loss is None
in inference mode -- is rejected by name.

CLI:
  input_spec.py smoke <source.py> --input-spec <spec.json> [--entry-point Model]
    Builds the module exactly as the capture tools would, runs config 0 once on
    CPU, checks its outputs with `output_leaves`, and prints their shapes. Exit
    0 ok, 3 no entry point, 2 it did not run or returned something the capture
    tools cannot record.
"""

import argparse
import json
import os
import sys
from pathlib import Path

DTYPES = {
  "float32",
  "float16",
  "bfloat16",
  "float64",
  "int64",
  "int32",
  "int16",
  "int8",
  "uint8",
  "bool",
}
FLOAT_DTYPES = {"float32", "float16", "bfloat16", "float64"}
INIT_KINDS = {"randn", "rand", "randint", "zeros", "ones"}


def validate_input_spec(spec, where="input"):
  """Returns a list of error strings; empty means the spec is usable.

  `where` prefixes every message with the job-file path of the field, so an
  error names what to edit.
  """
  errors = []
  if not isinstance(spec, dict):
    return [f"{where} must be an object"]

  init_args = spec.get("init_args")
  if init_args is not None and not isinstance(init_args, list):
    errors.append(f"{where}.init_args must be a list")
  init_kwargs = spec.get("init_kwargs")
  if init_kwargs is not None and not isinstance(init_kwargs, dict):
    errors.append(f"{where}.init_kwargs must be an object")

  configs = spec.get("configs")
  if configs is None:
    return errors
  if not isinstance(configs, list) or not configs:
    errors.append(f"{where}.configs must be a non-empty list")
    return errors

  for i, cfg in enumerate(configs):
    cw = f"{where}.configs[{i}]"
    if not isinstance(cfg, dict):
      errors.append(f"{cw} must be an object")
      continue
    inputs = cfg.get("inputs")
    if not isinstance(inputs, list) or not inputs:
      errors.append(f"{cw}.inputs must be a non-empty list of forward arguments")
      continue
    for j, arg in enumerate(inputs):
      errors.extend(_validate_arg(arg, f"{cw}.inputs[{j}]"))
  return errors


def _validate_arg(arg, aw):
  if not isinstance(arg, dict):
    return [f"{aw} must be an object"]
  if "value" in arg:
    extra = sorted(set(arg) & {"shape", "dtype", "init"})
    if extra:
      return [
        f"{aw} has both `value` (a static argument) and {extra} (a tensor); "
        "pick one"
      ]
    return []

  errors = []
  shape = arg.get("shape")
  if not isinstance(shape, list) or not all(
    isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
  ):
    errors.append(
      f"{aw}.shape must be a list of non-negative integers (or give `value` "
      "for a non-tensor argument)"
    )
  dtype = arg.get("dtype")
  if dtype not in DTYPES:
    errors.append(f"{aw}.dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    return errors

  init = arg.get("init", "randn" if dtype in FLOAT_DTYPES else None)
  if init is None:
    errors.append(
      f"{aw} is a {dtype} tensor with no `init`. Integer tensors are usually "
      "indices, and their valid range is part of the problem: give "
      "`init: \"randint\"` with `low` and `high`, or `zeros`/`ones`."
    )
  elif init not in INIT_KINDS:
    errors.append(f"{aw}.init must be one of {sorted(INIT_KINDS)}, got {init!r}")
  elif init in ("randn", "rand") and dtype not in FLOAT_DTYPES:
    errors.append(f"{aw}.init {init!r} needs a floating dtype, got {dtype!r}")
  elif init == "randint":
    low, high = arg.get("low"), arg.get("high")
    if not (isinstance(low, int) and isinstance(high, int) and low < high):
      errors.append(f"{aw}.init 'randint' needs integer `low` < `high`")
  scale = arg.get("scale")
  if scale is not None and not isinstance(scale, (int, float)):
    errors.append(f"{aw}.scale must be a number")
  return errors


def load_input_spec(path):
  """Reads and validates a spec file. Raises ValueError on a bad spec."""
  if not path:
    return None
  spec = json.loads(Path(path).read_text())
  errors = validate_input_spec(spec, where=str(path))
  if errors:
    raise ValueError("invalid input spec:\n  " + "\n  ".join(errors))
  return spec


def has_init(spec):
  return bool(spec) and ("init_args" in spec or "init_kwargs" in spec)


def init_inputs(spec):
  """The declared constructor arguments, in the `(args, kwargs)` convention."""
  return list(spec.get("init_args") or []), dict(spec.get("init_kwargs") or {})


def config_count(spec):
  return len(spec.get("configs") or []) if spec else 0


def build_args(spec, index, seed):
  """Materializes config `index` as a list of forward arguments (torch)."""
  import torch

  cfg = spec["configs"][index]
  gen = torch.Generator().manual_seed(seed)
  args = []
  for arg in cfg["inputs"]:
    if "value" in arg:
      args.append(arg["value"])
      continue
    dtype = getattr(torch, arg["dtype"])
    shape = tuple(arg["shape"])
    init = arg.get("init", "randn")
    if init == "randn":
      t = torch.randn(shape, generator=gen, dtype=torch.float32)
    elif init == "rand":
      t = torch.rand(shape, generator=gen, dtype=torch.float32)
    elif init == "randint":
      t = torch.randint(arg["low"], arg["high"], shape, generator=gen)
    elif init == "zeros":
      t = torch.zeros(shape)
    else:
      t = torch.ones(shape)
    if arg.get("scale") is not None:
      t = t * arg["scale"]
    # Sample in fp32/int64 and cast once, so a bf16 config draws the same
    # values a float32 config would, rounded -- not a different stream.
    args.append(t.to(dtype))
  return args


class MissingInitArgs(LookupError):
  """The module's constructor cannot be called with the arguments available."""


def check_init_args(cls, args, kwargs):
  """Binds the constructor arguments against `cls`'s signature, or raises.

  Without this a constructor with required parameters and nothing declared
  for them fails inside `cls(...)` with a TypeError, which the capture tools
  report as "the module would not run on CPU" and degrade past -- leaving the
  run with an unchecked port for a reason the user can fix in one line. A
  LookupError subclass, so every tool stops with exit 3 and asks instead.
  """
  import inspect

  try:
    sig = inspect.signature(cls)
  except (TypeError, ValueError):
    return  # no introspectable signature; let construction decide
  try:
    sig.bind(*args, **kwargs)
  except TypeError as e:
    raise MissingInitArgs(
      f"{cls.__name__}{sig} cannot be constructed from the arguments "
      f"available ({e}). Declare them in the job as input.init_args / "
      "input.init_kwargs, or name the source's factory with "
      "input.init_inputs_fn."
    ) from None


class UnsupportedOutput(TypeError):
  """forward() returned a leaf that is not a tensor."""


def output_leaves(out, path="output"):
  """Flattens a forward() result into its tensor leaves, or raises.

  Raises UnsupportedOutput naming the first non-tensor leaf by its path, e.g.
  `output[1] is NoneType` -- which is what the source has to stop returning.
  """
  import torch

  if isinstance(out, torch.Tensor):
    return [out]
  if isinstance(out, (list, tuple)):
    flat = []
    for i, item in enumerate(out):
      flat.extend(output_leaves(item, f"{path}[{i}]"))
    return flat
  if isinstance(out, dict):
    flat = []
    for key in sorted(out):
      flat.extend(output_leaves(out[key], f"{path}[{key!r}]"))
    return flat
  raise UnsupportedOutput(
    f"{path} is {type(out).__name__}; forward() may only return tensors, "
    "nested in lists, tuples or dicts. Drop it from the return value"
    + (" (a None output, e.g. an aux loss that is None in inference mode)"
       if out is None else "")
    + "."
  )


def resolve_entry(ns, name, source_module_name):
  """Looks up a named entry point. Returns ("module", cls) or ("function", fn).

  Only names defined in the user's file count: an `nn.Module` the file merely
  imported is not what it asked to have converted.
  """
  import torch.nn as nn

  obj = ns.get(name)
  if obj is None:
    defined = sorted(
      n
      for n, o in ns.items()
      if getattr(o, "__module__", None) == source_module_name
      and (isinstance(o, type) or callable(o))
    )
    raise LookupError(
      f"entry point {name!r} is not defined in the source. Defined here: "
      f"{', '.join(defined) or '(nothing)'}"
    )
  if isinstance(obj, type) and issubclass(obj, nn.Module):
    return "module", obj
  if callable(obj):
    return "function", obj
  raise LookupError(f"entry point {name!r} is neither an nn.Module nor callable")


def _smoke(args):
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
  import torch

  path = Path(args.source)
  ns = {"__name__": "__maxkernel_source__", "__file__": str(path.resolve())}
  sys.path.insert(0, str(path.parent.resolve()))
  try:
    exec(compile(path.read_text(), str(path), "exec"), ns)  # pylint: disable=exec-used
  finally:
    sys.path.pop(0)

  spec = load_input_spec(args.input_spec)
  kind, obj = resolve_entry(ns, args.entry_point, "__maxkernel_source__")
  torch.manual_seed(args.seed)
  if kind == "module":
    init_a, init_k = init_inputs(spec) if has_init(spec) else ([], {})
    check_init_args(obj, init_a, init_k)
    callee = obj(*init_a, **init_k).eval()
  else:
    callee = obj
  if not config_count(spec):
    raise LookupError("the spec declares no configs to run")
  fwd = build_args(spec, 0, args.seed)
  with torch.no_grad():
    out = callee(*fwd)
  leaves = output_leaves(out)
  print(f"SMOKE_OK entry={args.entry_point} kind={kind}")
  for i, o in enumerate(leaves):
    print(f"  out{i}: shape={list(o.shape)} dtype={str(o.dtype).replace('torch.', '')}")


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  sub = p.add_subparsers(dest="cmd", required=True)
  s = sub.add_parser("smoke", help="run config 0 of a spec once on CPU")
  s.add_argument("source")
  s.add_argument("--input-spec", required=True)
  s.add_argument("--entry-point", default="Model")
  s.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  try:
    _smoke(args)
  except MissingInitArgs as e:
    print(f"MISSING_INIT_ARGS: {e}", file=sys.stderr)
    return 3
  except LookupError as e:
    print(f"NO_ENTRY_POINT: {e}", file=sys.stderr)
    return 3
  except Exception as e:  # pylint: disable=broad-except
    print(f"SMOKE_FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    return 2
  return 0


if __name__ == "__main__":
  sys.exit(main())
