#!/usr/bin/env python3
"""Executes the user's PyTorch reference on CPU and records inputs + outputs.

WHY THIS EXISTS
---------------
For a non-JAX input, `<run_dir>/base.py` is a JAX *port* of the user's source,
written by an LLM, and it is the denominator of every number the run reports.
The worker's harness validation cannot check it: that phase binds `base.py` as
both `base_computation` and `opt_computation`, so it compares the baseline
against itself and can never detect a mistranslated port. A port that computes
the wrong thing produces five iterations of confident, wrong speedups.

That hole is unavoidable for CUDA -- a TPU host cannot run it. It is entirely
avoidable for PyTorch, because torch runs on the CPU in this very venv. So run
the user's actual module, keep the inputs and the answers, and let
`verify_port.py` check `base.py` against them on the TPU.

This tool only CAPTURES. It does not compare, does not write `base.py`, does
not touch the TPU, and has no LLM in the loop.

THE THREE HAZARDS IT IS BUILT AROUND
------------------------------------
1.  **Parameters.** An `nn.Module` holds `self.weight`, randomly initialized at
    construction. The JAX `computation` has no `self`, so that weight must
    arrive as an *argument*. Capturing inputs but not parameters makes the
    recorded outputs unreproducible noise. Every parameter and buffer is
    captured, in a stable order, and the manifest says which argument position
    each one takes.
2.  **Determinism.** Dropout or any RNG live in `forward()` means there is no
    golden value to capture. `model.eval()` and `torch.no_grad()` handle the
    common case; a two-run bitwise probe catches the rest, and exits rather
    than recording one sample and calling it truth.
3.  **bf16 has no numpy dtype.** Silently upcasting it on the way into the
    `.npz` would let the golden check pass against a port that is wrong about
    precision -- exactly the failure this tool exists to catch. Sub-fp16 dtypes
    are stored as their raw bit pattern plus a tag, and reconstituted through
    `ml_dtypes` (which JAX ships) on the reading side.

ARGUMENT ORDER CONTRACT
-----------------------
The assembled harness requires dynamic (traced) arguments first and static
(compile-time) arguments last -- see `tools/test_harness_template.py`, which
builds `static_argnums` as `range(len(dynamic_args), len(args))`. So the
canonical signature this tool records, and the one `base.py` must expose, is:

    computation(*forward_tensors, *parameters_and_buffers, *static_scalars)

Exit codes:
  0  golden captured
  2  torch missing, or the module will not run on CPU -- the caller should
     DEGRADE GRACEFULLY to self-comparison, not abort the run
  3  no discoverable entry point -- ask the user
  4  forward() is non-deterministic -- no golden value exists to capture
  5  the capture would exceed --max-bytes
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Force CPU before torch is imported. A module that would otherwise grab a GPU
# must not: the golden values have to be reproducible on the machine running
# the loop, which is a TPU host with no CUDA device.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# `--device xla` runs the reference on the TPU through torch_xla instead of in
# eager CPU. Detected before torch is imported, because torch_xla has to come
# in alongside torch rather than after it.
_WANT_XLA = "--device" in sys.argv and sys.argv[
  sys.argv.index("--device") + 1 : sys.argv.index("--device") + 2
] == ["xla"]

try:
  import numpy as np
except ImportError:  # pragma: no cover
  print("numpy is required", file=sys.stderr)
  sys.exit(2)

try:
  import torch
  import torch.nn as nn
except ImportError as e:
  print(f"DEGRADED: torch is not importable ({e})", file=sys.stderr)
  sys.exit(2)

_XLA = None
if _WANT_XLA:
  try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    _XLA = (torch_xla, xm)
  except ImportError as e:
    print(
      f"DEGRADED: --device xla needs torch_xla, which is not installed "
      f"({e}). requirements.txt pins torch==2.9.0 for it -- torch_xla's "
      "release must match the torch minor version. Install a build "
      "matching your libtpu, or use --device cpu.",
      file=sys.stderr,
    )
    sys.exit(2)


def _device():
  return _XLA[1].xla_device() if _XLA else torch.device("cpu")


def _sync():
  if _XLA:
    _XLA[0].sync(wait=True)


# ---------------------------------------------------------------------------
# dtype encoding
# ---------------------------------------------------------------------------

# Dtypes numpy cannot represent natively. Stored as the raw bit pattern under
# an integer view; the manifest carries the real dtype so the reader can
# reinterpret through ml_dtypes.
BIT_VIEW = {
  torch.bfloat16: ("bfloat16", torch.uint16),
  getattr(torch, "float8_e4m3fn", None): ("float8_e4m3fn", torch.uint8),
  getattr(torch, "float8_e5m2", None): ("float8_e5m2", torch.uint8),
}
BIT_VIEW.pop(None, None)


def encode_tensor(t):
  """Returns (np.ndarray, dtype_tag, encoding) for one tensor."""
  t = t.detach().cpu().contiguous()
  if t.dtype in BIT_VIEW:
    tag, view_dtype = BIT_VIEW[t.dtype]
    return t.view(view_dtype).numpy(), tag, "raw_bits"
  return t.numpy(), str(t.dtype).replace("torch.", ""), "native"


def to_float64(t):
  """Upcasts for numerical comparison, via float32 for the bit-view dtypes."""
  if t.dtype in BIT_VIEW:
    return t.detach().cpu().to(torch.float32).to(torch.float64)
  if t.is_floating_point():
    return t.detach().cpu().to(torch.float64)
  return t.detach().cpu()


# ---------------------------------------------------------------------------
# Source loading and entry-point discovery
# ---------------------------------------------------------------------------


def load_source(source_path):
  """Executes the user's file in its OWN namespace, never into globals().

  Same isolation rationale as `tools/assemble_test_harness.py`: two modules
  that both define a helper called `kernel` must not collide, and the user's
  file must not be able to rebind names this tool depends on.
  """
  path = Path(source_path)
  src = path.read_text()
  ns = {"__name__": "__maxkernel_source__", "__file__": str(path.resolve())}
  sys.path.insert(0, str(path.parent.resolve()))
  try:
    exec(compile(src, str(path), "exec"), ns)  # pylint: disable=exec-used
  finally:
    sys.path.pop(0)
  return ns


class EntryPoint:
  """A normalized handle on whatever shape the user's file happens to be."""

  def __init__(
    self,
    kind,
    name,
    model_cls=None,
    fn=None,
    init_inputs_fn=None,
    inputs_fn=None,
  ):
    self.kind = kind
    self.name = name
    self.model_cls = model_cls
    self.fn = fn
    self.init_inputs_fn = init_inputs_fn
    self.inputs_fn = inputs_fn


def discover_entry_point(ns):
  """Finds the callable the loop should reproduce.

  Never guesses between two equally plausible candidates -- an ambiguous file
  exits 3 and the orchestrator asks the user, because binding the wrong entry
  point silently benchmarks the wrong computation.
  """
  get_inputs = ns.get("get_inputs")
  get_init = ns.get("get_init_inputs")

  modules = {
    name: obj
    for name, obj in ns.items()
    if isinstance(obj, type)
    and issubclass(obj, nn.Module)
    and obj is not nn.Module
    and obj.__module__ == "__maxkernel_source__"
  }

  # KernelBench layout: a Model class plus the two input factories.
  if "Model" in modules and callable(get_inputs):
    return EntryPoint(
      "kernelbench",
      "Model.forward",
      model_cls=modules["Model"],
      init_inputs_fn=get_init,
      inputs_fn=get_inputs,
    )

  if len(modules) == 1:
    name, cls = next(iter(modules.items()))
    return EntryPoint(
      "module",
      f"{name}.forward",
      model_cls=cls,
      init_inputs_fn=get_init,
      inputs_fn=get_inputs,
    )

  if len(modules) > 1:
    if "Model" in modules:
      return EntryPoint(
        "module",
        "Model.forward",
        model_cls=modules["Model"],
        init_inputs_fn=get_init,
        inputs_fn=get_inputs,
      )
    raise LookupError(
      f"{len(modules)} nn.Module subclasses defined ({', '.join(sorted(modules))}) "
      "and none is named `Model`. Ask the user which one is being converted."
    )

  for candidate in ("computation", "forward", "main_computation", "run"):
    fn = ns.get(candidate)
    if callable(fn):
      return EntryPoint("function", candidate, fn=fn, inputs_fn=get_inputs)

  raise LookupError(
    "No nn.Module subclass and no module-level `computation`/`forward` "
    "function found."
  )


def materialize(entry, seed):
  """Instantiates the model and collects its parameters and buffers.

  Returns (callable, params) where `params` is an ordered name -> tensor map.
  The order is `named_parameters()` then `named_buffers()`, both in the order
  torch reports them, and it is stable across runs of the same file -- which
  is what lets the manifest pin each one to an argument position.
  """
  torch.manual_seed(seed)

  if entry.kind == "function":
    return entry.fn, {}

  init_args, init_kwargs = [], {}
  if callable(entry.init_inputs_fn):
    raw = entry.init_inputs_fn()
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
      init_args, init_kwargs = list(raw[0]), dict(raw[1])
    elif isinstance(raw, (list, tuple)):
      init_args = list(raw)
    elif isinstance(raw, dict):
      init_kwargs = dict(raw)

  model = entry.model_cls(*init_args, **init_kwargs)
  model.eval()
  if _XLA:
    model = model.to(_device())
    _sync()

  params = {}
  for name, p in model.named_parameters():
    params[name] = p.detach().cpu()
  for name, b in model.named_buffers():
    params[f"buffer.{name}"] = b.detach().cpu()
  return model, params


# Attributes nn.Module sets on itself. They describe the module's state, not
# the computation, and must not become arguments to `computation`.
MODULE_BOOKKEEPING = {"training", "call_super_init", "dump_patches"}


def collect_module_scalars(model):
  """Captures constructor scalars held as plain attributes on the module.

  `self.weight = nn.Parameter(...)` is a runtime array and shows up in
  `named_parameters()`. `self.eps = 1e-5` is a trace-time constant and shows
  up in neither `named_parameters()` nor `named_buffers()` -- but the JAX
  `computation` still needs it, as a `static_argnums` argument, because it has
  no `self` to read it from.

  Missing these is a silent port failure: the analyzer writes a `computation`
  that hard-codes a default `eps` while the golden outputs were produced with
  the user's value, and the discrepancy surfaces later as an unexplained
  correctness mismatch.
  """
  if not isinstance(model, nn.Module):
    return {}
  scalars = {}
  for name, value in sorted(vars(model).items()):
    if name.startswith("_") or name in MODULE_BOOKKEEPING:
      continue
    if isinstance(value, (int, float, bool, str)):
      scalars[name] = value
  return scalars


# ---------------------------------------------------------------------------
# Input configs
# ---------------------------------------------------------------------------


def is_tensor(x):
  return isinstance(x, torch.Tensor)


def split_dynamic_static(args):
  """Partitions call arguments into (tensors, scalars).

  Tensors are traced; scalars are `static_argnums` on the JAX side -- values
  the kernel branches on at trace time. Anything that is neither (a list, a
  dict) is treated as static, since it cannot be a traced array argument.
  """
  dynamic, static = [], []
  for a in args:
    (dynamic if is_tensor(a) else static).append(a)
  return dynamic, static


def build_configs(entry, seed, num_configs):
  """Builds the list of forward-argument sets to capture.

  Strongly prefers the source's own `get_inputs()`: a KernelBench task ships
  the shapes it was written for, and general_rules #6 forbids substituting our
  own problem dimensions.
  """
  if not callable(entry.inputs_fn):
    raise LookupError(
      "The source defines no `get_inputs()`, so there are no canonical input "
      "shapes to capture. Ask the user for representative inputs rather than "
      "inventing dimensions (general_rules #6)."
    )

  configs = []
  for i in range(num_configs):
    torch.manual_seed(seed + i)
    raw = entry.inputs_fn()
    if isinstance(raw, torch.Tensor):
      raw = [raw]
    if not isinstance(raw, (list, tuple)):
      raise TypeError(
        f"get_inputs() returned {type(raw).__name__}; expected a list or "
        "tuple of forward arguments."
      )
    configs.append(list(raw))
  return configs


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def flatten_outputs(out):
  """Flattens the forward() result to match jax.tree_util.tree_leaves ordering."""
  if isinstance(out, torch.Tensor):
    return [out]
  if isinstance(out, (list, tuple)):
    flat = []
    for item in out:
      flat.extend(flatten_outputs(item))
    return flat
  if isinstance(out, dict):
    flat = []
    for key in sorted(out):  # tree_leaves sorts dict keys
      flat.extend(flatten_outputs(out[key]))
    return flat
  raise TypeError(f"Unsupported output type: {type(out).__name__}")


def to_device(x):
  return x.to(_device()) if isinstance(x, torch.Tensor) else x


def run_forward(callee, args):
  with torch.no_grad():
    out = flatten_outputs(callee(*[to_device(a) for a in args]))
    _sync()
    # Bring results back to the host, so every comparison downstream is
    # host-side numpy regardless of which device produced it.
    return [o.detach().cpu() for o in out]


def probe_determinism(callee, args):
  """Runs forward twice and compares bitwise.

  Dropout, `torch.rand` inside forward, or any non-deterministic reduction
  means there is no single correct answer to record.
  """
  first = run_forward(callee, args)
  second = run_forward(callee, args)
  if len(first) != len(second):
    return False, "output arity differs between two identical calls"
  for i, (a, b) in enumerate(zip(first, second)):
    if a.shape != b.shape:
      return False, f"output {i} changed shape between two identical calls"
    if not torch.equal(
      a.detach().cpu().view(torch.int8 if a.dtype in BIT_VIEW else a.dtype),
      b.detach().cpu().view(torch.int8 if b.dtype in BIT_VIEW else b.dtype),
    ):
      return False, f"output {i} differs bitwise between two identical calls"
  return True, None


def run_fp64_probe(entry, model, args):
  # TPUs have no usable float64 path. On XLA the probe would either fail or
  # silently run in fp32 and report a divergence of zero -- which reads as
  # "perfectly conditioned" and is the opposite of the truth.
  if _XLA:
    return None
  """Re-runs the same graph in float64 to measure the reference's own spread.

  Advisory only. Many graphs will not survive the upcast (ops without a
  float64 CPU kernel, hard-coded dtypes); that is not a failure, it just means
  no tolerance recommendation can be derived.
  """
  try:
    if entry.kind == "function":
      f64_args = [to_float64(a) if is_tensor(a) else a for a in args]
      return run_forward(model, f64_args)
    f64_model = model.double()
    f64_args = [to_float64(a) if is_tensor(a) else a for a in args]
    out = run_forward(f64_model, f64_args)
    model.float()
    return out
  except Exception:  # pylint: disable=broad-except
    return None


def recommend_tolerances(native, f64):
  """Derives atol/rtol from the reference's own fp32-vs-fp64 divergence.

  Advisory. The run's real tolerances still come from the user (SKILL.md);
  this exists so that choice can be argued from the source's conditioning
  instead of guessed.
  """
  if f64 is None or len(native) != len(f64):
    return None

  worst_abs, worst_rel = 0.0, 0.0
  per_output = []
  for i, (a, b) in enumerate(zip(native, f64)):
    if not a.is_floating_point() and a.dtype not in BIT_VIEW:
      per_output.append({"index": i, "skipped": "non-floating output"})
      continue
    x = to_float64(a).numpy().astype(np.float64)
    y = to_float64(b).numpy().astype(np.float64)
    if x.shape != y.shape:
      per_output.append({"index": i, "skipped": "shape differs under float64"})
      continue

    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
      per_output.append({"index": i, "skipped": "no finite elements"})
      continue
    x, y = x[finite], y[finite]

    diff = np.abs(x - y)
    abs_max = float(diff.max()) if diff.size else 0.0

    # Relative error is only meaningful where the reference value is
    # meaningfully non-zero. A softmax row has entries at 1e-20; dividing by
    # those produces a "relative error" in the hundreds that says nothing
    # about the computation and, taken literally, would set a tolerance that
    # accepts any output at all. Measure rel error only on elements at least
    # REL_FLOOR of the tensor's own scale, and report how much was excluded.
    scale = float(np.abs(y).max()) if y.size else 0.0
    REL_FLOOR = 1e-6
    if scale > 0:
      significant = np.abs(y) >= REL_FLOOR * scale
    else:
      significant = np.zeros_like(y, dtype=bool)

    if significant.any():
      rel = diff[significant] / np.abs(y[significant])
      rel_max = float(rel.max())
      # Use a high percentile, not the max, as the recommendation driver. A
      # single element sitting on a cancellation (a near-zero result of a
      # large-magnitude subtraction) has an unbounded relative error that says
      # nothing about the computation as a whole. `jnp.allclose` tests
      # `|a-b| <= atol + rtol*|b|`, so atol is what covers those elements --
      # rtol should describe the bulk.
      rel_p999 = float(np.percentile(rel, 99.9))
      excluded = float(1.0 - significant.mean())
    else:
      rel_max = rel_p999 = 0.0
      excluded = 1.0

    worst_abs = max(worst_abs, abs_max)
    worst_rel = max(worst_rel, rel_p999)
    per_output.append(
      {
        "index": i,
        "max_abs_diff": abs_max,
        "max_rel_diff": rel_max,
        "p999_rel_diff": rel_p999,
        "tensor_scale": scale,
        "fraction_below_rel_floor": excluded,
      }
    )

  # Headroom over the reference's own error: the port is allowed to be about
  # an order of magnitude looser than the source's fp32-vs-fp64 spread before
  # the difference is more likely a mistranslation than accumulated rounding.
  # Clamped so a pathological output cannot recommend a tolerance that would
  # accept anything -- a tolerance of 1.0 is not a tolerance.
  FLOOR, CEILING = 1e-6, 0.5
  raw_atol, raw_rtol = worst_abs * 10.0, worst_rel * 10.0
  atol = float(min(max(raw_atol, FLOOR), CEILING))
  rtol = float(min(max(raw_rtol, FLOOR), CEILING))
  clamped = raw_atol > CEILING or raw_rtol > CEILING
  return {
    "atol": atol,
    "rtol": rtol,
    "clamped": clamped,
    "basis": (
      f"fp32-vs-fp64 divergence of the reference itself: max abs="
      f"{worst_abs:.3e}, p99.9 rel={worst_rel:.3e} (relative error over "
      f"elements >= {REL_FLOOR:g} of each tensor's scale; the p99.9 rather "
      "than the max, so one cancellation element cannot drive it). "
      "Recommendation is 10x each. jnp.allclose tests "
      "|a-b| <= atol + rtol*|b|, so these are a pair, not two independent "
      "thresholds" + (f"; clamped to {CEILING}" if clamped else "")
    ),
    "per_output": per_output,
  }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def describe(key, tensor):
  array, tag, encoding = encode_tensor(tensor)
  return array, {
    "key": key,
    "shape": list(tensor.shape),
    "dtype": tag,
    "encoding": encoding,
    "bytes": int(array.nbytes),
  }


def capture(
  source_path, seed, num_configs, dtype_policy, max_bytes, fp64_probe
):
  ns = load_source(source_path)
  entry = discover_entry_point(ns)
  model, params = materialize(entry, seed)
  module_scalars = collect_module_scalars(model)
  configs = build_configs(entry, seed, num_configs)

  if dtype_policy == "fp32":
    configs = [
      [a.float() if is_tensor(a) and a.is_floating_point() else a for a in cfg]
      for cfg in configs
    ]
    if entry.kind != "function":
      model = model.float()
      params = {
        k: (v.float() if v.is_floating_point() else v)
        for k, v in params.items()
      }

  arrays = {}
  manifest_configs = []
  total_bytes = 0

  for idx, args in enumerate(configs):
    dynamic, static = split_dynamic_static(args)

    ok, reason = probe_determinism(model, args)
    if not ok:
      raise RuntimeError(
        f"forward() is not deterministic: {reason}. There is no golden value "
        "to capture. Put the module in eval mode, seed any RNG it uses, or "
        "tell the loop to fall back to self-comparison."
      )

    outputs = run_forward(model, args)
    f64_outputs = run_fp64_probe(entry, model, args) if fp64_probe else None
    tolerance = recommend_tolerances(outputs, f64_outputs)

    dyn_meta = []
    for j, t in enumerate(dynamic):
      array, meta = describe(f"cfg{idx}/in{j}", t)
      arrays[meta["key"]] = array
      meta["role"] = "forward_input"
      meta["argnum"] = j
      dyn_meta.append(meta)
      total_bytes += meta["bytes"]

    # Parameters follow the forward tensors, still as DYNAMIC arguments: the
    # JAX `computation` has no `self` to read them from.
    offset = len(dynamic)
    for k, (name, t) in enumerate(params.items()):
      array, meta = describe(f"cfg{idx}/param/{name}", t)
      arrays[meta["key"]] = array
      meta["role"] = "parameter"
      meta["name"] = name
      meta["argnum"] = offset + k
      dyn_meta.append(meta)
      total_bytes += meta["bytes"]

    # Static arguments, in two groups: scalars passed to forward(), then
    # constructor scalars held on the module. Both are `static_argnums` on the
    # JAX side and both must appear in `computation`'s signature.
    static_meta = []
    static_offset = len(dynamic) + len(params)
    cursor = static_offset
    for value in static:
      static_meta.append(
        {
          "argnum": cursor,
          "name": None,
          "origin": "forward_argument",
          "value": value
          if isinstance(value, (int, float, bool, str, type(None)))
          else repr(value),
          "python_type": type(value).__name__,
        }
      )
      cursor += 1
    for name, value in module_scalars.items():
      static_meta.append(
        {
          "argnum": cursor,
          "name": name,
          "origin": "module_attribute",
          "value": value,
          "python_type": type(value).__name__,
        }
      )
      cursor += 1

    out_meta = []
    for j, t in enumerate(outputs):
      array, meta = describe(f"cfg{idx}/out{j}", t)
      arrays[meta["key"]] = array
      out_meta.append(meta)
      total_bytes += meta["bytes"]

    if f64_outputs is not None:
      for j, t in enumerate(f64_outputs):
        array, meta = describe(f"cfg{idx}/out64_{j}", t)
        arrays[meta["key"]] = array
        total_bytes += meta["bytes"]

    manifest_configs.append(
      {
        "index": idx,
        "dynamic": dyn_meta,
        "static": static_meta,
        "static_argnums": [m["argnum"] for m in static_meta],
        "outputs": out_meta,
        "tolerance_recommendation": tolerance,
      }
    )

    if total_bytes > max_bytes:
      raise MemoryError(
        f"capture would need {total_bytes / 1e6:.1f} MB, over the "
        f"{max_bytes / 1e6:.1f} MB cap. Re-run with a larger --max-bytes or "
        "fewer --num-configs."
      )

  manifest = {
    "source_path": str(Path(source_path).resolve()),
    "entry_point": {
      "kind": entry.kind,
      "name": entry.name,
      "parameter_order": list(params.keys()),
      "module_scalars": module_scalars,
    },
    "argument_contract": (
      "computation(*forward_tensors, *parameters_and_buffers, *static_scalars) "
      "-- dynamic arguments first, static last, as "
      "tools/test_harness_template.py requires"
    ),
    "seed": seed,
    "dtype_policy": dtype_policy,
    "torch_version": torch.__version__,
    "device": "xla" if _XLA else "cpu",
    "numpy_version": np.__version__,
    "deterministic": True,
    "total_bytes": total_bytes,
    "configs": manifest_configs,
    "degraded": None,
  }
  return arrays, manifest


def main():
  parser = argparse.ArgumentParser(
    description="Capture golden inputs/outputs from a PyTorch reference on CPU."
  )
  parser.add_argument("source_path")
  parser.add_argument("--out", required=True, help="path for the .npz")
  parser.add_argument(
    "--meta", required=True, help="path for the .json manifest"
  )
  parser.add_argument(
    "--device",
    default="cpu",
    choices=["cpu", "xla"],
    help="cpu (eager PyTorch -- the default, and the better "
    "semantic check) or xla (run the reference on the "
    "TPU through torch_xla). Never cuda: the loop runs "
    "on a TPU host.",
  )
  parser.add_argument(
    "--dtype-policy", default="preserve", choices=["preserve", "fp32"]
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num-configs", type=int, default=1)
  parser.add_argument("--max-bytes", type=int, default=256_000_000)
  parser.add_argument(
    "--no-fp64-probe",
    action="store_true",
    help="skip the float64 recomputation and tolerance advice",
  )
  args = parser.parse_args()

  try:
    arrays, manifest = capture(
      args.source_path,
      seed=args.seed,
      num_configs=args.num_configs,
      dtype_policy=args.dtype_policy,
      max_bytes=args.max_bytes,
      fp64_probe=not args.no_fp64_probe,
    )
  except LookupError as e:
    print(f"NO_ENTRY_POINT: {e}", file=sys.stderr)
    sys.exit(3)
  except RuntimeError as e:
    if "not deterministic" in str(e):
      print(f"NON_DETERMINISTIC: {e}", file=sys.stderr)
      sys.exit(4)
    print(f"DEGRADED: the module would not run on CPU: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(2)
  except MemoryError as e:
    print(f"TOO_LARGE: {e}", file=sys.stderr)
    sys.exit(5)
  except Exception as e:  # pylint: disable=broad-except
    print(f"DEGRADED: the module would not run on CPU: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(2)

  np.savez(args.out, **arrays)
  Path(args.meta).write_text(json.dumps(manifest, indent=2) + "\n")

  entry = manifest["entry_point"]
  cfg0 = manifest["configs"][0]
  print(
    f"GOLDEN_OK entry={entry['name']} kind={entry['kind']} "
    f"device={manifest['device']}"
  )
  print(f"  params captured : {len(entry['parameter_order'])}")
  print(f"  dynamic args    : {len(cfg0['dynamic'])}")
  print(f"  static_argnums  : {cfg0['static_argnums']}")
  print(f"  outputs         : {len(cfg0['outputs'])}")
  print(f"  total bytes     : {manifest['total_bytes']}")
  tol = cfg0.get("tolerance_recommendation")
  if tol:
    print(f"  recommended atol={tol['atol']:.3e} rtol={tol['rtol']:.3e}")
  sys.exit(0)


if __name__ == "__main__":
  main()
