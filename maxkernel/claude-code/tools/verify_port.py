#!/usr/bin/env python3
"""Checks the JAX port in base.py against the golden values from the source.

THE GATE THIS IMPLEMENTS
------------------------
`capture_torch_golden.py` recorded what the user's PyTorch module actually
computes. This tool checks that `<run_dir>/base.py` -- the JAX port that every
speedup in the run is measured against -- computes the same thing.

Without this, a non-JAX run has no way to detect a mistranslated port: the
worker's harness validation binds `base.py` as both sides of its comparison,
so it proves the harness runs and proves nothing about the port. Five
iterations optimizing against a wrong baseline is worse than no run at all,
because it ends in a confident, wrong speedup.

WHERE THIS RUNS: CPU BY DEFAULT, TPU WHEN THE DATA FITS
------------------------------------------------------
`--device cpu` (the default) is the right choice almost always:

1.  `base.py` is contractually pure JAX -- the worker's Phase 0.4 port (the
    `analyze-torch-source` reference) forbids putting Pallas in it, because the baseline is what a
    competent engineer writes *without* a custom kernel. Pure JAX runs on the
    CPU backend unchanged.
2.  This is a *semantic* check, not a performance one. Running it on CPU keeps
    TPU numerics out of the result: a failure means the port is wrong, not
    that the accelerator rounds differently. It also costs no TPU queue time.

`--device tpu` exists because "the port is right on CPU" and "the port is right
on the accelerator the run measures on" are not the same claim, and a dtype or
layout assumption can hold on one and not the other.

Its constraint is structural: `tpu_client.py` submits a *text* code file (its
`main()` opens the file in "r" mode and sends the source), so there is no
channel for a binary `.npz`. The golden arrays have to travel base64-inlined
inside the submitted source, which is only viable while they are small. Over
`--max-embed-bytes` the run refuses rather than submitting a source file of
tens of megabytes; `--device auto` falls back to CPU instead and records that
it did.

Exit codes:
  0  the port reproduces the golden values within tolerance
  1  MISMATCH -- the port disagrees with the source (worker follows fix-port)
  2  the port could not be executed at all (syntax, import, shape error)
  3  inputs missing or malformed (no golden file, no `computation`)
  5  --device tpu requested but the golden data exceeds --max-embed-bytes
"""

import argparse
import json

# This process runs the CPU check itself; the --device tpu path submits a
# generated script through tpu_client.py instead and never needs a local
# accelerator. Set before jax is imported either way.
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

try:
  import jax
  import jax.numpy as jnp
except ImportError as e:
  print(f"jax is required: {e}", file=sys.stderr)
  sys.exit(3)

try:
  import ml_dtypes
except ImportError:
  ml_dtypes = None


# Sub-fp16 dtypes were stored as raw bit patterns because numpy has no native
# representation for them. Reconstitute through ml_dtypes, which JAX ships.
RAW_BIT_DTYPES = {
  "bfloat16": "bfloat16",
  "float8_e4m3fn": "float8_e4m3fn",
  "float8_e5m2": "float8_e5m2",
}


def decode(array, meta):
  """Turns one stored array back into its true dtype."""
  if meta.get("encoding") != "raw_bits":
    return array
  tag = meta["dtype"]
  if ml_dtypes is None:
    raise RuntimeError(
      f"{meta['key']} is stored as raw {tag} bits but ml_dtypes is not "
      "importable, so it cannot be reconstituted."
    )
  target = getattr(ml_dtypes, RAW_BIT_DTYPES[tag])
  return array.view(target)


def load_computation(base_path):
  """Executes base.py in its own namespace and returns its `computation`."""
  src = Path(base_path).read_text()
  if "def computation" not in src:
    raise LookupError(
      f"{base_path} has no module-level `computation` -- the harness binds "
      "that name and refuses the file without it."
    )
  ns = {
    "__name__": "__maxkernel_base__",
    "__file__": str(Path(base_path).resolve()),
  }
  exec(compile(src, str(base_path), "exec"), ns)  # pylint: disable=exec-used
  fn = ns.get("computation")
  if not callable(fn):
    raise LookupError(
      f"{base_path} defines `computation` but it is not callable."
    )
  return fn


def build_args(npz, config):
  """Rebuilds the call arguments in the order the manifest recorded.

  Dynamic arguments (forward tensors, then parameters and buffers) come first,
  static scalars last -- the ordering `tools/test_harness_template.py` assumes
  when it computes `static_argnums`.
  """
  slots = {}
  for meta in config["dynamic"]:
    slots[meta["argnum"]] = jnp.asarray(decode(npz[meta["key"]], meta))
  for meta in config["static"]:
    slots[meta["argnum"]] = meta["value"]
  return tuple(slots[i] for i in sorted(slots))


def compare(expected, actual, atol, rtol):
  """Elementwise comparison with the worst offender located, not just counted."""
  exp = (
    np.asarray(expected, dtype=np.float64)
    if np.issubdtype(np.asarray(expected).dtype, np.number)
    else np.asarray(expected)
  )
  act = (
    np.asarray(actual, dtype=np.float64)
    if np.issubdtype(np.asarray(actual).dtype, np.number)
    else np.asarray(actual)
  )

  if exp.shape != act.shape:
    return {
      "ok": False,
      "reason": f"shape mismatch: expected {list(exp.shape)}, got {list(act.shape)}",
    }

  diff = np.abs(exp - act)
  tol = atol + rtol * np.abs(exp)
  bad = diff > tol
  n_bad = int(bad.sum())
  worst_flat = int(np.argmax(diff - tol)) if diff.size else 0
  worst_index = (
    list(map(int, np.unravel_index(worst_flat, diff.shape)))
    if diff.size
    else []
  )

  return {
    "ok": n_bad == 0,
    "elements": int(diff.size),
    "failing_elements": n_bad,
    "failing_fraction": float(n_bad / diff.size) if diff.size else 0.0,
    "max_abs_diff": float(diff.max()) if diff.size else 0.0,
    "worst_index": worst_index,
    "expected_at_worst": float(exp.flat[worst_flat]) if diff.size else 0.0,
    "actual_at_worst": float(act.flat[worst_flat]) if diff.size else 0.0,
  }


def diagnose(results):
  """Turns comparison statistics into a hypothesis a port repair can act on.

  A bare "mismatch" sends the repair hunting. The *shape* of the
  disagreement is usually diagnostic: everything wrong points at a different
  bug than a thin edge being wrong.
  """
  hints = []
  for r in results:
    if r.get("ok"):
      continue
    if "reason" in r:
      hints.append(
        f"output {r['index']}: {r['reason']} -- the port's output structure "
        "does not match the source's. Check the return signature and whether "
        "an output was transposed, split or merged."
      )
      continue
    frac = r["failing_fraction"]
    if frac > 0.95:
      hints.append(
        f"output {r['index']}: {frac:.0%} of elements disagree -- this is a "
        "wholesale semantic difference (wrong operation, wrong axis, missing "
        "scale factor), not accumulated rounding."
      )
    elif frac < 0.02:
      hints.append(
        f"output {r['index']}: only {frac:.2%} of elements disagree, worst at "
        f"index {r['worst_index']} -- look at boundary handling: masking, a "
        "ragged tail, an off-by-one in a window, or non-divisible tiling."
      )
    else:
      hints.append(
        f"output {r['index']}: {frac:.0%} of elements disagree (max abs diff "
        f"{r['max_abs_diff']:.3e}, expected {r['expected_at_worst']:.6g} vs "
        f"got {r['actual_at_worst']:.6g}) -- check accumulator dtype and "
        "reduction order before suspecting the algorithm."
      )
  return hints


def verify(base_path, golden_npz, golden_meta, atol, rtol):
  meta = json.loads(Path(golden_meta).read_text())
  npz = np.load(golden_npz)
  computation = load_computation(base_path)

  config_reports = []
  all_ok = True

  for config in meta["configs"]:
    args = build_args(npz, config)
    static_argnums = tuple(config["static_argnums"])
    try:
      raw = jax.jit(computation, static_argnums=static_argnums)(*args)
    except Exception as e:  # pylint: disable=broad-except
      raise RuntimeError(
        f"config {config['index']}: base.py's computation raised {type(e).__name__}: {e}"
      ) from e

    actual = [np.asarray(x) for x in jax.tree_util.tree_leaves(raw)]
    expected_meta = config["outputs"]

    if len(actual) != len(expected_meta):
      all_ok = False
      config_reports.append(
        {
          "index": config["index"],
          "ok": False,
          "outputs": [
            {
              "index": 0,
              "ok": False,
              "reason": (
                f"output count mismatch: source produced {len(expected_meta)} "
                f"tensors, port produced {len(actual)}"
              ),
            }
          ],
        }
      )
      continue

    results = []
    for j, om in enumerate(expected_meta):
      expected = decode(npz[om["key"]], om)
      r = compare(expected, actual[j], atol, rtol)
      r["index"] = j
      r["shape_expected"] = om["shape"]
      r["dtype_expected"] = om["dtype"]
      results.append(r)

    ok = all(r.get("ok") for r in results)
    all_ok = all_ok and ok
    config_reports.append(
      {"index": config["index"], "ok": ok, "outputs": results}
    )

  hints = []
  for cr in config_reports:
    if not cr["ok"]:
      hints.extend(diagnose(cr["outputs"]))

  return {
    "port_verified": all_ok,
    "base_path": str(Path(base_path).resolve()),
    "golden_npz": str(Path(golden_npz).resolve()),
    "atol": atol,
    "rtol": rtol,
    "backend": jax.default_backend(),
    "configs": config_reports,
    "diagnosis": hints,
  }


TPU_SCRIPT_TEMPLATE = """\
# Generated by tools/verify_port.py. Self-contained: tpu_client.py submits
# source text only, so the golden arrays travel base64-inlined below.
import base64, io, json, zlib
import numpy as np
import jax, jax.numpy as jnp
try:
  import ml_dtypes
except ImportError:
  ml_dtypes = None

_MANIFEST = json.loads({manifest!r})
ATOL, RTOL = {atol!r}, {rtol!r}
_npz = np.load(io.BytesIO(zlib.decompress(base64.b64decode({blob!r}))))

RAW_BITS = {{"bfloat16": "bfloat16", "float8_e4m3fn": "float8_e4m3fn",
             "float8_e5m2": "float8_e5m2"}}


def decode(arr, meta):
  if meta.get("encoding") != "raw_bits":
    return arr
  return arr.view(getattr(ml_dtypes, RAW_BITS[meta["dtype"]]))


_ns = {{}}
exec(compile({base_src!r}, "base.py", "exec"), _ns)
computation = _ns["computation"]

ok = True
for cfg in _MANIFEST["configs"]:
  slots = {{}}
  for m in cfg["dynamic"]:
    slots[m["argnum"]] = jnp.asarray(decode(_npz[m["key"]], m))
  for m in cfg["static"]:
    slots[m["argnum"]] = m["value"]
  args = tuple(slots[i] for i in sorted(slots))
  out = jax.jit(computation, static_argnums=tuple(cfg["static_argnums"]))(*args)
  leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]
  if len(leaves) != len(cfg["outputs"]):
    print("OUTPUT_COUNT_MISMATCH: expected %d got %d"
          % (len(cfg["outputs"]), len(leaves)))
    ok = False
    break
  for j, om in enumerate(cfg["outputs"]):
    exp = np.asarray(decode(_npz[om["key"]], om), dtype=np.float64)
    act = np.asarray(leaves[j], dtype=np.float64)
    if exp.shape != act.shape:
      print("SHAPE_MISMATCH out%d: %s vs %s" % (j, exp.shape, act.shape))
      ok = False
      continue
    d = np.abs(exp - act)
    bad = int((d > ATOL + RTOL * np.abs(exp)).sum())
    if bad:
      ok = False
      print("MISMATCH out%d: %d/%d elements, max_abs_diff %.6e"
            % (j, bad, d.size, float(d.max())))
print("PORT_CORRECTNESS: %s" % ok)
"""


def build_tpu_script(base_path, golden_npz, golden_meta, atol, rtol):
  """Renders a self-contained verification script with the golden data inlined."""
  import base64
  import zlib

  blob = base64.b64encode(
    zlib.compress(Path(golden_npz).read_bytes(), 6)
  ).decode("ascii")
  script = TPU_SCRIPT_TEMPLATE.format(
    manifest=Path(golden_meta).read_text(),
    blob=blob,
    atol=atol,
    rtol=rtol,
    base_src=Path(base_path).read_text(),
  )
  return script, len(blob)


def verify_on_tpu(
  base_path, golden_npz, golden_meta, atol, rtol, max_embed_bytes, script_out
):
  """Submits the verification to the TPU through the sanctioned client."""
  import subprocess

  script, blob_len = build_tpu_script(
    base_path, golden_npz, golden_meta, atol, rtol
  )
  if blob_len > max_embed_bytes:
    raise MemoryError(
      f"golden data is {blob_len / 1e6:.1f} MB once base64-encoded, over the "
      f"{max_embed_bytes / 1e6:.1f} MB embed cap. tpu_client.py submits "
      "source text only, so there is no way to ship it without inlining. Use "
      "--device cpu -- the check is just as valid there for a pure-JAX "
      "base.py -- or raise --max-embed-bytes."
    )

  path = Path(script_out or (Path(base_path).parent / "verify_port_tpu.py"))
  path.write_text(script, encoding="utf-8")

  proc = subprocess.run(
    [
      sys.executable,
      str(ROOT / "tools" / "tpu_client.py"),
      "--action",
      "correctness_test",
      "--code_file",
      str(path),
    ],
    capture_output=True,
    text=True,
    timeout=1800,
    check=False,
  )
  stdout = proc.stdout or ""
  return ("PORT_CORRECTNESS: True" in stdout), {
    "device": "tpu",
    "script": str(path),
    "embedded_bytes": blob_len,
    "client_returncode": proc.returncode,
    "stdout": stdout[-4000:],
    "stderr": (proc.stderr or "")[-2000:],
  }


def device_mismatch(golden_meta_path, ran_on):
  """Warns when the two halves of the comparison came from different devices.

  `capture_torch_golden.py` can record the reference on CPU or, with
  `--device xla`, on the TPU. This tool can run base.py on either too. Pairing
  them across devices is legal and sometimes deliberate, but it changes what a
  failure means: the disagreement then mixes a possible port error with the
  genuine numeric difference between two accelerators -- default matmul
  precision, reduction order, accumulation width. Read as a port bug, that
  sends a repair hunting for a semantic error that is not there.

  Returns a warning string, or None when the two sides agree.
  """
  try:
    golden_on = json.loads(Path(golden_meta_path).read_text()).get(
      "device", "cpu"
    )
  except Exception:  # pylint: disable=broad-except
    return None
  if golden_on == ran_on:
    return None
  # Name the two ways to make the sides agree -- moving the golden capture, or
  # moving this check -- rather than one, since which is cheaper depends on
  # what has already been computed.
  if ran_on == "cpu":
    fixes = (
      "re-capture on CPU with `capture_torch_golden.py --device cpu`, "
      "or re-run this check on the accelerator with `--device tpu`"
    )
  else:
    fixes = (
      "re-capture on the accelerator with "
      "`capture_torch_golden.py --device xla`, or re-run this check on "
      "the host with `--device cpu`"
    )
  return (
    f"device mismatch: the golden values were captured on {golden_on!r} and "
    f"base.py ran on {ran_on!r}. A failure here mixes a possible port error "
    f"with the numeric difference between two devices. To compare like with "
    f"like, {fixes}."
  )


def main():
  parser = argparse.ArgumentParser(
    description="Verify base.py against golden values captured from the source."
  )
  parser.add_argument("base_path")
  parser.add_argument("golden_npz")
  parser.add_argument("golden_meta")
  parser.add_argument("--atol", type=float, default=1e-2)
  parser.add_argument("--rtol", type=float, default=1e-2)
  parser.add_argument(
    "--device",
    default="cpu",
    choices=["cpu", "tpu", "auto"],
    help="cpu (default) runs base.py on the local JAX CPU "
    "backend; tpu submits a self-contained script "
    "through tpu_client.py; auto uses tpu when the "
    "golden data fits under --max-embed-bytes",
  )
  parser.add_argument(
    "--max-embed-bytes",
    type=int,
    default=8_000_000,
    help="ceiling on the base64-inlined golden payload for "
    "--device tpu (tpu_client.py submits source text only)",
  )
  parser.add_argument(
    "--emit-tpu-script",
    metavar="PATH",
    help="where to write the generated TPU script",
  )
  parser.add_argument("--out", help="write the verification report JSON here")
  args = parser.parse_args()

  for path in (args.base_path, args.golden_npz, args.golden_meta):
    if not Path(path).is_file():
      print(f"MISSING: {path}", file=sys.stderr)
      sys.exit(3)

  if args.device in ("tpu", "auto"):
    try:
      passed, info = verify_on_tpu(
        args.base_path,
        args.golden_npz,
        args.golden_meta,
        args.atol,
        args.rtol,
        args.max_embed_bytes,
        args.emit_tpu_script,
      )
    except MemoryError as e:
      if args.device == "tpu":
        print(f"TOO_LARGE_TO_EMBED: {e}", file=sys.stderr)
        sys.exit(5)
      # auto: the payload does not fit, so fall back and say so.
      print(f"NOTE: falling back to CPU -- {e}", file=sys.stderr)
    else:
      warning = device_mismatch(args.golden_meta, "tpu")
      report = {
        "port_verified": passed,
        "atol": args.atol,
        "rtol": args.rtol,
        "base_path": str(Path(args.base_path).resolve()),
        "device": "tpu",
        "device_warning": warning,
        **info,
      }
      if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
      if warning:
        print(f"WARNING: {warning}\n", file=sys.stderr)
      print(
        f"PORT_VERIFIED: {passed}  (device=tpu, "
        f"{info['embedded_bytes'] / 1e6:.1f} MB embedded)"
      )
      if not passed:
        print(info["stdout"])
      sys.exit(0 if passed else 1)

  try:
    report = verify(
      args.base_path, args.golden_npz, args.golden_meta, args.atol, args.rtol
    )
  except LookupError as e:
    print(f"MALFORMED: {e}", file=sys.stderr)
    sys.exit(3)
  except RuntimeError as e:
    print(f"PORT_UNRUNNABLE: {e}", file=sys.stderr)
    traceback.print_exc()
    if args.out:
      Path(args.out).write_text(
        json.dumps(
          {
            "port_verified": False,
            "error": str(e),
            "diagnosis": [
              "base.py could not be executed against the recorded inputs. This "
              "is a signature or shape error in the port, not a numerical one."
            ],
          },
          indent=2,
        )
        + "\n"
      )
    sys.exit(2)

  warning = device_mismatch(args.golden_meta, "cpu")
  report["device"] = "cpu"
  report["device_warning"] = warning
  if args.out:
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
  if warning:
    print(f"WARNING: {warning}\n", file=sys.stderr)

  if report["port_verified"]:
    n = sum(len(c["outputs"]) for c in report["configs"])
    print(
      f"PORT_VERIFIED: True  ({len(report['configs'])} configs, {n} outputs, "
      f"atol={args.atol:g} rtol={args.rtol:g}, backend={report['backend']})"
    )
    sys.exit(0)

  print("PORT_VERIFIED: False")
  for cr in report["configs"]:
    for r in cr["outputs"]:
      if r.get("ok"):
        continue
      if "reason" in r:
        print(f"  config {cr['index']} output {r['index']}: {r['reason']}")
      else:
        print(
          f"  config {cr['index']} output {r['index']}: "
          f"{r['failing_elements']}/{r['elements']} elements outside tolerance, "
          f"max abs diff {r['max_abs_diff']:.3e} at {r['worst_index']} "
          f"(expected {r['expected_at_worst']:.6g}, got {r['actual_at_worst']:.6g})"
        )
  print("\nDiagnosis:")
  for h in report["diagnosis"]:
    print(f"  - {h}")
  sys.exit(1)


if __name__ == "__main__":
  main()
