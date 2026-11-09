# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""What each compute unit on the chip is worth, and what feeds it.

Supplies answers for `reference/design/compute_units.md`. Everything here is a rate,
and a rate is only quotable next to the peak it is a fraction of — so
`mxu_peak_by_dtype` runs first and every later probe reports its result as a
fraction of that number measured *in the same process*. A probe that reports an
implausible rate is timing the host, not the chip.
"""

from __future__ import annotations

import harness

_GROUP = "compute_units"

# Large enough that one call is milliseconds, so dispatch overhead is noise.
_SQUARE = 8192


@harness.probe(
  _GROUP,
  "mxu_peak_by_dtype",
  "What matmul rate does XLA achieve per dtype? This is the denominator for every other rate.",
)
def _mxu_peak_by_dtype() -> str:
  import jax
  import jax.numpy as jnp

  size = _SQUARE
  total_flops = 2.0 * size**3

  # The `precision` column is not decoration. An f32 matmul that runs nearly
  # as fast as a bf16 one, on an MXU whose native input is bf16, is not doing
  # f32 arithmetic: default precision can decompose an f32 matmul into bf16
  # passes, so "float32" at DEFAULT is not measuring f32 arithmetic. HIGHEST
  # is what forces the real thing, and the gap between the two arms is the
  # number a design actually needs.
  candidates = [
    ("float32", "float32", "DEFAULT"),
    ("float32", "float32", "HIGHEST"),
    ("bfloat16", "float32", "DEFAULT"),
    ("bfloat16", "bfloat16", "DEFAULT"),
    ("float16", "float32", "DEFAULT"),
    ("int8", "int32", "DEFAULT"),
    ("float8_e4m3fn", "float32", "DEFAULT"),
    ("float8_e5m2", "float32", "DEFAULT"),
  ]

  findings = []
  for operand_name, accumulate_name, precision_name in candidates:
    operand_dtype = getattr(jnp, operand_name, None)
    label = f"{operand_name}->{accumulate_name}@{precision_name}"
    if operand_dtype is None:
      findings.append(f"{label}=dtype-absent")
      continue
    accumulate_dtype = getattr(jnp, accumulate_name)
    precision = getattr(jax.lax.Precision, precision_name)

    try:
      lhs = jnp.ones((size, size), dtype=operand_dtype)
      rhs = jnp.ones((size, size), dtype=operand_dtype)
      matmul = jax.jit(
        lambda a, b, acc=accumulate_dtype, prec=precision: jnp.dot(
          a, b, preferred_element_type=acc, precision=prec
        )
      )
      timing = harness.time_call(
        lambda: matmul(lhs, rhs), warmups=3, repeats=10
      )
    except Exception as error:  # noqa: BLE001 — an unsupported dtype is a finding.
      findings.append(f"{label}=NO({type(error).__name__})")
      continue

    tera_ops = total_flops / timing.median / 1e12
    findings.append(
      f"{label}={tera_ops:.1f}Tops/s(spread={timing.spread_pct:.0f}%)"
    )
    del lhs, rhs

  return f"{size}^3 jnp.dot, one device: " + " ".join(findings)


@harness.probe(
  _GROUP,
  "mxu_block_rows_cost",
  "Does a matmul pass over fewer than 128 rows cost less than a full 128-row pass?",
)
def _mxu_block_rows_cost() -> str:
  """Holds total work fixed and varies only the block height.

  If a sub-128-row pass really cost proportionally less, halving the block
  height would double the step count and leave wall time flat. Reporting the
  ratio against the 128-row arm makes the answer independent of the absolute
  rate, which drifts run to run.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  total_rows = 4096
  contraction = 2048
  columns = 2048

  lhs = jnp.ones((total_rows, contraction), dtype=jnp.bfloat16)
  rhs = jnp.ones((contraction, columns), dtype=jnp.bfloat16)

  def matmul_kernel(lhs_ref, rhs_ref, output_ref) -> None:
    output_ref[...] = jnp.dot(
      lhs_ref[...], rhs_ref[...], preferred_element_type=jnp.float32
    )

  def build(block_rows: int):
    return pl.pallas_call(
      matmul_kernel,
      grid=(total_rows // block_rows,),
      in_specs=[
        pl.BlockSpec((block_rows, contraction), lambda i: (i, 0)),
        pl.BlockSpec((contraction, columns), lambda i: (0, 0)),
      ],
      out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((total_rows, columns), jnp.float32),
    )

  timings = {}
  for block_rows in (8, 16, 32, 64, 128, 256, 512):
    call = jax.jit(build(block_rows))
    timings[block_rows] = harness.time_call(
      lambda call=call: call(lhs, rhs), warmups=3, repeats=15, batch=16
    )

  # The 128-row arm is the denominator, so if *it* is at the dispatch floor
  # every ratio below is a ratio of two host round trips.
  harness.require_above_dispatch_floor(timings[128], what="the 128-row arm")

  reference = timings[128].median
  ratios = " ".join(
    f"{rows}rows={timing.median / reference:.2f}x"
    for rows, timing in timings.items()
  )
  total_flops = 2.0 * total_rows * contraction * columns
  achieved = total_flops / reference / 1e12
  return (
    f"bf16 {total_rows}x{contraction}x{columns}, total work fixed, "
    f"wall time relative to the 128-row block: {ratios} "
    f"|| 128-row arm achieved {achieved:.1f}Tops/s "
    f"(cross-check against mxu_peak_by_dtype bfloat16->float32; "
    f"orders of magnitude below it means this probe timed the host) "
    f"|| spread at 128 rows = {timings[128].spread_pct:.0f}%"
  )


@harness.probe(
  _GROUP,
  "mxu_contraction_and_column_granularity",
  "Do sub-128 contraction depths and sub-128 column counts cost a full 128 as well?",
)
def _mxu_contraction_and_column_granularity() -> str:
  """The row axis and the two other axes need not have the same granularity."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = 512

  def matmul_kernel(lhs_ref, rhs_ref, output_ref) -> None:
    output_ref[...] = jnp.dot(
      lhs_ref[...], rhs_ref[...], preferred_element_type=jnp.float32
    )

  def time_shape(contraction: int, columns: int) -> harness.Timing:
    lhs = jnp.ones((rows, contraction), dtype=jnp.bfloat16)
    rhs = jnp.ones((contraction, columns), dtype=jnp.bfloat16)
    call = jax.jit(
      pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
      )
    )
    # Each call here is a single small matmul, short next to the host round
    # trip. Without a large `batch` every arm reports the round trip and the
    # sweep reads as flat, which looks like a finding and is not one.
    return harness.time_call(
      lambda: call(lhs, rhs), warmups=3, repeats=20, batch=256
    )

  contraction_arms = {
    depth: time_shape(depth, 512) for depth in (32, 64, 128, 256, 512)
  }
  column_arms = {
    width: time_shape(512, width) for width in (32, 64, 128, 256, 512)
  }

  harness.require_above_dispatch_floor(
    contraction_arms[128], what="the K=128 arm"
  )

  contraction_reference = contraction_arms[128].median
  column_reference = column_arms[128].median
  return (
    f"rows fixed at {rows}, bf16, batch=256 dispatches per timed region. "
    "contraction depth vs the 128 arm: "
    + " ".join(
      f"K={depth}:{timing.median / contraction_reference:.2f}x"
      for depth, timing in contraction_arms.items()
    )
    + " || columns vs the 128 arm: "
    + " ".join(
      f"N={width}:{timing.median / column_reference:.2f}x"
      for width, timing in column_arms.items()
    )
    + f" || per-call device time of the 128/128 arm = "
    f"{contraction_reference * 1e6:.2f}us against a {harness.dispatch_floor_seconds() * 1e6:.0f}us "
    "single-dispatch floor, which is the margin these ratios rest on"
  )


@harness.probe(
  _GROUP,
  "vpu_fma_rate",
  "What elementwise FMA rate does the VPU sustain on VMEM-resident data?",
)
def _vpu_fma_rate() -> str:
  """A dependent chain of FMAs, so nothing can be folded or reassociated."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows, columns = 512, 1024
  iterations = 2000
  batch = 8

  def fma_chain_kernel(source_ref, output_ref) -> None:
    def one_step(_, value):
      return value * 1.0000001 + 0.0000001

    output_ref[...] = jax.lax.fori_loop(
      0, iterations, one_step, source_ref[...]
    )

  source = jnp.full((rows, columns), 1.5, dtype=jnp.float32)
  call = jax.jit(
    pl.pallas_call(
      fma_chain_kernel,
      out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
    )
  )
  timing = harness.time_call(
    lambda: call(source), warmups=3, repeats=20, batch=batch
  )
  harness.require_above_dispatch_floor(timing, what="the FMA chain")

  total_flops = 2.0 * rows * columns * iterations
  tera_flops = total_flops / timing.median / 1e12
  return (
    f"f32 {rows}x{columns}, {iterations} dependent FMAs: "
    f"{tera_flops:.2f}Tflop/s ({timing}). "
    "Compare against mxu_peak_by_dtype float32 from the same run: the VPU "
    "is expected to be far slower, and a VPU rate near the MXU rate means "
    "the loop was folded away rather than executed."
  )


@harness.probe(
  _GROUP,
  "transcendental_relative_cost",
  "How much more does exp / tanh / rsqrt cost than a multiply, per element?",
)
def _transcendental_relative_cost() -> str:
  """Costs the ops a softmax or a normalization actually spends time in."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows, columns = 512, 1024
  iterations = 1000
  batch = 8

  def build(step_expression: str):
    def kernel(source_ref, output_ref) -> None:
      def one_step(_, value):
        return eval(  # noqa: S307 — expressions are literals below.
          step_expression,
          {"jnp": jnp, "jax": jax, "lax": jax.lax},
          {"v": value},
        )

      output_ref[...] = jax.lax.fori_loop(
        0, iterations, one_step, source_ref[...]
      )

    return jax.jit(
      pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32)
      )
    )

  source = jnp.full((rows, columns), 0.5, dtype=jnp.float32)
  arms = {
    "multiply": "v * 1.0000001",
    "exp": "jnp.exp(v * 0.0001) ",
    "tanh": "jnp.tanh(v)",
    "rsqrt": "lax.rsqrt(jnp.abs(v) + 1.0)",
    "log": "jnp.log(jnp.abs(v) + 1.0)",
    "divide": "v / 1.0000001",
    "sigmoid": "jax.nn.sigmoid(v)",
  }

  timings = {}
  for name, expression in arms.items():
    try:
      call = build(expression)
      timings[name] = harness.time_call(
        lambda call=call: call(source), warmups=3, repeats=15, batch=batch
      )
    except Exception as error:  # noqa: BLE001 — a missing lowering is a finding.
      timings[name] = None
      arms[name] = f"NO({type(error).__name__})"

  baseline = timings["multiply"]
  if baseline is None:
    raise harness.Invalid(
      "the multiply baseline itself failed; no ratio is quotable"
    )
  harness.require_above_dispatch_floor(baseline, what="the multiply baseline")

  parts = []
  for name, timing in timings.items():
    if timing is None:
      parts.append(f"{name}=unsupported")
    else:
      parts.append(f"{name}={timing.median / baseline.median:.2f}x")
  return (
    f"f32 {rows}x{columns}, {iterations} dependent ops, cost relative to a "
    f"multiply: {' '.join(parts)} || multiply arm: {baseline}"
  )


@harness.probe(
  _GROUP,
  "sparsecore_surface",
  "Is SparseCore reachable from Pallas on this stack at all?",
)
def _sparsecore_surface() -> str:
  """Cheap check before any design leans on offloading gathers to SparseCore."""
  import jax
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  names = [
    name
    for name in dir(pltpu)
    if "sparse" in name.lower()
    or "sc_" in name.lower()
    or name.startswith("SC")
  ]
  core_names = [name for name in dir(pltpu) if "core" in name.lower()]
  kinds = sorted({device.device_kind for device in jax.devices()})
  has_core_map = hasattr(pl, "core_map")

  # The absence of sparse-ish symbols on `pltpu` says nothing about
  # reachability: SparseCore is reached through a *separate* module
  # (`plsc.VectorSubcoreMesh`, used for example in
  # tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py). Import it.
  import importlib

  module_report = []
  for module_path in (
    "jax.experimental.pallas.tpu.sparsecore",
    "jax.experimental.pallas.sparsecore",
    "jax._src.pallas.mosaic.sparsecore",
  ):
    try:
      module = importlib.import_module(module_path)
    except ImportError as error:
      module_report.append(f"{module_path}=absent({type(error).__name__})")
      continue
    public = sorted(n for n in dir(module) if not n.startswith("_"))
    module_report.append(f"{module_path}=PRESENT{public}")

  # `TpuInfo.sparse_core` is None when the chip has no SparseCore, and
  # tokamax branches on exactly that to fall back to an XLA gather.
  try:
    sparse_core_info = pltpu.get_tpu_info().sparse_core
  except Exception as error:  # noqa: BLE001 — absence is the finding.
    sparse_core_info = f"unavailable({type(error).__name__})"

  return (
    f"device_kinds={kinds} pltpu sparse-ish symbols={names} "
    f"pltpu core-ish symbols={core_names} pl.core_map={has_core_map} "
    f"|| tpu_info.sparse_core={sparse_core_info!r} "
    f"|| modules: {' '.join(module_report)}"
  )
