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
"""What the hardware's tiling costs when your shapes do not match it.

Supplies answers for `reference/design/layout_precision.md`. The two trailing axes of
a TPU array are not symmetric — one maps to sublanes and one to lanes — and
almost every surprising layout cost traces back to that asymmetry. These probes
try to price it rather than describe it.
"""

from __future__ import annotations

import harness

_GROUP = "layout"


@harness.probe(
  _GROUP,
  "native_tile_granularity",
  "What does a block smaller than one native tile cost, per useful element?",
)
def _native_tile_granularity() -> str:
  """Sweeps the two trailing axes separately, because they are not alike.

  A 2D array asked for a ``(1, 128)`` block is rejected by Pallas outright,
  and the rejection is itself the first half of the answer, so it is
  recorded here rather than left as a crash — see `blockspec_constraint` in the returned detail.

  The way around it is the same one the error message names: a block may have
  a trailing dimension that is *not* a multiple of the native tile as long as
  it equals the whole array's extent on that axis. So the sweep runs over a
  3D array ``(steps, sublanes, lanes)`` blocked as ``(1, sublanes, lanes)``,
  where the two trailing block dims always equal the array's and are
  therefore legal at any size.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  # First: record the constraint, by provoking it deliberately.
  def scale_2d_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] * 2.0 + 1.0

  try:
    jax.block_until_ready(
      pl.pallas_call(
        scale_2d_kernel,
        grid=(16,),
        in_specs=[pl.BlockSpec((1, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
      )(jnp.ones((16, 128), jnp.float32))
    )
    constraint = "a (1,128) block over a (16,128) f32 array was ACCEPTED"
  except Exception as error:  # noqa: BLE001 — the rejection is the finding.
    constraint = (
      f"a (1,128) block over a (16,128) f32 array was REJECTED at trace "
      f"time: {type(error).__name__}: "
      f"{str(error).strip().splitlines()[0][:220]}"
    )

  # Then: the sweep, over a 3D array so any trailing extent is legal.
  steps = 16384
  batch = 8

  def scale_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] * 2.0 + 1.0

  def time_block(sublanes: int, lanes: int) -> harness.Timing:
    shape = (steps, sublanes, lanes)
    source = jnp.ones(shape, dtype=jnp.float32)
    call = jax.jit(
      pl.pallas_call(
        scale_kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
      )
    )
    timing = harness.time_call(
      lambda: call(source), warmups=3, repeats=10, batch=batch
    )
    del source
    return timing

  sublane_arms: dict[int, harness.Timing | None] = {}
  for rows in (1, 2, 4, 8, 16, 32):
    try:
      sublane_arms[rows] = time_block(rows, 128)
    except Exception:  # noqa: BLE001 — a rejected shape is a finding.
      sublane_arms[rows] = None

  lane_arms: dict[int, harness.Timing | None] = {}
  for cols in (8, 16, 32, 64, 128, 256):
    try:
      lane_arms[cols] = time_block(8, cols)
    except Exception:  # noqa: BLE001 — a rejected shape is a finding.
      lane_arms[cols] = None

  reference = sublane_arms.get(8)
  if reference is None:
    raise harness.Invalid(
      "the 8x128 reference arm did not run, so nothing here has a denominator"
    )
  harness.require_above_dispatch_floor(reference, what="the 8x128 arm")

  def per_element(
    arms: dict[int, harness.Timing | None], other_axis: int
  ) -> str:
    parts = []
    for size, timing in arms.items():
      if timing is None:
        parts.append(f"{size}:rejected")
        continue
      elements = steps * size * other_axis
      parts.append(f"{size}:{timing.median / elements * 1e9:.4f}ns/elem")
    return " ".join(parts)

  return (
    f"blockspec_constraint: {constraint} "
    f"|| sweep on a 3D (steps,sublanes,lanes) f32 array, {steps} grid steps, "
    f"block (1,sublanes,lanes), batch={batch}. "
    f"sublane axis at 128 lanes: {per_element(sublane_arms, 128)} "
    f"|| lane axis at 8 sublanes: {per_element(lane_arms, 8)} "
    "|| a FLAT ns/elem across a sweep means that axis is padded up to its "
    "native tile, so the smaller shapes pay full price; a FALLING ns/elem "
    "means the work really is proportional to the elements asked for. "
    f"|| the 8x128 arm: {reference}"
  )


@harness.probe(
  _GROUP,
  "lane_vs_sublane_reduction",
  "Is reducing over the last axis cheaper or dearer than reducing over the second-minor?",
)
def _lane_vs_sublane_reduction() -> str:
  """A reduction axis choice is often free to make at design time and not free to run."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = columns = 8192
  block = 512

  source = jnp.ones((rows, columns), dtype=jnp.float32)

  def build(axis: int):
    def reduce_kernel(source_ref, output_ref) -> None:
      reduced = jnp.sum(source_ref[...], axis=axis, keepdims=True)
      output_ref[...] = jnp.broadcast_to(reduced, output_ref.shape)

    return jax.jit(
      pl.pallas_call(
        reduce_kernel,
        grid=(rows // block, columns // block),
        in_specs=[pl.BlockSpec((block, block), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((block, block), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
      )
    )

  findings = {}
  for label, axis in (("lane_axis(-1)", -1), ("sublane_axis(-2)", -2)):
    call = build(axis)
    findings[label] = harness.time_call(
      lambda call=call: call(source), warmups=3, repeats=20, batch=4
    )

  lane = findings["lane_axis(-1)"]
  sublane = findings["sublane_axis(-2)"]
  harness.require_above_dispatch_floor(lane, what="the lane-axis arm")
  return (
    f"f32 {rows}x{columns}, {block}x{block} blocks, sum then broadcast back. "
    f"lane(-1)={lane} sublane(-2)={sublane} "
    f"|| sublane/lane = {sublane.median / lane.median:.2f}x"
  )


@harness.probe(
  _GROUP,
  "transpose_cost",
  "What does transposing an operand inside the kernel cost, against transposing it outside?",
)
def _transpose_cost() -> str:
  """Prices the choice between a relayout and a differently-shaped operand."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  size = 2048
  lhs = jnp.ones((size, size), dtype=jnp.bfloat16)
  rhs = jnp.ones((size, size), dtype=jnp.bfloat16)

  def plain_kernel(lhs_ref, rhs_ref, output_ref) -> None:
    output_ref[...] = jnp.dot(
      lhs_ref[...], rhs_ref[...], preferred_element_type=jnp.float32
    )

  def transposing_kernel(lhs_ref, rhs_ref, output_ref) -> None:
    output_ref[...] = jnp.dot(
      lhs_ref[...], rhs_ref[...].T, preferred_element_type=jnp.float32
    )

  def dimension_numbers_kernel(lhs_ref, rhs_ref, output_ref) -> None:
    output_ref[...] = jax.lax.dot_general(
      lhs_ref[...],
      rhs_ref[...],
      dimension_numbers=(((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
    )

  findings = {}
  for label, kernel in (
    ("no_transpose", plain_kernel),
    ("rhs_dot_T_in_kernel", transposing_kernel),
    ("contract_via_dimension_numbers", dimension_numbers_kernel),
  ):
    try:
      call = jax.jit(
        pl.pallas_call(
          kernel, out_shape=jax.ShapeDtypeStruct((size, size), jnp.float32)
        )
      )
      findings[label] = harness.time_call(
        lambda call=call: call(lhs, rhs), warmups=3, repeats=20, batch=48
      )
    except Exception as error:  # noqa: BLE001
      findings[label] = f"NO({type(error).__name__}: {str(error)[:150]})"

  baseline = findings["no_transpose"]
  if not isinstance(baseline, harness.Timing):
    raise harness.Invalid(f"the untransposed baseline failed: {baseline}")
  harness.require_above_dispatch_floor(
    baseline, what="the untransposed baseline"
  )

  parts = []
  for label, value in findings.items():
    if isinstance(value, harness.Timing):
      parts.append(f"{label}={value.median / baseline.median:.2f}x")
    else:
      parts.append(f"{label}={value}")
  return (
    f"bf16 {size}^3 single-block matmul, relative to no transpose: "
    f"{' '.join(parts)} || baseline {baseline}"
  )


@harness.probe(
  _GROUP,
  "mask_versus_pad",
  "For a shape that does not divide, is masking the tail cheaper than padding the array?",
)
def _mask_versus_pad() -> str:
  """The two ways to handle a tail, priced against each other at the same shape."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  block = 512
  real_rows = 3000
  padded_rows = 3072
  columns = 4096

  def masked_kernel(source_ref, output_ref) -> None:
    step = pl.program_id(0)
    row_index = step * block + jax.lax.broadcasted_iota(
      jnp.int32, source_ref.shape, 0
    )
    inside = row_index < real_rows
    output_ref[...] = jnp.where(inside, source_ref[...] * 2.0, 0.0)

  def unmasked_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] * 2.0

  def build(kernel, rows: int):
    steps = -(-rows // block)
    return jax.jit(
      pl.pallas_call(
        kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((block, columns), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block, columns), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
      )
    )

  ragged_source = jnp.ones((real_rows, columns), dtype=jnp.float32)
  padded_source = jnp.ones((padded_rows, columns), dtype=jnp.float32)

  # Build once, outside the timed region. Calling `build(...)` inside the
  # lambda passed to `time_call` re-traces and re-compiles the kernel every
  # iteration, so every arm times the compiler and the arms agree with each
  # other. Constructing a jitted callable is not free and must never happen
  # inside the thing being measured.
  masked_call = build(masked_kernel, real_rows)
  unmasked_ragged_call = build(unmasked_kernel, real_rows)
  padded_call = build(unmasked_kernel, padded_rows)

  batch = 48
  masked = harness.time_call(
    lambda: masked_call(ragged_source), warmups=3, repeats=20, batch=batch
  )
  unmasked_ragged = harness.time_call(
    lambda: unmasked_ragged_call(ragged_source),
    warmups=3,
    repeats=20,
    batch=batch,
  )
  padded = harness.time_call(
    lambda: padded_call(padded_source), warmups=3, repeats=20, batch=batch
  )
  harness.require_above_dispatch_floor(
    unmasked_ragged, what="the unmasked-ragged control"
  )

  return (
    f"f32 {real_rows}x{columns} with block rows={block} (does not divide; "
    f"{-(-real_rows // block)} steps), batch={batch}. "
    f"masked_in_kernel={masked} || same_grid_no_mask={unmasked_ragged} || "
    f"padded_to_{padded_rows}={padded} || mask overhead = "
    f"{masked.median / unmasked_ragged.median:.2f}x, "
    f"pad overhead vs unmasked ragged = {padded.median / unmasked_ragged.median:.2f}x. "
    "The unmasked-ragged arm is the control: it isolates the iota-and-select "
    "cost from the cost of the extra rows."
  )


@harness.probe(
  _GROUP,
  "dtype_conversion_cost",
  "What does converting between dtypes inside a kernel cost, per element?",
)
def _dtype_conversion_cost() -> str:
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = columns = 8192
  block = 512

  def build(source_name: str, target_name: str):
    source_dtype = getattr(jnp, source_name)
    target_dtype = getattr(jnp, target_name)

    def convert_kernel(source_ref, output_ref) -> None:
      output_ref[...] = source_ref[...].astype(target_dtype) * target_dtype(2)

    source = jnp.ones((rows, columns), dtype=source_dtype)
    call = jax.jit(
      pl.pallas_call(
        convert_kernel,
        grid=(rows // block, columns // block),
        in_specs=[pl.BlockSpec((block, block), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((block, block), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((rows, columns), target_dtype),
      )
    )
    return source, call

  findings = {}
  for source_name, target_name in (
    ("float32", "float32"),
    ("float32", "bfloat16"),
    ("bfloat16", "float32"),
    ("bfloat16", "bfloat16"),
    ("int8", "bfloat16"),
    ("int32", "float32"),
  ):
    label = f"{source_name}->{target_name}"
    if not hasattr(jnp, source_name) or not hasattr(jnp, target_name):
      findings[label] = "dtype-absent"
      continue
    try:
      source, call = build(source_name, target_name)
      findings[label] = harness.time_call(
        lambda call=call, source=source: call(source),
        warmups=3,
        repeats=15,
        batch=8,
      )
    except Exception as error:  # noqa: BLE001
      findings[label] = f"NO({type(error).__name__}: {str(error)[:120]})"

  baseline = findings["float32->float32"]
  if not isinstance(baseline, harness.Timing):
    raise harness.Invalid(f"the f32->f32 control failed: {baseline}")

  parts = []
  for label, value in findings.items():
    if isinstance(value, harness.Timing):
      parts.append(f"{label}={value.median / baseline.median:.2f}x")
    else:
      parts.append(f"{label}={value}")
  return (
    f"{rows}x{columns}, {block}x{block} blocks, one scale after the cast, "
    f"relative to f32->f32: {' '.join(parts)} || f32->f32 baseline {baseline} "
    "|| these arms move different numbers of bytes through HBM, so a ratio "
    "below 1.0 is bandwidth, not a cheaper convert"
  )


@harness.probe(
  _GROUP,
  "bitcast_and_subbyte",
  "Do pltpu.bitcast and sub-byte dtypes work in a kernel here?",
)
def _bitcast_and_subbyte() -> str:
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  findings = {}

  if hasattr(pltpu, "bitcast"):

    def bitcast_kernel(source_ref, output_ref) -> None:
      output_ref[...] = pltpu.bitcast(source_ref[...], jnp.int32)

    try:
      result = pl.pallas_call(
        bitcast_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
      )(jnp.ones((8, 128), jnp.float32))
      jax.block_until_ready(result)
      findings["pltpu.bitcast f32->i32"] = (
        f"yes (result[0,0]={int(result[0, 0])})"
      )
    except Exception as error:  # noqa: BLE001
      findings["pltpu.bitcast f32->i32"] = f"NO({type(error).__name__})"
  else:
    findings["pltpu.bitcast f32->i32"] = "symbol-absent"

  for dtype_name in ("int4", "uint4", "float8_e4m3fn", "float8_e5m2", "int8"):
    dtype = getattr(jnp, dtype_name, None)
    if dtype is None:
      findings[f"block of {dtype_name}"] = "dtype-absent"
      continue

    def copy_kernel(source_ref, output_ref) -> None:
      output_ref[...] = source_ref[...]

    try:
      source = jnp.ones((256, 256), dtype=dtype)
      result = pl.pallas_call(
        copy_kernel, out_shape=jax.ShapeDtypeStruct((256, 256), dtype)
      )(source)
      jax.block_until_ready(result)
      findings[f"block of {dtype_name}"] = "yes"
    except Exception as error:  # noqa: BLE001
      findings[f"block of {dtype_name}"] = f"NO({type(error).__name__})"

  return " | ".join(f"{key}={value}" for key, value in findings.items())


@harness.probe(
  _GROUP,
  "sublane_tiling_by_dtype",
  "Does the native sublane tile widen for narrow dtypes, and does it match what `get_tpu_info` reports?",
)
def _sublane_tiling_by_dtype() -> str:
  """Tests the one hardware constant most likely to be assumed instead of read.

  ``get_tpu_info().get_sublane_tiling`` (reported by
  `tpu_info_reported_constants`) is expected to follow
  ``8 * (32 // bitwidth)``, while the widely-copied idiom is a hardcoded
  ``NUM_SUBLANES = 8`` for every dtype. If the reported table is right, a bfloat16 block eight
  sublanes tall occupies a sixteen-sublane tile and half of every transfer is
  padding. That is a fact about *correct-looking* code, so it is worth
  timing on the target rather than citing.

  Total elements are held fixed across the arms of one dtype, so a flat
  ns/element curve means the block height did not matter and a step at the
  tile boundary means it did. Comparing across dtypes is deliberately not
  offered: a byte of int8 and a byte of f32 do not cost the same to move and
  the ratio would mean nothing.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  lanes = 128
  total_rows = 64 * 1024
  batch = 8

  def scale_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] + source_ref[...]

  def time_arm(dtype, sublanes: int) -> harness.Timing:
    steps = total_rows // sublanes
    shape = (steps, sublanes, lanes)
    source = jnp.ones(shape, dtype=dtype)
    call = jax.jit(
      pl.pallas_call(
        scale_kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
      )
    )
    timing = harness.time_call(
      lambda: call(source), warmups=3, repeats=10, batch=batch
    )
    del source
    return timing

  def report_tiling() -> str:
    try:
      getter = pltpu.get_tpu_info().get_sublane_tiling
    except AttributeError:
      return "get_sublane_tiling absent"
    # It is a bound method, not the dict its name suggests: printing it
    # gives the method's repr, a whole TpuInfo dump and not an answer. It
    # takes a dtype.
    parts = []
    for name in ("float32", "bfloat16", "int8", "int4"):
      dtype = getattr(jnp, name, None)
      if dtype is None:
        parts.append(f"{name}=dtype-absent")
        continue
      try:
        parts.append(f"{name}={getter(jnp.dtype(dtype))}")
      except Exception as error:  # noqa: BLE001
        parts.append(
          f"{name}=raised({type(error).__name__}: {str(error)[:60]})"
        )
    return "{" + ", ".join(parts) + "}"

  reported = report_tiling()

  findings = []
  guarded = False
  for dtype_name in ("float32", "bfloat16", "int8"):
    if not hasattr(jnp, dtype_name):
      findings.append(f"{dtype_name}=dtype-absent")
      continue
    dtype = getattr(jnp, dtype_name)
    arms: dict[int, harness.Timing | None] = {}
    first_error = ""
    for sublanes in (8, 16, 32, 64):
      try:
        arms[sublanes] = time_arm(dtype, sublanes)
      except Exception as error:  # noqa: BLE001 — a rejection is a finding.
        arms[sublanes] = None
        if not first_error:
          first_error = f"{type(error).__name__}: {str(error)[:200]}"

    # The tallest arm that ran is the denominator: it is the one certain to
    # be at least one whole native tile for every dtype in the table.
    reference = next(
      (arms[s] for s in (64, 32, 16, 8) if arms.get(s) is not None), None
    )
    if reference is None:
      findings.append(f"{dtype_name}=no arm ran ({first_error})")
      continue
    if not guarded:
      harness.require_above_dispatch_floor(
        reference, what=f"the {dtype_name} reference arm"
      )
      guarded = True

    parts = []
    for sublanes, timing in arms.items():
      if timing is None:
        parts.append(f"{sublanes}:rejected")
        continue
      parts.append(f"{sublanes}:{timing.median / reference.median:.2f}x")
    findings.append(
      f"{dtype_name} [{' '.join(parts)}] "
      f"(reference {reference.median * 1e6:.1f}us, "
      f"spread {reference.spread_pct:.0f}%)"
    )

  return (
    f"get_tpu_info reports {reported}. Elementwise add over a 3D "
    f"(steps, sublanes, {lanes}) array with block (1, sublanes, {lanes}), "
    f"total rows held at {total_rows} so every arm of a dtype moves the "
    f"same bytes; time relative to that dtype's tallest arm, batch={batch}: "
    + " || ".join(findings)
    + ". A ratio above 1.0 at a block height below the reported tiling is "
    "the padding waste made visible; a flat row means the block height did "
    "not matter and the reported table does not bind here."
  )
