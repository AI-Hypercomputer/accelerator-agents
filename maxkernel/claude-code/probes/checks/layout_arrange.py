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
"""Arrangements that are supposed to make a relayout not exist.

Supplies answers for `reference/design/layout_precision.md`, section
"Arranging so the relayout never exists". Its sibling module `layout.py` prices moves you already have; these
probes ask the constructive question instead — whether arranging the design a
particular way means the move is never generated.

Each probe is named after the arrangement it tests and is written so that the
arrangement **not** paying off is a visible result rather than a crash. Several
of these rules are cheap to state and expensive to believe: the compiler may
already be doing the transformation, in which case the rule is noise in a
reference file and should be deleted rather than promoted.
"""

from __future__ import annotations

import harness

_GROUP = "arrange"

# Every timing here is normalised against an arm timed in the same process, per
# `harness`'s third rule. Absolute microseconds appear only as the denominator.


def _pallas_call(*args, **kwargs):
  """``pl.pallas_call``, made traceable on CPU under ``PROBE_CPU_DRYRUN=1``.

  A `pallas_call` naming a TPU memory space raises ``Only interpret mode is
  supported on CPU backend`` at trace time, so without this the CPU dry run
  cannot reach the shape and arity errors it exists to catch — every arm dies
  on the backend check first and the probe looks uniformly broken whether or
  not it is. Setting the flag swaps in ``InterpretParams()``, which is the
  only spelling that works.

  **Nothing measured under the flag is a result.** Interpret mode does not
  invoke Mosaic, so a kernel with no TPU lowering runs happily and a probe
  written to return ``UNSUPPORTED`` returns ``PASS``. Its one use is "the
  script runs"; the flag is never set on hardware.

  The flag also collapses the grid to a single step. Interpret mode walks the
  grid in Python, so a 512-step probe that takes microseconds on a pod takes
  minutes on CPU and the whole group never finishes. One step exercises every
  `BlockSpec`, index map and kernel body, which is all the dry run is for.
  """
  from jax.experimental import pallas as pl

  if harness.env_flag("PROBE_CPU_DRYRUN"):
    from jax.experimental.pallas import tpu as pltpu

    kwargs.setdefault("interpret", pltpu.InterpretParams())
    grid = kwargs.get("grid")
    if isinstance(grid, tuple):
      kwargs["grid"] = (1,) * len(grid)
  return pl.pallas_call(*args, **kwargs)


@harness.probe(
  _GROUP,
  "expand_axis_leading_vs_sublane_vs_lane",
  "What does expanding a value cost along a leading, a sublane and a lane axis? (Its lane arm also changes spelling; pair it with `expand_blocked_vs_interleaved`.)",
)
def _expand_axis() -> str:
  """The arrangement: put a repeat/broadcast on a leading axis.

  All three arms read ``S x L`` elements and write ``R x S x L``, so they move
  the same bytes and differ only in which axis the expansion runs along. A
  ratio near 1.0 says the axis did not matter and the rule is not worth
  stating; a ratio well above 1.0 is the rule earning its place.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  steps, sublanes, lanes, repeats = 512, 64, 512, 4
  batch = 24

  source = jnp.ones((steps, sublanes, lanes), jnp.float32)

  # A BlockSpec of (1, sublanes, lanes) hands the kernel a ref of that shape,
  # leading 1 included, so every arm indexes it away before expanding and
  # writes through `output_ref[0]`. Broadcasting the un-squeezed ref raises
  # `Cannot broadcast to shape with fewer dimensions`.
  def leading_kernel(source_ref, output_ref) -> None:
    expanded = jnp.broadcast_to(source_ref[0][None], (repeats, sublanes, lanes))
    output_ref[0] = expanded * 1.5

  def sublane_kernel(source_ref, output_ref) -> None:
    expanded = jnp.repeat(source_ref[0], repeats, axis=0)
    output_ref[0] = expanded * 1.5

  def lane_kernel(source_ref, output_ref) -> None:
    expanded = jnp.repeat(source_ref[0], repeats, axis=1)
    output_ref[0] = expanded * 1.5

  def build(kernel, out_block, out_shape):
    return jax.jit(
      _pallas_call(
        kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec(
          out_block, lambda i: (i,) + (0,) * (len(out_block) - 1)
        ),
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
      )
    )

  arms: dict[str, object] = {}
  plans = (
    (
      "leading",
      leading_kernel,
      (1, repeats, sublanes, lanes),
      (steps, repeats, sublanes, lanes),
    ),
    (
      "sublane",
      sublane_kernel,
      (1, repeats * sublanes, lanes),
      (steps, repeats * sublanes, lanes),
    ),
    (
      "lane",
      lane_kernel,
      (1, sublanes, repeats * lanes),
      (steps, sublanes, repeats * lanes),
    ),
  )
  for label, kernel, out_block, out_shape in plans:
    try:
      call = build(kernel, out_block, out_shape)
      arms[label] = harness.time_call(
        lambda call=call: call(source), warmups=3, repeats=12, batch=batch
      )
    except Exception as error:  # noqa: BLE001 — a refusal is the finding.
      arms[label] = f"NO({type(error).__name__}: {str(error).strip()[:140]})"

  reference = arms["leading"]
  if not isinstance(reference, harness.Timing):
    raise harness.Invalid(f"the leading-axis arm did not run: {reference}")
  harness.require_above_dispatch_floor(reference, what="the leading-axis arm")

  parts = []
  for label, value in arms.items():
    if isinstance(value, harness.Timing):
      parts.append(f"{label}={value.median / reference.median:.2f}x")
    else:
      parts.append(f"{label}={value}")
  return (
    f"f32, {steps} grid steps, read ({sublanes},{lanes}) and write "
    f"{repeats}x that, batch={batch}. Every arm moves identical bytes; only "
    f"the expanded axis differs. Relative to the leading-axis arm: "
    f"{' '.join(parts)} || leading-axis arm {reference}"
  )


@harness.probe(
  _GROUP,
  "lane_split_alignment",
  "Does the cost of a lane-axis split depend on whether it lands on a 128 multiple?",
)
def _lane_split_alignment() -> str:
  """The arrangement: cut along lanes at a multiple of the lane width.

  Every arm reads and writes the same ``(sublanes, 512)`` block and performs
  the same two elementwise scales. Only the offset the block is split at
  changes, so the difference is the split, not the work or the traffic.
  The ``no_split`` arm applies one scale instead of two and is a floor rather
  than a like-for-like control; it is marked as such.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  steps, sublanes, lanes = 1024, 64, 512
  batch = 24
  source = jnp.ones((steps, sublanes, lanes), jnp.float32)

  def build(offset: int | None):
    if offset is None:

      def kernel(source_ref, output_ref) -> None:
        output_ref[...] = source_ref[...] * 1.5

    else:

      def kernel(source_ref, output_ref) -> None:
        block = source_ref[...]
        left = block[..., :offset] * 1.5
        right = block[..., offset:] * 2.5
        output_ref[...] = jnp.concatenate([left, right], axis=-1)

    return jax.jit(
      _pallas_call(
        kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((steps, sublanes, lanes), jnp.float32),
      )
    )

  arms: dict[str, object] = {}
  for label, offset in (
    ("no_split_FLOOR_not_a_control", None),
    ("split_at_256_aligned", 256),
    ("split_at_128_aligned", 128),
    ("split_at_64_half_tile", 64),
    ("split_at_100_unaligned", 100),
    ("split_at_130_unaligned", 130),
  ):
    try:
      call = build(offset)
      arms[label] = harness.time_call(
        lambda call=call: call(source), warmups=3, repeats=12, batch=batch
      )
    except Exception as error:  # noqa: BLE001
      arms[label] = f"NO({type(error).__name__}: {str(error).strip()[:140]})"

  reference = arms["split_at_128_aligned"]
  if not isinstance(reference, harness.Timing):
    raise harness.Invalid(f"the aligned 128 arm did not run: {reference}")
  harness.require_above_dispatch_floor(reference, what="the aligned-128 arm")

  parts = []
  for label, value in arms.items():
    if isinstance(value, harness.Timing):
      parts.append(f"{label}={value.median / reference.median:.2f}x")
    else:
      parts.append(f"{label}={value}")
  return (
    f"f32 ({sublanes},{lanes}) block, {steps} grid steps, split on the lane "
    f"axis then concatenated back, batch={batch}. Identical bytes and "
    f"identical arithmetic across every split arm. Relative to the "
    f"128-aligned split: {' '.join(parts)} || 128-aligned arm {reference} "
    "|| the no_split arm does one scale rather than two, so it is a floor "
    "and must not be quoted as the cost of splitting"
  )


@harness.probe(
  _GROUP,
  "concat_to_widen_narrow_matmul",
  "Do n narrow matmuls cost more than one matmul over their lane-concatenated operand?",
)
def _concat_to_widen_narrow_matmul() -> str:
  """The arrangement: widen a narrow contraction by concatenating siblings.

  Output width is held at 128 in every arm, so the MAC count is identical and
  only the number of matmul passes changes. ``w=128, n=1`` *is* the
  concatenated form and is therefore the denominator. HBM is held out of the
  comparison by a constant index map: every grid step revisits the same
  block, so the sweep prices MXU passes and not bandwidth.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows, contract, total_width = 512, 512, 128
  grid_steps, batch = 256, 12

  activations = jnp.ones((rows, contract), jnp.bfloat16)
  weights = jnp.ones((contract, total_width), jnp.bfloat16)

  def build(width: int):
    count = total_width // width

    def kernel(activation_ref, weight_ref, output_ref) -> None:
      activation = activation_ref[...]
      pieces = [
        jnp.dot(
          activation,
          weight_ref[:, index * width : (index + 1) * width],
          preferred_element_type=jnp.float32,
        )
        for index in range(count)
      ]
      output_ref[...] = (
        pieces[0] if count == 1 else jnp.concatenate(pieces, axis=-1)
      )

    return jax.jit(
      _pallas_call(
        kernel,
        grid=(grid_steps,),
        in_specs=[
          pl.BlockSpec((rows, contract), lambda i: (0, 0)),
          pl.BlockSpec((contract, total_width), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((rows, total_width), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, total_width), jnp.float32),
      )
    )

  arms: dict[str, object] = {}
  for width in (128, 64, 32, 16, 8):
    label = f"w{width}_n{total_width // width}"
    try:
      call = build(width)
      arms[label] = harness.time_call(
        lambda call=call: call(activations, weights),
        warmups=3,
        repeats=12,
        batch=batch,
      )
    except Exception as error:  # noqa: BLE001
      arms[label] = f"NO({type(error).__name__}: {str(error).strip()[:140]})"

  reference = arms[f"w{total_width}_n1"]
  if not isinstance(reference, harness.Timing):
    raise harness.Invalid(
      f"the concatenated (n=1) arm did not run: {reference}"
    )
  harness.require_above_dispatch_floor(reference, what="the concatenated arm")

  parts = []
  for label, value in arms.items():
    if isinstance(value, harness.Timing):
      parts.append(f"{label}={value.median / reference.median:.2f}x")
    else:
      parts.append(f"{label}={value}")
  return (
    f"bf16 ({rows},{contract}) @ ({contract},{total_width}) split into n "
    f"matmuls of width w, {grid_steps} grid steps on a resident block, "
    f"batch={batch}. MACs identical in every arm; n=1 IS the concatenated "
    f"form and is the denominator: {' '.join(parts)} "
    f"|| n=1 arm {reference} "
    "|| a ratio above 1.0 is what concatenating the siblings buys; a flat "
    "row means the compiler already merged them and the rule is redundant"
  )


@harness.probe(
  _GROUP,
  "weight_stationary_lhs_concat",
  "Does concatenating the varying operands of matmuls that share a stationary one pay, and at which row counts?",
)
def _weight_stationary_lhs_concat() -> str:
  """The arrangement: one matmul over concatenated LHS, one load of the RHS.

  Total rows are held at 512 across the sweep, so every arm performs the same
  MACs against the same right-hand side and only the number of passes over
  that shared operand changes. ``n=1`` is the fully concatenated form.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  total_rows, contract, width = 512, 512, 512
  # `batch` must lift the timed region well clear of the dispatch floor;
  # `require_above_dispatch_floor` withholds the result if it does not.
  grid_steps, batch = 64, 64

  activations = jnp.ones((total_rows, contract), jnp.bfloat16)
  shared = jnp.ones((contract, width), jnp.bfloat16)

  def build(rows_each: int):
    count = total_rows // rows_each

    def kernel(activation_ref, shared_ref, output_ref) -> None:
      stationary = shared_ref[...]
      pieces = [
        jnp.dot(
          activation_ref[index * rows_each : (index + 1) * rows_each, :],
          stationary,
          preferred_element_type=jnp.float32,
        )
        for index in range(count)
      ]
      output_ref[...] = (
        pieces[0] if count == 1 else jnp.concatenate(pieces, axis=0)
      )

    return jax.jit(
      _pallas_call(
        kernel,
        grid=(grid_steps,),
        in_specs=[
          pl.BlockSpec((total_rows, contract), lambda i: (0, 0)),
          pl.BlockSpec((contract, width), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((total_rows, width), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((total_rows, width), jnp.float32),
      )
    )

  arms: dict[str, object] = {}
  for rows_each in (512, 256, 128, 64, 32):
    label = f"m{rows_each}_n{total_rows // rows_each}"
    try:
      call = build(rows_each)
      arms[label] = harness.time_call(
        lambda call=call: call(activations, shared),
        warmups=3,
        repeats=12,
        batch=batch,
      )
    except Exception as error:  # noqa: BLE001
      arms[label] = f"NO({type(error).__name__}: {str(error).strip()[:140]})"

  reference = arms[f"m{total_rows}_n1"]
  if not isinstance(reference, harness.Timing):
    raise harness.Invalid(
      f"the concatenated (n=1) arm did not run: {reference}"
    )
  harness.require_above_dispatch_floor(reference, what="the concatenated arm")

  parts = []
  for label, value in arms.items():
    if isinstance(value, harness.Timing):
      parts.append(f"{label}={value.median / reference.median:.2f}x")
    else:
      parts.append(f"{label}={value}")
  return (
    f"bf16, n matmuls of m rows each against one shared "
    f"({contract},{width}) RHS, total rows held at {total_rows}, "
    f"{grid_steps} grid steps on a resident block, batch={batch}. "
    f"Relative to the single concatenated matmul: {' '.join(parts)} "
    f"|| n=1 arm {reference} "
    "|| a ratio above 1.0 is the cost of re-feeding the stationary operand"
  )


@harness.probe(
  _GROUP,
  "sliding_window_compact_vs_sublane",
  "Does a sliding window run differently with its windowed axis leading (sublane extent 1) than on sublanes?",
)
def _sliding_window_compact_vs_sublane() -> str:
  """The arrangement: give a windowed axis extent 1 so each tap is a leading index.

  Two spellings of the same causal depthwise convolution over the same
  elements with the same FLOPs. The compact arm carries the token axis as a
  leading axis with a trailing ``(1, channels)`` pair, so tap ``k`` is a
  leading-axis slice; the sublane arm carries tokens on the second-minor
  axis, so tap ``k`` is an unaligned sublane slice.

  Hypothesis: the compact layout requires a 32-bit element type, so the
  bf16 arm is run here too and its refusal, if any, recorded rather than
  cited.
  """
  import jax
  import jax.numpy as jnp

  sequences, tokens, channels, taps = 4, 512, 512, 4
  # A large batch, because clearing the dispatch floor is necessary and not
  # sufficient: the arms are expected to be close, and a probe whose arms sit
  # inside its own run-to-run spread has not discriminated. Compare the gap
  # between the arms against the printed spread before reading a winner.
  batch = 96
  weights = tuple(0.25 + 0.1 * index for index in range(taps))

  def compact_call(dtype):
    shape_in = (sequences, tokens + taps - 1, 1, channels)
    shape_out = (sequences, tokens, 1, channels)

    def kernel(source_ref, output_ref) -> None:
      block = source_ref[...]
      total = block[:, 0:tokens] * weights[0]
      for index in range(1, taps):
        total = total + block[:, index : index + tokens] * weights[index]
      output_ref[...] = total

    source = jnp.ones(shape_in, dtype)
    call = jax.jit(
      _pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(shape_out, dtype))
    )
    return source, call

  def sublane_call(dtype):
    shape_in = (sequences, tokens + taps - 1, channels)
    shape_out = (sequences, tokens, channels)

    def kernel(source_ref, output_ref) -> None:
      block = source_ref[...]
      total = block[:, 0:tokens] * weights[0]
      for index in range(1, taps):
        total = total + block[:, index : index + tokens] * weights[index]
      output_ref[...] = total

    source = jnp.ones(shape_in, dtype)
    call = jax.jit(
      _pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(shape_out, dtype))
    )
    return source, call

  arms: dict[str, object] = {}
  for label, builder, dtype in (
    ("compact_f32", compact_call, jnp.float32),
    ("sublane_f32", sublane_call, jnp.float32),
    ("compact_bf16", compact_call, jnp.bfloat16),
    ("sublane_bf16", sublane_call, jnp.bfloat16),
  ):
    try:
      source, call = builder(dtype)
      arms[label] = harness.time_call(
        lambda call=call, source=source: call(source),
        warmups=3,
        repeats=12,
        batch=batch,
      )
    except Exception as error:  # noqa: BLE001 — a refusal is the finding.
      arms[label] = f"NO({type(error).__name__}: {str(error).strip()[:180]})"

  reference = arms["compact_f32"]
  if not isinstance(reference, harness.Timing):
    raise harness.Invalid(f"the compact f32 arm did not run: {reference}")
  harness.require_above_dispatch_floor(reference, what="the compact f32 arm")

  parts = []
  for label, value in arms.items():
    if isinstance(value, harness.Timing):
      parts.append(f"{label}={value.median / reference.median:.2f}x")
    else:
      parts.append(f"{label}={value}")
  return (
    f"causal depthwise conv, {taps} taps, {sequences} sequences x {tokens} "
    f"tokens x {channels} channels, single block, batch={batch}. compact = "
    f"windowed axis leading with trailing (1,{channels}); sublane = "
    f"windowed axis second-minor. Identical elements and FLOPs. Relative to "
    f"compact_f32: {' '.join(parts)} || compact_f32 {reference}"
  )


@harness.probe(
  _GROUP,
  "degenerate_transpose_via_masked_sum",
  "How does a masked reduction compare with a real transpose for a degenerate axis, across widths?",
)
def _degenerate_transpose_via_masked_sum() -> str:
  """The arrangement: move a vector across the lane/sublane divide with arithmetic.

  Trades an XLU relayout for VPU work that never leaves the layout it started
  in. It can only pay while the moved value is small, so the sweep is the
  deliverable: the width at which the permute wins back is the number a design
  needs, not the ratio at one width.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  # `batch` must lift the timed region well clear of the dispatch floor;
  # `require_above_dispatch_floor` withholds the result if it does not.
  grid_steps, batch = 512, 64

  def build(width: int, form: str):
    def transpose_kernel(source_ref, output_ref) -> None:
      column = source_ref[...]  # (width, 1)
      output_ref[...] = column.T * 1.5  # (1, width)

    def masked_kernel(source_ref, output_ref) -> None:
      column = source_ref[...]  # (width, 1)
      spread = jnp.broadcast_to(column, (width, width))
      source_index = jax.lax.broadcasted_iota(jnp.int32, (width, width), 0)
      target_index = jax.lax.broadcasted_iota(jnp.int32, (width, width), 1)
      selected = jnp.where(source_index == target_index, spread, 0.0)
      output_ref[...] = jnp.sum(selected, axis=0, keepdims=True) * 1.5

    kernel = transpose_kernel if form == "transpose" else masked_kernel
    return jax.jit(
      _pallas_call(
        kernel,
        grid=(grid_steps,),
        in_specs=[pl.BlockSpec((width, 1), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((1, width), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((1, width), jnp.float32),
      )
    )

  findings = []
  guarded = False
  for width in (128, 256, 512):
    source = jnp.ones((width, 1), jnp.float32)
    arms: dict[str, object] = {}
    for form in ("transpose", "masked_sum"):
      try:
        call = build(width, form)
        arms[form] = harness.time_call(
          lambda call=call, source=source: call(source),
          warmups=3,
          repeats=12,
          batch=batch,
        )
      except Exception as error:  # noqa: BLE001
        arms[form] = f"NO({type(error).__name__}: {str(error).strip()[:120]})"

    transpose_arm = arms["transpose"]
    masked_arm = arms["masked_sum"]
    if not isinstance(transpose_arm, harness.Timing):
      findings.append(f"width{width}: transpose={transpose_arm}")
      continue
    if not guarded:
      harness.require_above_dispatch_floor(
        transpose_arm, what=f"the width={width} transpose arm"
      )
      guarded = True
    if isinstance(masked_arm, harness.Timing):
      findings.append(
        f"width{width}: masked_sum/transpose="
        f"{masked_arm.median / transpose_arm.median:.2f}x "
        f"(transpose {transpose_arm.median * 1e6:.1f}us)"
      )
    else:
      findings.append(f"width{width}: masked_sum={masked_arm}")

  return (
    f"f32 (width,1) -> (1,width) inside a kernel, {grid_steps} grid steps on "
    f"a resident block, batch={batch}. "
    + " || ".join(findings)
    + " || below 1.0 the masked reduction wins and the arrangement is worth "
    "taking; above 1.0 the real permute is cheaper at that width"
  )


@harness.probe(
  _GROUP,
  "elementwise_before_vs_after_expand",
  "Does the compiler already hoist elementwise work out of an expansion, or must the design do it?",
)
def _elementwise_before_vs_after_expand() -> str:
  """The arrangement: run elementwise work at the multiplicity it is stored at.

  Arithmetic says the ordering is worth the expansion factor. The question
  worth a job is whether the compiler has already taken it — if it has, the
  rule is noise and belongs deleted rather than promoted. The two arms compute
  the same function.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  steps, sublanes, lanes, repeats = 256, 64, 512, 8
  batch = 16
  source = jnp.ones((steps, sublanes, lanes), jnp.float32)

  def before_kernel(source_ref, output_ref) -> None:
    value = jnp.tanh(jnp.exp(source_ref[0] * 0.5))
    output_ref[0] = jnp.broadcast_to(value[None], (repeats, sublanes, lanes))

  def after_kernel(source_ref, output_ref) -> None:
    expanded = jnp.broadcast_to(source_ref[0][None], (repeats, sublanes, lanes))
    output_ref[0] = jnp.tanh(jnp.exp(expanded * 0.5))

  def build(kernel):
    return jax.jit(
      _pallas_call(
        kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec(
          (1, repeats, sublanes, lanes), lambda i: (i, 0, 0, 0)
        ),
        out_shape=jax.ShapeDtypeStruct(
          (steps, repeats, sublanes, lanes), jnp.float32
        ),
      )
    )

  arms: dict[str, object] = {}
  for label, kernel in (
    ("before_expand", before_kernel),
    ("after_expand", after_kernel),
  ):
    try:
      call = build(kernel)
      arms[label] = harness.time_call(
        lambda call=call: call(source), warmups=3, repeats=12, batch=batch
      )
    except Exception as error:  # noqa: BLE001
      arms[label] = f"NO({type(error).__name__}: {str(error).strip()[:140]})"

  reference = arms["before_expand"]
  if not isinstance(reference, harness.Timing):
    raise harness.Invalid(f"the before-expand arm did not run: {reference}")
  harness.require_above_dispatch_floor(reference, what="the before-expand arm")

  after = arms["after_expand"]
  ratio = (
    f"{after.median / reference.median:.2f}x"
    if isinstance(after, harness.Timing)
    else str(after)
  )
  return (
    f"f32, exp then tanh, expansion factor {repeats} on a leading axis, "
    f"{steps} grid steps, batch={batch}. Both arms write the same "
    f"{repeats}x output. after_expand/before_expand = {ratio} "
    f"|| before_expand {reference} "
    f"|| near {repeats:.2f}x means the ordering is the design's to make; "
    "near 1.00x means the compiler already hoisted it and the rule is "
    "redundant here"
  )


@harness.probe(
  _GROUP,
  "expand_before_vs_after_matmul",
  "Does expanding a matmul operand before the contraction cost in proportion to the extra MACs?",
)
def _expand_before_vs_after_matmul() -> str:
  """The arrangement: expand after the contraction, not before it.

  ``dot(repeat(A), B)`` and ``repeat(dot(A, B))`` compute the same thing and
  differ by the expansion factor in MACs. As above, the question is whether
  the compiler has already found it; the arms are written so that it having
  done so shows up as 1.00x.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows, contract, width, repeats = 256, 512, 512, 4
  # `batch` must lift the timed region well clear of the dispatch floor;
  # `require_above_dispatch_floor` withholds the result if it does not.
  grid_steps, batch = 64, 64

  activations = jnp.ones((rows, contract), jnp.bfloat16)
  weights = jnp.ones((contract, width), jnp.bfloat16)

  def before_kernel(activation_ref, weight_ref, output_ref) -> None:
    expanded = jnp.repeat(activation_ref[...][None], repeats, axis=0)
    output_ref[...] = jnp.einsum(
      "rmk,kn->rmn",
      expanded,
      weight_ref[...],
      preferred_element_type=jnp.float32,
    )

  def after_kernel(activation_ref, weight_ref, output_ref) -> None:
    product = jnp.dot(
      activation_ref[...], weight_ref[...], preferred_element_type=jnp.float32
    )
    output_ref[...] = jnp.broadcast_to(product[None], (repeats, rows, width))

  def build(kernel):
    return jax.jit(
      _pallas_call(
        kernel,
        grid=(grid_steps,),
        in_specs=[
          pl.BlockSpec((rows, contract), lambda i: (0, 0)),
          pl.BlockSpec((contract, width), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((repeats, rows, width), lambda i: (0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((repeats, rows, width), jnp.float32),
      )
    )

  arms: dict[str, object] = {}
  for label, kernel in (
    ("expand_before_matmul", before_kernel),
    ("expand_after_matmul", after_kernel),
  ):
    try:
      call = build(kernel)
      arms[label] = harness.time_call(
        lambda call=call: call(activations, weights),
        warmups=3,
        repeats=12,
        batch=batch,
      )
    except Exception as error:  # noqa: BLE001
      arms[label] = f"NO({type(error).__name__}: {str(error).strip()[:140]})"

  reference = arms["expand_after_matmul"]
  if not isinstance(reference, harness.Timing):
    raise harness.Invalid(f"the expand-after arm did not run: {reference}")
  harness.require_above_dispatch_floor(reference, what="the expand-after arm")

  before = arms["expand_before_matmul"]
  ratio = (
    f"{before.median / reference.median:.2f}x"
    if isinstance(before, harness.Timing)
    else str(before)
  )
  return (
    f"bf16 ({rows},{contract}) @ ({contract},{width}), expansion factor "
    f"{repeats}, {grid_steps} grid steps on a resident block, batch={batch}. "
    f"expand_before/expand_after = {ratio} || expand_after {reference} "
    f"|| near {repeats:.2f}x is the MAC ratio showing through; near 1.00x "
    "means the compiler sank the expansion past the contraction on its own"
  )


@harness.probe(
  _GROUP,
  "mask_index_dtype_match",
  "Does a select cost more when its index/iota width does not match the data it selects, and which widths lower?",
)
def _mask_index_dtype_match() -> str:
  """The arrangement: match a mask's or iota's itemsize to the data's.

  Ratios are taken strictly within one data dtype: an f32 element and a bf16
  element do not cost the same to move, so a cross-dtype ratio here would mean
  nothing. An iota dtype with no lowering is a finding and lands as its own
  arm rather than a crash.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  # One lane tile wide, not four. An int8 iota tops out at 127, so on a
  # 512-lane axis its values wrap and the comparison constant does not even
  # fit — the arm dies at trace time and measures nothing. 128 lanes is the
  # widest axis on which all three index dtypes are representable, so it is
  # the only width at which the three are comparable at all. The sublane
  # extent is raised to hold the block size constant.
  steps, sublanes, lanes = 1024, 256, 128
  batch = 24

  def build(data_name: str, index_name: str):
    data_dtype = getattr(jnp, data_name)
    index_dtype = getattr(jnp, index_name)
    limit = lanes // 2

    def kernel(source_ref, output_ref) -> None:
      block = source_ref[...]
      index = jax.lax.broadcasted_iota(
        index_dtype, block.shape, len(block.shape) - 1
      )
      inside = index < index_dtype(limit)
      output_ref[...] = jnp.where(inside, block * data_dtype(1.5), block)

    source = jnp.ones((steps, sublanes, lanes), data_dtype)
    call = jax.jit(
      _pallas_call(
        kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((steps, sublanes, lanes), data_dtype),
      )
    )
    return source, call

  findings = []
  guarded = False
  for data_name, index_names in (
    ("float32", ("int32", "int16", "int8")),
    ("bfloat16", ("int32", "int16", "int8")),
  ):
    arms: dict[str, object] = {}
    for index_name in index_names:
      if not hasattr(jnp, index_name) or not hasattr(jnp, data_name):
        arms[index_name] = "dtype-absent"
        continue
      try:
        source, call = build(data_name, index_name)
        arms[index_name] = harness.time_call(
          lambda call=call, source=source: call(source),
          warmups=3,
          repeats=12,
          batch=batch,
        )
      except Exception as error:  # noqa: BLE001 — a refusal is the finding.
        # 300, not 120: a Mosaic refusal puts the module and HLO names
        # before the reason, so 120 chars truncates mid-`Target doe`
        # and the refusal is unquotable — which is the one thing a
        # refusal arm exists to produce.
        arms[index_name] = (
          f"NO({type(error).__name__}: {str(error).strip()[:300]})"
        )

    reference = arms.get("int32")
    if not isinstance(reference, harness.Timing):
      findings.append(f"{data_name}: int32 reference did not run ({reference})")
      continue
    if not guarded:
      harness.require_above_dispatch_floor(
        reference, what=f"the {data_name}/int32 arm"
      )
      guarded = True
    parts = []
    for index_name, value in arms.items():
      if isinstance(value, harness.Timing):
        parts.append(
          f"iota_{index_name}={value.median / reference.median:.2f}x"
        )
      else:
        parts.append(f"iota_{index_name}={value}")
    findings.append(
      f"{data_name} data [{' '.join(parts)}] "
      f"(int32 arm {reference.median * 1e6:.1f}us, "
      f"spread {reference.spread_pct:.0f}%)"
    )

  return (
    f"select over a ({sublanes},{lanes}) block, {steps} grid steps, "
    f"batch={batch}, iota on the lane axis. One lane tile wide because an "
    f"int8 iota cannot represent an index past 127, so a wider lane axis "
    f"excludes that arm entirely. Ratios are within one data "
    f"dtype only — an f32 and a bf16 element do not move for the same price. "
    + " || ".join(findings)
    + " || f32 data with an int32 iota and bf16 data with an int16 iota are "
    "the matched pairs; a ratio above 1.0 on a mismatched pair is the repack"
  )


@harness.probe(
  _GROUP,
  "expand_blocked_vs_interleaved",
  "Is the cost of an expansion set by the axis it runs along, or by whether it interleaves?",
)
def _expand_blocked_vs_interleaved() -> str:
  """Separates the *axis* of an expansion from its *spelling*.

  `expand_axis_leading_vs_sublane_vs_lane` varies both at once — its lane arm
  is an element-interleaving ``jnp.repeat`` and its leading arm is a
  broadcast — so whatever it measures gets attributed to the axis, which is
  what its name says. This probe holds the spelling fixed and varies the
  axis, then holds the axis fixed and varies the spelling, so the two can be
  told apart.

  The two spellings, for a repeat factor ``R`` along lanes:

  * **interleaved** — ``jnp.repeat(x, R, axis=-1)``: ``a a a a b b b b ...``.
    Equivalently a broadcast into an axis inserted *after* the lane axis.
  * **blocked** — broadcast into an axis inserted *before* the lane axis,
    then reshape: ``a b c ... a b c ...``. Same bytes, same element count,
    a different permutation.

  A fifth arm repeats on ``axis=1`` of a **rank-3** block, which is nominally
  the sublane axis and so ought to be cheap if the axis is what matters.

  Every arm reads ``S x L`` elements and writes ``R`` times that, so all arms
  move identical bytes and differ only in the permutation requested. The
  leading-axis broadcast is the denominator, matching the older probe so the
  two are comparable.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  steps, sublanes, lanes, repeats = 512, 64, 256, 4
  batch = 24
  # A rank-3 block with the same element count, so the rank-3 arm is not
  # measuring a different amount of work: 8 * 16 * 128 == 64 * 256.
  c_dim, h_dim, d_dim = 8, 16, 128
  assert c_dim * h_dim * d_dim == sublanes * lanes

  def leading_kernel(source_ref, output_ref) -> None:
    block = source_ref[0]
    output_ref[0] = (
      jnp.broadcast_to(block[None], (repeats, sublanes, lanes)) * 1.5
    )

  def sublane_interleaved_kernel(source_ref, output_ref) -> None:
    output_ref[0] = jnp.repeat(source_ref[0], repeats, axis=0) * 1.5

  def sublane_blocked_kernel(source_ref, output_ref) -> None:
    block = source_ref[0]
    expanded = jnp.broadcast_to(block[None], (repeats, sublanes, lanes))
    output_ref[0] = expanded.reshape(repeats * sublanes, lanes) * 1.5

  def lane_interleaved_kernel(source_ref, output_ref) -> None:
    output_ref[0] = jnp.repeat(source_ref[0], repeats, axis=1) * 1.5

  def lane_blocked_kernel(source_ref, output_ref) -> None:
    block = source_ref[0]
    # Broadcast into an axis *before* the lane axis, then flatten it in.
    expanded = jnp.broadcast_to(block[:, None, :], (sublanes, repeats, lanes))
    output_ref[0] = expanded.reshape(sublanes, repeats * lanes) * 1.5

  def rank3_sublane_repeat_kernel(source_ref, output_ref) -> None:
    # `axis=1` of a rank-3 array is the sublane axis. An axis-based rule
    # predicts this is cheap; a spelling-based one predicts it is not.
    output_ref[0] = jnp.repeat(source_ref[0], repeats, axis=1) * 1.5

  def build(kernel, in_block, in_shape, out_block, out_shape, dtype):
    source = jnp.ones(in_shape, dtype)
    call = jax.jit(
      _pallas_call(
        kernel,
        grid=(steps,),
        in_specs=[
          pl.BlockSpec(in_block, lambda i: (i,) + (0,) * (len(in_block) - 1))
        ],
        out_specs=pl.BlockSpec(
          out_block, lambda i: (i,) + (0,) * (len(out_block) - 1)
        ),
        out_shape=jax.ShapeDtypeStruct(out_shape, dtype),
      )
    )
    return call, source

  flat_in_block = (1, sublanes, lanes)
  flat_in_shape = (steps, sublanes, lanes)
  rank3_in_block = (1, c_dim, h_dim, d_dim)
  rank3_in_shape = (steps, c_dim, h_dim, d_dim)

  plans = (
    (
      "leading",
      leading_kernel,
      flat_in_block,
      flat_in_shape,
      (1, repeats, sublanes, lanes),
      (steps, repeats, sublanes, lanes),
    ),
    (
      "sublane_interleaved",
      sublane_interleaved_kernel,
      flat_in_block,
      flat_in_shape,
      (1, repeats * sublanes, lanes),
      (steps, repeats * sublanes, lanes),
    ),
    (
      "sublane_blocked",
      sublane_blocked_kernel,
      flat_in_block,
      flat_in_shape,
      (1, repeats * sublanes, lanes),
      (steps, repeats * sublanes, lanes),
    ),
    (
      "lane_interleaved",
      lane_interleaved_kernel,
      flat_in_block,
      flat_in_shape,
      (1, sublanes, repeats * lanes),
      (steps, sublanes, repeats * lanes),
    ),
    (
      "lane_blocked",
      lane_blocked_kernel,
      flat_in_block,
      flat_in_shape,
      (1, sublanes, repeats * lanes),
      (steps, sublanes, repeats * lanes),
    ),
    (
      "rank3_repeat_axis1",
      rank3_sublane_repeat_kernel,
      rank3_in_block,
      rank3_in_shape,
      (1, c_dim, repeats * h_dim, d_dim),
      (steps, c_dim, repeats * h_dim, d_dim),
    ),
  )

  findings: list[str] = []
  reference_note = ""
  for dtype, label in ((jnp.float32, "f32"), (jnp.bfloat16, "bf16")):
    arms: dict[str, object] = {}
    for name, kernel, in_block, in_shape, out_block, out_shape in plans:
      try:
        call, source = build(
          kernel, in_block, in_shape, out_block, out_shape, dtype
        )
        arms[name] = harness.time_call(
          lambda call=call, source=source: call(source),
          warmups=3,
          repeats=12,
          batch=batch,
        )
      except Exception as error:  # noqa: BLE001 — a refusal is the finding.
        arms[name] = f"NO({type(error).__name__}: {str(error).strip()[:120]})"

    reference = arms["leading"]
    if not isinstance(reference, harness.Timing):
      raise harness.Invalid(
        f"the {label} leading-axis denominator did not run: {reference}"
      )
    harness.require_above_dispatch_floor(
      reference, what=f"the {label} leading-axis arm"
    )
    if label == "f32":
      reference_note = f"f32 leading-axis arm {reference}"

    parts = []
    for name, value in arms.items():
      if isinstance(value, harness.Timing):
        parts.append(f"{name}={value.median / reference.median:.2f}x")
      else:
        parts.append(f"{name}={value}")
    findings.append(f"{label}: " + " ".join(parts))

  return (
    f"grid={steps}, source block ({sublanes},{lanes}) written {repeats}x, "
    f"identical bytes in every arm; rank-3 arm is "
    f"({c_dim},{h_dim},{d_dim}) with the same element count. Relative to "
    f"the leading-axis broadcast. "
    + " || ".join(findings)
    + f" || {reference_note} || If blocked and interleaved differ while the "
    "two blocked arms do not, the axis is not the lever and the spelling is"
  )
