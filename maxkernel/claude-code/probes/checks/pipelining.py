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
"""Whether the transfers hide behind the compute, and what it costs when they do not.

Supplies answers for `reference/design/pipelining.md`.

**Name the denominator.** "Bandwidth" on this chip is at least three different
numbers — what a large XLA copy achieves, what a Pallas grid achieves, and what
a hand-rolled DMA achieves — and a roofline that silently switches between them
is wrong by more than the effect most optimizations chase. `hbm_bandwidth_arms`
measures all three in one probe, side by side, so that the reference file has to
say which one it means.
"""

from __future__ import annotations

import harness

_GROUP = "pipelining"


@harness.probe(
  _GROUP,
  "hbm_bandwidth_arms",
  "What HBM bandwidth is achievable, measured three different ways?",
)
def _hbm_bandwidth_arms() -> str:
  """Three arms, because a roofline needs to name which one it used."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  elements = 256 * 1024 * 1024  # 1 GiB of f32.
  rows = elements // 1024
  block_rows = 1024
  source = jnp.ones((rows, 1024), dtype=jnp.float32)
  moved_bytes = 2 * source.size * 4  # Read once, write once.

  def copy_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...]

  xla_copy = jax.jit(lambda value: value * 1.0)
  # Batched so the timed region clears the host round trip
  # (`harness.dispatch_floor_seconds`); otherwise part of every bandwidth
  # number is host overhead charged to HBM.
  xla_timing = harness.time_call(
    lambda: xla_copy(source), warmups=3, repeats=10, batch=4
  )

  grid_call = jax.jit(
    pl.pallas_call(
      copy_kernel,
      grid=(rows // block_rows,),
      in_specs=[pl.BlockSpec((block_rows, 1024), lambda i: (i, 0))],
      out_specs=pl.BlockSpec((block_rows, 1024), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
    )
  )
  grid_timing = harness.time_call(
    lambda: grid_call(source), warmups=3, repeats=10, batch=4
  )

  read_only = jax.jit(lambda value: jnp.sum(value, dtype=jnp.float32))
  read_timing = harness.time_call(
    lambda: read_only(source), warmups=3, repeats=10, batch=4
  )
  harness.require_above_dispatch_floor(read_timing, what="the read-only arm")

  def rate(timing: harness.Timing, total_bytes: int) -> float:
    return total_bytes / timing.median / 1e12

  return (
    f"{source.nbytes / 2**30:.2f} GiB f32 array. "
    f"xla_read_write={rate(xla_timing, moved_bytes):.3f}TB/s ({xla_timing}) | "
    f"pallas_grid_read_write={rate(grid_timing, moved_bytes):.3f}TB/s "
    f"(block {block_rows}x1024, {grid_timing}) | "
    f"xla_read_only_sum={rate(read_timing, source.nbytes):.3f}TB/s ({read_timing}) "
    "|| the read-only arm counts only the read, so it is the closest thing "
    "here to a peak figure; the read-write arms are what a real kernel sees"
  )


@harness.probe(
  _GROUP,
  "block_size_bandwidth_curve",
  "How large does a block have to be before a copy reaches full bandwidth?",
)
def _block_size_bandwidth_curve() -> str:
  """The bandwidth side of the block-size squeeze, as an actual curve."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = 65536
  columns = 1024
  source = jnp.ones((rows, columns), dtype=jnp.float32)
  moved_bytes = 2 * source.nbytes

  def copy_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...]

  findings = {}
  for block_rows in (8, 32, 128, 512, 2048, 8192):
    call = jax.jit(
      pl.pallas_call(
        copy_kernel,
        grid=(rows // block_rows,),
        in_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
      )
    )
    try:
      timing = harness.time_call(
        lambda call=call: call(source), warmups=2, repeats=8, batch=4
      )
    except Exception as error:  # noqa: BLE001
      findings[block_rows] = f"NO({type(error).__name__})"
      continue
    block_kib = block_rows * columns * 4 / 1024
    findings[block_rows] = (
      f"{moved_bytes / timing.median / 1e12:.3f}TB/s ({block_kib:.0f}KiB/block)"
    )

  return (
    f"f32 {rows}x{columns} copy, read+write counted, block rows swept: "
    + " | ".join(f"{rows_}={value}" for rows_, value in findings.items())
    + " || the knee is where a block stops being too small to amortize a "
    "transfer; compare the top of this curve against hbm_bandwidth_arms"
  )


@harness.probe(
  _GROUP,
  "arithmetic_intensity_crossover",
  "At how many FLOPs per byte does a kernel stop being bandwidth-bound?",
)
def _arithmetic_intensity_crossover() -> str:
  """Locates the ridge point empirically instead of dividing two spec sheets."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = 16384
  columns = 1024
  block_rows = 1024
  source = jnp.ones((rows, columns), dtype=jnp.float32)
  moved_bytes = 2 * source.nbytes

  def build(work_per_element: int):
    def work_kernel(source_ref, output_ref) -> None:
      def one_step(_, value):
        return value * 1.0000001 + 0.0000001

      output_ref[...] = jax.lax.fori_loop(
        0, work_per_element, one_step, source_ref[...]
      )

    return jax.jit(
      pl.pallas_call(
        work_kernel,
        grid=(rows // block_rows,),
        in_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
      )
    )

  findings = {}
  for work_per_element in (1, 2, 4, 8, 16, 32, 64, 128):
    call = build(work_per_element)
    timing = harness.time_call(
      lambda call=call: call(source), warmups=2, repeats=10, batch=8
    )
    intensity = 2.0 * work_per_element / 8.0  # 8 bytes moved per element.
    findings[work_per_element] = (
      f"{intensity:.2f}flop/byte "
      f"bw={moved_bytes / timing.median / 1e12:.3f}TB/s "
      f"compute={2.0 * source.size * work_per_element / timing.median / 1e12:.2f}Tflop/s"
    )

  return (
    f"f32 {rows}x{columns}, {block_rows}-row blocks, N dependent FMAs per "
    "element. As N rises the bandwidth figure should fall away from the "
    "hbm_bandwidth_arms value and the compute figure should flatten at the "
    "vpu_fma_rate value; the crossover is the ridge point. "
    + " | ".join(f"N={n}: {value}" for n, value in findings.items())
  )


@harness.probe(
  _GROUP,
  "pipeline_depth_and_warmup",
  "How many grid steps does a pipeline need before it reaches steady state?",
)
def _pipeline_depth_and_warmup() -> str:
  """Prologue and epilogue are the part of a pipeline that never overlaps."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  block_rows = 1024
  columns = 1024

  def scale_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] * 2.0

  findings = {}
  for steps in (1, 2, 4, 8, 16, 32, 64):
    rows = steps * block_rows
    source = jnp.ones((rows, columns), dtype=jnp.float32)
    call = jax.jit(
      pl.pallas_call(
        scale_kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
      )
    )
    # Deliberately unbatched, and the only probe in the suite that is.
    # This one is *about* the fixed cost: the affine fit below separates
    # the per-step slope from the intercept, and batching would amortize
    # away the very intercept being measured. The fit's own intercept is
    # the cross-check -- it should land near `dispatch_floor`.
    timing = harness.time_call(
      lambda call=call: call(source), warmups=3, repeats=15
    )
    findings[steps] = timing
    del source

  # A perfectly pipelined kernel is affine in the step count: fit the two
  # ends and report how far the short grids sit above the line.
  slope = (findings[64].median - findings[4].median) / (64 - 4)
  intercept = findings[4].median - 4 * slope
  excess = {
    steps: (timing.median - (intercept + steps * slope)) * 1e6
    for steps, timing in findings.items()
  }
  return (
    f"f32 {block_rows}x{columns} blocks, one scale per element, grid steps swept: "
    + " | ".join(
      f"{steps}:{timing.median * 1e6:.1f}us"
      for steps, timing in findings.items()
    )
    + f" || affine fit over steps 4..64: {slope * 1e6:.2f}us/step + "
    f"{intercept * 1e6:.1f}us fixed. residual above that line, in us: "
    + " ".join(f"{steps}:{value:+.1f}" for steps, value in excess.items())
    + " || the fixed term is dispatch plus prologue/epilogue; a kernel whose "
    "whole grid is a few steps pays it in full"
  )


@harness.probe(
  _GROUP,
  "emit_pipeline_vs_blockspec_grid",
  "Does an inner pltpu.emit_pipeline beat the BlockSpec-managed grid at the same partition?",
)
def _emit_pipeline_vs_blockspec_grid() -> str:
  """The managed grid is the default; this asks what hand-rolling buys."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  if not hasattr(pltpu, "emit_pipeline"):
    raise harness.Unsupported("pltpu.emit_pipeline does not exist on this jax")

  rows = 16384
  columns = 1024
  block_rows = 1024
  source = jnp.ones((rows, columns), dtype=jnp.float32)
  moved_bytes = 2 * source.nbytes

  def scale_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] * 2.0

  managed = jax.jit(
    pl.pallas_call(
      scale_kernel,
      grid=(rows // block_rows,),
      in_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
      out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
    )
  )
  managed_timing = harness.time_call(
    lambda: managed(source), warmups=3, repeats=10, batch=32
  )

  def outer_kernel(source_hbm_ref, output_hbm_ref) -> None:
    pipeline = pltpu.emit_pipeline(
      scale_kernel,
      grid=(rows // block_rows,),
      in_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
      out_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
    )
    pipeline(source_hbm_ref, output_hbm_ref)

  emitted = jax.jit(
    pl.pallas_call(
      outer_kernel,
      in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
      out_specs=pl.BlockSpec(memory_space=pl.ANY),
      out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
    )
  )
  emitted_timing = harness.time_call(
    lambda: emitted(source), warmups=3, repeats=10, batch=32
  )
  harness.require_above_dispatch_floor(
    managed_timing, what="the BlockSpec-managed arm"
  )

  reference = jnp.asarray(source) * 2.0
  emitted_result = emitted(source)
  max_error = float(jnp.max(jnp.abs(emitted_result - reference)))
  if max_error != 0.0:
    raise harness.Invalid(
      f"emit_pipeline arm produced wrong values (max_abs_error={max_error}); "
      "no timing from it is quotable"
    )

  return (
    f"f32 {rows}x{columns} scale, {block_rows}-row blocks, identical partition. "
    f"blockspec_grid={moved_bytes / managed_timing.median / 1e12:.3f}TB/s "
    f"({managed_timing}) | "
    f"emit_pipeline={moved_bytes / emitted_timing.median / 1e12:.3f}TB/s "
    f"({emitted_timing}) | emit_pipeline/managed = "
    f"{emitted_timing.median / managed_timing.median:.2f}x. "
    "emit_pipeline output verified exact against the managed arm."
  )


@harness.probe(
  _GROUP,
  "manual_dma_vs_managed",
  "Does a hand-rolled make_async_copy beat the BlockSpec-managed transfer?",
)
def _manual_dma_vs_managed() -> str:
  """The lowest rung of hand-rolling, with a serial arm as the negative control.

  The serial arm matters: a comparison timed below the dispatch floor can make
  hand-rolling look like a win when both arms are mostly overhead. If the
  serial arm is not clearly slower than both pipelined arms, the whole probe
  is measuring overhead.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  if not hasattr(pltpu, "make_async_copy"):
    raise harness.Unsupported(
      "pltpu.make_async_copy does not exist on this jax"
    )

  rows = 8192
  columns = 1024
  block_rows = 512
  steps = rows // block_rows
  source = jnp.ones((rows, columns), dtype=jnp.float32)
  moved_bytes = 2 * source.nbytes

  def scale_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] * 2.0

  managed = jax.jit(
    pl.pallas_call(
      scale_kernel,
      grid=(steps,),
      in_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
      out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
    )
  )

  def serial_kernel(
    source_hbm_ref, output_hbm_ref, in_buffer, out_buffer, sem
  ) -> None:
    """One block at a time, no overlap at all: the negative control."""

    def one_step(step, _):
      slice_in = pltpu.make_async_copy(
        source_hbm_ref.at[pl.ds(step * block_rows, block_rows)],
        in_buffer,
        sem,
      )
      slice_in.start()
      slice_in.wait()
      out_buffer[...] = in_buffer[...] * 2.0
      slice_out = pltpu.make_async_copy(
        out_buffer,
        output_hbm_ref.at[pl.ds(step * block_rows, block_rows)],
        sem,
      )
      slice_out.start()
      slice_out.wait()
      return 0

    jax.lax.fori_loop(0, steps, one_step, 0)

  serial = jax.jit(
    pl.pallas_call(
      serial_kernel,
      in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
      out_specs=pl.BlockSpec(memory_space=pl.ANY),
      out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
      scratch_shapes=[
        pltpu.VMEM((block_rows, columns), jnp.float32),
        pltpu.VMEM((block_rows, columns), jnp.float32),
        pltpu.SemaphoreType.DMA,
      ],
    )
  )

  managed_timing = harness.time_call(
    lambda: managed(source), warmups=3, repeats=10, batch=32
  )
  try:
    serial_result = serial(source)
    jax.block_until_ready(serial_result)
  except Exception as error:  # noqa: BLE001
    return (
      harness.Verdict.FAIL,
      f"the hand-rolled serial arm did not run: {type(error).__name__}: "
      f"{str(error)[:300]}. The managed arm reached "
      f"{moved_bytes / managed_timing.median / 1e12:.3f}TB/s ({managed_timing}).",
    )

  reference = source * 2.0
  max_error = float(jnp.max(jnp.abs(serial_result - reference)))
  if max_error != 0.0:
    raise harness.Invalid(
      f"the serial DMA arm produced wrong values (max_abs_error={max_error})"
    )
  serial_timing = harness.time_call(
    lambda: serial(source), warmups=3, repeats=10, batch=32
  )
  harness.require_above_dispatch_floor(
    managed_timing, what="the BlockSpec-managed arm"
  )

  return (
    f"f32 {rows}x{columns} scale, {block_rows}-row blocks, {steps} steps. "
    f"blockspec_managed={moved_bytes / managed_timing.median / 1e12:.3f}TB/s "
    f"({managed_timing}) | "
    f"hand_rolled_serial_no_overlap="
    f"{moved_bytes / serial_timing.median / 1e12:.3f}TB/s ({serial_timing}) | "
    f"serial/managed = {serial_timing.median / managed_timing.median:.2f}x. "
    "The serial arm is the control, not a proposal: it is what the managed "
    "grid buys you over no overlap at all. If it is not clearly slower, "
    "this shape is too small to be measuring transfer overlap."
  )
