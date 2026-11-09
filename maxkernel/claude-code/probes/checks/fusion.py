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
"""What widening the kernel boundary actually buys, and what it costs.

Supplies answers for `reference/design/fusion_boundary.md`.

The claim a fusion decision rests on is that the intermediate avoids a round
trip to HBM. XLA already fuses a great deal on its own, so the honest question
is not "is fused faster than unfused" but "is a hand-written fused kernel faster
than what XLA does with the same graph". Every arm here therefore has an XLA
arm, and it is a real comparison rather than a straw man.
"""

from __future__ import annotations

import harness

_GROUP = "fusion"


@harness.probe(
  _GROUP,
  "elementwise_chain_xla_vs_pallas",
  "Does XLA already fuse a chain of elementwise ops as well as one Pallas kernel would?",
)
def _elementwise_chain_xla_vs_pallas() -> str:
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = 16384
  columns = 1024
  block_rows = 1024
  source = jnp.ones((rows, columns), dtype=jnp.float32)
  moved_bytes = 2 * source.nbytes

  findings = {}
  for chain_length in (1, 4, 16):

    def apply_chain(value, chain_length=chain_length):
      for _ in range(chain_length):
        value = jnp.tanh(value) * 1.5 + 0.25
      return value

    xla_call = jax.jit(apply_chain)
    # One elementwise pass over this array is short next to the host round
    # trip; at batch=1 the pallas/xla ratio would be a ratio between two
    # numbers that are both mostly dispatch.
    xla_timing = harness.time_call(
      lambda xla_call=xla_call: xla_call(source),
      warmups=3,
      repeats=10,
      batch=24,
    )

    def chain_kernel(source_ref, output_ref, chain_length=chain_length) -> None:
      value = source_ref[...]
      for _ in range(chain_length):
        value = jnp.tanh(value) * 1.5 + 0.25
      output_ref[...] = value

    pallas_call = jax.jit(
      pl.pallas_call(
        chain_kernel,
        grid=(rows // block_rows,),
        in_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
      )
    )
    pallas_timing = harness.time_call(
      lambda pallas_call=pallas_call: pallas_call(source),
      warmups=3,
      repeats=10,
      batch=24,
    )
    if chain_length == 1:
      harness.require_above_dispatch_floor(
        xla_timing, what="the length=1 XLA arm"
      )

    difference = float(jnp.max(jnp.abs(xla_call(source) - pallas_call(source))))
    findings[chain_length] = (
      f"xla={xla_timing.median * 1e6:.0f}us "
      f"({moved_bytes / xla_timing.median / 1e12:.3f}TB/s) "
      f"pallas={pallas_timing.median * 1e6:.0f}us "
      f"pallas/xla={pallas_timing.median / xla_timing.median:.2f}x "
      f"max_abs_diff={difference:.2e}"
    )

  return (
    f"f32 {rows}x{columns}, chain of `tanh(x)*1.5+0.25`, "
    f"pallas block rows={block_rows}. "
    + " | ".join(f"length={n}: {value}" for n, value in findings.items())
    + " || a ratio near 1.0 at every length means XLA fused the chain "
    "itself and there is nothing for a kernel to win here"
  )


@harness.probe(
  _GROUP,
  "matmul_epilogue_fusion",
  "Is a matmul with its bias-and-activation epilogue fused into one kernel faster than XLA's version?",
)
def _matmul_epilogue_fusion() -> str:
  """The archetypal fusion decision, priced.

  Correctness is checked against the XLA arm before any timing is quoted: a
  fused kernel that skips the epilogue benchmarks beautifully.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = contraction = columns = 4096
  block_rows = block_columns = 512
  block_contraction = 512

  lhs = jnp.ones((rows, contraction), dtype=jnp.bfloat16) * 0.01
  rhs = jnp.ones((contraction, columns), dtype=jnp.bfloat16) * 0.01
  bias = jnp.arange(columns, dtype=jnp.float32).reshape(1, columns) * 0.001

  def xla_graph(lhs_value, rhs_value, bias_value):
    product = jnp.dot(lhs_value, rhs_value, preferred_element_type=jnp.float32)
    return jax.nn.gelu(product + bias_value, approximate=True)

  xla_call = jax.jit(xla_graph)
  # If either arm sits within a small multiple of the host round trip, part
  # of each figure is host overhead and the fused/xla ratio is biased toward
  # 1. Batching keeps both arms well clear of it; the floor check below owns
  # the verdict.
  xla_timing = harness.time_call(
    lambda: xla_call(lhs, rhs, bias), warmups=3, repeats=10, batch=4
  )

  def fused_kernel(
    lhs_ref, rhs_ref, bias_ref, output_ref, accumulator_ref
  ) -> None:
    step = pl.program_id(2)
    last_step = pl.num_programs(2) - 1

    @pl.when(step == 0)
    def _initialize() -> None:
      accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

    accumulator_ref[...] += jnp.dot(
      lhs_ref[...], rhs_ref[...], preferred_element_type=jnp.float32
    )

    @pl.when(step == last_step)
    def _epilogue() -> None:
      output_ref[...] = jax.nn.gelu(
        accumulator_ref[...] + bias_ref[...], approximate=True
      )

  from jax.experimental.pallas import tpu as pltpu

  fused_call = jax.jit(
    pl.pallas_call(
      fused_kernel,
      grid=(
        rows // block_rows,
        columns // block_columns,
        contraction // block_contraction,
      ),
      in_specs=[
        pl.BlockSpec((block_rows, block_contraction), lambda i, j, k: (i, k)),
        pl.BlockSpec(
          (block_contraction, block_columns), lambda i, j, k: (k, j)
        ),
        pl.BlockSpec((1, block_columns), lambda i, j, k: (0, j)),
      ],
      out_specs=pl.BlockSpec(
        (block_rows, block_columns), lambda i, j, k: (i, j)
      ),
      out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
      scratch_shapes=[pltpu.VMEM((block_rows, block_columns), jnp.float32)],
    )
  )

  fused_result = fused_call(lhs, rhs, bias)
  xla_result = xla_call(lhs, rhs, bias)
  max_error = float(jnp.max(jnp.abs(fused_result - xla_result)))
  relative = max_error / float(jnp.max(jnp.abs(xla_result)) + 1e-30)
  if relative > 1e-3:
    raise harness.Invalid(
      f"fused kernel disagrees with the XLA arm: max_abs_error={max_error:.3e} "
      f"relative={relative:.3e}. No timing from this probe is quotable."
    )

  fused_timing = harness.time_call(
    lambda: fused_call(lhs, rhs, bias), warmups=3, repeats=10, batch=4
  )
  harness.require_above_dispatch_floor(xla_timing, what="the XLA arm")
  total_flops = 2.0 * rows * contraction * columns
  return (
    f"bf16 {rows}x{contraction}x{columns} matmul + f32 bias + tanh-gelu. "
    f"xla={xla_timing.median * 1e6:.0f}us "
    f"({total_flops / xla_timing.median / 1e12:.1f}Tops/s) | "
    f"fused_pallas={fused_timing.median * 1e6:.0f}us "
    f"({total_flops / fused_timing.median / 1e12:.1f}Tops/s) | "
    f"fused/xla={fused_timing.median / xla_timing.median:.2f}x | "
    f"correctness vs XLA: max_abs_error={max_error:.3e} relative={relative:.3e}"
  )


@harness.probe(
  _GROUP,
  "intermediate_hbm_roundtrip",
  "What does materializing an intermediate to HBM actually cost, at a known size?",
)
def _intermediate_hbm_roundtrip() -> str:
  """The number a fusion argument is really made of, isolated.

  Two Pallas kernels with an HBM intermediate between them, against one
  kernel doing the same arithmetic. Both arms use the same block size, so the
  only difference is the round trip.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = 16384
  columns = 1024
  block_rows = 1024
  source = jnp.ones((rows, columns), dtype=jnp.float32)
  intermediate_bytes = 2 * source.nbytes  # Written once, read once.

  def first_half(source_ref, output_ref) -> None:
    output_ref[...] = jnp.tanh(source_ref[...]) * 1.5

  def second_half(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] * source_ref[...] + 0.25

  def whole(source_ref, output_ref) -> None:
    value = jnp.tanh(source_ref[...]) * 1.5
    output_ref[...] = value * value + 0.25

  def build(kernel):
    return pl.pallas_call(
      kernel,
      grid=(rows // block_rows,),
      in_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
      out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
    )

  split_call = jax.jit(
    lambda value: build(second_half)(build(first_half)(value))
  )
  fused_call = jax.jit(build(whole))

  difference = float(jnp.max(jnp.abs(split_call(source) - fused_call(source))))
  if difference > 1e-5:
    raise harness.Invalid(
      f"the two arms compute different things (max_abs_diff={difference:.3e})"
    )

  # The saving this probe reports is a *difference* between two arms, which
  # is the shape of measurement the dispatch floor damages most: a constant
  # round trip added to both cancels in the difference but inflates both
  # denominators. Batching keeps it out of both.
  split_timing = harness.time_call(
    lambda: split_call(source), warmups=3, repeats=10, batch=24
  )
  fused_timing = harness.time_call(
    lambda: fused_call(source), warmups=3, repeats=10, batch=24
  )
  harness.require_above_dispatch_floor(fused_timing, what="the one-kernel arm")
  saved = split_timing.median - fused_timing.median
  return (
    f"f32 {rows}x{columns} ({source.nbytes / 2**20:.0f} MiB), same "
    f"{block_rows}-row block in both arms. "
    f"two_kernels_via_hbm={split_timing.median * 1e6:.0f}us | "
    f"one_kernel={fused_timing.median * 1e6:.0f}us | "
    f"saved={saved * 1e6:.0f}us for an intermediate of "
    f"{intermediate_bytes / 2**20:.0f} MiB of traffic, implying "
    f"{intermediate_bytes / saved / 1e12:.3f}TB/s of avoided traffic "
    "-- cross-check that against hbm_bandwidth_arms: a figure far above it "
    "means the saving is not only the round trip, and one far below means "
    "the arms differ in something else too"
  )


@harness.probe(
  _GROUP,
  "operand_count_vmem_pressure",
  "How does the largest workable block shrink as a kernel takes on more operands?",
)
def _operand_count_vmem_pressure() -> str:
  """The capacity price of a wider kernel, which is what limits fusion."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  columns = 1024

  def build(operand_count: int, block_rows: int):
    def sum_kernel(*refs) -> None:
      output_ref = refs[-1]
      total = refs[0][...]
      for ref in refs[1:-1]:
        total = total + ref[...]
      output_ref[...] = total

    rows = block_rows * 4
    specs = [
      pl.BlockSpec((block_rows, columns), lambda i: (i, 0))
      for _ in range(operand_count)
    ]
    call = pl.pallas_call(
      sum_kernel,
      grid=(4,),
      in_specs=specs,
      out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
    )
    operands = [
      jnp.ones((rows, columns), dtype=jnp.float32) for _ in range(operand_count)
    ]
    return lambda: jax.block_until_ready(call(*operands))

  findings = {}
  for operand_count in (1, 2, 4, 8):

    def attempt(block_rows: int, operand_count=operand_count) -> None:
      build(operand_count, block_rows)()

    largest_ok = None
    first_fail = None
    for block_rows in (512, 1024, 2048, 4096, 8192, 16384):
      try:
        attempt(block_rows)
      except Exception:  # noqa: BLE001 — the boundary is the measurement.
        first_fail = block_rows
        break
      largest_ok = block_rows
    if largest_ok is None:
      findings[operand_count] = "nothing fits"
      continue
    working_mib = (operand_count + 1) * largest_ok * columns * 4 / 2**20
    findings[operand_count] = (
      f"largest_ok={largest_ok}rows (first_fail={first_fail}) "
      f"=> {working_mib:.1f}MiB of single-buffered blocks"
    )

  return (
    f"f32, {columns} columns, N inputs summed into one output, block rows "
    "laddered (not bisected -- this probe wants the trend, not the exact "
    "boundary; `vmem_scratch_bisect` owns the boundary). "
    + " | ".join(f"N={n}: {value}" for n, value in findings.items())
    + " || the implied MiB figures should be roughly constant across N if "
    "VMEM is the binding constraint"
  )


@harness.probe(
  _GROUP,
  "fusion_enables_dma_compute_overlap",
  "Does fusing the producer in let a hand-issued accumulator DMA hide behind compute?",
)
def _fusion_enables_dma_compute_overlap() -> str:
  """The pipelining argument for fusion, isolated from the traffic argument.

  Every other fusion probe here prices fusion as *avoided HBM traffic*. This
  one prices something `BlockSpec` cannot express at all: an accumulator that
  lives in HBM, is read-modified-written once per grid step, and has its read
  DMA issued *before* the compute so the round trip hides behind it. That is
  the structure of tokamax's cross-entropy backward kernel, which starts the
  `w_grad` read before computing the softmax
  [tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py:945-951].

  Three arms, and the two ratios answer different questions:

  * ``late`` and ``early`` are the *same kernel* with the *same* traffic, and
    differ only in where ``read.start()`` sits relative to the compute. Their
    ratio is the overlap win alone, and it is the number this probe exists
    for. It is only available because the compute is inside the kernel.
  * ``split`` is the honest unfused alternative -- compute to an HBM
    intermediate, reduce it separately. Its ratio against ``early`` is the
    total, and it mixes the overlap win with the intermediate's traffic.

  ``late == early`` is a real possible answer, and would mean Mosaic already
  hoists the DMA issue. It is reported, not suppressed.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  outer_steps = 8
  inner_tiles = 4
  block_rows = 512
  columns = 1024
  total_blocks = outer_steps * inner_tiles
  tile_bytes = block_rows * columns * 4

  source = jnp.full(
    (total_blocks * block_rows, columns), 0.5, dtype=jnp.float32
  )
  accumulator_shape = jax.ShapeDtypeStruct(
    (inner_tiles * block_rows, columns), jnp.float32
  )

  def chain(value, length):
    for _ in range(length):
      value = jnp.tanh(value) * 1.5 + 0.25
    return value

  def build_fused(chain_length: int, *, early: bool, read: bool = True):
    """One kernel: managed input pipeline, hand-rolled HBM accumulator.

    ``read=False`` drops the accumulator read entirely and overwrites the
    tile instead. That arm computes the **wrong answer on purpose** and is
    never correctness-checked -- it exists only as a cost floor, to say
    whether the read was ever on the critical path. Without it, an
    ``early == late`` result cannot be told apart from a read that was free
    in both arms.
    """

    def kernel(x_ref, accumulator_ref, tile_ref, read_sem, write_sem) -> None:
      outer = pl.program_id(0)
      inner = pl.program_id(1)
      # The accumulator tile is chosen by the *inner* axis and the
      # reduction runs over the *outer* one, so between two visits to the
      # same tile the VMEM scratch has been used by every other tile.
      # That is what makes the read back from HBM necessary rather than
      # decorative -- with a single tile the scratch would just persist.
      accumulator_slice = accumulator_ref.at[
        pl.ds(inner * block_rows, block_rows), :
      ]
      read_copy = pltpu.make_async_copy(accumulator_slice, tile_ref, read_sem)
      write = pltpu.make_async_copy(tile_ref, accumulator_slice, write_sem)

      if read and early:

        @pl.when(outer != 0)
        def _start_read_early() -> None:
          read_copy.start()

      value = chain(x_ref[...], chain_length)

      if read and not early:

        @pl.when(outer != 0)
        def _start_read_late() -> None:
          read_copy.start()

      if read:

        @pl.when(outer == 0)
        def _first_visit() -> None:
          tile_ref[...] = value

        @pl.when(outer != 0)
        def _later_visit() -> None:
          read_copy.wait()
          tile_ref[...] = tile_ref[...] + value
      else:
        tile_ref[...] = value

      write.start()
      write.wait()

    return jax.jit(
      pl.pallas_call(
        kernel,
        grid=(outer_steps, inner_tiles),
        in_specs=[
          pl.BlockSpec(
            (block_rows, columns),
            lambda o, i: (o * inner_tiles + i, 0),
          )
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        out_shape=accumulator_shape,
        scratch_shapes=[
          pltpu.VMEM((block_rows, columns), jnp.float32),
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary")
        ),
      )
    )

  def build_split(chain_length: int):
    """Two kernels: the producer materializes, a second pass reduces."""

    def producer(x_ref, output_ref) -> None:
      output_ref[...] = chain(x_ref[...], chain_length)

    producer_call = pl.pallas_call(
      producer,
      grid=(total_blocks,),
      in_specs=[pl.BlockSpec((block_rows, columns), lambda i: (i, 0))],
      out_specs=pl.BlockSpec((block_rows, columns), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
    )

    def whole(value):
      staged = producer_call(value)
      staged = staged.reshape(outer_steps, inner_tiles * block_rows, columns)
      return jnp.sum(staged, axis=0)

    return jax.jit(whole)

  findings = {}
  for chain_length in (1, 4, 16):
    early_call = build_fused(chain_length, early=True)
    late_call = build_fused(chain_length, early=False)
    noread_call = build_fused(chain_length, early=True, read=False)
    split_call = build_split(chain_length)

    early_result = early_call(source)
    late_result = late_call(source)
    split_result = split_call(source)
    scale = float(jnp.max(jnp.abs(split_result))) + 1e-30
    early_error = float(jnp.max(jnp.abs(early_result - split_result))) / scale
    late_error = float(jnp.max(jnp.abs(late_result - split_result))) / scale
    if max(early_error, late_error) > 1e-5:
      raise harness.Invalid(
        f"chain={chain_length}: the fused arms disagree with the split "
        f"arm (early={early_error:.3e} late={late_error:.3e} relative). "
        "A kernel whose DMA ordering is wrong accumulates stale tiles "
        "and is fast for exactly that reason, so no timing is quotable."
      )

    early_timing = harness.time_call(
      lambda call=early_call: call(source), warmups=3, repeats=10, batch=8
    )
    late_timing = harness.time_call(
      lambda call=late_call: call(source), warmups=3, repeats=10, batch=8
    )
    noread_timing = harness.time_call(
      lambda call=noread_call: call(source), warmups=3, repeats=10, batch=8
    )
    split_timing = harness.time_call(
      lambda call=split_call: call(source), warmups=3, repeats=10, batch=8
    )
    harness.require_above_dispatch_floor(
      early_timing, what=f"the chain={chain_length} early arm"
    )
    findings[chain_length] = (
      f"early={early_timing.median * 1e6:.0f}us "
      f"(spread {early_timing.spread_pct:.0f}%) "
      f"late={late_timing.median * 1e6:.0f}us "
      f"noread={noread_timing.median * 1e6:.0f}us "
      f"split={split_timing.median * 1e6:.0f}us | "
      f"late/early={late_timing.median / early_timing.median:.2f}x "
      f"early/noread={early_timing.median / noread_timing.median:.2f}x "
      f"split/early={split_timing.median / early_timing.median:.2f}x"
    )

  return (
    f"f32, {outer_steps} outer x {inner_tiles} inner grid, "
    f"{block_rows}x{columns} tiles ({tile_bytes / 2**20:.0f} MiB each, "
    f"read+written once per step). "
    + " | ".join(f"chain={n}: {v}" for n, v in findings.items())
    + " || late/early is the overlap win on its own -- same kernel, same "
    "bytes, only the DMA issue point moves. early/noread says whether the "
    "read was ever on the critical path (noread drops it and is "
    "deliberately WRONG, a cost floor only): late/early=1.00 with "
    "early/noread>1 means Mosaic hoists the start; late/early=1.00 with "
    "early/noread=1.00 means the read was free in both arms and this "
    "probe has said nothing about hoisting. split/early is the total "
    "against not fusing at all, and also carries the intermediate traffic"
  )


@harness.probe(
  _GROUP,
  "gmm_tgmm_grid_axis_conflict",
  "What does fusing a matmul and its transposed-lhs twin into one grid cost?",
)
def _gmm_tgmm_grid_axis_conflict() -> str:
  """Why `gmm` and `tgmm` are separate kernel families, priced.

  tokamax keeps them apart and says only that a consequence follows
  (`fuse_act has to be None in tgmm`), never why. The structural reason is
  testable without reading another line of tokamax: the two contract over
  *different* axes.

      gmm : out[m, n] = sum_k lhs[m, k] * rhs[k, n]
      tgmm: dw [k, n] = sum_m lhs[m, k] * dout[m, n]

  One grid has one innermost axis. Make it `k` and `gmm` accumulates in a
  single (bm, bn) tile while `tgmm` is reducing over `m`, the *outermost*
  axis -- so its accumulator cannot be a tile, it has to be the whole
  (K, N) result, resident for the entire grid. Make it `m` and the roles
  swap. There is no ordering that serves both.

  Two arms: the split that tokamax ships, and the fused kernel that pays
  O(K*N) of VMEM instead of O(bk*bn). Swept over size, because the
  interesting part is not the time at one shape -- it is the shape at which
  the fused arm stops compiling and the split arm does not.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  block = 512
  findings = {}
  timed_any = False
  compile_boundary_found = False

  for size in (1024, 2048, 4096):
    rows = contraction = columns = size
    lhs = jnp.full((rows, contraction), 0.01, dtype=jnp.bfloat16)
    rhs = jnp.full((contraction, columns), 0.02, dtype=jnp.bfloat16)
    dout = jnp.full((rows, columns), 0.03, dtype=jnp.bfloat16)
    grid_m, grid_n, grid_k = (
      rows // block,
      columns // block,
      contraction // block,
    )

    # --- split arm: each contraction gets to be innermost -------------
    def gmm_kernel(lhs_ref, rhs_ref, out_ref, acc_ref) -> None:
      step = pl.program_id(2)

      @pl.when(step == 0)
      def _init() -> None:
        acc_ref[...] = jnp.zeros_like(acc_ref)

      acc_ref[...] += jnp.dot(
        lhs_ref[...], rhs_ref[...], preferred_element_type=jnp.float32
      )

      @pl.when(step == pl.num_programs(2) - 1)
      def _store() -> None:
        out_ref[...] = acc_ref[...]

    gmm_call = pl.pallas_call(
      gmm_kernel,
      grid=(grid_m, grid_n, grid_k),  # k innermost
      in_specs=[
        pl.BlockSpec((block, block), lambda m, n, k: (m, k)),
        pl.BlockSpec((block, block), lambda m, n, k: (k, n)),
      ],
      out_specs=pl.BlockSpec((block, block), lambda m, n, k: (m, n)),
      out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
      scratch_shapes=[pltpu.VMEM((block, block), jnp.float32)],
    )

    def tgmm_kernel(lhs_ref, dout_ref, dw_ref, acc_ref) -> None:
      step = pl.program_id(2)

      @pl.when(step == 0)
      def _init() -> None:
        acc_ref[...] = jnp.zeros_like(acc_ref)

      acc_ref[...] += jax.lax.dot_general(
        lhs_ref[...],
        dout_ref[...],
        (((0,), (0,)), ((), ())),  # contract the m axis
        preferred_element_type=jnp.float32,
      )

      @pl.when(step == pl.num_programs(2) - 1)
      def _store() -> None:
        dw_ref[...] = acc_ref[...]

    tgmm_call = pl.pallas_call(
      tgmm_kernel,
      grid=(grid_k, grid_n, grid_m),  # m innermost
      in_specs=[
        pl.BlockSpec((block, block), lambda k, n, m: (m, k)),
        pl.BlockSpec((block, block), lambda k, n, m: (m, n)),
      ],
      out_specs=pl.BlockSpec((block, block), lambda k, n, m: (k, n)),
      out_shape=jax.ShapeDtypeStruct((contraction, columns), jnp.float32),
      scratch_shapes=[pltpu.VMEM((block, block), jnp.float32)],
    )
    split_call = jax.jit(lambda a, b, d: (gmm_call(a, b), tgmm_call(a, d)))

    # --- fused arm: one grid, so dw must stay resident ----------------
    def fused_kernel(
      lhs_ref, rhs_ref, dout_ref, out_ref, dw_ref, acc_ref
    ) -> None:
      m_index, n_index, k_index = (pl.program_id(i) for i in range(3))

      @pl.when(k_index == 0)
      def _init_out() -> None:
        acc_ref[...] = jnp.zeros_like(acc_ref)

      acc_ref[...] += jnp.dot(
        lhs_ref[...], rhs_ref[...], preferred_element_type=jnp.float32
      )

      @pl.when(k_index == pl.num_programs(2) - 1)
      def _store_out() -> None:
        out_ref[...] = acc_ref[...]

      # The tgmm half. Its reduction axis is m, the outermost, so the
      # write target is a slice of a full-size resident block rather
      # than a tile the pipeline could stage.
      partial = jax.lax.dot_general(
        lhs_ref[...],
        dout_ref[...],
        (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
      )
      row = pl.ds(k_index * block, block)
      column = pl.ds(n_index * block, block)

      @pl.when(m_index == 0)
      def _init_dw() -> None:
        dw_ref[row, column] = partial

      @pl.when(m_index != 0)
      def _accumulate_dw() -> None:
        dw_ref[row, column] += partial

    fused_call = jax.jit(
      pl.pallas_call(
        fused_kernel,
        grid=(grid_m, grid_n, grid_k),
        in_specs=[
          pl.BlockSpec((block, block), lambda m, n, k: (m, k)),
          pl.BlockSpec((block, block), lambda m, n, k: (k, n)),
          pl.BlockSpec((block, block), lambda m, n, k: (m, n)),
        ],
        out_specs=[
          pl.BlockSpec((block, block), lambda m, n, k: (m, n)),
          # The whole of dw as one block, revisited at every grid
          # point. This is the cost the split avoids.
          pl.BlockSpec((contraction, columns), lambda m, n, k: (0, 0)),
        ],
        out_shape=[
          jax.ShapeDtypeStruct((rows, columns), jnp.float32),
          jax.ShapeDtypeStruct((contraction, columns), jnp.float32),
        ],
        scratch_shapes=[pltpu.VMEM((block, block), jnp.float32)],
      )
    )

    resident_mib = contraction * columns * 4 / 2**20
    tile_mib = block * block * 4 / 2**20

    reference_out = jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
    reference_dw = jax.lax.dot_general(
      lhs, dout, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )

    def relative_error(got, want):
      return float(jnp.max(jnp.abs(got - want))) / (
        float(jnp.max(jnp.abs(want))) + 1e-30
      )

    try:
      split_out, split_dw = split_call(lhs, rhs, dout)
      split_error = max(
        relative_error(split_out, reference_out),
        relative_error(split_dw, reference_dw),
      )
    except Exception as error:  # noqa: BLE001 — the boundary is the result.
      findings[size] = (
        f"dw_resident={resident_mib:.0f}MiB vs tile={tile_mib:.0f}MiB | "
        f"split FAILED: {type(error).__name__}"
      )
      continue

    try:
      fused_out, fused_dw = fused_call(lhs, rhs, dout)
      fused_error = max(
        relative_error(fused_out, reference_out),
        relative_error(fused_dw, reference_dw),
      )
    except Exception as error:  # noqa: BLE001 — the boundary is the result.
      message = str(error).split("\n")[0][:120]
      compile_boundary_found = True
      findings[size] = (
        f"dw_resident={resident_mib:.0f}MiB vs tile={tile_mib:.0f}MiB | "
        f"split ok (rel_err={split_error:.1e}) | "
        f"fused DID NOT COMPILE: {type(error).__name__}: {message}"
      )
      continue

    if max(split_error, fused_error) > 1e-2:
      findings[size] = (
        f"dw_resident={resident_mib:.0f}MiB | arms disagree with XLA "
        f"(split={split_error:.1e} fused={fused_error:.1e}); no timing"
      )
      continue

    split_timing = harness.time_call(
      lambda: split_call(lhs, rhs, dout), warmups=3, repeats=10, batch=16
    )
    fused_timing = harness.time_call(
      lambda: fused_call(lhs, rhs, dout), warmups=3, repeats=10, batch=16
    )
    # Per size, not for the probe. The smallest size is the one most likely
    # to sit under the floor, and it is also the one this probe cares least
    # about -- letting its `Invalid` escape would abandon the larger sizes,
    # which are the whole point. Withhold that size's number and continue.
    try:
      harness.require_above_dispatch_floor(
        split_timing, what=f"the size={size} split arm"
      )
    except harness.Invalid as under_floor:
      findings[size] = (
        f"dw_resident={resident_mib:.0f}MiB vs tile={tile_mib:.0f}MiB | "
        f"timing withheld: {under_floor}"
      )
      continue
    timed_any = True
    findings[size] = (
      f"dw_resident={resident_mib:.0f}MiB vs tile={tile_mib:.0f}MiB | "
      f"split={split_timing.median * 1e6:.0f}us "
      f"fused={fused_timing.median * 1e6:.0f}us "
      f"fused/split={fused_timing.median / split_timing.median:.2f}x | "
      f"rel_err split={split_error:.1e} fused={fused_error:.1e}"
    )

  if not timed_any and not compile_boundary_found:
    raise harness.Invalid(
      "no size produced either a timing above the dispatch floor or a "
      "compile boundary, so this probe established nothing: "
      + " | ".join(f"size={n}: {v}" for n, v in findings.items())
    )

  return (
    f"bf16 MxKxN square, f32 accumulate, {block}x{block} blocks. "
    "gmm contracts k, tgmm contracts m; the split arm gives each its own "
    "innermost axis, the fused arm shares one grid and holds all of dw "
    "resident. "
    + " | ".join(f"size={n}: {v}" for n, v in findings.items())
    + " || the fused arm failing to compile at a size the split arm "
    "handles is the answer: the split is not a style choice, it is what "
    "the VMEM budget leaves"
  )
