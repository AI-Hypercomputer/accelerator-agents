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
"""How much fits, what a grid step costs, and what the traversal order buys.

Supplies answers for `reference/design/partitioning.md`. The block size is
squeezed between capacity from one side and per-invocation overhead from the
other, and both sides are measured here.

Two habits this module holds to:

* **Bisect, do not ladder.** A geometric ladder brackets a boundary; quoting
  its last passing rung as "the limit" can be off by a whole ladder step.
  Every limit probe here ladders to find a bracket and then bisects inside it.
* **Whatever you allocate, you must also read — at a different offset than you
  wrote.** A scratch that is written and never read is dead and never gets
  allocated, so the probe measures nothing and reports a suspiciously round
  number. Reading back the same offset you just wrote is no better: store-to-
  load forwarding removes the read.
"""

from __future__ import annotations

from collections.abc import Callable

import harness

_GROUP = "partitioning"


def _largest_that_works(
  attempt: Callable[[int], None],
  ladder: tuple[int, ...],
) -> tuple[int | None, int | None, str]:
  """Ladders up to bracket a boundary, then bisects to find it exactly.

  Args:
    attempt: Called with a size; must raise if that size does not work.
    ladder: Increasing sizes to try, coarsely.

  Returns:
    ``(largest_ok, smallest_failing, error_text)``. ``largest_ok`` is None if
    even the first rung failed; ``smallest_failing`` is None if the whole
    ladder passed, which means the boundary was never bracketed and no limit
    should be quoted.
  """
  largest_ok: int | None = None
  smallest_failing: int | None = None
  error_text = ""

  for size in ladder:
    try:
      attempt(size)
    except Exception as error:  # noqa: BLE001 — the failure is the measurement.
      smallest_failing = size
      error_text = f"{type(error).__name__}: {str(error).strip()[:200]}"
      break
    largest_ok = size

  if smallest_failing is None:
    return largest_ok, None, "ladder never failed: no boundary was bracketed"

  low = largest_ok if largest_ok is not None else 0
  high = smallest_failing
  while low + 1 < high:
    middle = (low + high) // 2
    try:
      attempt(middle)
    except Exception:  # noqa: BLE001
      high = middle
    else:
      low = middle
  return (low if low > 0 else None), high, error_text


@harness.probe(
  _GROUP,
  "vmem_scratch_bisect",
  "How large a VMEM scratch can a kernel actually allocate and use?",
)
def _vmem_scratch_bisect() -> str:
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  def attempt_mib(mebibytes: int) -> None:
    rows = (mebibytes * 1024 * 1024) // (128 * 4)

    def kernel(output_ref, scratch_ref) -> None:
      # Write at one traced offset and read at a different one, so
      # neither dead-code elimination nor store-to-load forwarding can
      # make the allocation disappear.
      step = pl.program_id(0)
      scratch_ref[pl.ds(step, 8), :] = jnp.full((8, 128), 1.0, jnp.float32)
      output_ref[...] = scratch_ref[pl.ds(step + 8, 8), :]

    result = pl.pallas_call(
      kernel,
      grid=(2,),
      out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
      scratch_shapes=[pltpu.VMEM((rows, 128), jnp.float32)],
    )()
    jax.block_until_ready(result)

  largest_ok, smallest_failing, error_text = _largest_that_works(
    attempt_mib, (1, 4, 8, 16, 24, 32, 48, 64, 96, 128)
  )
  if largest_ok is None:
    raise harness.Invalid(f"even the smallest rung failed: {error_text}")
  return (
    f"f32 VMEM scratch, bisected: largest_ok={largest_ok}MiB "
    f"first_fail={smallest_failing}MiB || {error_text}"
  )


@harness.probe(
  _GROUP,
  "vmem_bytes_per_element_by_dtype",
  "Does a narrower dtype buy proportionally more VMEM, or is there a floor?",
)
def _vmem_bytes_per_element_by_dtype() -> str:
  """Answers whether an int8 block is really 4x cheaper than an f32 one.

  Reported as element counts as well as bytes, because the question a block
  size actually asks is "how many elements fit", and a `max(itemsize, 2)`
  byte model and a true `itemsize` model differ exactly there.

  Two parts of this probe should not be simplified away.

  The **two independent SMEM offsets** keep the scratch alive. Deriving the
  write and read offsets from the same `program_id` (e.g. ``pl.ds(step, 8)``
  and ``pl.ds(step + 8, 8)``) can be enough for an f32 scratch but not for
  narrower dtypes, where the compiler may prove the store dead and the ladder
  then climbs far past physical VMEM without a rejection. Here the write
  index and the read index arrive as separate SMEM values and the caller
  passes ``(units - 1, 0)``, so the compiler cannot prove they coincide, the
  store stays live, and the buffer reaches the scoped stack allocator.

  The **capacity guard** is the other half. That failure mode does not look
  like a failure — it looks like a large, self-consistent answer. A ceiling
  at or above physical `vmem_capacity_bytes` means the *scoped* allocator
  never ran and what was measured is a static shape check, so the number is
  withheld rather than footnoted.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  # One unit is 32x128, a whole sublane tile for every dtype tested: 32 is
  # the tiling for int8 and a multiple of bf16's 16 and f32's 8. Sizing the
  # unit this way keeps the arms comparable in tiles rather than in rows.
  rows_per_unit = 32

  def attempt_for(dtype_name: str) -> Callable[[int], None]:
    dtype = jnp.dtype(getattr(jnp, dtype_name))

    def attempt(units: int) -> None:
      def kernel(offsets_ref, output_ref, scratch_ref) -> None:
        write_at = offsets_ref[0]
        read_at = offsets_ref[1]
        scratch_ref[write_at] = jnp.full((rows_per_unit, 128), 1, dtype)
        output_ref[...] = scratch_ref[read_at]

      result = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_shape=jax.ShapeDtypeStruct((rows_per_unit, 128), dtype),
        scratch_shapes=[pltpu.VMEM((units, rows_per_unit, 128), dtype)],
      )(jnp.array([units - 1, 0], jnp.int32))
      jax.block_until_ready(result)

    return attempt

  # The ladder has to reach past the boundary for the *narrowest* dtype, not
  # the widest: if a narrow dtype really does buy proportionally more VMEM,
  # its boundary sits at 2x (bf16) or 4x (int8) the f32 one.
  ladder = (16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768)
  capacity = pltpu.get_tpu_info().vmem_capacity_bytes

  findings = {}
  suspect = []
  for dtype_name in ("float32", "bfloat16", "int8"):
    if not hasattr(jnp, dtype_name):
      findings[dtype_name] = "dtype-absent"
      continue
    itemsize = jnp.dtype(getattr(jnp, dtype_name)).itemsize
    try:
      largest_ok, smallest_failing, _ = _largest_that_works(
        attempt_for(dtype_name), ladder
      )
    except Exception as error:  # noqa: BLE001
      findings[dtype_name] = f"NO({type(error).__name__})"
      continue
    if smallest_failing is None or largest_ok is None:
      findings[dtype_name] = (
        f"no boundary bracketed up to {ladder[-1]} units "
        f"({ladder[-1] * rows_per_unit * 128 * itemsize / 2**20:.0f}MiB "
        "requested)"
      )
      suspect.append(dtype_name)
      continue
    elements = largest_ok * rows_per_unit * 128
    byte_ceiling = elements * itemsize
    findings[dtype_name] = (
      f"{largest_ok} units (first_fail={smallest_failing}), "
      f"{elements} elements, {byte_ceiling} bytes "
      f"= {byte_ceiling / 2**20:.3f}MiB at {itemsize}B/element"
    )
    if byte_ceiling >= capacity:
      suspect.append(dtype_name)

  if suspect:
    raise harness.Invalid(
      f"the ceiling reported for {suspect} reaches or exceeds physical "
      f"vmem_capacity_bytes={capacity}, which means the scoped allocator "
      "never ran and this is a static shape check wearing a capacity's "
      "clothes. Every figure is withheld -- including the dtype ratio, "
      "which is self-consistent precisely because all three arms hit the "
      f"same static bound. Arms as measured: {findings}"
    )

  return (
    f"largest usable VMEM scratch of shape (units, {rows_per_unit}, 128), "
    "written and read at two independent SMEM offsets, bisected per dtype: "
    + " || ".join(f"{name}={value}" for name, value in findings.items())
    + f" || guard: every arm is below physical vmem_capacity_bytes="
    f"{capacity}. Read the ratio, not the absolute: f32:bf16:int8 element "
    "counts of 1:2:4 mean bytes-per-element tracks itemsize, while 1:2:2 "
    "means a 2-byte floor"
  )


@harness.probe(
  _GROUP,
  "smem_scratch_bisect",
  "How many int32 scalars fit in an SMEM scratch?",
)
def _smem_scratch_bisect() -> str:
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  def attempt(elements: int) -> None:
    def kernel(output_ref, scratch_ref) -> None:
      step = pl.program_id(0)
      scratch_ref[step] = step + 1
      output_ref[0] = scratch_ref[step + 1]

    result = pl.pallas_call(
      kernel,
      grid=(2,),
      out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
      out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
      scratch_shapes=[pltpu.SMEM((elements,), jnp.int32)],
    )()
    jax.block_until_ready(result)

  largest_ok, smallest_failing, error_text = _largest_that_works(
    attempt, (1024, 4096, 16384, 65536, 262144, 1048576)
  )
  if largest_ok is None:
    raise harness.Invalid(f"even the smallest rung failed: {error_text}")
  return (
    f"int32 SMEM scratch, bisected: largest_ok={largest_ok} elements "
    f"({largest_ok * 4 / 1024:.1f} KiB) first_fail={smallest_failing} "
    f"|| {error_text}"
  )


@harness.probe(
  _GROUP,
  "dispatch_floor",
  "What does one pallas_call cost before it does any work, and what does one extra grid step cost?",
)
def _dispatch_floor() -> str:
  """The floor that makes small kernels not worth writing.

  Both halves matter to a block-size choice: the per-call cost says whether
  to fuse, and the per-grid-step cost says how small a block can get before
  the grid itself dominates.

  `harness.dispatch_floor_seconds` measures the per-process dispatch floor
  with the same method as the single-call arm here.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  def copy_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...]

  source = jnp.ones((8, 128), dtype=jnp.float32)
  single_call = jax.jit(
    pl.pallas_call(
      copy_kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )
  )
  single = harness.time_call(lambda: single_call(source), warmups=5, repeats=50)

  identity = jax.jit(lambda value: value + 1.0)
  xla_floor = harness.time_call(lambda: identity(source), warmups=5, repeats=50)

  def build_grid(steps: int):
    return jax.jit(
      pl.pallas_call(
        copy_kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
      )
    )

  # A ladder that re-reads one resident block does not price a grid step: no
  # DMA is issued and no address arithmetic is done per step, so it measures
  # the loop counter. The second ladder walks a distinct block each step,
  # which is what a real kernel does.
  def build_walking_grid(steps: int):
    return jax.jit(
      pl.pallas_call(
        copy_kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((steps * 8, 128), jnp.float32),
      )
    )

  grid_timings = {}
  walk_timings = {}
  for steps in (1, 16, 64, 256, 1024, 4096):
    call = build_grid(steps)
    grid_timings[steps] = harness.time_call(
      lambda call=call: call(source), warmups=3, repeats=20
    )
    walk_source = jnp.ones((steps * 8, 128), dtype=jnp.float32)
    walk_call = build_walking_grid(steps)
    walk_timings[steps] = harness.time_call(
      lambda call=walk_call, arg=walk_source: call(arg), warmups=3, repeats=20
    )

  def _ladder(timings) -> str:
    return " ".join(
      f"{steps}:{timing.median * 1e6:.1f}us"
      for steps, timing in timings.items()
    )

  resident_ns = (
    (grid_timings[4096].median - grid_timings[1].median) / (4096 - 1) * 1e9
  )
  walking_ns = (
    (walk_timings[4096].median - walk_timings[1].median) / (4096 - 1) * 1e9
  )
  return (
    f"one 8x128 f32 copy pallas_call: {single} || bare jitted XLA add for "
    f"comparison: {xla_floor} || grid-step ladder, SAME resident block every "
    f"step (issues no DMA, so this is a loop-counter floor and NOT the cost "
    f"of a grid step): {_ladder(grid_timings)} -> {resident_ns:.0f}ns/step || "
    f"grid-step ladder, DISTINCT block every step (real DMA and address "
    f"arithmetic; this is the number to use): {_ladder(walk_timings)} -> "
    f"{walking_ns:.0f}ns/step. Cross-check the second against "
    "`ragged_tail_cost` and the affine fit in reference/design/pipelining.md. "
    "These are host-observed round trips, so the single-call number is an upper "
    "bound on the device-side cost."
  )


@harness.probe(
  _GROUP,
  "matmul_block_size_sweep",
  "Which block shape wins for a fixed matmul, and how much is between best and worst?",
)
def _matmul_block_size_sweep() -> str:
  """The spread here is what a design pays for choosing a block size badly."""
  import itertools

  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows, contraction, columns = 4096, 4096, 4096

  lhs = jnp.ones((rows, contraction), dtype=jnp.bfloat16)
  rhs = jnp.ones((contraction, columns), dtype=jnp.bfloat16)

  def matmul_kernel(lhs_ref, rhs_ref, output_ref) -> None:
    step = pl.program_id(2)

    @pl.when(step == 0)
    def _initialize() -> None:
      output_ref[...] = jnp.zeros_like(output_ref)

    output_ref[...] += jnp.dot(
      lhs_ref[...], rhs_ref[...], preferred_element_type=jnp.float32
    )

  def build(block_rows: int, block_columns: int, block_contraction: int):
    return jax.jit(
      pl.pallas_call(
        matmul_kernel,
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
        ],
        out_specs=pl.BlockSpec(
          (block_rows, block_columns), lambda i, j, k: (i, j)
        ),
        out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
      )
    )

  total_flops = 2.0 * rows * contraction * columns
  findings = {}
  for block_rows, block_columns, block_contraction in itertools.product(
    (256, 512, 1024), (256, 512, 1024), (256, 512, 1024)
  ):
    # An f32 accumulator block plus two bf16 operand blocks, double
    # buffered, has to fit; skip the shapes that obviously cannot.
    working_bytes = 2 * (
      4 * block_rows * block_columns
      + 2 * block_rows * block_contraction
      + 2 * block_contraction * block_columns
    )
    if working_bytes > 48 * 1024 * 1024:
      continue
    label = f"{block_rows}x{block_columns}x{block_contraction}"
    try:
      call = build(block_rows, block_columns, block_contraction)
      timing = harness.time_call(
        lambda call=call: call(lhs, rhs), warmups=2, repeats=8
      )
    except Exception as error:  # noqa: BLE001 — a shape that will not compile is a finding.
      findings[label] = f"NO({type(error).__name__})"
      continue
    findings[label] = total_flops / timing.median / 1e12

  numeric = {
    label: rate for label, rate in findings.items() if isinstance(rate, float)
  }
  if not numeric:
    raise harness.Invalid("no block shape compiled; nothing to rank")

  ranked = sorted(numeric.items(), key=lambda item: -item[1])
  best_label, best_rate = ranked[0]
  worst_label, worst_rate = ranked[-1]
  rejected = [
    label for label, rate in findings.items() if not isinstance(rate, float)
  ]
  return (
    f"bf16 {rows}x{contraction}x{columns}, f32 accumulate, "
    f"grid (m, n, k) with k innermost. best={best_label} at {best_rate:.1f}Tops/s, "
    f"worst={worst_label} at {worst_rate:.1f}Tops/s, "
    f"best/worst={best_rate / worst_rate:.2f}x. "
    f"full ranking: {[(label, round(rate, 1)) for label, rate in ranked]} "
    f"|| would-not-compile: {rejected}"
  )


@harness.probe(
  _GROUP,
  "grid_traversal_order",
  "Does the order the grid is walked change wall time, holding the partition fixed?",
)
def _grid_traversal_order() -> str:
  """Same blocks, same work, different re-fetch counts from HBM."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = contraction = columns = 4096
  block = 512

  lhs = jnp.ones((rows, contraction), dtype=jnp.bfloat16)
  rhs = jnp.ones((contraction, columns), dtype=jnp.bfloat16)

  def build(accumulate_axis: int):
    """accumulate_axis is the grid position of k: 2 = innermost."""

    def matmul_kernel(lhs_ref, rhs_ref, output_ref) -> None:
      @pl.when(pl.program_id(accumulate_axis) == 0)
      def _initialize() -> None:
        output_ref[...] = jnp.zeros_like(output_ref)

      output_ref[...] += jnp.dot(
        lhs_ref[...], rhs_ref[...], preferred_element_type=jnp.float32
      )

    steps = rows // block
    if accumulate_axis == 2:  # grid (m, n, k)
      grid = (steps, steps, steps)
      lhs_map = lambda i, j, k: (i, k)
      rhs_map = lambda i, j, k: (k, j)
      out_map = lambda i, j, k: (i, j)
    elif accumulate_axis == 0:  # grid (k, m, n) — k outermost
      grid = (steps, steps, steps)
      lhs_map = lambda k, i, j: (i, k)
      rhs_map = lambda k, i, j: (k, j)
      out_map = lambda k, i, j: (i, j)
    else:  # grid (n, m, k) — the two parallel axes swapped
      grid = (steps, steps, steps)
      lhs_map = lambda j, i, k: (i, k)
      rhs_map = lambda j, i, k: (k, j)
      out_map = lambda j, i, k: (i, j)

    return jax.jit(
      pl.pallas_call(
        matmul_kernel,
        grid=grid,
        in_specs=[
          pl.BlockSpec((block, block), lhs_map),
          pl.BlockSpec((block, block), rhs_map),
        ],
        out_specs=pl.BlockSpec((block, block), out_map),
        out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
      )
    )

  total_flops = 2.0 * rows * contraction * columns
  findings = {}
  for label, accumulate_axis in (
    ("(m,n,k) k innermost", 2),
    ("(n,m,k) k innermost", 1),
    ("(k,m,n) k outermost", 0),
  ):
    try:
      call = build(accumulate_axis)
      timing = harness.time_call(
        lambda call=call: call(lhs, rhs), warmups=2, repeats=10
      )
    except Exception as error:  # noqa: BLE001
      findings[label] = f"NO({type(error).__name__}: {str(error)[:120]})"
      continue
    findings[label] = f"{total_flops / timing.median / 1e12:.1f}Tops/s"

  return (
    f"bf16 {rows}^3, {block}x{block} blocks, identical partition, only the "
    f"traversal order differs: "
    + " | ".join(f"{label}={value}" for label, value in findings.items())
    + ". The k-outermost arm re-reads and re-writes the whole f32 output "
    "every k step, so it is the arm that shows what traversal order costs."
  )


@harness.probe(
  _GROUP,
  "dimension_semantics_effect",
  "Does labelling grid axes `parallel` instead of `arbitrary` change anything measurable?",
)
def _dimension_semantics_effect() -> str:
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  rows = columns = 4096
  block = 512

  source = jnp.ones((rows, columns), dtype=jnp.float32)

  def scale_kernel(source_ref, output_ref) -> None:
    output_ref[...] = jnp.tanh(source_ref[...]) * 2.0

  params_class = getattr(pltpu, "CompilerParams", None) or getattr(
    pltpu, "TPUCompilerParams", None
  )
  if params_class is None:
    raise harness.Unsupported(
      "neither pltpu.CompilerParams nor TPUCompilerParams exists"
    )

  def build(semantics: tuple[str, str]):
    return jax.jit(
      pl.pallas_call(
        scale_kernel,
        grid=(rows // block, columns // block),
        in_specs=[pl.BlockSpec((block, block), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((block, block), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
        compiler_params=params_class(dimension_semantics=semantics),
      )
    )

  findings = {}
  for label, semantics in (
    ("arbitrary,arbitrary", ("arbitrary", "arbitrary")),
    ("parallel,parallel", ("parallel", "parallel")),
    ("parallel,arbitrary", ("parallel", "arbitrary")),
  ):
    try:
      # Build outside the timed region, and batch so the region clears
      # the host round trip; unbatched, a small elementwise kernel sits
      # near the dispatch floor and the arms cannot be told apart.
      call = build(semantics)
      timing = harness.time_above_floor(
        lambda call=call: call(source),
        what=f"the {label} arm",
        repeats=15,
        batch=32,
      )
    except Exception as error:  # noqa: BLE001
      findings[label] = f"NO({type(error).__name__}: {str(error)[:120]})"
      continue
    findings[label] = timing

  baseline = findings.get("arbitrary,arbitrary")
  if not isinstance(baseline, harness.Timing):
    raise harness.Invalid(f"the arbitrary baseline failed: {baseline}")
  harness.require_above_dispatch_floor(
    baseline, what="the all-arbitrary baseline"
  )

  parts = []
  for label, value in findings.items():
    if isinstance(value, harness.Timing):
      parts.append(f"{label}={value.median / baseline.median:.3f}x")
    else:
      parts.append(f"{label}={value}")
  return (
    f"f32 {rows}x{columns} elementwise tanh, {block}x{block} blocks, "
    f"time relative to all-arbitrary: {' '.join(parts)} || baseline "
    f"{baseline} || spread of the baseline bounds how large a difference "
    "this probe can actually resolve. Read the ratios against "
    f"is_megacore={pltpu.get_tpu_info().is_megacore} and "
    f"num_cores={pltpu.get_tpu_info().num_cores} from "
    "`tpu_info_reported_constants`: `parallel` is what lets Mosaic split "
    "the grid across cores, so on a chip presenting one core per device "
    "there is nothing for it to split and a null result is the expected "
    "one rather than evidence the annotation never matters."
  )


@harness.probe(
  _GROUP,
  "ragged_tail_cost",
  "What does a grid whose last block is partial cost, against one that divides evenly?",
)
def _ragged_tail_cost() -> str:
  """A tail block is padded to the full block, so it costs a whole step."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  block = 512
  # A single call at this size can sit near the host round trip, so
  # `harness.time_above_floor` below amortizes it by batching rather than by
  # growing the shape. Growing the contraction
  # and column extents to 4096 instead would not fit: a (512, 4096) bf16
  # block plus a (4096, 4096) bf16 block plus a (512, 4096) f32 output, all
  # double-buffered, exceeds a typical default scoped VMEM limit.
  contraction = columns = 2048

  def matmul_kernel(lhs_ref, rhs_ref, output_ref) -> None:
    output_ref[...] = jnp.dot(
      lhs_ref[...], rhs_ref[...], preferred_element_type=jnp.float32
    )

  def time_rows(rows: int) -> tuple[harness.Timing, int]:
    lhs = jnp.ones((rows, contraction), dtype=jnp.bfloat16)
    rhs = jnp.ones((contraction, columns), dtype=jnp.bfloat16)
    steps = -(-rows // block)
    call = jax.jit(
      pl.pallas_call(
        matmul_kernel,
        grid=(steps,),
        in_specs=[
          pl.BlockSpec((block, contraction), lambda i: (i, 0)),
          pl.BlockSpec((contraction, columns), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block, columns), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
      )
    )
    timing = harness.time_above_floor(
      lambda: call(lhs, rhs), what=f"the {rows}-row arm", repeats=15, batch=32
    )
    return timing, steps

  findings = {}
  even_arm: harness.Timing | None = None
  for rows in (2048, 2049, 2560, 3072):
    try:
      timing, steps = time_rows(rows)
    except Exception as error:  # noqa: BLE001
      findings[rows] = f"NO({type(error).__name__}: {str(error)[:120]})"
      continue
    if rows == 2048:
      even_arm = timing
    useful_flops = 2.0 * rows * contraction * columns
    findings[rows] = (
      f"{steps}steps {timing.median * 1e6:.0f}us "
      f"{useful_flops / timing.median / 1e12:.1f}Tops/s_useful "
      f"(spread {timing.spread_pct:.0f}%)"
    )

  if even_arm is None:
    raise harness.Invalid(
      f"the evenly-dividing 2048 arm failed: {findings[2048]}"
    )
  harness.require_above_dispatch_floor(
    even_arm, what="the evenly-dividing 2048 arm"
  )

  return (
    f"bf16, block rows={block}, K=N={contraction}. rows -> "
    + " | ".join(f"{rows}: {value}" for rows, value in findings.items())
    + f". 2048 divides evenly ({2048 // block} steps); 2049 adds one step "
    "for a single useful row, so the gap between those two arms is the "
    "cost of a ragged tail. 2560 is the even-dividing control that shows "
    "what a genuinely fifth step costs."
  )


@harness.probe(
  _GROUP,
  "pallas_call_intercept_chained_clock",
  "Is there a fixed per-pallas_call cost that survives batching, and is it device time?",
)
def _pallas_call_intercept_chained_clock() -> str:
  """Separates per-call *device* cost from per-call *host dispatch*.

  The dispatch floor (`harness.dispatch_floor_seconds`) is a host round
  trip, and `time_call`'s ``batch=``
  amortizes it away. The question here is whether anything is left underneath
  once it is gone.

  **The discriminating instrument is a chained clock.** ``M`` invocations are
  nested inside one ``jit``, each consuming the previous one's output, and
  the whole chain is timed and divided by ``M``. A ``tpu_custom_call`` is
  opaque to XLA, so the chain cannot be folded, reassociated or CSE'd — the
  device runs ``M`` kernels back to back with one dispatch between the host
  and all of them. Host cost therefore divides away by ``M``; a per-call
  *device* cost does not. An unchained arm at the same shape is measured
  beside it, and the gap between the two is the dispatch.

  Sweeping the grid length at fixed bytes per step and fitting
  ``T_per_call = intercept + slope * steps`` over the chained arm gives the
  intercept directly. Expressed in grid steps — ``intercept / slope`` — it
  says how much of a kernel has to exist before the intercept stops being
  the largest term in it. Below that count, fusing two ``pallas_call``s is
  worth more than any block-size tuning, because fusing deletes an intercept
  and re-blocking cannot.

  A roofline that predicts a per-call time below the measured intercept is
  predicting something the hardware will not produce.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  sublanes, lanes = 8, 4096  # 128 KiB per step read, 128 KiB written
  bytes_per_step = sublanes * lanes * 4 * 2
  grid_lengths = (8, 16, 32, 64, 128, 256)
  chain = 8

  def kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] * 1.5

  def one_call(steps: int):
    return pl.pallas_call(
      kernel,
      grid=(steps,),
      in_specs=[pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0))],
      out_specs=pl.BlockSpec((1, sublanes, lanes), lambda i: (i, 0, 0)),
      out_shape=jax.ShapeDtypeStruct((steps, sublanes, lanes), jnp.float32),
    )

  def build_chained(steps: int):
    call = one_call(steps)

    @jax.jit
    def chained(x):
      # Each call consumes the previous output. The custom call is opaque
      # to XLA, so this dependency cannot be optimized away.
      for _ in range(chain):
        x = call(x)
      return x

    return chained

  per_call: dict[int, harness.Timing] = {}
  chained_region: dict[int, harness.Timing] = {}
  unchained: dict[int, harness.Timing] = {}
  failures: list[str] = []

  for steps in grid_lengths:
    source = jnp.ones((steps, sublanes, lanes), jnp.float32)
    try:
      chained = build_chained(steps)
      region = harness.time_call(
        lambda chained=chained, source=source: chained(source),
        warmups=3,
        repeats=12,
        batch=1,
      )
      harness.require_above_dispatch_floor(
        region, what=f"the chained arm at {steps} steps"
      )
      chained_region[steps] = region
      per_call[steps] = harness.Timing(
        median=region.median / chain,
        minimum=region.minimum / chain,
        maximum=region.maximum / chain,
        repeats=region.repeats,
        batch=chain,
      )

      single = jax.jit(one_call(steps))
      unchained[steps] = harness.time_call(
        lambda single=single, source=source: single(source),
        warmups=3,
        repeats=12,
        batch=1,
      )
    except harness.Invalid:
      raise
    except Exception as error:  # noqa: BLE001 — a refusal is the finding.
      failures.append(
        f"{steps}: {type(error).__name__}: {str(error).strip()[:100]}"
      )

  if len(per_call) < 3:
    raise harness.Invalid(
      "fewer than three grid lengths produced a chained measurement, so "
      f"no intercept can be fitted. failures: {'; '.join(failures)}"
    )

  xs = [float(s) for s in sorted(per_call)]
  ys = [per_call[int(s)].median for s in xs]
  n = float(len(xs))
  mean_x = sum(xs) / n
  mean_y = sum(ys) / n
  denominator = sum((x - mean_x) ** 2 for x in xs)
  slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
  intercept = mean_y - slope * mean_x
  steps_of_intercept = intercept / slope if slope > 0 else float("nan")

  longest = max(per_call)
  dispatch_gap = unchained[longest].median - per_call[longest].median

  rows = " | ".join(
    f"{s}: chained {per_call[s].median * 1e6:.1f}us/call, "
    f"unchained {unchained[s].median * 1e6:.1f}us/call"
    for s in sorted(per_call)
  )
  return (
    f"f32, {bytes_per_step // 1024} KiB per grid step (r+w), grid swept "
    f"{grid_lengths}, chain={chain} calls inside one jit. "
    f"Fit over the chained column: intercept "
    f"**{intercept * 1e6:.1f}us per pallas_call**, slope "
    f"{slope * 1e9:.0f}ns per grid step, so the intercept is worth "
    f"**{steps_of_intercept:.0f} grid steps** at this block size. "
    f"At {longest} steps the unchained arm is {dispatch_gap * 1e6:.1f}us/call "
    f"slower than the chained one; that difference is host dispatch and it "
    f"is a separate cost, not the same one twice. {rows}"
    + (f" || failures: {'; '.join(failures)}" if failures else "")
  )
