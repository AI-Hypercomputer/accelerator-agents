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
"""Open questions in `reference/design/sparsecore.md`, one probe per question.

Same group (`sparsecore`) and same reference file as `checks/sparsecore.py`;
a separate module only because that one is already long. Each probe below
answers exactly one open question, named in its docstring.

Two conventions here that the older module does not have:

* **Every sweep prints its arm as it finishes**, with ``flush=True``, instead of
  accumulating a return string. A probe that hangs on arm 4 of 6 should say so.
  The suite's normal three-line report still happens at the end for the arms
  that completed.
* **Anything that can wedge the Mosaic core is ``destructive=True``** and so is
  excluded from a group run. A hang in ``subcore_barrier`` leaves an empty
  log; it must not take the other probes in the same process with it.
"""

from __future__ import annotations

import harness

from checks.sparsecore import _doubling_kernel, _sc, _vector_mesh

_GROUP = "sparsecore"


def _say(line: str) -> None:
  """Prints one arm's result the moment it exists.

  The suite writes to a file, not a tty, so stdout is block-buffered and a
  process that never exits flushes nothing at all, not even the suite
  banner. Run the suite under ``python -u``; this belt-and-braces flush
  covers the case where somebody forgets.
  """
  print(f"      . {line}", flush=True)


def _short(error: BaseException, limit: int = 180) -> str:
  return f"{type(error).__name__}: {str(error).strip().replace(chr(10), ' ')[:limit]}"


# --------------------------------------------------------------------------
# Does a wide Get lower with needs_layout_passes=True?
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_wide_get_layout_passes",
  "Is the (num_lanes,) shape rule a property of the SparseCore, or only of needs_layout_passes=False?",
)
def _sc_wide_get_layout_passes() -> str:
  """Tests whether the "shapes are (num_lanes,)" rule belongs to the flag.

  Hypothesis from reading the lowering: this is a property of the *flag*, not
  of the hardware. `_check_aval_is_supported` in sc_lowering.py is called from
  exactly two sites, `Get` and `Swap`, both under
  ``if not ctx.lowering_context.needs_layout_passes:``. So a wide `Get` ought
  to be refused with the flag off and accepted with it on.

  That is a prediction from source, and needs a run to confirm.
  The 2x2 below is the method: {lane-group, wide} x {flag off, flag on},
  each summing a buffer and checked against `np.sum`. The interesting cell is
  wide/on — if it lowers, the lane-group loop is a workaround for the flag
  rather than for the unit, and a kernel that does not need the flag may read
  a whole buffer.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  wide = 4096

  rng = np.random.default_rng(7)
  arms = []
  for width in (vec, wide):
    host = rng.integers(0, 4, size=width).astype(np.int32)
    x = jnp.asarray(host)
    want = int(host.sum())
    for layout_passes in (False, True):
      label = f"width={width},needs_layout_passes={layout_passes}"

      @pl.kernel(
        out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
        mesh=_vector_mesh(num_cores=1, num_subcores=1),
        scratch_types=[
          pltpu.VMEM((width,), jnp.int32),
          pltpu.VMEM((vec,), jnp.int32),
          pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(needs_layout_passes=layout_passes),
      )
      def kernel(x_ref, o_ref, buf, obuf, sem):
        pltpu.async_copy(x_ref, buf, sem).wait()
        # The whole point: one `Get` of the entire buffer. At
        # width=num_lanes this is a native shape; at width=4096 it is the
        # shape the flag-off check refuses.
        obuf[...] = jnp.full((vec,), jnp.sum(buf[...]), jnp.int32)
        pltpu.sync_copy(obuf, o_ref)

      try:
        got = int(np.asarray(jax.jit(kernel)(x))[0])
        arm = f"{label}:ok" if got == want else f"{label}:WRONG({got}!={want})"
      except Exception as error:  # noqa: BLE001 — the refusal is the finding.
        arm = f"{label}:NO({_short(error)})"
      _say(arm)
      arms.append(arm)

  return (
    f"one `jnp.sum(buf[...])` per cell, 1c x 1sc, num_lanes={vec}: "
    + " | ".join(arms)
  )


@harness.probe(
  _GROUP,
  "sc_wide_get_without_reduction",
  "With needs_layout_passes=True, does a wide Get lower when the kernel is a copy or an elementwise map rather than a reduction?",
)
def _sc_wide_get_without_reduction() -> str:
  """Isolates the wide read from the reduction in `sc_wide_get_layout_passes`.

  That probe uses `jnp.sum(buf[...])` as the wide read. `jnp.sum` lowers to
  `tpu.scan`, and if infer-vector-layout refuses `tpu.scan` at any width
  (including the native one), the flag-on cells fail at the *reduction*, not
  at the read, and never reach the question.

  This asks it with two bodies that contain no reduction at all: a straight
  copy, and an elementwise multiply. If either lowers at width=4096 with the
  flag on, then the lane-group loop is a workaround for `needs_layout_passes`
  and not for the vector unit, and a kernel that has no reduction in it may
  read a whole buffer at once.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  wide = 4096

  bodies = {
    # `o = x` -- a wide Get feeding a wide Swap, the minimum that exercises
    # both of the two call sites the shape check lives at.
    "copy": (lambda buf: buf[...], lambda h: h),
    # `o = 2*x` -- adds an arithmetic op that is elementwise, so unlike
    # tpu.scan it has an obvious per-lane layout.
    "mul2": (lambda buf: buf[...] * 2, lambda h: h * 2),
  }

  rng = np.random.default_rng(17)
  arms = []
  for name, (device_body, host_body) in bodies.items():
    for width in (vec, wide):
      host = rng.integers(0, 100, size=width).astype(np.int32)
      x = jnp.asarray(host)
      want = host_body(host)
      for layout_passes in (False, True):
        label = f"{name},width={width},nlp={layout_passes}"

        @pl.kernel(
          out_type=jax.ShapeDtypeStruct((width,), jnp.int32),
          mesh=_vector_mesh(num_cores=1, num_subcores=1),
          scratch_types=[
            pltpu.VMEM((width,), jnp.int32),
            pltpu.VMEM((width,), jnp.int32),
            pltpu.SemaphoreType.DMA,
          ],
          compiler_params=pltpu.CompilerParams(
            needs_layout_passes=layout_passes
          ),
        )
        def kernel(x_ref, o_ref, buf, obuf, sem, _b=device_body):
          pltpu.async_copy(x_ref, buf, sem).wait()
          obuf[...] = _b(buf)
          pltpu.sync_copy(obuf, o_ref)

        try:
          got = np.asarray(jax.jit(kernel)(x))
          arm = f"{label}:ok" if np.array_equal(got, want) else f"{label}:WRONG"
        except Exception as error:  # noqa: BLE001
          arm = f"{label}:NO({_short(error)})"
        _say(arm)
        arms.append(arm)

  return (
    f"no reduction in any cell, 1c x 1sc, num_lanes={vec}, wide={wide} "
    f"(nlp = needs_layout_passes): " + " | ".join(arms)
  )


# --------------------------------------------------------------------------
# Gather cost against index count at fixed row width
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_gather_index_count_scaling",
  "At a fixed row width, how does the SparseCore-versus-XLA gather comparison vary with the number of indices gathered?",
)
def _sc_gather_index_count_scaling() -> str:
  """Varies the index count at a fixed row width.

  `sc_gather_row_width_scaling` holds the index count at 2^18 and varies the
  row; this holds the row at 1 f32 and varies the index count over 256x. Two
  things it can find. A win that grows with index count is a fixed cost being
  amortized, so a ratio measured at one index count is a property of that
  count rather than of the gather. A win that *shrinks* means the table is
  falling out of subcore VMEM, or that XLA's own launch overhead was most of
  what the small arms measured.

  Every arm sizes `batch` by measuring, not by predicting -- see
  `harness.time_above_floor`. Neither arm's cost need be linear in the index
  count (a fixed cost on the SC side, a possible step on the XLA side), so no
  fixed batch is right across the sweep.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  workers = info.num_cores * info.num_subcores
  width, chunk = 1, 1024
  # 256 KiB of f32; must fit in SparseCoreInfo.vmem_capacity_bytes.
  table_rows = 1 << 16

  rng = np.random.default_rng(11)
  table = jnp.asarray(rng.standard_normal((table_rows, width)), jnp.float32)
  flat = jnp.asarray(np.asarray(table).reshape(-1).copy())

  arms = []
  floor_candidate = None
  for log2_count in (12, 14, 16, 18, 20):
    num_indices = 1 << log2_count
    num_chunks = num_indices // chunk
    indices = jnp.asarray(
      rng.integers(0, table_rows, size=num_indices), jnp.int32
    )

    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((width, num_indices), jnp.float32),
      mesh=_vector_mesh(
        num_cores=info.num_cores, num_subcores=info.num_subcores
      ),
      scratch_types=[
        pltpu.VMEM((table_rows * width,), jnp.float32),
        pltpu.VMEM((chunk,), jnp.int32),
        pltpu.VMEM((width, chunk), jnp.float32),
        pltpu.SemaphoreType.DMA,
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def sc_gather(table_ref, idx_ref, o_ref, tbl, ibuf, obuf, sem):
      pltpu.async_copy(table_ref, tbl, sem).wait()
      worker = jax.lax.axis_index(
        "core"
      ) * info.num_subcores + jax.lax.axis_index("subcore")

      @pl.loop(worker, num_chunks, step=workers)
      def _(c):
        pltpu.async_copy(idx_ref.at[pl.ds(c * chunk, chunk)], ibuf, sem).wait()

        # The table is flat and the row is 1 wide, so the index vector
        # *is* the address vector. See the width probe for why a 2-D
        # table does not lower.
        @plsc.parallel_loop(0, chunk, vec)
        def _(i):
          obuf[0, pl.ds(i, vec)] = plsc.load_gather(tbl, [ibuf[pl.ds(i, vec)]])

        pltpu.sync_copy(obuf, o_ref.at[:, pl.ds(c * chunk, chunk)])

    try:
      sc_fn = jax.jit(sc_gather)
      xla_fn = jax.jit(lambda t, i: t[i])
      got = np.asarray(sc_fn(flat, indices)).T
      want = np.asarray(xla_fn(table, indices))
      if not np.array_equal(got, want):
        arm = f"n=2^{log2_count}:WRONG"
      else:
        sc_timing = harness.time_above_floor(
          lambda: sc_fn(flat, indices),
          what=f"SC n=2^{log2_count}",
        )
        xla_timing = harness.time_above_floor(
          lambda: xla_fn(table, indices),
          what=f"XLA n=2^{log2_count}",
        )
        for timing, who in (
          (sc_timing, "SC"),
          (xla_timing, "XLA"),
        ):
          if (
            floor_candidate is None
            or timing.region_median < floor_candidate[1].region_median
          ):
            floor_candidate = (f"{who} n=2^{log2_count}", timing)
        arm = (
          f"n=2^{log2_count}(batch={sc_timing.batch}/{xla_timing.batch}):"
          f"{sc_timing.median * 1e6:.2f}us "
          f"{num_indices * 4 / sc_timing.median / 1e9:.0f}GB/s "
          f"(XLA/SC={xla_timing.median / sc_timing.median:.2f}x, "
          f"XLA {xla_timing.median * 1e6:.2f}us)"
        )
    except Exception as error:  # noqa: BLE001
      arm = f"n=2^{log2_count}:NO({_short(error)})"
    _say(arm)
    arms.append(arm)

  if all(":NO(" in arm for arm in arms):
    raise harness.Unsupported("no index count lowered: " + " | ".join(arms))
  if floor_candidate is not None:
    label, timing = floor_candidate
    harness.require_above_dispatch_floor(
      timing, what=f"the shortest timed region ({label})"
    )

  return (
    f"1-wide f32 rows from a {table_rows * 4 // 1024} KiB table in SC VMEM, "
    f"{info.num_cores}c x {info.num_subcores}sc, chunk={chunk} fixed, "
    f"index count swept 256x: " + " | ".join(arms)
  )


# --------------------------------------------------------------------------
# Gather width curve with chunk held constant
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_gather_width_curve_fixed_chunk",
  "Does the gather width curve keep its shape when chunk is held constant instead of shrinking with the row?",
)
def _sc_gather_width_curve_fixed_chunk() -> str:
  """Removes the `chunk` confound from the row-width curve.

  `sc_gather_row_width_scaling` sizes `chunk` to hold the output staging
  buffer at 128 KiB, so chunk falls 1024/1024/1024/512/128 as the row widens.
  The two widest arms therefore differ from the narrow ones in *two* ways,
  and the widest arms are where the confound is worst. This re-runs the same
  five widths at chunk=128 throughout, which is the value the widest arm
  already used, so every arm pays the same per-chunk DMA overhead.

  If the curve keeps its shape, the granule reading stands. If the narrow
  arms lose their win at chunk=128, part of that win was chunk size.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  workers = info.num_cores * info.num_subcores
  num_indices = 1 << 18
  chunk = 128  # fixed -- this is the whole point of the probe
  num_chunks = num_indices // chunk

  rng = np.random.default_rng(13)
  arms = []
  floor_candidate = None
  for width in (1, 4, 16, 64, 256):
    table_rows = (1 << 16) // width  # 256 KiB in every arm
    table = jnp.asarray(rng.standard_normal((table_rows, width)), jnp.float32)
    flat = jnp.asarray(np.asarray(table).reshape(-1).copy())
    indices = jnp.asarray(
      rng.integers(0, table_rows, size=num_indices), jnp.int32
    )

    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((width, num_indices), jnp.float32),
      mesh=_vector_mesh(
        num_cores=info.num_cores, num_subcores=info.num_subcores
      ),
      scratch_types=[
        pltpu.VMEM((table_rows * width,), jnp.float32),
        pltpu.VMEM((chunk,), jnp.int32),
        pltpu.VMEM((width, chunk), jnp.float32),
        pltpu.SemaphoreType.DMA,
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def sc_gather(table_ref, idx_ref, o_ref, tbl, ibuf, obuf, sem, _w=width):
      pltpu.async_copy(table_ref, tbl, sem).wait()
      worker = jax.lax.axis_index(
        "core"
      ) * info.num_subcores + jax.lax.axis_index("subcore")

      @pl.loop(worker, num_chunks, step=workers)
      def _(c):
        pltpu.async_copy(idx_ref.at[pl.ds(c * chunk, chunk)], ibuf, sem).wait()

        @plsc.parallel_loop(0, chunk, vec)
        def _(i):
          base = ibuf[pl.ds(i, vec)] * _w
          for col in range(_w):
            obuf[col, pl.ds(i, vec)] = plsc.load_gather(tbl, [base + col])

        pltpu.sync_copy(obuf, o_ref.at[:, pl.ds(c * chunk, chunk)])

    try:
      sc_fn = jax.jit(sc_gather)
      xla_fn = jax.jit(lambda t, i: t[i])
      got = np.asarray(sc_fn(flat, indices)).T
      want = np.asarray(xla_fn(table, indices))
      if not np.array_equal(got, want):
        arm = f"w={width}:WRONG"
      else:
        sc_timing = harness.time_above_floor(
          lambda: sc_fn(flat, indices), what=f"SC w={width}", batch=16
        )
        xla_timing = harness.time_above_floor(
          lambda: xla_fn(table, indices), what=f"XLA w={width}", batch=16
        )
        if (
          floor_candidate is None
          or sc_timing.region_median < floor_candidate[1].region_median
        ):
          floor_candidate = (f"w={width}", sc_timing)
        gathered = num_indices * width * 4
        arm = (
          f"w={width}:{gathered / sc_timing.median / 1e9:.0f}GB/s "
          f"{sc_timing.median * 1e6:.1f}us "
          f"(XLA/SC={xla_timing.median / sc_timing.median:.2f}x)"
        )
    except Exception as error:  # noqa: BLE001
      arm = f"w={width}:NO({_short(error)})"
    _say(arm)
    arms.append(arm)

  if all(":NO(" in arm for arm in arms):
    raise harness.Unsupported("no width lowered: " + " | ".join(arms))
  if floor_candidate is not None:
    label, timing = floor_candidate
    harness.require_above_dispatch_floor(
      timing, what=f"the shortest timed region ({label})"
    )

  return (
    f"{num_indices} gathers from a 256 KiB f32 table, "
    f"{info.num_cores}c x {info.num_subcores}sc, **chunk={chunk} in every "
    f"arm** (compare sc_gather_row_width_scaling, where chunk falls "
    f"1024/1024/1024/512/128): " + " | ".join(arms)
  )


# --------------------------------------------------------------------------
# SC/TC overlap against what the TC neighbour is bound by
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_tc_overlap_vs_neighbour_kind",
  "Does SparseCore work still hide behind the TensorCore when the TensorCore neighbour is memory-bound rather than compute-bound?",
)
def _sc_tc_overlap_vs_neighbour_kind() -> str:
  """Varies whether the TensorCore neighbour is compute- or memory-bound.

  An overlap score against a bf16 matmul is a score against a compute-bound
  neighbour that leaves HBM largely to the SparseCore. The offload
  argument inverts if the neighbour is memory-bound: SC streaming and a TC
  elementwise pass are then competing for the same bandwidth, and hiding one
  behind the other is not something the hardware can do however independent
  the programs are.

  Same score as the existing probe -- `(sc + tc - both) / min(sc, tc)`,
  1.0 = every microsecond the durations allowed to be hidden was hidden --
  computed against three neighbours in one run so the SC arm is identical
  across them.
  """
  import jax
  import jax.numpy as jnp

  n, chunk = 1 << 24, 2048  # 64 MiB in, 64 MiB out, the usual SC arm
  x = jnp.arange(n, dtype=jnp.float32)
  sc_kernel = _doubling_kernel(n, chunk, num_cores=2, num_subcores=16)
  sc_only = jax.jit(lambda x: sc_kernel(x))
  sc_timing = harness.time_call(lambda: sc_only(x), batch=4)
  harness.require_above_dispatch_floor(sc_timing, what="SC arm")
  _say(f"SC arm alone: {sc_timing}")

  m = 8192
  a = jnp.ones((m, m), jnp.bfloat16)
  b = jnp.ones((m, m), jnp.bfloat16)
  # 256 MiB read + 256 MiB written per pass -- an elementwise map with no
  # reuse, which is the canonical memory-bound TensorCore shape.
  big = jnp.arange(1 << 26, dtype=jnp.float32)

  # Every neighbour takes its operands as jit *arguments*. Closing over
  # `jnp.ones` and calling `jax.jit(lambda: a @ b)` invites XLA to constant
  # fold the whole thing at compile time, which reads as a TensorCore that
  # costs nothing and an overlap score of 1.0 for free.
  neighbours = {
    "compute-bound(bf16 8192^3 matmul)": (
      (a, b),
      jax.jit(lambda p, q: p @ q),
      jax.jit(lambda x, p, q: (sc_kernel(x), p @ q)),
    ),
    "memory-bound(512MiB elementwise)": (
      (big,),
      jax.jit(lambda v: v * 2.0 + 1.0),
      jax.jit(lambda x, v: (sc_kernel(x), v * 2.0 + 1.0)),
    ),
    # A reduction reads without writing much, so it loads HBM roughly half
    # as hard as the elementwise arm at a similar duration. If the score
    # tracks read+write volume rather than "is it memory bound", this arm
    # sits between the other two rather than beside one of them.
    "memory-bound(256MiB reduction)": (
      (big,),
      jax.jit(lambda v: v.sum()),
      jax.jit(lambda x, v: (sc_kernel(x), v.sum())),
    ),
  }

  arms = []
  for label, (operands, tc_only, both_fn) in neighbours.items():
    try:
      # The neighbour arm sets its own batch: the three neighbours differ
      # in cost, and a fixed batch that suits one leaves another under the
      # dispatch floor, where the harness withholds it.
      tc_timing = harness.time_above_floor(
        lambda _o=operands, _f=tc_only: _f(*_o), what=f"TC {label}"
      )
      both_timing = harness.time_above_floor(
        lambda _o=operands, _f=both_fn: _f(x, *_o),
        what=f"both {label}",
        batch=4,
      )
      harness.require_above_dispatch_floor(tc_timing, what=f"TC {label}")
      harness.require_above_dispatch_floor(both_timing, what=f"both {label}")
      sc, tc, both = (
        sc_timing.median,
        tc_timing.median,
        both_timing.median,
      )
      hidden = sc + tc - both
      score = hidden / min(sc, tc)
      arm = (
        f"{label}: score={score:.2f} "
        f"(sc={sc * 1e6:.1f}us tc={tc * 1e6:.1f}us "
        f"both={both * 1e6:.1f}us hidden={hidden * 1e6:.1f}us of "
        f"{min(sc, tc) * 1e6:.1f}us hideable; SC adds "
        f"{(both - tc) * 1e6:.1f}us to the step)"
      )
    except harness.Invalid as error:
      arm = f"{label}:INVALID({error})"
    except Exception as error:  # noqa: BLE001
      arm = f"{label}:NO({_short(error)})"
    _say(arm)
    arms.append(arm)

  return (
    f"SC 2*x over n={n} f32 on 2c x 16sc ({sc_timing.median * 1e6:.1f}us "
    f"alone) beside three TensorCore neighbours, score = "
    f"(sc+tc-both)/min(sc,tc): " + " | ".join(arms)
  )


# --------------------------------------------------------------------------
# Scatter-add fanned out across every vector subcore
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_scatter_add_fanout",
  "Does fanning a SparseCore scatter-add histogram across all vector subcores change how it compares with XLA's scatter-add?",
)
def _sc_scatter_add_fanout() -> str:
  """Fans the scatter-add histogram out across subcores, without a barrier.

  `sc_scatter_add_vs_xla` puts the whole histogram on one subcore. Whether
  its comparison with XLA is a statement about the SparseCore or about using
  one subcore of it is the open question, and `sc_subcore_scaling` cannot
  answer it because that measures work with no combine at all.

  Privatize instead of coordinating: every subcore keeps its own 4096-bucket
  histogram in its own VMEM (16 KiB, well inside
  `SparseCoreInfo.vmem_capacity_bytes`), and the cross-subcore fold is a
  dense sum done afterwards on the TensorCore. That is a real implementation
  strategy, not a dodge -- but it is not free, and the fold is *inside* the
  timed jit for exactly that reason. It also sidesteps `subcore_barrier`,
  which is the reason this probe can run in a shard with others.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  buckets, num_updates, chunk = 4096, 1 << 18, 2048
  num_chunks = num_updates // chunk

  rng = np.random.default_rng(1)  # same seed as the single-subcore probe
  indices = jnp.asarray(rng.integers(0, buckets, size=num_updates), jnp.int32)
  xla_fn = jax.jit(lambda idx: jnp.zeros((buckets,), jnp.int32).at[idx].add(1))
  want = np.asarray(xla_fn(indices))
  # 40 repeats, not 20: every SC/XLA ratio in the probe divides by this
  # arm, so its median is worth extra samples against a wide spread.
  xla_timing = harness.time_above_floor(
    lambda: xla_fn(indices), what="XLA scatter-add arm", batch=48, repeats=40
  )
  harness.require_above_dispatch_floor(xla_timing, what="XLA scatter-add arm")
  _say(f"XLA arm: {xla_timing}")

  arms = []
  baseline = None
  for num_cores, num_subcores in (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (2, 16),
  ):
    workers = num_cores * num_subcores
    label = f"{num_cores}c/{num_subcores}sc"

    # 1-D output, sliced by worker. `o_ref.at[worker]` on a 2-D ref is a
    # rank reduction and can hit a `tpu.memref_squeeze` refusal; folding the
    # row into the offset is the same layout without the squeeze.
    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((workers * buckets,), jnp.int32),
      mesh=_vector_mesh(num_cores=num_cores, num_subcores=num_subcores),
      scratch_types=[
        pltpu.VMEM((buckets,), jnp.int32),
        pltpu.VMEM((chunk,), jnp.int32),
        pltpu.SemaphoreType.DMA,
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def sc_hist(idx_ref, o_ref, hist, ibuf, sem, _ns=num_subcores, _w=workers):
      worker = jax.lax.axis_index("core") * _ns + jax.lax.axis_index("subcore")

      @plsc.parallel_loop(0, buckets, vec)
      def _(i):
        hist[pl.ds(i, vec)] = jnp.zeros(vec, jnp.int32)

      @pl.loop(worker, num_chunks, step=_w)
      def _(c):
        pltpu.async_copy(idx_ref.at[pl.ds(c * chunk, chunk)], ibuf, sem).wait()

        @plsc.parallel_loop(0, chunk, vec)
        def _(i):
          digits = ibuf[pl.ds(i, vec)]
          counts, mask = plsc.scan_count(digits)
          plsc.addupdate_scatter(hist, [digits], counts, mask=mask)

      pltpu.sync_copy(hist, o_ref.at[pl.ds(worker * buckets, buckets)])

    try:
      # The dense fold is inside the jit: privatizing the histogram moves
      # a cost, it does not remove one, and a number that excluded the
      # fold would be comparing a partial result against a complete one.
      fn = jax.jit(
        lambda idx, _k=sc_hist, _w=workers: (
          _k(idx).reshape(_w, buckets).sum(axis=0)
        )
      )
      got = np.asarray(fn(indices))
      if not np.array_equal(got, want):
        bad = int((got != want).sum())
        arm = f"{label}:WRONG({bad}/{buckets} buckets)"
      else:
        timing = harness.time_call(lambda: fn(indices), batch=4)
        if baseline is None:
          baseline = timing
        arm = (
          f"{label}:{timing.median * 1e6:.1f}us "
          f"{baseline.median / timing.median:.2f}x-vs-1sc "
          f"XLA/SC={xla_timing.median / timing.median:.2f}x"
        )
    except Exception as error:  # noqa: BLE001
      arm = f"{label}:NO({_short(error)})"
    _say(arm)
    arms.append(arm)

  if all(":NO(" in arm or ":WRONG(" in arm for arm in arms):
    raise harness.Unsupported(
      "no fanned-out histogram ran: " + " | ".join(arms)
    )

  return (
    f"{num_updates} int32 updates into {buckets} buckets, one private "
    f"histogram per subcore plus a dense TC fold inside the timed jit "
    f"(XLA/SC>1 means SC wins; compare sc_scatter_add_vs_xla): "
    f"XLA {xla_timing.median * 1e6:.1f}us || " + " | ".join(arms)
  )


# --------------------------------------------------------------------------
# Do the addupdate spellings share the store_compressed mask gap?
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_addupdate_variants",
  "Do `plsc.addupdate` and `plsc.addupdate_compressed` lower, and do they share any masking restriction with `store_compressed`?",
)
def _sc_addupdate_variants() -> str:
  """Tests whether the two `addupdate` spellings lower.

  Hypothesis: `store_compressed` and `load_expanded` may refuse a masked call
  with `Not implemented: Mask is not supported` (check with
  `sc_primitive_lowering`). The two `addupdate` spellings matter differently: `addupdate` takes no mask at all, so it
  should lower if anything does, and if it does it is a cheaper accumulate
  than `addupdate_scatter` for the case where the index vector is the
  identity. `addupdate_compressed` takes the same `mask=` keyword as
  `store_compressed`, so it is the direct test of whether the gap is in the
  mask machinery or in that one primitive.

  Both bind `swap_p` with `add=True` (sc_primitives.py), so a
  common refusal would point at the swap lowering rather than at either name.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  x_i32 = jnp.arange(vec, dtype=jnp.int32)

  def run(body, inputs, *, layout_passes):
    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
      mesh=_vector_mesh(num_cores=1, num_subcores=1),
      scratch_types=[
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.SemaphoreType.DMA,
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=layout_passes),
    )
    def kernel(x_ref, o_ref, buf, acc, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      body(buf, acc)
      pltpu.sync_copy(acc, o_ref)

    return np.asarray(jax.jit(kernel)(inputs))

  def _addupdate(buf, acc):
    acc[...] = jnp.full((vec,), 100, jnp.int32)
    # No mask, no indices: a straight atomic add of a whole lane group.
    plsc.addupdate(acc.at[pl.ds(0, vec)], buf[...])

  def _addupdate_compressed(buf, acc):
    acc[...] = jnp.zeros((vec,), jnp.int32)
    v = buf[...]
    # Same mask convention as store_compressed: the i-th True in `mask`
    # adds v[that lane] into acc[i], packed to the front.
    plsc.addupdate_compressed(acc.at[pl.ds(0, vec)], v, mask=(v % 2) == 0)

  checks = {
    "addupdate": (_addupdate, np.arange(vec) + 100),
    "addupdate_compressed": (
      _addupdate_compressed,
      np.concatenate(
        [np.arange(0, vec, 2), np.zeros(vec - vec // 2, np.int32)]
      ),
    ),
  }

  arms = []
  for name, (body, expected) in checks.items():
    # Both flag settings, because "does it lower" and "does it lower with
    # layout inference on" can be different answers for gather/scatter
    # primitives in this surface.
    for layout_passes in (False, True):
      label = f"{name}(needs_layout_passes={layout_passes})"
      try:
        out = run(body, x_i32, layout_passes=layout_passes)
        arm = (
          f"{label}:ok"
          if np.array_equal(out, expected)
          else f"{label}:ran,out={out.tolist()},want={expected.tolist()}"
        )
      except Exception as error:  # noqa: BLE001
        arm = f"{label}:NO({_short(error)})"
      _say(arm)
      arms.append(arm)

  return (
    f"one tiny 1c x 1sc kernel per spelling, vec={vec}, output staged "
    f"through VMEM: " + " | ".join(arms)
  )


# --------------------------------------------------------------------------
# Scalar+vector MPMD in one kernel: which scratch spaces it accepts
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_mpmd_vmem_scratch",
  "Can a scalar+vector MPMD SparseCore kernel allocate VMEM scratch, or is it limited to the SMEM the scalar subcore can also reach?",
)
def _sc_mpmd_vmem_scratch() -> str:
  """Tests which scratch spaces a scalar+vector MPMD kernel can allocate.

  With more than one mesh in a kernel, an unscoped scratch type can be
  refused with an error of the form `MPMD map with more than one mesh
  requires scratch_type to have a core_type specified`.

  **A plain `pltpu.SMEM(...)` or `pltpu.VMEM(...)` has no mesh attached, and
  with two meshes in the kernel the lowering cannot guess which one it
  belongs to.** The spelling that attaches one is the matmul operator:
  `MemorySpace.__matmul__` in core.py returns
  `CoreMemorySpace(memory_space, mesh)`, so `pltpu.SMEM @ scalar_mesh` is the
  core-scoped space and `(pltpu.SMEM @ scalar_mesh)(shape, dtype)` is the
  scratch type. `SemaphoreType.DMA @ mesh` (core.py) returns the
  already-constructed ref rather than a callable, which is a wart worth
  knowing up front.

  Which spaces each mesh will accept is not guesswork either --
  `Mesh.supported_memory_spaces` lists them, and `CoreMemorySpace.__post_init__`
  checks against it:

    * `VectorSubcoreMesh`: VMEM, VMEM_SHARED, SMEM, SEMAPHORE (sc_core.py)
    * `ScalarSubcoreMesh`: VMEM_SHARED, SMEM, SEMAPHORE (sc_core.py)

  Read from source, toolchain-dependent. If so, the scalar subcore cannot
  hold plain VMEM, and `VMEM_SHARED` is the one non-semaphore space *both*
  meshes accept -- which makes it the candidate for handing data between
  them.

  Four arms, control first:

  1. control -- core-scoped SMEM for each body, the minimum that should work;
  2. VMEM for the vector body, SMEM for the scalar one -- the configuration
     that makes MPMD useful, since SMEM is small;
  3. VMEM_SHARED for both -- the candidate shared channel;
  4. plain unscoped `pltpu.VMEM`, the negative control expected to produce
     the error above.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  scalar_mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1)
  vector_mesh = plsc.VectorSubcoreMesh(
    core_axis_name="core",
    subcore_axis_name="subcore",
    num_cores=1,
    num_subcores=1,
  )
  x = jnp.arange(vec, dtype=jnp.int32)

  def _try(label, scratch_types, scalar_body, vector_body):
    try:
      kernel = pl.kernel(
        [scalar_body, vector_body],
        out_type=(
          jax.ShapeDtypeStruct((vec,), jnp.int32),
          jax.ShapeDtypeStruct((vec,), jnp.int32),
        ),
        mesh=[scalar_mesh, vector_mesh],
        scratch_types=scratch_types,
      )
      a, b = jax.jit(kernel)(x, x)
      a, b = np.asarray(a), np.asarray(b)
      ok = np.array_equal(a, np.arange(vec) + 1) and np.array_equal(
        b, np.arange(vec) * 2
      )
      arm = f"{label}:{'ok' if ok else f'WRONG(a={a.tolist()},b={b.tolist()})'}"
    except Exception as error:  # noqa: BLE001
      arm = f"{label}:NO({_short(error)})"
    _say(arm)
    return arm

  # Every body in an MPMD kernel is handed the whole ref list -- both inputs,
  # both outputs, then all the scratch -- not just the refs it touches. The
  # two bodies below therefore take the same signature and each ignores the
  # other's buffer.
  def scalar_body(x1, _x2, o1, _o2, sbuf, _vbuf, sem):
    pltpu.async_copy(x1, sbuf, sem).wait()

    @pl.loop(0, vec)
    def _(i):
      sbuf[i] = sbuf[i] + 1

    pltpu.async_copy(sbuf, o1, sem).wait()

  def vector_body_scalarloop(_x1, x2, _o1, o2, _sbuf, vbuf, sem):
    """For SMEM/VMEM_SHARED buffers: index one element at a time."""
    pltpu.async_copy(x2, vbuf, sem).wait()

    @pl.loop(0, vec)
    def _(i):
      vbuf[i] = vbuf[i] * 2

    pltpu.async_copy(vbuf, o2, sem).wait()

  def vector_body_lanes(_x1, x2, _o1, o2, _sbuf, vbuf, sem):
    """For a real VMEM buffer: a lane group at a time, as VMEM requires."""
    pltpu.async_copy(x2, vbuf, sem).wait()

    @plsc.parallel_loop(0, vec, vec)
    def _(i):
      vbuf[pl.ds(i, vec)] = vbuf[pl.ds(i, vec)] * 2

    pltpu.sync_copy(vbuf, o2)

  def scoped(space, mesh):
    """`space @ mesh` -> CoreMemorySpace, then call it for the ref type."""
    return (space @ mesh)((vec,), jnp.int32)

  arms = []
  plan = []

  # 1. Control first: core-scoped SMEM for each body. Both meshes list SMEM
  #    in supported_memory_spaces, so if this is refused the spelling is
  #    wrong and nothing below it means anything.
  plan.append(
    (
      "control: SMEM@scalar + SMEM@vector",
      lambda: [
        scoped(pltpu.SMEM, scalar_mesh),
        scoped(pltpu.SMEM, vector_mesh),
        pltpu.SemaphoreType.DMA @ vector_mesh,
      ],
      scalar_body,
      vector_body_scalarloop,
    )
  )

  # 2. The configuration that makes MPMD worth having: the vector body gets
  #    real VMEM. Predicted to fail -- pallas_call_registration.py maps
  #    CoreMemorySpace(VMEM, mesh) and raises `Invalid core type for VMEM`
  #    for anything but CoreType.TC -- even though the mesh itself lists VMEM
  #    as supported. If both are true, that is a lowering gap, not a design.
  plan.append(
    (
      "SMEM@scalar + VMEM@vector",
      lambda: [
        scoped(pltpu.SMEM, scalar_mesh),
        scoped(pltpu.VMEM, vector_mesh),
        pltpu.SemaphoreType.DMA @ vector_mesh,
      ],
      scalar_body,
      vector_body_lanes,
    )
  )

  # 3. VMEM_SHARED, the only non-semaphore space both meshes accept, and so
  #    the candidate channel for handing data from one subcore type to the
  #    other.
  plan.append(
    (
      "VMEM_SHARED@scalar + VMEM_SHARED@vector",
      lambda: [
        scoped(pltpu.MemorySpace.VMEM_SHARED, scalar_mesh),
        scoped(pltpu.MemorySpace.VMEM_SHARED, vector_mesh),
        pltpu.SemaphoreType.DMA @ vector_mesh,
      ],
      scalar_body,
      vector_body_scalarloop,
    )
  )

  # 4. Negative control: plain unscoped scratch, the spelling expected to
  #    produce the `missing core_type` error. It shows that the error is
  #    about the *spelling* and not about MPMD.
  plan.append(
    (
      "unscoped pltpu.SMEM + pltpu.VMEM",
      lambda: [
        pltpu.SMEM((vec,), jnp.int32),
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.SemaphoreType.DMA,
      ],
      scalar_body,
      vector_body_lanes,
    )
  )

  for label, build, sbody, vbody in plan:
    try:
      scratch_types = build()
    except Exception as error:  # noqa: BLE001 — the scratch type itself may be refused.
      arm = f"{label}:NO(building the scratch type: {_short(error)})"
      _say(arm)
      arms.append(arm)
      continue
    arms.append(_try(label, scratch_types, sbody, vbody))

  if not arms[0].endswith(":ok"):
    raise harness.Invalid(
      "the core-scoped SMEM control did not run, so nothing here is a "
      "statement about what an MPMD kernel can allocate: " + " | ".join(arms)
    )

  return (
    f"pl.kernel(mesh=[ScalarSubcoreMesh, VectorSubcoreMesh]) with four "
    f"scratch configurations, core-scoped with `space @ mesh`, vec={vec}: "
    + " | ".join(arms)
  )


# --------------------------------------------------------------------------
# subcore_barrier and fetch_and_add: minimal reproducer
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_subcore_barrier_minimal",
  "Does a bare `subcore_barrier()` complete, and does that depend on the subcore count?",
  destructive=True,
)
def _sc_subcore_barrier_minimal() -> str:
  """A minimal `subcore_barrier` reproducer that escalates one step at a time.

  A hang in `subcore_barrier` can leave a **completely empty log**, and an
  empty log cannot distinguish "the primitive deadlocks" from "this call
  pattern deadlocks". This probe is built to make that distinction:

  * The arms escalate. A kernel with one barrier and *nothing else* comes
    first, at 2 subcores, then 4, 8, 16, 32; `fetch_and_add` is a separate
    later arm; barrier plus fetch_and_add is last. The first arm that does not
    return names the boundary.
  * Every arm prints before it starts and after it finishes, flushed. A hang
    then leaves "starting 8sc" as its last line instead of an empty file,
    which is the whole difference between a diagnosis and a guess.
  * `destructive=True`. A wedged Mosaic core poisons every later probe in the
    process, and this one can wedge the *device*. It must be `--pick`ed,
    run alone in its own process, and wrapped in a wall-clock `timeout`.

  A kernel with a barrier and no other work still has to be *correct*, not
  just terminate: the output is a constant written by subcore 0, so a run
  that returns the wrong constant means the emit raced the barrier.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  _SENTINEL = 12345

  def bare_barrier(num_subcores, num_barriers):
    """One barrier (or several), no fetch_and_add, no loop, no DMA in."""

    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
      mesh=_vector_mesh(num_cores=1, num_subcores=num_subcores),
      scratch_types=[pltpu.VMEM((vec,), jnp.int32)],
    )
    def kernel(o_ref, obuf):
      subcore = jax.lax.axis_index("subcore")
      for _ in range(num_barriers):
        plsc.subcore_barrier()

      @pl.when(subcore == 0)
      def _():
        obuf[...] = jnp.full((vec,), _SENTINEL, jnp.int32)
        pltpu.sync_copy(obuf, o_ref)

    got = int(np.asarray(jax.jit(kernel)())[0])
    return "ok" if got == _SENTINEL else f"WRONG({got})"

  def fetch_add_only(num_subcores):
    """fetch_and_add with no barrier at all.

    Racy by construction -- subcore 0 may not have zeroed the accumulator
    when another subcore adds into it -- so the tally is *not* checked
    against num_subcores. The question is only whether the primitive
    executes and returns, separately from `subcore_barrier`.
    """
    _LOCAL, _GLOBAL = 0, 1

    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
      mesh=_vector_mesh(num_cores=1, num_subcores=num_subcores),
      scratch_types=[
        pltpu.SMEM((2,), jnp.int32),
        pltpu.VMEM((vec,), jnp.int32),
      ],
    )
    def kernel(o_ref, acc, obuf):
      subcore = jax.lax.axis_index("subcore")
      acc[_LOCAL] = jnp.int32(0)
      acc[_GLOBAL] = jnp.int32(0)
      plsc.fetch_and_add(acc.at[_GLOBAL], jnp.int32(1), subcore_id=0)

      @pl.when(subcore == 0)
      def _():
        obuf[...] = jnp.full((vec,), acc[_GLOBAL], jnp.int32)
        pltpu.sync_copy(obuf, o_ref)

    got = int(np.asarray(jax.jit(kernel)())[0])
    # No verdict on the value: without a barrier the read is racy and any
    # tally in [0, num_subcores] is a legitimate outcome.
    return f"ran(tally={got}, racy-by-design)"

  def barrier_and_fetch_add(num_subcores):
    """The full pattern: init, barrier, fetch_and_add, barrier, emit."""
    _LOCAL, _GLOBAL = 0, 1

    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
      mesh=_vector_mesh(num_cores=1, num_subcores=num_subcores),
      scratch_types=[
        pltpu.SMEM((2,), jnp.int32),
        pltpu.VMEM((vec,), jnp.int32),
      ],
    )
    def kernel(o_ref, acc, obuf):
      subcore = jax.lax.axis_index("subcore")
      acc[_LOCAL] = jnp.int32(0)
      acc[_GLOBAL] = jnp.int32(0)
      plsc.subcore_barrier()
      plsc.fetch_and_add(acc.at[_GLOBAL], jnp.int32(1), subcore_id=0)
      plsc.subcore_barrier()

      @pl.when(subcore == 0)
      def _():
        obuf[...] = jnp.full((vec,), acc[_GLOBAL], jnp.int32)
        pltpu.sync_copy(obuf, o_ref)

    got = int(np.asarray(jax.jit(kernel)())[0])
    return (
      f"ok(tally={got})"
      if got == num_subcores
      else f"WRONG(tally={got},want={num_subcores})"
    )

  plan = [
    ("bare-barrier 2sc x1", lambda: bare_barrier(2, 1)),
    ("bare-barrier 4sc x1", lambda: bare_barrier(4, 1)),
    ("bare-barrier 8sc x1", lambda: bare_barrier(8, 1)),
    ("bare-barrier 16sc x1", lambda: bare_barrier(16, 1)),
    (
      f"bare-barrier {info.num_subcores}sc x1",
      lambda: bare_barrier(info.num_subcores, 1),
    ),
    ("bare-barrier 8sc x3", lambda: bare_barrier(8, 3)),
    ("fetch_and_add-only 8sc", lambda: fetch_add_only(8)),
    ("barrier+fetch_and_add 2sc", lambda: barrier_and_fetch_add(2)),
    ("barrier+fetch_and_add 8sc", lambda: barrier_and_fetch_add(8)),
  ]

  arms = []
  for label, run in plan:
    # Printed *before* the call. This line is the diagnosis if the process
    # never gets to print the next one.
    print(f"      > starting {label}", flush=True)
    try:
      arm = f"{label}:{run()}"
    except Exception as error:  # noqa: BLE001 — a refusal is a finding.
      arm = f"{label}:NO({_short(error)})"
    _say(arm)
    arms.append(arm)

  if all(":NO(" in arm for arm in arms):
    raise harness.Unsupported(
      "no cross-subcore coordination arm ran: " + " | ".join(arms)
    )
  return (
    f"escalating from a bare subcore_barrier to the init/barrier/"
    f"fetch_and_add/barrier pattern, 1 core, num_subcores<={info.num_subcores}: "
    + " | ".join(arms)
  )


# --------------------------------------------------------------------------
# Wide Get against the lane-group loop: faster, or only shorter?
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_wide_get_vs_lane_loop",
  "Where a wide Get lowers, is it faster than the lane-group loop, or only shorter?",
  # On some chips the wide spelling aborts the whole process (SIGABRT), which
  # would take every later probe in a group run down with it.
  destructive=True,
)
def _sc_wide_get_vs_lane_loop() -> str:
  """Times the wide spelling against the lane-group loop.

  `sc_wide_get_without_reduction` asks whether `obuf[...] = buf[...] * 2`
  over a wide buffer lowers with `needs_layout_passes` left at its `True`
  default. It does not time anything, and "lowers" is not "is fast": the
  compiler may emit the same `num_lanes`-wide loop underneath, in which case
  the wide spelling buys readability and nothing else.

  That matters because the wide spelling is only available to a kernel
  containing no op that forces the flag off -- no `load_gather`, no
  `scan_count`, no `jnp.sum` -- which is a small corner of the design space.
  Worth knowing whether the corner pays before anyone designs into it.

  Three arms over the same work, `o = x*2 + 1` on int32:

    1. wide, flag ON   -- one statement over the whole buffer
    2. loop, flag ON   -- lane groups, same flag, so the *only* variable
                          between arms 1 and 2 is the spelling
    3. loop, flag OFF  -- the spelling most kernels in this suite use

  Arm 2 is what makes this a controlled comparison. Reporting 1 against 3
  alone would confound the spelling with the flag, and the flag changes which
  compiler passes run.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  # 128 KiB of int32 per buffer, two buffers; must fit in
  # SparseCoreInfo.vmem_capacity_bytes.
  n = 32768

  rng = np.random.default_rng(23)
  host = rng.integers(0, 1000, size=n).astype(np.int32)
  x = jnp.asarray(host)
  want = host * 2 + 1

  def build(spelling, layout_passes):
    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((n,), jnp.int32),
      mesh=_vector_mesh(num_cores=1, num_subcores=1),
      scratch_types=[
        pltpu.VMEM((n,), jnp.int32),
        pltpu.VMEM((n,), jnp.int32),
        pltpu.SemaphoreType.DMA,
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=layout_passes),
    )
    def kernel(x_ref, o_ref, buf, obuf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      if spelling == "wide":
        obuf[...] = buf[...] * 2 + 1
      else:

        @pl.loop(0, n, step=vec)
        def _(i):
          obuf[pl.ds(i, vec)] = buf[pl.ds(i, vec)] * 2 + 1

      pltpu.sync_copy(obuf, o_ref)

    return kernel

  plan = [
    ("wide,nlp=True", "wide", True),
    ("loop,nlp=True", "loop", True),
    ("loop,nlp=False", "loop", False),
  ]

  arms = []
  medians = {}
  floor_candidate = None
  for label, spelling, layout_passes in plan:
    try:
      fn = jax.jit(build(spelling, layout_passes))
      got = np.asarray(fn(x))
      if not np.array_equal(got, want):
        arm = f"{label}:WRONG"
      else:
        timing = harness.time_above_floor(lambda _f=fn: _f(x), what=label)
        medians[label] = timing.median
        if (
          floor_candidate is None
          or timing.region_median < floor_candidate[1].region_median
        ):
          floor_candidate = (label, timing)
        arm = (
          f"{label}:{timing.median * 1e6:.2f}us "
          f"spread={timing.spread_pct:.0f}% batch={timing.batch}"
        )
    except Exception as error:  # noqa: BLE001
      arm = f"{label}:NO({_short(error)})"
    _say(arm)
    arms.append(arm)

  if all(":NO(" in arm for arm in arms):
    raise harness.Unsupported("no spelling lowered: " + " | ".join(arms))
  if floor_candidate is not None:
    label, timing = floor_candidate
    harness.require_above_dispatch_floor(
      timing, what=f"the shortest timed region ({label})"
    )

  verdict = ""
  if "wide,nlp=True" in medians and "loop,nlp=True" in medians:
    ratio = medians["loop,nlp=True"] / medians["wide,nlp=True"]
    verdict = f" || loop/wide at the same flag = {ratio:.2f}x"

  return (
    f"o = x*2+1 over {n} int32 in VMEM, 1c x 1sc, num_lanes={vec}; the "
    f"only variable between the first two arms is the spelling: "
    + " | ".join(arms)
    + verdict
  )


# --------------------------------------------------------------------------
# Where does XLA's gather cost step, and is it a capacity threshold?
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_xla_gather_cliff_bracket",
  "At what index count does the XLA gather's cost step sharply, and does that threshold move when the table size changes?",
)
def _sc_xla_gather_cliff_bracket() -> str:
  """Locates any sharp step in XLA gather cost, and tests what moves it.

  Hypothesis: XLA's 1-wide gather has a fast path that stops applying past
  some index count, so its cost is flat, then steps sharply, then grows
  linearly again. `sc_gather_index_count_scaling` samples whole octaves; this
  probe sweeps index count in half-octave steps so a step can be bracketed
  tightly, which is what a reader needs to know whether a gather comparison
  measured at one index count applies to their workload.

  Second question: is the step about capacity? If the output and index
  buffers are what overflows, the step lands at the same index count for any
  table size; if the table participates, a small table holds out longer. So
  the sweep runs at two table sizes far apart (65536 rows and 256 rows of
  f32). A step that does not move under that change is evidence the table is
  not what overflows.

  XLA only: `sc_gather_index_count_scaling` covers the SparseCore arm.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np

  rng = np.random.default_rng(29)
  # Half-octave steps, 2^12 .. 2^20.
  counts = sorted(
    {int(m * 2**e) for e in range(12, 20) for m in (1, 1.5)} | {2**20}
  )
  arms = []
  per_table = {}
  floor_candidate = None

  for table_rows, table_label in ((65536, "256KiB"), (256, "1KiB")):
    table = jnp.asarray(rng.standard_normal(table_rows).astype(np.float32))
    medians = []
    for num_indices in counts:
      label = f"{table_label},n={num_indices}"
      try:
        indices = jnp.asarray(
          rng.integers(0, table_rows, size=num_indices), jnp.int32
        )
        fn = jax.jit(lambda t, i: t[i])
        np.asarray(fn(table, indices))  # compile + fault early
        timing = harness.time_above_floor(
          lambda _f=fn, _i=indices: _f(table, _i), what=label
        )
        medians.append((num_indices, timing.median))
        if (
          floor_candidate is None
          or timing.region_median < floor_candidate[1].region_median
        ):
          floor_candidate = (label, timing)
        arm = f"{label}:{timing.median * 1e6:.2f}us"
      except Exception as error:  # noqa: BLE001
        arm = f"{label}:NO({_short(error)})"
      _say(arm)
      arms.append(arm)
    per_table[table_label] = medians

  if all(":NO(" in arm for arm in arms):
    raise harness.Unsupported("no arm ran: " + " | ".join(arms))
  if floor_candidate is not None:
    label, timing = floor_candidate
    harness.require_above_dispatch_floor(
      timing, what=f"the shortest timed region ({label})"
    )

  # Report the largest ratio between adjacent index counts in each row. A
  # cost proportional to index count grows by sqrt(2) per half-octave step,
  # so an adjacent ratio of 4 or more is a step, not proportional growth,
  # wherever in the range it sits. Comparing against the row minimum instead
  # would flag plain linear growth across a range this wide.
  verdicts = []
  for table_label, medians in per_table.items():
    if len(medians) < 2:
      continue
    ratio, below, above = max(
      (m1 / max(m0, 1e-12), n0, n1)
      for (n0, m0), (n1, m1) in zip(medians, medians[1:])
    )
    verdicts.append(
      f"{table_label}: step between n={below} and n={above} "
      f"({ratio:.1f}x adjacent)"
      if ratio >= 4.0
      else f"{table_label}: no step in this range "
      f"(largest adjacent ratio {ratio:.2f}x at n={below}->{above})"
    )

  return (
    "XLA `table[indices]` only, 1-wide f32 rows, index count swept in "
    "half-octave steps from 2^12 to 2^20, at two table sizes (65536 and 256 "
    "rows): " + " | ".join(arms) + " || " + " ; ".join(verdicts)
  )


# --------------------------------------------------------------------------
# How a DMA inside an MPMD kernel names its core
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_mpmd_dma_core_id",
  "How does a DMA inside a two-mesh MPMD SparseCore kernel name its core?",
)
def _sc_mpmd_dma_core_id() -> str:
  """Tests DMA spellings inside a scalar+vector MPMD kernel.

  Once the scratch is scoped with `(space @ mesh)(shape, dtype)` (see
  `sc_mpmd_vmem_scratch`), a DMA inside the kernel can be refused one layer
  down with an error of the form:

      VerificationError: ... 'tpu.enqueue_dma' op Core id must be specified

  The source says where that `None` comes from. In
  `sc_lowering.py:_dma_start_lowering_rule` the core index is initialised to
  `None` and only ever assigned inside `if device_id is not None:` -- which
  then derives the destination mesh from the *semaphore's* memory space
  (`isinstance(sem_aval.memory_space, pallas_core.CoreMemorySpace)`) and
  resolves the axes through `tc_lowering._device_id_to_logical`. That
  function fills any axis the caller left out with `lax.axis_index(name)`
  (lowering.py). So:

    * `pltpu.async_copy` takes no `device_id` at all -- it cannot ever set
      the core id;
    * `pltpu.make_async_remote_copy(..., device_id=..., device_id_type=MESH)`
      does, and a *dict* device_id with no JAX-mesh axes in it returns
      `logical_device_id=None` -- "purely local communication"
      (primitives.py) -- while still populating the core axes.

  That last point is the whole hypothesis: **a "remote" copy addressed to
  your own core is the spelling for a local DMA that names its core.**

  Second hypothesis, independent and cheaper: both meshes in
  `sc_mpmd_vmem_scratch` are built with the axis name `"core"`. `_device_id_to_logical`
  computes `mpmd_core_axis_names = core_axis_names - spmd_core_axis_names`
  by *name*, so two meshes sharing one is at best ambiguous. Arms 2 and 3
  give the scalar mesh a distinct axis name and change nothing else.

  Held fixed across every arm: the scratch is core-scoped SMEM on both sides
  -- the control configuration of `sc_mpmd_vmem_scratch` -- so the only
  variable is how the copy is issued.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  x = jnp.arange(vec, dtype=jnp.int32)
  want_a, want_b = np.arange(vec) + 1, np.arange(vec) * 2

  def meshes(shared_axis_name):
    scalar_axis = "core" if shared_axis_name else "score"
    return (
      plsc.ScalarSubcoreMesh(axis_name=scalar_axis, num_cores=1),
      plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=1,
        num_subcores=1,
      ),
      scalar_axis,
    )

  def run(label, *, shared_axis_name, copy):
    scalar_mesh, vector_mesh, scalar_axis = meshes(shared_axis_name)

    def scalar_body(x1, _x2, o1, _o2, sbuf, _vbuf, sem, sem2):
      copy(x1, sbuf, sem, sem2, {scalar_axis: 0})

      @pl.loop(0, vec)
      def _(i):
        sbuf[i] = sbuf[i] + 1

      copy(sbuf, o1, sem, sem2, {scalar_axis: 0})

    def vector_body(_x1, x2, _o1, o2, _sbuf, vbuf, sem, sem2):
      copy(x2, vbuf, sem, sem2, {"core": 0, "subcore": 0})

      @pl.loop(0, vec)
      def _(i):
        vbuf[i] = vbuf[i] * 2

      copy(vbuf, o2, sem, sem2, {"core": 0, "subcore": 0})

    kernel = pl.kernel(
      [scalar_body, vector_body],
      out_type=(
        jax.ShapeDtypeStruct((vec,), jnp.int32),
        jax.ShapeDtypeStruct((vec,), jnp.int32),
      ),
      mesh=[scalar_mesh, vector_mesh],
      scratch_types=[
        (pltpu.SMEM @ scalar_mesh)((vec,), jnp.int32),
        (pltpu.SMEM @ vector_mesh)((vec,), jnp.int32),
        pltpu.SemaphoreType.DMA @ vector_mesh,
        pltpu.SemaphoreType.DMA @ vector_mesh,
      ],
    )
    a, b = jax.jit(kernel)(x, x)
    a, b = np.asarray(a), np.asarray(b)
    if np.array_equal(a, want_a) and np.array_equal(b, want_b):
      return "ok"
    return f"WRONG(a={a.tolist()},b={b.tolist()})"

  def local(src, dst, sem, _sem2, _dev):
    pltpu.async_copy(src, dst, sem).wait()

  def sync(src, dst, _sem, _sem2, _dev):
    pltpu.sync_copy(src, dst)

  def remote_named(src, dst, sem, sem2, dev):
    pltpu.make_async_remote_copy(
      src,
      dst,
      sem2,
      sem,
      device_id=dev,
      device_id_type=pl.DeviceIdType.MESH,
    ).start()
    pltpu.make_async_remote_copy(
      src,
      dst,
      sem2,
      sem,
      device_id=dev,
      device_id_type=pl.DeviceIdType.MESH,
    ).wait()

  def remote_empty(src, dst, sem, sem2, _dev):
    pltpu.make_async_remote_copy(
      src,
      dst,
      sem2,
      sem,
      device_id={},
      device_id_type=pl.DeviceIdType.MESH,
    ).start()
    pltpu.make_async_remote_copy(
      src,
      dst,
      sem2,
      sem,
      device_id={},
      device_id_type=pl.DeviceIdType.MESH,
    ).wait()

  plan = [
    # Baseline: plain async_copy with both meshes sharing the axis name, the
    # configuration the other arms each change one thing from.
    ("async_copy, shared axis name", True, local),
    ("async_copy, distinct axis names", False, local),
    ("sync_copy, distinct axis names", False, sync),
    ("remote_copy device_id={axes:0}", False, remote_named),
    ("remote_copy device_id={}", False, remote_empty),
  ]

  arms = []
  for label, shared, copy in plan:
    try:
      arm = f"{label}:{run(label, shared_axis_name=shared, copy=copy)}"
    except Exception as error:  # noqa: BLE001
      arm = f"{label}:NO({_short(error)})"
    _say(arm)
    arms.append(arm)

  if all(":NO(" in arm for arm in arms):
    raise harness.Unsupported(
      "no DMA spelling lowered inside an MPMD kernel: " + " | ".join(arms)
    )
  return (
    "scalar+vector MPMD, core-scoped SMEM on both sides held fixed, only "
    "the DMA spelling varying, 1 core x 1 subcore: " + " | ".join(arms)
  )
