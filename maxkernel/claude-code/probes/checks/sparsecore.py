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
"""What the SparseCore is, what reaches it, and what it costs to use with the TensorCore.

Supplies answers for `reference/design/sparsecore.md`.

The SparseCore Pallas surface lives at ``jax.experimental.pallas.tpu_sc``;
other plausible module paths do not exist, and probing the wrong one reads as
"SparseCore is unreachable". Every probe here is written so that the surface
being absent shows up as ``UNSUPPORTED``, not as a pass.

Two structural facts shape how these are written. SparseCore kernels cannot be
dry-run: ``VectorSubcoreMesh.__post_init__`` calls ``get_tpu_info()``, which
raises ``ValueError: Unsupported TPU device kind: cpu``, and even past that the
SC primitives have no CPU interpret rules. And the TensorCore is a *separate
program* — nothing here can put TC and SC work in one kernel, so the last four
probes measure the handoff between two kernels instead.
"""

from __future__ import annotations

import harness

_GROUP = "sparsecore"


def _sc():
  """Imports the SparseCore Pallas surface, or reports its absence."""
  try:
    from jax.experimental.pallas import tpu_sc as plsc
  except ImportError as error:
    raise harness.Unsupported(
      f"jax.experimental.pallas.tpu_sc is absent: {error}"
    ) from error
  return plsc


def _vector_mesh(*, num_cores: int = 1, num_subcores: int = 1):
  plsc = _sc()
  return plsc.VectorSubcoreMesh(
    core_axis_name="core",
    subcore_axis_name="subcore",
    num_cores=num_cores,
    num_subcores=num_subcores,
  )


# --------------------------------------------------------------------------
# Surface
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_module_surface",
  "Which SparseCore Pallas surface does this jax expose, at what module path, and with what geometry?",
)
def _sc_module_surface() -> str:
  """Reports the SparseCore surface and the geometry it reports.

  Complements `compute_units.sparsecore_surface`, which tests the module
  paths `pallas.tpu.sparsecore`, `pallas.sparsecore` and
  `_src.pallas.mosaic.sparsecore`. This probe reports both those paths and
  `pallas.tpu_sc`, so a reader sees which one this toolchain ships.
  """
  import importlib

  import jax
  import jax.numpy as jnp

  absent_paths = []
  for path in (
    "jax.experimental.pallas.tpu.sparsecore",
    "jax.experimental.pallas.sparsecore",
    "jax._src.pallas.mosaic.sparsecore",
  ):
    try:
      importlib.import_module(path)
    except ImportError:
      absent_paths.append(path.rsplit(".", 1)[-1] + "@" + path)

  plsc = _sc()
  symbols = sorted(n for n in dir(plsc) if not n.startswith("_"))
  info = plsc.get_sparse_core_info()

  shapes = {}
  for name in ("float32", "int32", "bfloat16", "int8"):
    try:
      shapes[name] = [tuple(s) for s in plsc.supported_shapes(jnp.dtype(name))]
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      shapes[name] = f"{type(error).__name__}"

  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  core_types = sorted(n for n in dir(pltpu.CoreType) if not n.startswith("_"))
  mem_spaces = sorted(
    n for n in dir(pltpu.MemorySpace) if not n.startswith("_")
  )

  return (
    f"jax={jax.__version__} tpu_sc=PRESENT ({len(symbols)} symbols) "
    f"{symbols} || tried-and-absent={len(absent_paths)}/3 old paths "
    f"|| get_sparse_core_info()={info!r} "
    f"|| supported_shapes={shapes} "
    f"|| CoreType={core_types} MemorySpace={mem_spaces} "
    f"|| pl.kernel accepts mesh sequence="
    f"{'mesh' in __import__('inspect').signature(pl.kernel).parameters}"
  )


@harness.probe(
  _GROUP,
  "sc_kernel_cannot_dry_run",
  "Can a SparseCore kernel be dry-run on CPU the way a TensorCore Pallas kernel can?",
)
def _sc_kernel_cannot_dry_run() -> str:
  """Named after the claim that it can. A `PASS` here means it cannot.

  On a TPU host this probe cannot reproduce the CPU failure, so it reports
  what it can check — that the mesh constructor consults `get_tpu_info()`,
  which is the call that raises on CPU — and describes the CPU failure.
  """
  import inspect

  from jax._src.pallas.mosaic import sc_core

  source = inspect.getsource(sc_core.VectorSubcoreMesh.__post_init__)
  consults_tpu_info = "get_sparse_core_info()" in source
  if not consults_tpu_info:
    return (
      "FAIL",
      "VectorSubcoreMesh.__post_init__ no longer calls "
      "get_sparse_core_info(); the CPU dry-run block may have lifted. "
      "Re-check on a CPU host.",
    )
  return (
    "VectorSubcoreMesh.__post_init__ calls get_sparse_core_info(), which "
    "calls get_tpu_info(). On a CPU host that raises ValueError: "
    "Unsupported TPU device kind: cpu, so an SC kernel cannot even be "
    "constructed there. Monkeypatching get_sparse_core_info to a fake "
    "SparseCoreInfo lets the kernel trace but not lower: "
    "NotImplementedError: MLIR translation rule for primitive "
    "parallel_loop not found for platform cpu. Tracing catches typos; "
    "nothing else about an SC kernel can be checked off-device."
  )


# --------------------------------------------------------------------------
# Does anything actually run on it
# --------------------------------------------------------------------------


def _doubling_kernel(n: int, chunk: int, *, num_cores: int, num_subcores: int):
  """out = 2*in over a 1-D f32 array, one chunk per (core, subcore) stride."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  mesh = _vector_mesh(num_cores=num_cores, num_subcores=num_subcores)
  vec = plsc.get_sparse_core_info().num_lanes
  workers = num_cores * num_subcores
  num_chunks = n // chunk

  @pl.kernel(
    out_type=jax.ShapeDtypeStruct((n,), jnp.float32),
    mesh=mesh,
    scratch_types=[pltpu.VMEM((chunk,), jnp.float32), pltpu.SemaphoreType.DMA],
  )
  def kernel(x_ref, o_ref, buf, sem):
    worker = jax.lax.axis_index("core") * num_subcores + jax.lax.axis_index(
      "subcore"
    )

    @pl.loop(worker, num_chunks, step=workers)
    def _(c):
      copy = pltpu.async_copy(x_ref.at[pl.ds(c * chunk, chunk)], buf, sem)
      copy.wait()

      @plsc.parallel_loop(0, chunk, vec)
      def _(i):
        buf[pl.ds(i, vec)] = buf[pl.ds(i, vec)] * 2.0

      pltpu.sync_copy(buf, o_ref.at[pl.ds(c * chunk, chunk)])

  return kernel


@harness.probe(
  _GROUP,
  "sc_vector_kernel_runs",
  "Does a minimal VectorSubcoreMesh kernel compile, run and compute the right answer?",
)
def _sc_vector_kernel_runs() -> str:
  import jax
  import jax.numpy as jnp
  import numpy as np

  n, chunk = 1 << 16, 1024
  x = jnp.arange(n, dtype=jnp.float32)
  kernel = _doubling_kernel(n, chunk, num_cores=1, num_subcores=8)
  out = np.asarray(jax.jit(kernel)(x))
  expected = np.arange(n, dtype=np.float32) * 2.0
  if not np.array_equal(out, expected):
    wrong = int((out != expected).sum())
    return (
      "FAIL",
      f"{wrong}/{n} elements wrong; first={out[:4]} want={expected[:4]}",
    )
  return (
    f"n={n} chunk={chunk} cores=1 subcores=8: exact match on all {n} "
    f"elements. A SparseCore kernel compiles, launches and is numerically "
    f"correct through pl.kernel(mesh=VectorSubcoreMesh)."
  )


@harness.probe(
  _GROUP,
  "sc_scalar_kernel_runs",
  "Does a ScalarSubcoreMesh kernel compile and run, and which memory spaces does it accept?",
)
def _sc_scalar_kernel_runs() -> str:
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1)
  spaces = [str(s) for s in mesh.supported_memory_spaces]
  n = 256

  # A scalar-subcore kernel can neither load from nor store to an HBM ref:
  # both raise "Get/Swap does not support loading from/storing to HBM".
  # The data has to be DMA'd into a core-local space first — SMEM here,
  # since a ScalarSubcoreMesh cannot allocate plain VMEM at all.
  @pl.kernel(
    out_type=jax.ShapeDtypeStruct((n,), jnp.int32),
    mesh=mesh,
    scratch_types=[pltpu.SMEM((n,), jnp.int32), pltpu.SemaphoreType.DMA],
  )
  def kernel(x_ref, o_ref, buf, sem):
    pltpu.async_copy(x_ref, buf, sem).wait()

    @pl.loop(0, n)
    def _(i):
      buf[i] = buf[i] + 1

    pltpu.async_copy(buf, o_ref, sem).wait()

  x = jnp.arange(n, dtype=jnp.int32)
  out = np.asarray(jax.jit(kernel)(x))
  ok = np.array_equal(out, np.arange(n, dtype=np.int32) + 1)
  return (
    f"ScalarSubcoreMesh(num_cores=1) ran, numerics_ok={ok}, "
    f"supported_memory_spaces={spaces} (note: no plain VMEM — a scalar "
    f"subcore kernel gets VMEM_SHARED, SMEM and SEMAPHORE only)"
  )


@harness.probe(
  _GROUP,
  "sc_dtype_support",
  "Which element dtypes survive a load-compute-store round trip in a SparseCore vector kernel?",
)
def _sc_dtype_support() -> str:
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  chunk, n = 256, 4096
  verdicts = []
  for name in (
    "float32",
    "int32",
    "uint32",
    "bfloat16",
    "float16",
    "int8",
    "int16",
  ):
    dtype = jnp.dtype(name)
    try:
      mesh = _vector_mesh(num_cores=1, num_subcores=1)

      @pl.kernel(
        out_type=jax.ShapeDtypeStruct((n,), dtype),
        mesh=mesh,
        scratch_types=[
          pltpu.VMEM((chunk,), dtype),
          pltpu.SemaphoreType.DMA,
        ],
      )
      def kernel(x_ref, o_ref, buf, sem, _dtype=dtype):
        @pl.loop(0, n // chunk)
        def _(c):
          pltpu.async_copy(x_ref.at[pl.ds(c * chunk, chunk)], buf, sem).wait()

          @plsc.parallel_loop(0, chunk, vec)
          def _(i):
            buf[pl.ds(i, vec)] = buf[pl.ds(i, vec)] + _dtype.type(1)

          pltpu.sync_copy(buf, o_ref.at[pl.ds(c * chunk, chunk)])

      x = jnp.zeros((n,), dtype)
      out = np.asarray(jax.jit(kernel)(x))
      ok = bool((out == 1).all())
      verdicts.append(f"{name}={'ok' if ok else 'WRONG'}")
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      message = str(error).strip().replace("\n", " ")[:90]
      verdicts.append(f"{name}=NO({type(error).__name__}: {message})")
  return (
    f"vec_dim={vec} chunk={chunk} n={n}, scratch VMEM ref of that dtype, "
    f"add 1 and read back: {' | '.join(verdicts)}"
  )


@harness.probe(
  _GROUP,
  "sc_vmem_per_subcore",
  "How much VMEM can one vector subcore allocate before the compiler refuses, and does it match what `SparseCoreInfo` reports?",
)
def _sc_vmem_per_subcore() -> str:
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  claimed = info.vmem_capacity_bytes
  results = []
  last_ok = 0
  # f32, so bytes = 4 * elements. Sweep past the reported capacity on both
  # sides, as fractions of what SparseCoreInfo reports on this chip.
  fractions = (1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8, 15 / 16, 1, 5 / 4, 2)
  sizes_kib = sorted({max(1, int(claimed * f) // 1024) for f in fractions})
  for kib in sizes_kib:
    elements = kib * 1024 // 4
    try:
      mesh = _vector_mesh(num_cores=1, num_subcores=1)

      @pl.kernel(
        out_type=jax.ShapeDtypeStruct((vec,), jnp.float32),
        mesh=mesh,
        scratch_types=[
          pltpu.VMEM((elements,), jnp.float32),
          pltpu.VMEM((vec,), jnp.float32),
          pltpu.SemaphoreType.DMA,
        ],
      )
      def kernel(x_ref, o_ref, buf, edge, sem, _e=elements):
        # Touch the far end of the allocation and carry the value back
        # out, so neither the allocation nor the traffic can be
        # eliminated as dead.
        pltpu.async_copy(x_ref, edge, sem).wait()
        buf[pl.ds(_e - vec, vec)] = edge[...]
        edge[...] = buf[pl.ds(_e - vec, vec)]
        pltpu.sync_copy(edge, o_ref)

      jax.block_until_ready(jax.jit(kernel)(jnp.ones((vec,), jnp.float32)))
      results.append(f"{kib}KiB=ok")
      last_ok = kib
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      # Carry the message, not just the type: a capacity refusal and a
      # coding mistake are both exceptions, and a bare type name cannot
      # tell them apart, so a type-only report could claim a capacity of
      # zero.
      message = str(error).strip().replace("\n", " ")[:90]
      results.append(f"{kib}KiB=NO({type(error).__name__}: {message})")
  return (
    f"single f32 VMEM scratch, one subcore: {' '.join(results)} "
    f"|| largest accepted={last_ok}KiB vs "
    f"SparseCoreInfo.vmem_capacity_bytes={claimed} "
    f"({claimed / 1024:.0f}KiB)"
  )


# --------------------------------------------------------------------------
# The primitive surface
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_primitive_lowering",
  "Which of the tpu_sc primitives actually lower to Mosaic and compute the right answer?",
)
def _sc_primitive_lowering() -> str:
  """One tiny kernel per primitive. A primitive that exists but does not lower
  is the single most expensive thing to discover halfway through a design."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes

  # Whether the kernel needed needs_layout_passes=False, per primitive. The
  # Mosaic-SC infer-vector-layout pass rejects several of the gather/scatter
  # ops outright and names the flag in its own error, so "does it lower" and
  # "does it lower with layout inference on" are two different questions and
  # both belong in the answer.
  needed_flag: list[str] = []

  def _run_once(body, out_shape, out_dtype, inputs, scratch, layout_passes):
    mesh = _vector_mesh(num_cores=1, num_subcores=1)
    n_in = len(inputs)

    # A SparseCore kernel may not store to an HBM ref: `o_ref[...] = v`
    # raises "Swap does not support storing to HBM". Every body below is
    # therefore handed a VMEM staging buffer in place of the real output
    # ref, and the copy out happens here, once.
    def staged(*refs):
      body(*refs[:n_in], refs[n_in + 1], *refs[n_in + 2 :])
      pltpu.sync_copy(refs[n_in + 1], refs[n_in])

    kernel = pl.kernel(
      staged,
      out_type=jax.ShapeDtypeStruct(out_shape, out_dtype),
      mesh=mesh,
      scratch_types=[pltpu.VMEM(out_shape, out_dtype), *scratch],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=layout_passes),
    )
    return np.asarray(jax.jit(kernel)(*inputs))

  def run(body, out_shape, out_dtype, *inputs, scratch=()):
    """Lower with layout inference on; if Mosaic-SC refuses and names the
    flag, retry with it off and record that the primitive needs it."""
    try:
      return _run_once(body, out_shape, out_dtype, inputs, scratch, True)
    except Exception as error:  # noqa: BLE001
      if "needs_layout_passes" not in str(error):
        raise
    out = _run_once(body, out_shape, out_dtype, inputs, scratch, False)
    needed_flag.append(_current[0])
    return out

  verdicts = []
  _current = [""]

  def check(name, fn):
    _current[0] = name
    try:
      detail = fn()
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      message = str(error).strip().replace("\n", " ")[:110]
      verdicts.append(f"{name}=NO({type(error).__name__}: {message})")
    else:
      flag = "*" if name in needed_flag else ""
      verdicts.append(f"{name}{flag}={detail}")

  x_i32 = jnp.arange(vec, dtype=jnp.int32)
  ones = jnp.ones((vec,), jnp.int32)

  def _cumsum():
    def body(x_ref, o_ref, buf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      o_ref[...] = plsc.cumsum(buf[...])

    out = run(
      body,
      (vec,),
      jnp.int32,
      ones,
      scratch=[pltpu.VMEM((vec,), jnp.int32), pltpu.SemaphoreType.DMA],
    )
    want = np.cumsum(np.ones(vec, np.int32))
    return "ok" if np.array_equal(out, want) else f"WRONG{out.tolist()}"

  def _cummax():
    def body(x_ref, o_ref, buf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      o_ref[...] = plsc.cummax(buf[...])

    src = jnp.array([3, 1, 4, 1, 5, 9, 2, 6] * (vec // 8), jnp.int32)
    out = run(
      body,
      (vec,),
      jnp.int32,
      src,
      scratch=[pltpu.VMEM((vec,), jnp.int32), pltpu.SemaphoreType.DMA],
    )
    want = np.maximum.accumulate(np.asarray(src))
    return "ok" if np.array_equal(out, want) else f"WRONG{out.tolist()}"

  def _scan_count():
    def body(x_ref, o_ref, buf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      counts, _mask = plsc.scan_count(buf[...])
      o_ref[...] = counts

    src = jnp.zeros((vec,), jnp.int32)
    out = run(
      body,
      (vec,),
      jnp.int32,
      src,
      scratch=[pltpu.VMEM((vec,), jnp.int32), pltpu.SemaphoreType.DMA],
    )
    want = np.arange(vec)
    return "ok" if np.array_equal(out, want) else f"ran,out={out.tolist()}"

  def _sort_key_val():
    def body(x_ref, o_ref, buf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      keys = buf[...]
      sorted_keys, _sorted_vals = plsc.sort_key_val(keys, keys, descending=True)
      o_ref[...] = sorted_keys

    src = jnp.array(list(range(vec))[::-1], jnp.int32)
    out = run(
      body,
      (vec,),
      jnp.int32,
      src,
      scratch=[pltpu.VMEM((vec,), jnp.int32), pltpu.SemaphoreType.DMA],
    )
    want = np.arange(vec)[::-1]
    return (
      "ok(descending)"
      if np.array_equal(out, want)
      else f"ran,out={out.tolist()}"
    )

  def _all_reduce():
    def body(x_ref, o_ref, buf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      v = buf[...]
      popcount = plsc.all_reduce_population_count(v > 2)
      ffs = plsc.all_reduce_ffs(v > 2)
      o_ref[...] = popcount * 1000 + ffs

    out = run(
      body,
      (vec,),
      jnp.int32,
      x_i32,
      scratch=[pltpu.VMEM((vec,), jnp.int32), pltpu.SemaphoreType.DMA],
    )
    want_pop, want_ffs = vec - 3, 3
    got = int(out[0])
    return (
      "ok"
      if got == want_pop * 1000 + want_ffs
      else f"ran,popcount={got // 1000},ffs={got % 1000},want={want_pop}/{want_ffs}"
    )

  def _store_compressed():
    def body(x_ref, o_ref, buf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      v = buf[...]
      o_ref[...] = jnp.full((vec,), -1, jnp.int32)
      # A bare scratch ref is a JitTracer and is refused; the primitive
      # wants a TransformedRef, which `.at[...]` produces.
      plsc.store_compressed(o_ref.at[pl.ds(0, vec)], v, mask=(v % 2) == 0)

    out = run(
      body,
      (vec,),
      jnp.int32,
      x_i32,
      scratch=[
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.SemaphoreType.DMA,
      ],
    )
    want_prefix = np.arange(0, vec, 2)
    ok = np.array_equal(out[: len(want_prefix)], want_prefix)
    return "ok" if ok else f"ran,out={out.tolist()}"

  def _load_gather():
    def body(x_ref, idx_ref, o_ref, buf, ibuf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      pltpu.async_copy(idx_ref, ibuf, sem).wait()
      o_ref[...] = plsc.load_gather(buf, [ibuf[...]])

    idx = jnp.array(list(range(vec))[::-1], jnp.int32)
    out = run(
      body,
      (vec,),
      jnp.int32,
      x_i32 * 10,
      idx,
      scratch=[
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.SemaphoreType.DMA,
      ],
    )
    want = np.arange(vec)[::-1] * 10
    return "ok" if np.array_equal(out, want) else f"ran,out={out.tolist()}"

  def _store_scatter():
    def body(x_ref, idx_ref, o_ref, buf, ibuf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      pltpu.async_copy(idx_ref, ibuf, sem).wait()
      o_ref[...] = jnp.zeros((vec,), jnp.int32)
      plsc.store_scatter(o_ref.at[pl.ds(0, vec)], [ibuf[...]], buf[...])

    idx = jnp.array(list(range(vec))[::-1], jnp.int32)
    out = run(
      body,
      (vec,),
      jnp.int32,
      x_i32,
      idx,
      scratch=[
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.SemaphoreType.DMA,
      ],
    )
    want = np.arange(vec)[::-1]
    return "ok" if np.array_equal(out, want) else f"ran,out={out.tolist()}"

  def _addupdate_scatter():
    def body(x_ref, idx_ref, o_ref, buf, ibuf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      pltpu.async_copy(idx_ref, ibuf, sem).wait()
      o_ref[...] = jnp.zeros((vec,), jnp.int32)
      digits = ibuf[...]
      # scan_count gives each lane its *1-based* occurrence count among
      # duplicates plus a mask selecting the last lane of each run, so
      # one scatter-add of `counts` at those lanes adds the whole run
      # without a race. `counts + 1` is the tempting spelling and is
      # wrong: it over-counts every run by one, so bucket 0 would read
      # `vec + 1`.
      counts, mask = plsc.scan_count(digits)
      plsc.addupdate_scatter(
        o_ref.at[pl.ds(0, vec)], [digits], counts, mask=mask
      )

    # Every index is 0, so a correct histogram puts `vec` in bucket 0.
    idx = jnp.zeros((vec,), jnp.int32)
    out = run(
      body,
      (vec,),
      jnp.int32,
      x_i32,
      idx,
      scratch=[
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.SemaphoreType.DMA,
      ],
    )
    return "ok(histogram)" if int(out[0]) == vec else f"ran,out={out.tolist()}"

  def _subcore_barrier_and_fetch_add():
    num_subcores = 8
    mesh = _vector_mesh(num_cores=1, num_subcores=num_subcores)

    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
      mesh=mesh,
      scratch_types=[
        pltpu.SMEM((1,), jnp.int32),
        # `(vec,)`: a scalar store into VMEM is a swap and is refused.
        pltpu.VMEM((vec,), jnp.int32),
      ],
    )
    def kernel(o_ref, tally, obuf):
      subcore = jax.lax.axis_index("subcore")

      @pl.when(subcore == 0)
      def _():
        tally[0] = jnp.int32(0)

      plsc.subcore_barrier()
      plsc.fetch_and_add(tally, jnp.int32(1), subcore_id=0)
      plsc.subcore_barrier()

      @pl.when(subcore == 0)
      def _():
        obuf[...] = jnp.full((vec,), tally[0], jnp.int32)
        pltpu.sync_copy(obuf, o_ref)

    out = np.asarray(jax.jit(kernel)())
    return (
      f"ok(subcores={num_subcores} each added 1, tally={int(out[0])})"
      if int(out[0]) == num_subcores
      else f"ran,tally={int(out[0])},want={num_subcores}"
    )

  def _load_expanded():
    def body(x_ref, o_ref, buf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      mask = (jnp.arange(vec) % 2) == 0
      o_ref[...] = plsc.load_expanded(buf.at[pl.ds(0, vec)], mask=mask)

    out = run(
      body,
      (vec,),
      jnp.int32,
      x_i32,
      scratch=[pltpu.VMEM((vec,), jnp.int32), pltpu.SemaphoreType.DMA],
    )
    # Even positions should hold ref[0], ref[1], ref[2], ...
    want = np.arange(vec // 2)
    ok = np.array_equal(out[::2], want)
    return "ok" if ok else f"ran,out={out.tolist()}"

  def _pack_unpack():
    def body(x_ref, o_ref, buf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      v = buf[...]
      packed = plsc.pack(
        v.astype(jnp.int32),
        v.astype(jnp.int32),
        format=plsc.PackFormat.INTERLEAVED,
        preferred_element_type=jnp.int16,
      )
      a, _b = plsc.unpack(
        packed,
        format=plsc.PackFormat.INTERLEAVED,
        preferred_element_type=jnp.int32,
      )
      o_ref[...] = a

    out = run(
      body,
      (vec,),
      jnp.int32,
      x_i32,
      scratch=[pltpu.VMEM((vec,), jnp.int32), pltpu.SemaphoreType.DMA],
    )
    return (
      "ok" if np.array_equal(out, np.arange(vec)) else f"ran,out={out.tolist()}"
    )

  def _bitcast():
    def body(x_ref, o_ref, buf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      o_ref[...] = plsc.bitcast(buf[...], jnp.float32).view(jnp.int32)

    out = run(
      body,
      (vec,),
      jnp.int32,
      x_i32,
      scratch=[pltpu.VMEM((vec,), jnp.int32), pltpu.SemaphoreType.DMA],
    )
    return (
      "ok" if np.array_equal(out, np.arange(vec)) else f"ran,out={out.tolist()}"
    )

  check("cumsum", _cumsum)
  check("cummax", _cummax)
  check("scan_count", _scan_count)
  check("sort_key_val", _sort_key_val)
  check("all_reduce_ffs+popcount", _all_reduce)
  check("store_compressed", _store_compressed)
  check("load_expanded", _load_expanded)
  check("load_gather", _load_gather)
  check("store_scatter", _store_scatter)
  check("addupdate_scatter", _addupdate_scatter)
  # `subcore_barrier+fetch_and_add` is deliberately not checked here. It can
  # hang the device, and a hang takes every other primitive in this probe
  # down with it, none of their verdicts reaching the log. It is checked by
  # `sc_subcore_barrier_minimal` in checks/sparsecore_gaps.py, which is
  # `destructive=True` and so runs in its own process.
  # `_subcore_barrier_and_fetch_add` above is not called; it shows the
  # shape of that check.
  check("pack/unpack", _pack_unpack)
  check("bitcast", _bitcast)

  return (
    f"vec_dim={vec}, one kernel per primitive, output staged through VMEM "
    f"(* = only lowers with needs_layout_passes=False): "
    f"{' | '.join(verdicts)}"
  )


@harness.probe(
  _GROUP,
  "sc_pl_ops_inside_kernel",
  "Which ordinary jnp/lax constructs survive inside a SparseCore kernel body?",
)
def _sc_pl_ops_inside_kernel() -> str:
  """SparseCore is not a small TensorCore. Anything assuming (8, 128) tiles,
  an MXU, or a reduction across lanes has to be checked rather than assumed."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  verdicts = []

  def run_once(compute, layout_passes):
    mesh = _vector_mesh(num_cores=1, num_subcores=1)

    def body(x_ref, o_ref, buf, obuf, sem):
      pltpu.async_copy(x_ref, buf, sem).wait()
      obuf[...] = compute(buf[...])
      pltpu.sync_copy(obuf, o_ref)

    kernel = pl.kernel(
      body,
      out_type=jax.ShapeDtypeStruct((vec,), jnp.float32),
      mesh=mesh,
      scratch_types=[
        pltpu.VMEM((vec,), jnp.float32),
        pltpu.VMEM((vec,), jnp.float32),
        pltpu.SemaphoreType.DMA,
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=layout_passes),
    )
    return np.asarray(jax.jit(kernel)(jnp.arange(vec, dtype=jnp.float32) + 1.0))

  def attempt(name, compute):
    # Retry the infer-vector-layout refusals with the flag their own
    # diagnostic names. Without this the probe cannot tell "absent from the
    # SparseCore" from "needs needs_layout_passes=False", and would record
    # ops as absent that had only been asked the wrong way.
    try:
      out = run_once(compute, True)
      verdicts.append(f"{name}=ok({out[0]:.3g})")
      return
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      first = error
      if "needs_layout_passes" not in str(error):
        message = str(error).strip().replace("\n", " ")[:120]
        verdicts.append(f"{name}=NO({type(error).__name__}: {message})")
        return
    try:
      out = run_once(compute, False)
      verdicts.append(f"{name}*=ok({out[0]:.3g})")
    except Exception as error:  # noqa: BLE001
      message = str(error).strip().replace("\n", " ")[:120]
      verdicts.append(
        f"{name}=NO(layout-pass refusal {type(first).__name__}, and "
        f"with the flag off {type(error).__name__}: {message})"
      )

  attempt("add", lambda v: v + 1.0)
  attempt("mul", lambda v: v * 2.0)
  attempt("div", lambda v: v / 2.0)
  attempt("exp", jnp.exp)
  attempt("log", jnp.log)
  attempt("rsqrt", jax.lax.rsqrt)
  attempt("tanh", jnp.tanh)
  attempt("select", lambda v: jnp.where(v > 2, v, -v))
  attempt("max_reduce", lambda v: jnp.full((vec,), jnp.max(v)))
  attempt("sum_reduce", lambda v: jnp.full((vec,), jnp.sum(v)))
  attempt("jnp.cumsum", lambda v: jnp.cumsum(v))
  attempt("sort", lambda v: jnp.sort(v))
  attempt("iota", lambda v: v + jnp.arange(vec, dtype=jnp.float32))
  attempt("bf16_cast", lambda v: v.astype(jnp.bfloat16).astype(jnp.float32))
  # A 1-D dot is the cheapest thing that reaches dot_general without also
  # needing a 2-D iota, which the SparseCore refuses first and which would
  # otherwise mask the answer to "is there a matmul at all".
  attempt("dot_general", lambda v: jnp.full((vec,), jnp.dot(v, v)))

  return (
    f"vec_dim={vec} f32 (* = only lowers with needs_layout_passes=False): "
    + " | ".join(verdicts)
  )


# --------------------------------------------------------------------------
# Scaling across subcores and cores
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_subcore_scaling",
  "Does SparseCore work scale with subcore and core count, and by how much?",
)
def _sc_subcore_scaling() -> str:
  import jax
  import jax.numpy as jnp

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  # Large enough that the all-cores, all-subcores arm (the fastest, and the
  # denominator of the speedup) has real device time to measure; the floor
  # guard below owns the verdict.
  n, chunk = 1 << 23, 2048
  x = jnp.arange(n, dtype=jnp.float32)

  arms = []
  for cores in (1, info.num_cores):
    for subcores in (1, 2, 4, 8, info.num_subcores):
      if subcores > info.num_subcores:
        continue
      kernel = jax.jit(
        _doubling_kernel(n, chunk, num_cores=cores, num_subcores=subcores)
      )
      timing = harness.time_call(lambda k=kernel: k(x), batch=4)
      arms.append((cores, subcores, timing))

  slowest = max(t.median for _, _, t in arms)
  # Guard the *fastest* arm: it is the one the speedup ratio divides by, and
  # the one that host dispatch reaches first. A slow arm clearing the floor
  # says nothing about it.
  harness.require_above_dispatch_floor(
    min(arms, key=lambda a: a[2].median)[2], what="fastest SC arm"
  )
  parts = [f"{c}c/{s}sc:{slowest / t.median:.2f}x" for c, s, t in arms]
  reference = min(arms, key=lambda a: a[2].median)[2]
  return (
    f"2*x over n={n} f32, chunk={chunk} per subcore, speedup relative to "
    f"the slowest arm: {' '.join(parts)} || slowest arm "
    f"{slowest * 1e6:.1f}us, fastest arm {reference}"
  )


@harness.probe(
  _GROUP,
  "sc_gather_vs_xla_gather",
  "Is a SparseCore gather faster than the XLA gather it would replace, and by how much?",
)
def _sc_gather_vs_xla_gather() -> str:
  """The canonical reason to reach for SparseCore. If this is not a win at
  the sizes a design cares about, none of the rest of this file matters."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  # The table is capped by per-subcore SC VMEM, so it is sized to half of
  # `SparseCoreInfo.vmem_capacity_bytes` (rounded down to a power of two);
  # measure the usable limit with `sc_vmem_per_subcore`. The index count is
  # what buys device time above the dispatch floor.
  table_rows = 1 << ((info.vmem_capacity_bytes // 8).bit_length() - 1)
  num_indices = 1 << 18
  chunk = 1024

  rng = np.random.default_rng(0)
  table = jnp.asarray(rng.standard_normal(table_rows), jnp.float32)
  indices = jnp.asarray(
    rng.integers(0, table_rows, size=num_indices), jnp.int32
  )

  mesh = _vector_mesh(num_cores=info.num_cores, num_subcores=info.num_subcores)
  workers = info.num_cores * info.num_subcores
  num_chunks = num_indices // chunk

  @pl.kernel(
    out_type=jax.ShapeDtypeStruct((num_indices,), jnp.float32),
    mesh=mesh,
    scratch_types=[
      pltpu.VMEM((table_rows,), jnp.float32),
      pltpu.VMEM((chunk,), jnp.int32),
      pltpu.VMEM((chunk,), jnp.float32),
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

      @plsc.parallel_loop(0, chunk, vec)
      def _(i):
        obuf[pl.ds(i, vec)] = plsc.load_gather(tbl, [ibuf[pl.ds(i, vec)]])

      pltpu.sync_copy(obuf, o_ref.at[pl.ds(c * chunk, chunk)])

  sc_fn = jax.jit(sc_gather)
  xla_fn = jax.jit(lambda t, i: t[i])

  want = np.asarray(xla_fn(table, indices))
  got = np.asarray(sc_fn(table, indices))
  if not np.array_equal(got, want):
    wrong = int((got != want).sum())
    raise harness.Invalid(
      f"SC gather disagrees with XLA on {wrong}/{num_indices} elements; "
      f"no timing is reportable from a kernel that computed the wrong thing."
    )

  sc_timing = harness.time_call(lambda: sc_fn(table, indices), batch=8)
  xla_timing = harness.time_above_floor(
    lambda: xla_fn(table, indices), what="XLA gather arm", batch=8
  )
  harness.require_above_dispatch_floor(xla_timing, what="XLA gather arm")

  ratio = xla_timing.median / sc_timing.median
  return (
    f"table={table_rows} f32 rows ({table_rows * 4 // 1024}KiB, resident in "
    f"SC VMEM), "
    f"{num_indices} random int32 indices, chunk={chunk}, "
    f"{info.num_cores}c x {info.num_subcores}sc: SC gather is {ratio:.2f}x "
    f"the XLA gather (>1 means SC wins). SC {sc_timing} || XLA {xla_timing}"
  )


# --------------------------------------------------------------------------
# Piping SparseCore and TensorCore together
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_tc_one_kernel_two_meshes",
  "Can one pl.kernel host a TensorCore mesh and a SparseCore mesh at once?",
)
def _sc_tc_one_kernel_two_meshes() -> str:
  """Named after the claim that it can. `pl.kernel` takes a *sequence* of
  meshes, which is the MPMD path, so the API shape suggests yes. The lowering
  is where the answer actually is."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  tc_mesh = pltpu.create_tensorcore_mesh("tc")
  sc_mesh = _vector_mesh(num_cores=1, num_subcores=1)

  def tc_body(x_ref, o_ref):
    o_ref[...] = x_ref[...] + 1.0

  def sc_body(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0

  try:
    kernel = pl.kernel(
      [tc_body, sc_body],
      out_type=(
        jax.ShapeDtypeStruct((8, 128), jnp.float32),
        jax.ShapeDtypeStruct((vec,), jnp.float32),
      ),
      mesh=[tc_mesh, sc_mesh],
    )
    jax.block_until_ready(
      jax.jit(kernel)(
        jnp.ones((8, 128), jnp.float32), jnp.ones((vec,), jnp.float32)
      )
    )
  except NotImplementedError as error:
    return (
      "UNSUPPORTED",
      f"pl.kernel(mesh=[TensorCoreMesh, VectorSubcoreMesh]) is refused: "
      f"{str(error).strip()[:200]}. TC and SC work cannot share one "
      f"Pallas kernel on this jax; they are two programs and the handoff "
      f"is at the XLA level.",
    )
  except Exception as error:  # noqa: BLE001 — any other refusal is a finding too.
    return (
      "UNSUPPORTED",
      f"{type(error).__name__}: {str(error).strip()[:250]}",
    )
  return (
    "FAIL",
    "A mixed TC+SC pl.kernel compiled and ran. This contradicts the "
    "NotImplementedError in pallas_call_registration.py; re-read the "
    "reference file, the restriction has been lifted.",
  )


@harness.probe(
  _GROUP,
  "sc_two_sparsecore_meshes_one_kernel",
  "Can one `pl.kernel` host both SparseCore meshes (scalar subcore and vector subcore) at once?",
)
def _sc_two_sparsecore_meshes_one_kernel() -> str:
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  # The compatibility check requires a shared core axis name and core count.
  scalar_mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1)
  vector_mesh = plsc.VectorSubcoreMesh(
    core_axis_name="core",
    subcore_axis_name="subcore",
    num_cores=1,
    num_subcores=1,
  )

  # Every body in an MPMD kernel is handed the whole ref list — both inputs,
  # both outputs, then the scratch — not just the ones it touches. Staging is
  # through SMEM because a ScalarSubcoreMesh cannot allocate plain VMEM and
  # the scratch list is shared between the two programs.
  def scalar_body(x1_ref, _x2_ref, o1_ref, _o2_ref, buf, sem):
    pltpu.async_copy(x1_ref, buf, sem).wait()

    @pl.loop(0, vec)
    def _(i):
      buf[i] = buf[i] + 1

    pltpu.async_copy(buf, o1_ref, sem).wait()

  def vector_body(_x1_ref, x2_ref, _o1_ref, o2_ref, buf, sem):
    pltpu.async_copy(x2_ref, buf, sem).wait()

    @pl.loop(0, vec)
    def _(i):
      buf[i] = buf[i] * 2

    pltpu.async_copy(buf, o2_ref, sem).wait()

  kernel = pl.kernel(
    [scalar_body, vector_body],
    out_type=(
      jax.ShapeDtypeStruct((vec,), jnp.int32),
      jax.ShapeDtypeStruct((vec,), jnp.int32),
    ),
    mesh=[scalar_mesh, vector_mesh],
    scratch_types=[pltpu.SMEM((vec,), jnp.int32), pltpu.SemaphoreType.DMA],
  )
  x = jnp.arange(vec, dtype=jnp.int32)
  a, b = jax.jit(kernel)(x, x)
  a, b = np.asarray(a), np.asarray(b)
  ok = np.array_equal(a, np.arange(vec) + 1) and np.array_equal(
    b, np.arange(vec) * 2
  )
  return (
    f"pl.kernel(mesh=[ScalarSubcoreMesh, VectorSubcoreMesh]) compiled and "
    f"ran, both results correct={ok}. Two SparseCore programs do share one "
    f"kernel launch; the axis names and core counts must agree."
  )


@harness.probe(
  _GROUP,
  "sc_tc_independent_overlap",
  "When independent SparseCore and TensorCore work is issued in one jit, does the SC time hide behind the TC time?",
)
def _sc_tc_independent_overlap() -> str:
  """The piping question, asked the only way this stack allows: two separate
  programs in one XLA computation. Full overlap reads as
  `both ~= max(sc, tc)`; serialization reads as `both ~= sc + tc`."""
  import jax
  import jax.numpy as jnp

  # Both arms are sized to carry real device time on their own, and to land
  # within a small factor of each other, since a matmul far larger or smaller
  # than its partner cannot show overlap either way. The floor guard below
  # owns the verdict if they do not.
  info = _sc().get_sparse_core_info()
  n, chunk = 1 << 24, 2048
  x = jnp.arange(n, dtype=jnp.float32)
  sc_kernel = _doubling_kernel(
    n, chunk, num_cores=info.num_cores, num_subcores=info.num_subcores
  )

  m = 8192
  a = jnp.ones((m, m), jnp.bfloat16)
  b = jnp.ones((m, m), jnp.bfloat16)

  sc_only = jax.jit(lambda x: sc_kernel(x))
  tc_only = jax.jit(lambda a, b: a @ b)

  @jax.jit
  def both(x, a, b):
    return sc_kernel(x), a @ b

  sc_timing = harness.time_above_floor(
    lambda: sc_only(x), what="SC arm", batch=4
  )
  tc_timing = harness.time_above_floor(
    lambda: tc_only(a, b), what="TC arm", batch=4
  )
  both_timing = harness.time_above_floor(
    lambda: both(x, a, b), what="combined arm", batch=4
  )
  for timing, what in (
    (sc_timing, "SC arm"),
    (tc_timing, "TC arm"),
    (both_timing, "combined arm"),
  ):
    harness.require_above_dispatch_floor(timing, what=what)

  serial = sc_timing.median + tc_timing.median
  parallel = max(sc_timing.median, tc_timing.median)
  overlap = (serial - both_timing.median) / max(serial - parallel, 1e-12)
  return (
    f"SC 2*x over n={n} f32 on {info.num_cores}c x {info.num_subcores}sc; "
    f"TC bf16 {m}x{m} matmul. "
    f"both/(sc+tc)={both_timing.median / serial:.3f} "
    f"both/max(sc,tc)={both_timing.median / parallel:.3f} "
    f"overlap_fraction={overlap:.2f} (1.0 = SC fully hidden behind TC, "
    f"0.0 = fully serialized) || SC {sc_timing} || TC {tc_timing} "
    f"|| both {both_timing}"
  )


@harness.probe(
  _GROUP,
  "sc_tc_dependent_handoff",
  "What does a SparseCore result cost to hand to the TensorCore in the same jit?",
)
def _sc_tc_dependent_handoff() -> str:
  """The overlap probe measures the best case: nothing depends on anything.
  A real pipeline is SC-gather then TC-matmul on the gathered rows, which
  cannot overlap within one step and pays a program switch instead. This is
  the number that decides whether an SC stage is worth putting in a chain."""
  import jax
  import jax.numpy as jnp

  # n must be at least m*m for the reshape below, and both arms must clear
  # the dispatch floor on their own.
  m = 4096
  n, chunk = m * m, 2048
  x = jnp.arange(n, dtype=jnp.float32)
  info = _sc().get_sparse_core_info()
  sc_kernel = _doubling_kernel(
    n, chunk, num_cores=info.num_cores, num_subcores=info.num_subcores
  )

  b = jnp.ones((m, m), jnp.bfloat16)

  def tc_stage(v):
    # Consume the SC output as the matmul LHS so the dependency is real.
    lhs = v[: m * m].reshape(m, m).astype(jnp.bfloat16)
    return lhs @ b

  sc_only = jax.jit(lambda x: sc_kernel(x))
  tc_only = jax.jit(lambda v: tc_stage(v))
  chained = jax.jit(lambda x: tc_stage(sc_kernel(x)))

  sc_out = sc_only(x)
  sc_timing = harness.time_above_floor(
    lambda: sc_only(x), what="SC stage", batch=4
  )
  tc_timing = harness.time_above_floor(
    lambda: tc_only(sc_out), what="TC stage", batch=8
  )
  chain_timing = harness.time_above_floor(
    lambda: chained(x), what="chained", batch=4
  )
  for timing, what in (
    (sc_timing, "SC stage"),
    (tc_timing, "TC stage"),
    (chain_timing, "chained"),
  ):
    harness.require_above_dispatch_floor(timing, what=what)

  serial = sc_timing.median + tc_timing.median
  return (
    f"SC 2*x over n={n} f32 on {info.num_cores}c x {info.num_subcores}sc, "
    f"then a bf16 {m}x{m} matmul that "
    f"consumes its output. chained/(sc+tc)={chain_timing.median / serial:.3f} "
    f"(1.0 = the handoff is free and the two stages simply add; >1 = the "
    f"program switch costs extra) || SC {sc_timing} || TC {tc_timing} "
    f"|| chained {chain_timing}"
  )


# --------------------------------------------------------------------------
# What is worth offloading: SparseCore against the unit it would relieve
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "sc_vs_tc_streaming_bandwidth",
  "On a purely memory-bound stream, what fraction of HBM bandwidth does SparseCore reach against the TensorCore?",
)
def _sc_vs_tc_streaming_bandwidth() -> str:
  """The first question any offload decision needs. If SparseCore cannot pull
  a stream at a comparable rate, moving bandwidth-bound work there does not
  relieve the TensorCore, it just makes the work slower."""
  import jax
  import jax.numpy as jnp

  # 128 MiB per array. The TC arm is a bare `v * 2.0`, so it is fast enough
  # that only a large n plus a large batch keeps it off the dispatch floor.
  n, chunk = 1 << 25, 2048
  x = jnp.arange(n, dtype=jnp.float32)
  bytes_moved = 2 * n * 4  # one read, one write

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  sc_fn = jax.jit(
    _doubling_kernel(
      n, chunk, num_cores=info.num_cores, num_subcores=info.num_subcores
    )
  )
  tc_fn = jax.jit(lambda v: v * 2.0)

  sc_timing = harness.time_above_floor(
    lambda: sc_fn(x), what="SC stream arm", batch=4
  )
  tc_timing = harness.time_above_floor(
    lambda: tc_fn(x), what="TC stream arm", batch=32
  )
  harness.require_above_dispatch_floor(sc_timing, what="SC stream arm")
  harness.require_above_dispatch_floor(tc_timing, what="TC stream arm")

  # `mem_bw_bytes_per_second` from get_tpu_info(), reported not assumed.
  from jax.experimental.pallas import tpu as pltpu

  peak = pltpu.get_tpu_info().mem_bw_bytes_per_second
  sc_bw = bytes_moved / sc_timing.median
  tc_bw = bytes_moved / tc_timing.median
  return (
    f"2*x over n={n} f32 ({bytes_moved / 2**20:.0f} MiB moved), "
    f"{info.num_cores}c x {info.num_subcores}sc chunk={chunk}: "
    f"SC {sc_bw / 1e9:.0f} GB/s ({100 * sc_bw / peak:.1f}% of the reported "
    f"{peak / 1e12:.2f} TB/s peak), XLA-on-TC {tc_bw / 1e9:.0f} GB/s "
    f"({100 * tc_bw / peak:.1f}%). SC/TC bandwidth ratio="
    f"{sc_bw / tc_bw:.2f}x || SC {sc_timing} || TC {tc_timing}"
  )


@harness.probe(
  _GROUP,
  "sc_vs_tc_transcendental",
  "On VPU-shaped elementwise math, how does SparseCore compare with the TensorCore VPU?",
)
def _sc_vs_tc_transcendental() -> str:
  """A stream of `exp` is the shape of work that saturates the VPU rather
  than HBM. Offloading it to SparseCore only helps if the SC does it at a
  rate worth the transfer."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  n, chunk = 1 << 23, 2048
  depth = 8  # dependent exps per element, to lift it clear of HBM
  workers = info.num_cores * info.num_subcores
  num_chunks = n // chunk

  mesh = _vector_mesh(num_cores=info.num_cores, num_subcores=info.num_subcores)

  @pl.kernel(
    out_type=jax.ShapeDtypeStruct((n,), jnp.float32),
    mesh=mesh,
    scratch_types=[pltpu.VMEM((chunk,), jnp.float32), pltpu.SemaphoreType.DMA],
  )
  def sc_exp(x_ref, o_ref, buf, sem):
    worker = jax.lax.axis_index(
      "core"
    ) * info.num_subcores + jax.lax.axis_index("subcore")

    @pl.loop(worker, num_chunks, step=workers)
    def _(c):
      pltpu.async_copy(x_ref.at[pl.ds(c * chunk, chunk)], buf, sem).wait()

      @plsc.parallel_loop(0, chunk, vec)
      def _(i):
        v = buf[pl.ds(i, vec)]
        for _ in range(depth):
          v = jnp.exp(v * 0.5)
        buf[pl.ds(i, vec)] = v

      pltpu.sync_copy(buf, o_ref.at[pl.ds(c * chunk, chunk)])

  def tc_exp(v):
    for _ in range(depth):
      v = jnp.exp(v * 0.5)
    return v

  x = jnp.full((n,), 0.5, jnp.float32)
  sc_fn, tc_fn = jax.jit(sc_exp), jax.jit(tc_exp)

  got, want = np.asarray(sc_fn(x)), np.asarray(tc_fn(x))
  if not np.allclose(got, want, rtol=1e-3, atol=1e-5):
    raise harness.Invalid(
      f"SC exp chain disagrees with XLA (max abs diff "
      f"{np.max(np.abs(got - want)):.3g}); no timing is reportable."
    )

  sc_timing = harness.time_above_floor(
    lambda: sc_fn(x), what="SC exp arm", batch=4
  )
  tc_timing = harness.time_above_floor(
    lambda: tc_fn(x), what="TC exp arm", batch=16
  )
  harness.require_above_dispatch_floor(sc_timing, what="SC exp arm")
  harness.require_above_dispatch_floor(tc_timing, what="TC exp arm")

  return (
    f"{depth} dependent exp(0.5*v) over n={n} f32, "
    f"{info.num_cores}c x {info.num_subcores}sc chunk={chunk}: "
    f"TC/SC={tc_timing.median / sc_timing.median:.3f} (>1 means SC is "
    f"faster). SC {sc_timing} || XLA-on-TC {tc_timing}"
  )


@harness.probe(
  _GROUP,
  "sc_scatter_add_vs_xla",
  "Is a SparseCore scatter-add faster than the XLA scatter-add it would replace?",
)
def _sc_scatter_add_vs_xla() -> str:
  """Histogram / segment-sum / embedding-gradient shape -- the second canonical
  SparseCore workload after gather.

  Hypothesis: read/modify/write does not behave like read. The accumulation
  serialises through `scan_count`, and XLA's dense scatter-add may be a
  well-optimised TensorCore path rather than a slow gather-shaped one, so a
  SparseCore win on gather (`sc_gather_vs_xla_gather`) does not imply a win
  here. Compare the two probes before assuming "SparseCore is good at
  data-dependent addressing" covers scatter-add."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  buckets, num_updates, chunk = 4096, 1 << 18, 2048

  rng = np.random.default_rng(1)
  indices = jnp.asarray(rng.integers(0, buckets, size=num_updates), jnp.int32)

  # One subcore owns the whole histogram: a cross-subcore reduction is a
  # different measurement and would confound this one.
  mesh = _vector_mesh(num_cores=1, num_subcores=1)
  num_chunks = num_updates // chunk

  @pl.kernel(
    out_type=jax.ShapeDtypeStruct((buckets,), jnp.int32),
    mesh=mesh,
    scratch_types=[
      pltpu.VMEM((buckets,), jnp.int32),
      pltpu.VMEM((chunk,), jnp.int32),
      pltpu.SemaphoreType.DMA,
    ],
    compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
  )
  def sc_hist(idx_ref, o_ref, hist, ibuf, sem):
    @plsc.parallel_loop(0, buckets, vec)
    def _(i):
      hist[pl.ds(i, vec)] = jnp.zeros(vec, jnp.int32)

    @pl.loop(0, num_chunks)
    def _(c):
      pltpu.async_copy(idx_ref.at[pl.ds(c * chunk, chunk)], ibuf, sem).wait()

      @plsc.parallel_loop(0, chunk, vec)
      def _(i):
        digits = ibuf[pl.ds(i, vec)]
        counts, mask = plsc.scan_count(digits)
        plsc.addupdate_scatter(hist, [digits], counts, mask=mask)

    pltpu.sync_copy(hist, o_ref)

  def xla_hist(idx):
    return jnp.zeros((buckets,), jnp.int32).at[idx].add(1)

  sc_fn, xla_fn = jax.jit(sc_hist), jax.jit(xla_hist)
  got, want = np.asarray(sc_fn(indices)), np.asarray(xla_fn(indices))
  if not np.array_equal(got, want):
    wrong = int((got != want).sum())
    raise harness.Invalid(
      f"SC histogram disagrees with XLA in {wrong}/{buckets} buckets "
      f"(SC total {got.sum()} vs {want.sum()}); no timing is reportable."
    )

  # Asymmetric batches on purpose. The XLA arm can be launch-bound, with a
  # per-call cost that falls as the batch grows, so a fixed batch guess is
  # unreliable; `time_above_floor` re-times it at a larger batch if needed.
  # The SC arm is a single-subcore loop over every update and carries far
  # more device time per call.
  sc_timing = harness.time_call(lambda: sc_fn(indices), batch=4)
  xla_timing = harness.time_above_floor(
    lambda: xla_fn(indices), what="XLA scatter-add arm", batch=48
  )
  harness.require_above_dispatch_floor(xla_timing, what="XLA scatter-add arm")

  return (
    f"{num_updates} int32 updates into {buckets} buckets, one subcore, "
    f"chunk={chunk}: XLA/SC={xla_timing.median / sc_timing.median:.2f}x "
    f"(>1 means SC wins). SC {sc_timing} || XLA {xla_timing}"
  )


@harness.probe(
  _GROUP,
  "sc_data_dependent_control_flow",
  "Can a SparseCore kernel do data-dependent work, such as a while loop whose trip count comes from the data?",
)
def _sc_data_dependent_control_flow() -> str:
  """The structural reason SparseCore exists. A TensorCore kernel pays for the
  worst case of every lane; if SC can branch on the data, ragged and
  early-exit work belongs there even when the raw rates are unflattering."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  mesh = _vector_mesh(num_cores=1, num_subcores=1)

  @pl.kernel(
    out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
    mesh=mesh,
    scratch_types=[
      pltpu.VMEM((vec,), jnp.int32),
      pltpu.VMEM((vec,), jnp.int32),
      pltpu.SemaphoreType.DMA,
    ],
  )
  def kernel(x_ref, o_ref, buf, obuf, sem):
    pltpu.async_copy(x_ref, buf, sem).wait()

    # Trip count is read out of the data, not known at trace time.
    def cond(carry):
      i, _acc = carry
      return i < buf[...][0]

    def body(carry):
      i, acc = carry
      return i + 1, acc + 1

    _i, acc = jax.lax.while_loop(cond, body, (jnp.int32(0), jnp.int32(0)))
    obuf[...] = jnp.full((vec,), acc, jnp.int32)
    pltpu.sync_copy(obuf, o_ref)

  trips = 37
  x = jnp.full((vec,), trips, jnp.int32)
  out = np.asarray(jax.jit(kernel)(x))
  if int(out[0]) != trips:
    return (
      "FAIL",
      f"data-dependent while_loop ran {int(out[0])} times, want {trips}",
    )
  return (
    f"jax.lax.while_loop with a trip count read from VMEM ran exactly "
    f"{trips} iterations inside a VectorSubcoreMesh kernel. Data-dependent "
    f"control flow works on SparseCore."
  )


@harness.probe(
  _GROUP,
  "sc_gather_row_width_scaling",
  "How does SparseCore gather throughput depend on the width of the row gathered, relative to the DMA granule `SparseCoreInfo` reports?",
)
def _sc_gather_row_width_scaling() -> str:
  """Embedding-lookup shape. A gather of narrow rows is granule-bound, not
  bandwidth-bound, and the width where that stops is the number that decides
  whether a table is worth putting on SparseCore at all."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  # Enough indices that each arm carries real device time, matching
  # `sc_gather_vs_xla_gather`; too few and every arm is round-trip overhead,
  # which the floor guard below turns into INVALID.
  num_indices = 1 << 18
  workers = info.num_cores * info.num_subcores
  # Budget per-subcore VMEM from SparseCoreInfo: half for the table, a
  # quarter for the output staging buffer (powers of two, rounded down).
  table_elems = 1 << ((info.vmem_capacity_bytes // 8).bit_length() - 1)
  obuf_bytes = 1 << ((info.vmem_capacity_bytes // 4).bit_length() - 1)

  arms = []
  slowest_floor_candidate = None
  rng = np.random.default_rng(2)
  for width in (1, 4, 16, 64, 256):
    # obuf is (width, chunk) f32 and has to share subcore VMEM with the
    # table; hold it at `obuf_bytes` by shrinking chunk as the rows widen,
    # rather than letting the widest arm ask for more than the subcore has.
    chunk = min(1024, obuf_bytes // (4 * width))
    num_chunks = num_indices // chunk
    # Hold the table at a fixed byte size so every arm has the same VMEM
    # residency; only the row width — and so the granule count — changes.
    # The table must fit in per-subcore VMEM; `sc_vmem_per_subcore` measures
    # how much of `vmem_capacity_bytes` is usable.
    table_rows = table_elems // width
    table = jnp.asarray(rng.standard_normal((table_rows, width)), jnp.float32)
    # **The gather table is kept genuinely 1-D.** Two 2-D spellings can be
    # refused. `tbl.at[:, col]` on a (rows, width) ref is an integer index
    # applied after a slice:
    #   NotImplementedError: Integer indexing of refs that follows a
    #   non-trivial slice is not supported on SC
    # Transposing to (width, rows) puts the integer first, `tbl.at[col]`,
    # which can hit the next check:
    #   NotImplementedError: Cannot confirm that the memref is memory
    #   tile-aligned since its producer op tpu.memref_squeeze is not
    #   supported by this check
    # Rank reduction of a 2-D ref is the problem in both, not the index
    # order (toolchain-dependent). So flatten the table and do the
    # addressing in the index vector: `row * width + col` over a 1-D ref,
    # which is also the real memory layout of an embedding table and so the
    # access pattern whose granule behaviour is the question.
    flat = jnp.asarray(np.asarray(table).reshape(-1).copy())
    indices = jnp.asarray(
      rng.integers(0, table_rows, size=num_indices), jnp.int32
    )
    mesh = _vector_mesh(
      num_cores=info.num_cores, num_subcores=info.num_subcores
    )

    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((width, num_indices), jnp.float32),
      mesh=mesh,
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
        arms.append(f"w={width}:WRONG")
        continue
      sc_timing = harness.time_above_floor(
        lambda: sc_fn(flat, indices), what=f"SC gather w={width}", batch=16
      )
      xla_timing = harness.time_call(lambda: xla_fn(table, indices), batch=16)
      # The granule question is about narrow rows, so the narrowest arm
      # that lowered is the one that has to be a real measurement. Keep
      # the fastest SC timing and hold the whole probe to it below.
      if (
        slowest_floor_candidate is None
        or sc_timing.median < slowest_floor_candidate[1].median
      ):
        slowest_floor_candidate = (f"w={width}", sc_timing)
      gathered = num_indices * width * 4
      # Carry `chunk`: it is not constant across arms (VMEM forces it
      # down as the rows widen), so it is a confound in the curve and a
      # reader has to be able to see it.
      arms.append(
        f"w={width}(chunk={chunk}):"
        f"{gathered / sc_timing.median / 1e9:.0f}GB/s "
        f"{sc_timing.median * 1e6:.1f}us "
        f"(XLA/SC={xla_timing.median / sc_timing.median:.2f}x)"
      )
    except Exception as error:  # noqa: BLE001 — a refusal at a width is a finding.
      # Carry the message, not just the class: `NotImplementedError` alone
      # cannot distinguish "this width is too narrow" from "this kernel
      # never lowered at any width".
      arms.append(f"w={width}:NO({type(error).__name__}: {error})")

  if all(":NO(" in arm for arm in arms):
    # Every arm failed, so there is no scaling curve here and a PASS would
    # be a green verdict over an empty measurement — the failure mode that
    # nobody re-reads. Report the refusal as the result it is.
    raise harness.Unsupported(
      "no width lowered, so the granule curve is unmeasured: "
      + " | ".join(arms)
    )

  if slowest_floor_candidate is not None:
    label, timing = slowest_floor_candidate
    harness.require_above_dispatch_floor(
      timing, what=f"the fastest SC gather arm ({label})"
    )

  return (
    f"{num_indices} random rows gathered from a {table_elems * 4 // 1024}KiB "
    f"f32 table held in SC "
    f"VMEM, {info.num_cores}c x {info.num_subcores}sc, "
    f"dma_granule_size_bytes={info.dma_granule_size_bytes}: "
    f"{' | '.join(arms)}"
  )


@harness.probe(
  _GROUP,
  "sc_smem_capacity",
  "How much SMEM can a SparseCore kernel allocate, and what dtypes does it hold?",
)
def _sc_smem_capacity() -> str:
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  results = []
  last_ok = 0
  for words in (64, 256, 1024, 4096, 16384, 65536):
    try:
      mesh = _vector_mesh(num_cores=1, num_subcores=1)

      @pl.kernel(
        out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
        mesh=mesh,
        scratch_types=[
          pltpu.SMEM((words,), jnp.int32),
          pltpu.VMEM((vec,), jnp.int32),
        ],
      )
      def kernel(o_ref, smem, obuf, _w=words):
        smem[_w - 1] = jnp.int32(7)
        obuf[...] = jnp.full((vec,), smem[_w - 1], jnp.int32)
        pltpu.sync_copy(obuf, o_ref)

      jax.block_until_ready(jax.jit(kernel)())
      results.append(f"{words}w=ok")
      last_ok = words
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      message = str(error).strip().replace("\n", " ")[:90]
      results.append(f"{words}w=NO({type(error).__name__}: {message})")

  dtypes = []
  for name in ("int32", "float32", "int8", "bfloat16"):
    try:
      mesh = _vector_mesh(num_cores=1, num_subcores=1)
      dtype = jnp.dtype(name)

      @pl.kernel(
        out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
        mesh=mesh,
        scratch_types=[
          pltpu.SMEM((8,), dtype),
          pltpu.VMEM((vec,), jnp.int32),
        ],
      )
      def kernel(o_ref, smem, obuf, _d=dtype):
        smem[0] = _d.type(1)
        obuf[...] = jnp.full((vec,), smem[0].astype(jnp.int32), jnp.int32)
        pltpu.sync_copy(obuf, o_ref)

      jax.block_until_ready(jax.jit(kernel)())
      dtypes.append(f"{name}=ok")
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      message = str(error).strip().replace("\n", " ")[:90]
      dtypes.append(f"{name}=NO({type(error).__name__}: {message})")

  return (
    f"int32 SMEM scratch on one vector subcore: {' '.join(results)} "
    f"|| largest accepted={last_ok} words ({last_ok * 4 / 1024:.0f}KiB) "
    f"|| dtypes: {' '.join(dtypes)}"
  )


@harness.probe(
  _GROUP,
  "sc_multi_subcore_reduction",
  "Does a cross-subcore reduction built from subcore_barrier and fetch_and_add work, and how does it scale?",
  # A subcore_barrier deadlock hangs the device and produces an empty log, so
  # this probe runs in its own process, never beside probes whose results are
  # wanted. `sc_subcore_barrier_minimal` isolates which call pattern hangs.
  destructive=True,
)
def _sc_multi_subcore_reduction() -> str:
  """The blocker for every fanned-out SparseCore kernel.

  Both worked kernels in `sc-kernels` use one vector subcore, and
  `sc_subcore_scaling` measures what that leaves on the table. Fanning them
  out needs a cross-subcore combine, and the only two primitives for one are
  `subcore_barrier` and `fetch_and_add`. The swap target must be a scalar SMEM
  ref: a VMEM target is refused with `Swap only supports scalars in SMEM`,
  which is a statement about the kernel, not about the primitives.

  This runs the smallest kernel that genuinely needs both. Every subcore sums
  a stripe into its own SMEM slot, barriers, and folds that total into
  subcore 0's SMEM with `fetch_and_add`; subcore 0 emits. Correctness is
  checked against `np.sum`, so a barrier that does not actually order the
  initialization against the adds shows up as a wrong total rather than as a
  plausible one.

  The scaling arm is what `sc_subcore_scaling` cannot supply. That probe is
  embarrassingly-parallel streaming with no combine; this has a serialized
  fold at the end, and the gap between the two curves is the cost of
  coordinating.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  info = plsc.get_sparse_core_info()
  vec = info.num_lanes
  n = 1 << 22
  chunk = 4096
  num_chunks = n // chunk
  # Small values: the fold is int32 and 4M elements of anything larger
  # overflows, which would read as a barrier bug rather than as arithmetic.
  rng = np.random.default_rng(23)
  host = rng.integers(0, 4, size=n).astype(np.int32)
  x = jnp.asarray(host)
  want = int(host.sum())

  _LOCAL, _GLOBAL = 0, 1
  arms = []
  baseline = None
  best = None
  for num_subcores in (1, 2, 4, 8, info.num_subcores):
    mesh = _vector_mesh(num_cores=1, num_subcores=num_subcores)

    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
      mesh=mesh,
      scratch_types=[
        pltpu.VMEM((chunk,), jnp.int32),
        # `fetch_and_add` targets a scalar SMEM ref specifically, and
        # VMEM is not an alternative. Slot 0 is this subcore's own
        # running total; slot 1 is only ever read on subcore 0, where
        # it is the accumulator every subcore adds into.
        pltpu.SMEM((2,), jnp.int32),
        # `(vec,)`, not `(1,)`: a scalar store into a VMEM ref is a
        # swap and is refused. Every emit is a full lane group.
        pltpu.VMEM((vec,), jnp.int32),
        pltpu.SemaphoreType.DMA,
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def kernel(x_ref, o_ref, buf, acc, obuf, sem, _ns=num_subcores):
      subcore = jax.lax.axis_index("subcore")
      acc[_LOCAL] = jnp.int32(0)
      acc[_GLOBAL] = jnp.int32(0)
      # Without this barrier a fast subcore could add into subcore 0's
      # accumulator before subcore 0 has zeroed it. The docstring on
      # `fetch_and_add` names exactly this hazard.
      plsc.subcore_barrier()

      # The running sum lives in SMEM rather than in a `pl.loop`
      # `init_carry` because `fetch_and_add` needs it in SMEM anyway --
      # it targets a scalar SMEM ref on the destination subcore.
      @pl.loop(subcore, num_chunks, step=_ns)
      def _(c):
        pltpu.async_copy(x_ref.at[pl.ds(c * chunk, chunk)], buf, sem).wait()

        # A whole-buffer read may not lower here. The SC vector unit
        # is `num_lanes` wide and a `Get` can be refused at any other
        # shape (`NotImplementedError: Get only supports int32 arrays
        # of shapes [(num_lanes,)]`; see `sc_wide_get_layout_passes`).
        # Reduce a lane group at a time. This is `pl.loop`, not `parallel_loop`,
        # because the iterations share the SMEM accumulator.
        @pl.loop(0, chunk, step=vec)
        def _(i):
          acc[_LOCAL] = acc[_LOCAL] + jnp.sum(buf[pl.ds(i, vec)])

      plsc.subcore_barrier()
      plsc.fetch_and_add(acc.at[_GLOBAL], acc[_LOCAL], subcore_id=0)
      plsc.subcore_barrier()

      @pl.when(subcore == 0)
      def _():
        obuf[...] = jnp.full((vec,), acc[_GLOBAL], jnp.int32)
        pltpu.sync_copy(obuf, o_ref)

    try:
      fn = jax.jit(kernel)
      got = int(np.asarray(fn(x))[0])
      if got != want:
        arms.append(f"{num_subcores}sc:WRONG(got={got},want={want})")
        continue
      timing = harness.time_call(lambda: fn(x), batch=16)
      if num_subcores == 1:
        baseline = timing
      best = timing
      speedup = (
        f" {baseline.median / timing.median:.1f}x"
        if baseline is not None
        else ""
      )
      arms.append(
        f"{num_subcores}sc:{timing.median * 1e6:.1f}us "
        f"{n * 4 / timing.median / 1e9:.0f}GB/s{speedup}"
      )
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      message = str(error).strip().replace("\n", " ")[:180]
      arms.append(f"{num_subcores}sc:NO({type(error).__name__}: {message})")

  if all(":NO(" in arm or ":WRONG(" in arm for arm in arms):
    raise harness.Unsupported(
      "no cross-subcore reduction ran, so fan-out is still unevidenced: "
      + " | ".join(arms)
    )
  if best is not None:
    harness.require_above_dispatch_floor(
      best, what="the widest cross-subcore reduction"
    )

  return (
    f"sum of {n} int32 ({n * 4 / (1 << 20):.0f} MiB) folded across subcores "
    f"with subcore_barrier + fetch_and_add into SMEM, 1 core: "
    f"{' | '.join(arms)}"
  )


@harness.probe(
  _GROUP,
  "sc_pl_loop_init_carry",
  "Does pl.loop carry a value between iterations on the SparseCore, or does a running scalar have to live in SMEM?",
)
def _sc_pl_loop_init_carry() -> str:
  """Checks whether `pl.loop(init_carry=...)` works on the SparseCore.

  It is tempting to assume `pl.loop` "carries nothing" and route every
  running scalar through SMEM, as both worked kernels in `sc-kernels` do.
  `pl.loop` takes `init_carry` and returns the final carry, with the shape

      @pl.loop(0, n, step=num_lanes, init_carry=(value,))
      def body(i, carry): ...

  but the signature is not evidence that it lowers on SC. Three arms, because
  the useful answer is not yes/no:

  * `scalar` -- an int32 carry, the case the SMEM workaround exists for.
  * `vector` -- a `(num_lanes,)` carry, which is what would let an
    accumulation stay in registers instead of round-tripping VMEM.
  * `smem` -- the SMEM spelling the kernels use today, as the control.

  All three compute the same sum, so a carry that lowers but drops values
  shows up as a wrong answer rather than as a pass.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  iters = 64
  want = iters * (iters - 1) // 2

  def _run(build):
    out = np.asarray(jax.jit(build())())
    return int(out[0])

  def _scalar_carry():
    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
      mesh=_vector_mesh(num_cores=1, num_subcores=1),
      scratch_types=[pltpu.VMEM((vec,), jnp.int32)],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def kernel(o_ref, obuf):
      @pl.loop(0, iters, init_carry=(jnp.int32(0),))
      def total(i, carry):
        return (carry[0] + i,)

      obuf[...] = jnp.full((vec,), total[0], jnp.int32)
      pltpu.sync_copy(obuf, o_ref)

    return kernel

  def _vector_carry():
    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
      mesh=_vector_mesh(num_cores=1, num_subcores=1),
      scratch_types=[pltpu.VMEM((vec,), jnp.int32)],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def kernel(o_ref, obuf):
      @pl.loop(0, iters, init_carry=(jnp.zeros((vec,), jnp.int32),))
      def total(i, carry):
        return (carry[0] + jnp.full((vec,), i, jnp.int32),)

      obuf[...] = total[0]
      pltpu.sync_copy(obuf, o_ref)

    return kernel

  def _smem_carry():
    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((vec,), jnp.int32),
      mesh=_vector_mesh(num_cores=1, num_subcores=1),
      scratch_types=[
        pltpu.SMEM((1,), jnp.int32),
        pltpu.VMEM((vec,), jnp.int32),
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def kernel(o_ref, smem, obuf):
      smem[0] = jnp.int32(0)

      @pl.loop(0, iters)
      def _(i):
        smem[0] = smem[0] + i

      obuf[...] = jnp.full((vec,), smem[0], jnp.int32)
      pltpu.sync_copy(obuf, o_ref)

    return kernel

  results = []
  for name, build in (
    ("scalar", _scalar_carry),
    ("vector", _vector_carry),
    ("smem", _smem_carry),
  ):
    try:
      got = _run(build)
      results.append(
        f"{name}=ok({got})"
        if got == want
        else f"{name}=WRONG(got={got},want={want})"
      )
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      message = str(error).strip().replace("\n", " ")[:150]
      results.append(f"{name}=NO({type(error).__name__}: {message})")

  if all("=NO(" in r for r in results):
    # Every arm failing, the SMEM control included, is a defect in the probe
    # (for example a scalar store into the VMEM output buffer, refused with
    # `Swap only supports scalars in SMEM`), not a finding about
    # `init_carry`. A verdict comes from how many arms produced a number.
    raise harness.Unsupported(
      "no loop-carry spelling ran, including the SMEM control, so this "
      "says nothing about init_carry: " + " | ".join(results)
    )

  return (
    f"pl.loop over {iters} iterations summing 0..{iters - 1} (want {want}), "
    f"one vector subcore: {' | '.join(results)}"
  )


@harness.probe(
  _GROUP,
  "sc_bf16_widen_and_pack",
  "How does a SparseCore kernel consume and produce bf16, on a unit whose arithmetic is f32?",
)
def _sc_bf16_widen_and_pack() -> str:
  """The idiom every bf16 tensor on this unit has to go through.

  `sc_pl_ops_inside_kernel` says bf16 survives a VMEM round trip, which is
  not the same as being computable. The SparseCore's arithmetic is f32/int32,
  so a bf16 tensor is carried as uint32 words holding two values each, and
  the widen is done in the bit pattern: bf16 *is* the top 16 bits of the
  corresponding f32, so shifting the wanted half up and masking the low 16
  bits off gives an f32 with no conversion instruction at all. Coming back
  out, `plsc.pack(..., INTERLEAVED)` re-pairs two f32 vectors.

  This idiom appears in production SparseCore kernels, but reading it in a
  source file is not evidence that it lowers, and it does not say which
  element of the pair `INTERLEAVED` puts where. The probe checks both orderings and reports which one is
  right, since guessing it wrong produces a plausible-looking tensor with
  every pair transposed.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  plsc = _sc()
  vec = plsc.get_sparse_core_info().num_lanes
  pairs = vec * 64
  n = pairs * 2

  rng = np.random.default_rng(31)
  host_bf16 = jnp.asarray(rng.standard_normal(n).astype(np.float32)).astype(
    jnp.bfloat16
  )
  # Two bf16 per uint32 word, in memory order: on a little-endian host the
  # even element lands in the low 16 bits.
  host = np.asarray(jax.device_get(host_bf16))
  words = jnp.asarray(host.view(np.uint32))
  want_words = jnp.asarray(host.astype(np.float32) * 2.0).astype(jnp.bfloat16)
  want_words = np.asarray(jax.device_get(want_words)).view(np.uint32)

  _HIGH_HALF_MASK = jnp.uint32(0xFFFF0000)

  def _build(swap: bool):
    @pl.kernel(
      out_type=jax.ShapeDtypeStruct((pairs,), jnp.uint32),
      mesh=_vector_mesh(num_cores=1, num_subcores=1),
      scratch_types=[
        pltpu.VMEM((pairs,), jnp.uint32),
        pltpu.VMEM((pairs,), jnp.uint32),
        pltpu.SemaphoreType.DMA,
      ],
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def kernel(x_ref, o_ref, xbuf, obuf, sem):
      pltpu.async_copy(x_ref, xbuf, sem).wait()

      @plsc.parallel_loop(0, pairs, vec)
      def _(i):
        w = xbuf[pl.ds(i, vec)]
        # bf16 is the high 16 bits of the f32 with the same value, so
        # the widen is a shift and a mask -- no convert instruction.
        first = plsc.bitcast(
          jnp.bitwise_and(jnp.left_shift(w, 16), _HIGH_HALF_MASK),
          jnp.float32,
        )
        second = plsc.bitcast(jnp.bitwise_and(w, _HIGH_HALF_MASK), jnp.float32)
        first = first * 2.0
        second = second * 2.0
        lo, hi = (second, first) if swap else (first, second)
        obuf[pl.ds(i, vec)] = plsc.bitcast(
          plsc.pack(lo, hi, format=plsc.PackFormat.INTERLEAVED),
          jnp.uint32,
        )

      pltpu.sync_copy(obuf, o_ref)

    return kernel

  results = []
  matched = None
  for swap in (False, True):
    label = "pack(second,first)" if swap else "pack(first,second)"
    try:
      got = np.asarray(jax.jit(_build(swap))(words))
      if np.array_equal(got, want_words):
        results.append(f"{label}=ok")
        matched = label
      else:
        bad = int((got != want_words).sum())
        results.append(f"{label}=MISMATCH({bad}/{pairs} words)")
    except Exception as error:  # noqa: BLE001 — the refusal is the finding.
      message = str(error).strip().replace("\n", " ")[:150]
      results.append(f"{label}=NO({type(error).__name__}: {message})")

  if matched is None:
    raise harness.Unsupported(
      "no bf16 widen/pack ordering reproduced the reference, so the "
      "idiom is unestablished: " + " | ".join(results)
    )

  return (
    f"{n} bf16 widened to f32 via shift+mask+bitcast, scaled by 2, packed "
    f"back with plsc.pack(INTERLEAVED), one vector subcore: "
    f"{' | '.join(results)} || correct ordering is {matched}, where "
    f"`first` is the low 16 bits of the uint32 word"
  )
