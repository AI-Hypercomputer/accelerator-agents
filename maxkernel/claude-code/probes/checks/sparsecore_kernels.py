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
"""Two whole SparseCore kernels, timed against the TensorCore doing the same work.

Supplies answers for `reference/design/sparsecore.md`. The `sparsecore` group answers
"what reaches the SparseCore and what does one primitive cost"; this group
answers the question that actually decides an offload: **when you move a real
algorithm off the TensorCore, do you win?**

Two algorithms, both picked because they are the shape SparseCore is built for
— data-dependent indexing over a large array, with almost no arithmetic:

* `sc_topk_*` — MSB radix-select top-k. The exercise. Its TensorCore control is
  `jax.lax.top_k`, which on TPU lowers to a sort-based reduction over the whole
  array and is the thing an LLM sampler spends its vocabulary pass on.
* `sc_moe_route_*` — the MoE routing prologue: count tokens per expert, prefix
  the counts into offsets, and emit the stable permutation that groups tokens
  by expert. Its TensorCore control is `jnp.argsort` plus a scatter-add. This
  is not a toy: it is the prologue in front of every grouped-matmul MoE layer,
  and it occupies the VPU while the MXU has nothing to do.

Both kernels are written for **one vector subcore of one SparseCore**, and that
is a deliberate floor, not an oversight. A single subcore is
``1 / (num_cores * num_subcores)`` of the chip's SparseCore capacity (read
both from ``SparseCoreInfo``), so a ratio taken here understates what the fully
fanned-out version would do — and a single-subcore kernel needs no
`subcore_barrier`, no `fetch_and_add` and no cross-subcore prefix sum, which is
where a SparseCore kernel gets hard. Read every ratio in this file as a lower
bound with the fan-out left on the table. `reference/design/sparsecore.md`
records what the fan-out would take.
"""

from __future__ import annotations

import harness

_GROUP = "sc-kernels"

_INT32_MIN = -2147483648
_INT32_MAX = 2147483647


def _sc():
  """Imports the SparseCore Pallas surface, or reports its absence."""
  try:
    from jax.experimental.pallas import tpu_sc as plsc
  except ImportError as error:
    raise harness.Unsupported(
      f"jax.experimental.pallas.tpu_sc is absent: {error}"
    ) from error
  return plsc


def _sc_geometry() -> tuple[int, int]:
  """Returns ``(num_lanes, vmem_capacity_bytes)`` of one vector subcore.

  Read from ``SparseCoreInfo`` at run time, never at import time, so that
  ``run_probes.py --list`` works on a machine without a TPU. ``num_lanes`` is
  the SIMD width every vector op below is a multiple of (the SC compiler has
  no "long" vector the way the TensorCore's ``(8, 128)`` tile does), and
  ``vmem_capacity_bytes`` is the working-set budget that caps ``n``.

  Raises:
    harness.Unsupported: If ``tpu_sc`` is absent or reports no SparseCore.
  """
  plsc = _sc()
  try:
    info = plsc.get_sparse_core_info()
  except Exception as error:  # noqa: BLE001 — no SparseCore is the finding.
    raise harness.Unsupported(
      f"SparseCoreInfo unavailable: {type(error).__name__}: {error}"
    ) from error
  return int(info.num_lanes), int(info.vmem_capacity_bytes)


def _one_subcore_mesh():
  """A mesh of exactly one vector subcore on one SparseCore."""
  return _sc().VectorSubcoreMesh(
    core_axis_name="core",
    subcore_axis_name="subcore",
    num_cores=1,
    num_subcores=1,
  )


# --------------------------------------------------------------------------
# Order-preserving float32 -> int32
# --------------------------------------------------------------------------
#
# Radix select needs keys whose *bit pattern* sorts the way the values do. A
# raw `bitcast(f32 -> i32)` almost works: positive floats come out as
# increasing positive ints, and every negative float comes out negative, so
# negatives sort below positives. What it gets wrong is the negatives among
# themselves — -2.0 has a larger magnitude and therefore a *larger* unsigned
# bit pattern than -1.0, so as int32 it compares greater when it should compare
# less.
#
# `INT32_MIN - b` for b < 0 reverses exactly that range and leaves it negative,
# which fixes the order without disturbing the positives. The map is its own
# inverse (see `_from_ordered_key`), so the kernel can carry keys in this form
# throughout and convert back only when it writes the answer. +0.0 and -0.0
# both land on 0, which is correct: they are equal as floats.


def _to_ordered_key(x_f32):
  """f32 -> int32 whose signed order is the float order."""
  import jax.numpy as jnp
  from jax.experimental.pallas import tpu_sc as plsc

  b = plsc.bitcast(x_f32, jnp.int32)
  return jnp.where(b >= 0, b, jnp.int32(_INT32_MIN) - b)


def _from_ordered_key(s_i32):
  """The inverse of `_to_ordered_key` — the same expression, which is the point."""
  import jax.numpy as jnp
  from jax.experimental.pallas import tpu_sc as plsc

  b = jnp.where(s_i32 >= 0, s_i32, jnp.int32(_INT32_MIN) - s_i32)
  return plsc.bitcast(b, jnp.float32)


def _digit(s_i32, shift: int):
  """Byte `shift` of an ordered key, renumbered so that *ascending index means
  descending key*.

  Two corrections are folded in here and both are easy to get wrong. The top
  byte of a negative int32 is 0x80..0xFF and of a positive one 0x00..0x7F, so
  a plain byte extract would rank every negative float above every positive
  one; XOR-ing 0x80 into the top byte undoes the sign bias and turns the
  signed order into the unsigned order the histogram wants. Then `255 - d`
  flips it, so bucket 0 holds the largest keys and a forward prefix sum walks
  the array from the top down.
  """

  d = (s_i32 >> shift) & 0xFF
  if shift == 24:
    d = d ^ 0x80
  return 255 - d


def _popcount(mask):
  """Number of true lanes in a `num_lanes`-wide mask, as a scalar."""
  import jax.numpy as jnp
  from jax.experimental.pallas import tpu_sc as plsc

  return jnp.max(plsc.all_reduce_population_count(mask))


def _compact_dest(mask, base, limit: int):
  """Destination index for each true lane of `mask` in a compacted array.

  This is `store_compressed` written out by hand, as a fallback. Whether
  `store_compressed` lowers depends on how it is called: appending at a
  running scalar offset into a VMEM scratch is the spelling expected to
  lower, while writing direct to the kernel's output ref may be refused (the
  mask is a required keyword). Check with probe `sc_primitive_lowering` and
  see `reference/design/sparsecore.md`. Prefer the primitive and fall back to
  this only when your own call refuses; this fallback costs an extra pass
  over the mask.

  An inclusive prefix sum over the mask gives each true lane its 1-based rank
  among the true lanes, so `base + rank - 1` is where it goes. Lanes that are
  false get a meaningless index and are discarded by the same mask passed to
  `store_scatter`. The clamp to `limit - 1` keeps an overrun inside the buffer
  rather than corrupting whatever follows it: the caller is expected to detect
  the overrun separately, from the high-water mark.
  """
  import jax.numpy as jnp
  from jax.experimental.pallas import tpu_sc as plsc

  rank = plsc.cumsum(jnp.where(mask, 1, 0))
  return jnp.minimum(base + rank - 1, limit - 1)


def _first_true(mask):
  """Index of the first true lane, or `num_lanes` if there is none, as a scalar."""
  import jax.numpy as jnp
  from jax.experimental.pallas import tpu_sc as plsc

  return jnp.max(plsc.all_reduce_ffs(mask))


# --------------------------------------------------------------------------
# The top-k kernel
# --------------------------------------------------------------------------


def _build_topk(n: int, k: int):
  """Builds a jitted SparseCore top-k over `n` float32 keys.

  The algorithm is MSB radix select: bucket every key by its top byte, find
  the bucket the k-th largest falls in, narrow to that bucket, and repeat on
  the next byte down. Four bytes later the surviving range has collapsed to a
  single value, and that value is the threshold. A final pass emits
  everything strictly above it plus exactly as many ties as reach k.

  **Rounds do not compact — they re-scan.** The textbook radix select copies
  each round's survivors into a candidate buffer, which is faster: pass one
  touches all `n` keys and every later pass touches only about `n/256` of
  them. It cannot be made correct here. The buffer has to be sized for the
  worst case, which is every key landing in one bucket, which is `n` — and
  `n` int32 candidates plus the `n` int32 keys plus a pong buffer does not
  fit in one subcore's VMEM (``SparseCoreInfo.vmem_capacity_bytes``) at the
  sizes worth offloading. Sizing it optimistically is worse than slow: inputs
  with heavy ties, or a wide value range, overflow it, and an overflowed
  candidate buffer returns a wrong answer rather than an error.

  Re-scanning costs four passes over `n` instead of about one, and needs no
  candidate buffer at all: the surviving set is described by an inclusive
  range `[lo, hi]` in ordered-key space, carried in two SMEM scalars. Every
  round narrows it by one byte. That is the whole state.

  **The result is not ordered.** `jax.lax.top_k` returns its k values sorted
  descending; this returns the same *set*, in the order the keys appear in
  the input. Sorting k elements afterwards is cheap, and the callers that
  matter — MoE routing, sampling with renormalization — do not need it. The
  correctness probe sorts both sides before comparing, and says so.

  Args:
    n: Length of the input. Must be a multiple of `num_lanes`.
    k: How many to return. Must be a multiple of `num_lanes` and at most `n`.

  Returns:
    A jitted `f32[n] -> (f32[k], i32[k])`.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  vec, vmem_bytes = _sc_geometry()
  if n % vec or k % vec or k > n:
    raise ValueError(f"n={n} k={k} must be multiples of {vec} with k <= n")

  # Landing zone for the input DMA. Streaming in chunks rather than mapping
  # the whole f32 array keeps one copy of the data in VMEM instead of two.
  chunk = min(n, 1024)

  working = n * 4 + chunk * 4 + 256 * 4 + 2 * (k + vec) * 4
  if working > vmem_bytes:
    raise harness.Unsupported(
      f"n={n} k={k} needs {working} B of subcore VMEM, over the "
      f"{vmem_bytes} B budget"
    )

  out_type = (
    jax.ShapeDtypeStruct((k,), jnp.float32),
    jax.ShapeDtypeStruct((k,), jnp.int32),
  )

  scratch = [
    pltpu.VMEM((n,), jnp.int32),  # ordered keys
    pltpu.VMEM((chunk,), jnp.float32),  # DMA landing zone
    pltpu.VMEM((256,), jnp.int32),  # digit histogram
    pltpu.VMEM((k + vec,), jnp.float32),  # output values
    pltpu.VMEM((k + vec,), jnp.int32),  # output indices
    pltpu.SMEM((8,), jnp.int32),  # scalar carries
    pltpu.SemaphoreType.DMA,
  ]

  # Named slots in the SMEM scratch. Every one of these is a value that has
  # to survive a `pl.loop` across several loops, where an `init_carry` does
  # not reach. SparseCore SMEM is small, so keep the slot count minimal;
  # probe `sc_smem_capacity` measures the capacity on the target.
  _R, _LO, _HI, _RUN, _FOUND, _COUNT, _OFF, _USED = range(8)

  @jax.jit
  def topk(x):
    # needs_layout_passes=False is mandatory, not a tuning knob: the
    # Mosaic-SC infer-vector-layout pass rejects scan_count,
    # addupdate_scatter, store_scatter and indexed vector loads outright,
    # and names this flag in its own error text.
    @pl.kernel(
      out_type=out_type,
      mesh=_one_subcore_mesh(),
      scratch_types=scratch,
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def kernel(x_hbm, out_v, out_i, keys, land, hist, buf_v, buf_i, smem, sem):
      lane = jnp.arange(vec, dtype=jnp.int32)

      # ---- load and re-key -------------------------------------------
      @pl.loop(0, n // chunk)
      def _(c):
        pltpu.sync_copy(x_hbm.at[pl.ds(c * chunk, chunk)], land)

        @pl.loop(0, chunk // vec)
        def _(j):
          at = j * vec
          keys[pl.ds(c * chunk + at, vec)] = _to_ordered_key(
            land[pl.ds(at, vec)]
          )

      smem[_R] = jnp.int32(k)
      smem[_LO] = jnp.int32(_INT32_MIN)
      smem[_HI] = jnp.int32(_INT32_MAX)

      # ---- four rounds of radix select -------------------------------
      for round_index in range(4):
        shift = 24 - 8 * round_index

        @pl.loop(0, 16)
        def _(c):
          hist[pl.ds(c * vec, vec)] = jnp.zeros(vec, jnp.int32)

        # Histogram over the keys still in range. `scan_count` is what
        # makes this correct: a scatter-add of ones would drop
        # duplicates landing in the same `num_lanes`-wide vector, because the
        # lanes race. `scan_count` returns each lane's *1-based*
        # occurrence count plus a mask selecting the last lane of each
        # run, so scattering `counts` — not `counts + 1`, which
        # over-counts every run by one — adds the run atomically.
        @pl.loop(0, n // vec)
        def _(c):
          s = keys[pl.ds(c * vec, vec)]
          eligible = jnp.logical_and(s >= smem[_LO], s <= smem[_HI])
          d = _digit(s, shift)
          counts, last = plsc.scan_count(d, mask=eligible)
          plsc.addupdate_scatter(
            hist, (d,), counts, mask=jnp.logical_and(last, eligible)
          )

        # Walk the 256 buckets top-down and stop at the first whose
        # cumulative count reaches `r`. Buckets are already numbered so
        # that ascending index is descending key.
        smem[_RUN] = jnp.int32(0)
        smem[_FOUND] = jnp.int32(255)

        @pl.loop(0, 16)
        def _(c):
          h = hist[pl.ds(c * vec, vec)]
          inclusive = plsc.cumsum(h) + smem[_RUN]
          hit = _first_true(inclusive >= smem[_R])
          smem[_FOUND] = jnp.where(
            jnp.logical_and(hit < vec, smem[_FOUND] == 255),
            jnp.minimum(c * vec + hit, 255),
            smem[_FOUND],
          )
          smem[_RUN] = smem[_RUN] + jnp.sum(h)

        # How many keys sit strictly above the chosen bucket. Those are
        # settled; the rest of `r` has to come out of the bucket, which
        # is what the next round refines.
        smem[_COUNT] = jnp.int32(0)

        @pl.loop(0, 16)
        def _(c):
          h = hist[pl.ds(c * vec, vec)]
          above = (c * vec + lane) < smem[_FOUND]
          smem[_COUNT] = smem[_COUNT] + jnp.sum(jnp.where(above, h, 0))

        smem[_R] = smem[_R] - smem[_COUNT]

        # Narrow [lo, hi] by the byte just decided. The top byte is a
        # separate case only because `_digit` folds the sign flip into
        # it: there d = 127 - (s >> 24), so the chosen bucket fixes the
        # arithmetic-shifted top byte directly, in [-128, 127], and
        # `<< 24` of that cannot overflow int32. Lower bytes are plain.
        if round_index == 0:
          smem[_LO] = (127 - smem[_FOUND]) << 24
        else:
          smem[_LO] = smem[_LO] + ((255 - smem[_FOUND]) << shift)
        smem[_HI] = smem[_LO] + jnp.int32((1 << shift) - 1)

      # Four rounds have collapsed the range to one value.
      threshold = smem[_LO]

      # ---- emit ------------------------------------------------------
      # Everything strictly above the threshold is in, unconditionally.
      # Ties are rationed: `cumsum` over the tie mask gives each tie its
      # rank within the chunk, `_USED` carries the rank across chunks,
      # and only the first `_R` of them are taken. Without the ration an
      # input of 4096 equal keys would overrun the output.
      smem[_OFF] = jnp.int32(0)
      smem[_USED] = jnp.int32(0)

      @pl.loop(0, n // vec)
      def _(c):
        at = c * vec
        values = keys[pl.ds(at, vec)]
        above = values > threshold
        tied = values == threshold
        rank = plsc.cumsum(jnp.where(tied, 1, 0)) + smem[_USED]
        take = jnp.logical_and(tied, rank <= smem[_R])
        chosen = jnp.logical_or(above, take)
        dest = _compact_dest(chosen, smem[_OFF], k + vec)
        plsc.store_scatter(
          buf_v, (dest,), _from_ordered_key(values), mask=chosen
        )
        plsc.store_scatter(buf_i, (dest,), at + lane, mask=chosen)
        smem[_OFF] = smem[_OFF] + _popcount(chosen)
        smem[_USED] = smem[_USED] + _popcount(take)

      pltpu.sync_copy(buf_v.at[pl.ds(0, k)], out_v)
      pltpu.sync_copy(buf_i.at[pl.ds(0, k)], out_i)

    return kernel(x)

  return topk


# --------------------------------------------------------------------------
# The MoE routing kernel
# --------------------------------------------------------------------------


def _build_moe_route(tokens: int, experts: int):
  """Builds a jitted SparseCore MoE routing prologue.

  Given `expert_id[t]` for each token, produces the tokens-per-expert counts
  and the stable permutation that groups tokens by expert — the two things a
  grouped-matmul MoE layer needs before it can call the MXU at all.

  Three SparseCore primitives carry the whole thing, and each replaces
  something the TensorCore has no good answer for:

  * `scan_count` turns a `num_lanes`-wide vector of expert ids into per-lane
    duplicate ranks, which is a within-vector histogram the VPU cannot do.
  * `addupdate_scatter` accumulates those into a 1-D counter array at
    data-dependent indices.
  * `store_scatter` writes each token id to `offset[e] + rank`, a fully
    data-dependent destination.

  The TensorCore control is `jnp.argsort` — a full comparison sort of the
  token axis to recover an ordering that is already implicit in the ids.

  Args:
    tokens: Number of tokens. Must be a multiple of `num_lanes`.
    experts: Number of experts. Must be a multiple of `num_lanes`.

  Returns:
    A jitted `i32[tokens] -> (i32[experts] counts, i32[tokens] permutation)`.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  vec, vmem_bytes = _sc_geometry()
  if tokens % vec or experts % vec:
    raise ValueError(
      f"tokens={tokens} experts={experts} must be multiples of {vec}"
    )

  working = (2 * tokens + 3 * experts) * 4
  if working > vmem_bytes:
    raise harness.Unsupported(
      f"tokens={tokens} experts={experts} needs {working} B of subcore "
      f"VMEM, over the {vmem_bytes} B budget"
    )

  out_type = (
    jax.ShapeDtypeStruct((experts,), jnp.int32),
    jax.ShapeDtypeStruct((tokens,), jnp.int32),
  )
  scratch = [
    pltpu.VMEM((tokens,), jnp.int32),  # expert ids
    pltpu.VMEM((tokens,), jnp.int32),  # permutation being built
    pltpu.VMEM((experts,), jnp.int32),  # counts
    pltpu.VMEM((experts,), jnp.int32),  # exclusive-prefix offsets
    pltpu.VMEM((experts,), jnp.int32),  # running fill per expert
    pltpu.SMEM((1,), jnp.int32),
    pltpu.SemaphoreType.DMA,
  ]

  @jax.jit
  def route(expert_ids):
    # See the note in _build_topk: the layout-inference pass refuses every
    # scatter/gather primitive this kernel is built out of.
    @pl.kernel(
      out_type=out_type,
      mesh=_one_subcore_mesh(),
      scratch_types=scratch,
      compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
    )
    def kernel(
      e_hbm, out_counts, out_perm, ids, perm, counts, offsets, fill, smem, sem
    ):
      lane = jnp.arange(vec, dtype=jnp.int32)

      pltpu.sync_copy(e_hbm, ids)

      @pl.loop(0, experts // vec)
      def _(c):
        counts[pl.ds(c * vec, vec)] = jnp.zeros(vec, jnp.int32)
        fill[pl.ds(c * vec, vec)] = jnp.zeros(vec, jnp.int32)

      @pl.loop(0, tokens // vec)
      def _(c):
        d = ids[pl.ds(c * vec, vec)]
        # `run` is 1-based: the first occurrence in the vector gets 1,
        # so at the last lane of a run `run` *is* the run length.
        # Scattering `run + 1` would over-count every expert by one per
        # vector.
        run, last = plsc.scan_count(d)
        plsc.addupdate_scatter(counts, (d,), run, mask=last)

      # Exclusive prefix over the expert axis, carried in SMEM because a
      # `pl.loop` has no carry of its own.
      smem[0] = jnp.int32(0)

      @pl.loop(0, experts // vec)
      def _(c):
        h = counts[pl.ds(c * vec, vec)]
        offsets[pl.ds(c * vec, vec)] = plsc.cumsum(h) - h + smem[0]
        smem[0] = smem[0] + jnp.sum(h)

      # Place each token. `fill` holds how many tokens of each expert have
      # already been placed by earlier vectors; `run` disambiguates the
      # duplicates inside this one. Together they make the permutation
      # stable, which is what `argsort(kind="stable")` gives on the other
      # side and what makes the two comparable at all.
      @pl.loop(0, tokens // vec)
      def _(c):
        d = ids[pl.ds(c * vec, vec)]
        run, last = plsc.scan_count(d)
        base = plsc.load_gather(offsets, (d,)) + plsc.load_gather(fill, (d,))
        # `run - 1` for the same reason: `run` counts from 1, and the
        # first token of an expert in this vector belongs at `base + 0`.
        plsc.store_scatter(perm, (base + run - 1,), c * vec + lane)
        plsc.addupdate_scatter(fill, (d,), run, mask=last)

      pltpu.sync_copy(counts, out_counts)
      pltpu.sync_copy(perm, out_perm)

    return kernel(expert_ids)

  return route


# --------------------------------------------------------------------------
# Probes: top-k
# --------------------------------------------------------------------------


def _topk_reference(x, k):
  import jax
  import jax.numpy as jnp

  values, indices = jax.lax.top_k(x, k)
  return jnp.sort(values), jnp.sort(indices)


@harness.probe(
  _GROUP,
  "sc_topk_correctness",
  "Does a SparseCore radix-select top-k return the same set as jax.lax.top_k?",
)
def _sc_topk_correctness() -> str:
  """Correctness first, and separately from any timing.

  Six inputs, chosen for what they break rather than for coverage: plain
  random f32 exercises the ordinary path; a signed input and an all-negative
  input exercise the negative-key transform, which is the part of a radix
  select that is wrong in most first drafts; and a heavily quantized input
  forces ties, which is what the tie rationing in the emit pass exists for.

  The `all-equal` case is not padding. Every one of the `n` keys ties at the
  threshold, so the emit pass is asked for `n` writes into a `k`-element
  buffer and the ration is the only thing stopping it. This is the case that
  breaks a compacting design that sizes a candidate buffer from an estimate
  (it silently returns the wrong answer when the estimate is exceeded); the
  re-scan design here has no such buffer.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np

  _sc()
  cases = []
  for label, n, k, maker in (
    ("uniform", 4096, 64, lambda rng, n: rng.random(n, dtype=np.float32)),
    (
      "signed",
      4096,
      64,
      lambda rng, n: rng.standard_normal(n).astype(np.float32),
    ),
    (
      "ties",
      4096,
      64,
      lambda rng, n: np.round(rng.standard_normal(n) * 4).astype(np.float32),
    ),
    (
      "wide",
      16384,
      128,
      lambda rng, n: rng.standard_normal(n).astype(np.float32),
    ),
    ("all-equal", 4096, 64, lambda rng, n: np.full(n, 1.5, np.float32)),
    (
      "all-neg",
      4096,
      64,
      lambda rng, n: -np.abs(rng.standard_normal(n)).astype(np.float32),
    ),
  ):
    rng = np.random.default_rng(0xC0FFEE + len(label))
    x = jnp.asarray(maker(rng, n))
    topk = _build_topk(n, k)
    values, indices = jax.block_until_ready(topk(x))
    want_v, want_i = _topk_reference(x, k)
    got_v, got_i = jnp.sort(values), jnp.sort(indices)

    # Values must match exactly -- a radix select does no arithmetic on the
    # keys, so anything but bit equality is a bug, not a tolerance.
    value_ok = bool(jnp.array_equal(got_v, want_v))
    # Indices may legitimately differ when values tie: two equal keys are
    # interchangeable and the two algorithms break the tie differently.
    index_ok = bool(jnp.array_equal(got_i, want_i)) or bool(
      jnp.array_equal(jnp.sort(x[got_i]), want_v)
    )
    cases.append(
      f"{label}(n={n},k={k}):{'ok' if value_ok and index_ok else 'MISMATCH'}"
    )

  bad = [c for c in cases if "ok" not in c]
  if bad:
    return (
      "FAIL",
      "SparseCore top-k disagrees with lax.top_k: " + ", ".join(cases),
    )
  return (
    "value sets identical to jax.lax.top_k on all six inputs (indices "
    "compared by the values they select, since ties are interchangeable); "
    "result is unordered, both sides sorted before comparison: "
    + ", ".join(cases)
  )


@harness.probe(
  _GROUP,
  "sc_topk_vs_lax_topk",
  "How does SparseCore radix-select top-k compare with jax.lax.top_k on the TensorCore?",
)
def _sc_topk_vs_lax_topk() -> str:
  """The offload question for top-k, with the TensorCore arm in the same process.

  Correctness is re-checked here and gates the timing: `jax.lax.top_k` and a
  kernel that returns the wrong 64 elements cost different amounts, and the
  wrong one is usually cheaper.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np

  _sc()
  rows = []
  for n, k in ((4096, 64), (16384, 64), (16384, 128), (65536, 64)):
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal(n).astype(np.float32))
    try:
      topk = _build_topk(n, k)
    except harness.Unsupported as error:
      rows.append(f"n={n},k={k}: skipped ({error})")
      continue

    values, _ = jax.block_until_ready(topk(x))
    want_v, _ = _topk_reference(x, k)
    if not bool(jnp.array_equal(jnp.sort(values), want_v)):
      raise harness.Invalid(
        f"n={n} k={k}: SparseCore top-k returned the wrong set; timing withheld"
      )

    reference = jax.jit(lambda a, kk=k: jax.lax.top_k(a, kk))
    sc = harness.time_call(lambda: topk(x), batch=32)
    tc = harness.time_call(lambda: reference(x), batch=32)
    harness.require_above_dispatch_floor(sc, what=f"sc topk n={n}")
    harness.require_above_dispatch_floor(tc, what=f"tc topk n={n}")
    rows.append(
      f"n={n},k={k}: sc={sc.median * 1e6:.1f}us tc={tc.median * 1e6:.1f}us "
      f"ratio={sc.median / tc.median:.2f}x spread={sc.spread_pct:.0f}/{tc.spread_pct:.0f}%"
    )

  return (
    "ratio is SparseCore / lax.top_k, so below 1.00 means the SparseCore "
    "wins outright; one vector subcore of one SparseCore, no fan-out. "
    + "; ".join(rows)
  )


@harness.probe(
  _GROUP,
  "sc_topk_frees_the_mxu",
  "Does moving top-k to the SparseCore actually free the MXU, or just move the time?",
)
def _sc_topk_frees_the_mxu() -> str:
  """The question the ratio above does not answer.

  A SparseCore top-k that is *slower* than `lax.top_k` can still be the right
  call, because it runs on a unit the matmul is not using. The arms:

  * `tc_only` — the matmul alone. The floor.
  * `tc_plus_lax` — matmul and `lax.top_k` in one jit. Both want the VPU/MXU
    pipeline, so this is the status quo cost.
  * `tc_plus_sc` — matmul and the SparseCore top-k in one jit.

  What decides the offload is `tc_plus_sc - tc_only` against
  `tc_plus_lax - tc_only`: how much the top-k adds to a step that is already
  doing matmul work. If the SparseCore arm adds near zero, the top-k became
  free even if it is individually slower.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np

  _sc()
  n, k = 16384, 64
  rng = np.random.default_rng(11)
  x = jnp.asarray(rng.standard_normal(n).astype(np.float32))
  # bf16 4096x4096: a small matmul is launch-bound, so "does the top-k hide
  # behind it" would be asked of a matmul that is itself mostly waiting, and
  # its timed region risks falling under the dispatch floor. This shape
  # matches `sc_tc_dependent_handoff`, so the two probes' TC arms are
  # comparable.
  a = jnp.asarray(rng.standard_normal((4096, 4096))).astype(jnp.bfloat16)
  b = jnp.asarray(rng.standard_normal((4096, 4096))).astype(jnp.bfloat16)

  topk = _build_topk(n, k)
  values, _ = jax.block_until_ready(topk(x))
  want_v, _ = _topk_reference(x, k)
  if not bool(jnp.array_equal(jnp.sort(values), want_v)):
    raise harness.Invalid(
      "SparseCore top-k returned the wrong set; timing withheld"
    )

  @jax.jit
  def tc_only(a, b):
    return a @ b

  @jax.jit
  def tc_plus_lax(a, b, x):
    return a @ b, jax.lax.top_k(x, k)

  @jax.jit
  def tc_plus_sc(a, b, x):
    return a @ b, topk(x)

  only = harness.time_call(lambda: tc_only(a, b), batch=16)
  with_lax = harness.time_call(lambda: tc_plus_lax(a, b, x), batch=16)
  with_sc = harness.time_call(lambda: tc_plus_sc(a, b, x), batch=16)
  for timing, what in (
    (only, "matmul"),
    (with_lax, "matmul+lax.top_k"),
    (with_sc, "matmul+sc.top_k"),
  ):
    harness.require_above_dispatch_floor(timing, what=what)

  added_lax = with_lax.median - only.median
  added_sc = with_sc.median - only.median
  verdict = (
    "SparseCore top-k is nearly free alongside the matmul"
    if added_sc < 0.5 * added_lax
    else "SparseCore top-k does not hide behind the matmul"
  )
  return (
    f"matmul(4096x4096 bf16)={only.median * 1e6:.1f}us; "
    f"+lax.top_k={with_lax.median * 1e6:.1f}us (adds {added_lax * 1e6:.1f}us); "
    f"+sc.top_k={with_sc.median * 1e6:.1f}us (adds {added_sc * 1e6:.1f}us); "
    f"added_sc/added_lax={added_sc / added_lax if added_lax else float('nan'):.2f}; "
    f"{verdict}. n={n} k={k}, one subcore, spreads "
    f"{only.spread_pct:.0f}/{with_lax.spread_pct:.0f}/{with_sc.spread_pct:.0f}%"
  )


# --------------------------------------------------------------------------
# Probes: MoE routing
# --------------------------------------------------------------------------


def _moe_reference(expert_ids, experts):
  import jax
  import jax.numpy as jnp

  @jax.jit
  def reference(ids):
    counts = jnp.zeros((experts,), jnp.int32).at[ids].add(1)
    permutation = jnp.argsort(ids, stable=True).astype(jnp.int32)
    return counts, permutation

  return reference


@harness.probe(
  _GROUP,
  "sc_moe_route_correctness",
  "Does the SparseCore routing kernel produce the same counts and stable permutation as argsort?",
)
def _sc_moe_route_correctness() -> str:
  """Exact equality is the bar here, not a tolerance.

  Both sides are stable, so the permutation is unique: the SparseCore version
  is stable because `fill` counts placements across vectors and `scan_count`
  orders the duplicates inside one, and `argsort(stable=True)` is stable by
  construction. An unstable kernel would still produce a *valid* grouping and
  a wrong permutation, which is exactly the bug that survives a loose check.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np

  _sc()
  cases = []
  for label, tokens, experts, maker in (
    ("uniform", 4096, 128, lambda rng, t, e: rng.integers(0, e, t)),
    # A skewed routing distribution is the realistic one and the one that
    # stresses the per-expert counters: half the tokens on one expert.
    (
      "skewed",
      4096,
      128,
      lambda rng, t, e: np.where(rng.random(t) < 0.5, 0, rng.integers(0, e, t)),
    ),
    ("few-experts", 8192, 16, lambda rng, t, e: rng.integers(0, e, t)),
  ):
    rng = np.random.default_rng(0x5EED + len(label))
    ids = jnp.asarray(maker(rng, tokens, experts).astype(np.int32))
    route = _build_moe_route(tokens, experts)
    counts, permutation = jax.block_until_ready(route(ids))
    want_counts, want_perm = jax.block_until_ready(
      _moe_reference(ids, experts)(ids)
    )
    ok = bool(jnp.array_equal(counts, want_counts)) and bool(
      jnp.array_equal(permutation, want_perm)
    )
    cases.append(
      f"{label}(T={tokens},E={experts}):{'ok' if ok else 'MISMATCH'}"
    )

  if any("ok" not in c for c in cases):
    return (
      "FAIL",
      "SparseCore routing disagrees with argsort: " + ", ".join(cases),
    )
  return (
    "counts and stable permutation identical to bincount+argsort: "
    + ", ".join(cases)
  )


@harness.probe(
  _GROUP,
  "sc_moe_route_vs_xla",
  "How does SparseCore MoE routing compare with argsort on the TensorCore?",
)
def _sc_moe_route_vs_xla() -> str:
  """The offload question for the MoE prologue."""
  import jax
  import jax.numpy as jnp
  import numpy as np

  _sc()
  rows = []
  for tokens, experts in ((1024, 128), (4096, 128), (16384, 128), (4096, 256)):
    rng = np.random.default_rng(13)
    ids = jnp.asarray(rng.integers(0, experts, tokens).astype(np.int32))
    try:
      route = _build_moe_route(tokens, experts)
    except harness.Unsupported as error:
      rows.append(f"T={tokens},E={experts}: skipped ({error})")
      continue
    reference = _moe_reference(ids, experts)

    counts, permutation = jax.block_until_ready(route(ids))
    want_counts, want_perm = jax.block_until_ready(reference(ids))
    if not (
      bool(jnp.array_equal(counts, want_counts))
      and bool(jnp.array_equal(permutation, want_perm))
    ):
      raise harness.Invalid(
        f"T={tokens} E={experts}: routing kernel disagrees with argsort; "
        "timing withheld"
      )

    sc = harness.time_call(lambda: route(ids), batch=32)
    tc = harness.time_call(lambda: reference(ids), batch=32)
    harness.require_above_dispatch_floor(sc, what=f"sc route T={tokens}")
    harness.require_above_dispatch_floor(tc, what=f"xla route T={tokens}")
    rows.append(
      f"T={tokens},E={experts}: sc={sc.median * 1e6:.1f}us "
      f"xla={tc.median * 1e6:.1f}us ratio={sc.median / tc.median:.2f}x "
      f"spread={sc.spread_pct:.0f}/{tc.spread_pct:.0f}%"
    )

  return (
    "ratio is SparseCore / (bincount + stable argsort on TC), so below 1.00 "
    "means the SparseCore wins outright; one vector subcore, no fan-out. "
    + "; ".join(rows)
  )


@harness.probe(
  _GROUP,
  "sc_moe_route_frees_the_mxu",
  "Does moving MoE routing to the SparseCore free the MXU for the expert matmul?",
)
def _sc_moe_route_frees_the_mxu() -> str:
  """The same three-arm test as `sc_topk_frees_the_mxu`, on the routing prologue.

  This is the more interesting of the two, because in a real MoE layer the
  routing genuinely *precedes* the matmul it feeds — so an overlap seen
  here is only realizable across steps, or against a matmul from a different
  layer. The probe measures the independent case and says so; the dependent
  case is `sparsecore.sc_tc_dependent_handoff`.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np

  _sc()
  tokens, experts = 4096, 128
  rng = np.random.default_rng(17)
  ids = jnp.asarray(rng.integers(0, experts, tokens).astype(np.int32))
  # bf16 4096x4096 for the same reason as `sc_topk_frees_the_mxu`. It is
  # also the more honest control: an MoE routing prologue sits in front of
  # a grouped matmul of roughly this size, not a launch-bound one.
  a = jnp.asarray(rng.standard_normal((4096, 4096))).astype(jnp.bfloat16)
  b = jnp.asarray(rng.standard_normal((4096, 4096))).astype(jnp.bfloat16)

  route = _build_moe_route(tokens, experts)
  reference = _moe_reference(ids, experts)
  counts, permutation = jax.block_until_ready(route(ids))
  want_counts, want_perm = jax.block_until_ready(reference(ids))
  if not (
    bool(jnp.array_equal(counts, want_counts))
    and bool(jnp.array_equal(permutation, want_perm))
  ):
    raise harness.Invalid(
      "routing kernel disagrees with argsort; timing withheld"
    )

  @jax.jit
  def tc_only(a, b):
    return a @ b

  @jax.jit
  def tc_plus_xla(a, b, ids):
    return a @ b, reference(ids)

  @jax.jit
  def tc_plus_sc(a, b, ids):
    return a @ b, route(ids)

  only = harness.time_call(lambda: tc_only(a, b), batch=16)
  with_xla = harness.time_call(lambda: tc_plus_xla(a, b, ids), batch=16)
  with_sc = harness.time_call(lambda: tc_plus_sc(a, b, ids), batch=16)
  for timing, what in (
    (only, "matmul"),
    (with_xla, "matmul+argsort"),
    (with_sc, "matmul+sc.route"),
  ):
    harness.require_above_dispatch_floor(timing, what=what)

  added_xla = with_xla.median - only.median
  added_sc = with_sc.median - only.median
  return (
    f"matmul(4096x4096 bf16)={only.median * 1e6:.1f}us; "
    f"+argsort={with_xla.median * 1e6:.1f}us (adds {added_xla * 1e6:.1f}us); "
    f"+sc.route={with_sc.median * 1e6:.1f}us (adds {added_sc * 1e6:.1f}us); "
    f"added_sc/added_xla={added_sc / added_xla if added_xla else float('nan'):.2f}; "
    f"T={tokens} E={experts}, independent (routing does not feed this matmul), "
    f"one subcore, spreads "
    f"{only.spread_pct:.0f}/{with_xla.spread_pct:.0f}/{with_sc.spread_pct:.0f}%"
  )


@harness.probe(
  _GROUP,
  "sc_moe_route_offload_sweep",
  "Does the routing-offload comparison hold across a different routing shape and a different neighbour?",
)
def _sc_moe_route_offload_sweep() -> str:
  """Replicates `sc_moe_route_frees_the_mxu` across shapes.

  That probe compares how much the SparseCore routing prologue adds to a step
  already running a bf16 4096x4096 matmul against how much XLA's `argsort`
  adds. It is one routing shape (T=4096, E=128) beside one matmul size, so a
  result from it alone is a data point, not a rule.

  Two things could make it fragile, and they pull in opposite directions:

  * **The shadow.** Any win is the SparseCore hiding behind the matmul, so a
    shorter matmul should erode it and a longer one should widen it. If the
    ratio is flat across matmul size, the mechanism is not what we think.
  * **The routing shape.** Expert count drives the SC kernel's serialized
    `scan_count` fold and the XLA arm's `argsort` differently, so the two
    arms need not move together.

  Sweeping both says whether "offload the routing" is a rule or a data point.
  Matmul sizes start at 3072 so that 16 of them clear the dispatch floor;
  `require_above_dispatch_floor` withholds any arm that does not.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np

  _sc()
  rng = np.random.default_rng(29)
  arms = []
  for tokens, experts in ((4096, 128), (8192, 16)):
    ids = jnp.asarray(rng.integers(0, experts, tokens).astype(np.int32))
    route = _build_moe_route(tokens, experts)
    reference = _moe_reference(ids, experts)
    counts, permutation = jax.block_until_ready(route(ids))
    want_counts, want_perm = jax.block_until_ready(reference(ids))
    if not (
      bool(jnp.array_equal(counts, want_counts))
      and bool(jnp.array_equal(permutation, want_perm))
    ):
      raise harness.Invalid(
        f"routing kernel disagrees with argsort at T={tokens} "
        f"E={experts}; timing withheld"
      )

    for dim in (3072, 4096, 8192):
      a = jnp.asarray(rng.standard_normal((dim, dim))).astype(jnp.bfloat16)
      b = jnp.asarray(rng.standard_normal((dim, dim))).astype(jnp.bfloat16)

      @jax.jit
      def tc_only(a, b):
        return a @ b

      @jax.jit
      def tc_plus_xla(a, b, ids, _r=reference):
        return a @ b, _r(ids)

      @jax.jit
      def tc_plus_sc(a, b, ids, _r=route):
        return a @ b, _r(ids)

      only = harness.time_call(lambda: tc_only(a, b), batch=16)
      with_xla = harness.time_call(lambda: tc_plus_xla(a, b, ids), batch=16)
      with_sc = harness.time_call(lambda: tc_plus_sc(a, b, ids), batch=16)
      for timing, what in (
        (only, f"matmul{dim}"),
        (with_xla, f"matmul{dim}+argsort"),
        (with_sc, f"matmul{dim}+sc.route"),
      ):
        harness.require_above_dispatch_floor(timing, what=what)

      added_xla = with_xla.median - only.median
      added_sc = with_sc.median - only.median
      # A negative added time is noise swamping the signal, not a free
      # kernel, and a ratio computed from one is meaningless.
      if added_xla <= 0 or added_sc <= 0:
        arms.append(
          f"T{tokens}/E{experts}/mm{dim}:NOISE"
          f"(added_xla={added_xla * 1e6:.1f}us,"
          f"added_sc={added_sc * 1e6:.1f}us)"
        )
        continue
      arms.append(
        f"T{tokens}/E{experts}/mm{dim}:mm={only.median * 1e6:.0f}us "
        f"+xla={added_xla * 1e6:.1f}us +sc={added_sc * 1e6:.1f}us "
        f"ratio={added_sc / added_xla:.2f} "
        f"(spreads {only.spread_pct:.0f}/{with_xla.spread_pct:.0f}/"
        f"{with_sc.spread_pct:.0f}%)"
      )

  return (
    "added step time of the routing prologue beside an independent bf16 "
    "matmul, one subcore, ratio<1 means the SparseCore offload wins: "
    + " | ".join(arms)
  )
