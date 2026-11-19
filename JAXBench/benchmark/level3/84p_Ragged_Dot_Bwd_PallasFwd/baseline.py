# Copyright 2026 Google LLC. All Rights Reserved.
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
# ==============================================================================
"""Ragged dot backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

Ragged dot is the grouped matmul (GMM) of Mixture-of-Experts layers. The rows
of `lhs` are sorted by group (expert) and `group_sizes[g]` consecutive rows
belong to group `g`. Each row is multiplied by the weight matrix of its group:
`out[m] = lhs[m] @ rhs[g(m)]`.

`forward_pallas(lhs, rhs, group_sizes)` is the Tokamax Pallas TPU forward
kernel `gmm`, with tile sizes from the Tokamax heuristic. It returns
`(out, residuals)`. What the Tokamax forward saves for its VJP is the ragged
dot output before the activation:
  residuals  `[M, N]`  pre-activation output
This configuration has no activation, so `residuals` is `out` itself, and
Tokamax's VJP does not read it.

`workload(lhs, rhs, group_sizes, dout)` returns the gradients `(dlhs, drhs)`
for the output cotangent `dout`. It is the correctness oracle: JAX autodiff
(`jax.vjp`) of the pure-JAX reference `forward`, which calls
`jax.lax.ragged_dot_general`. This file contains no backward kernel and no
hand-derived backward pass. `group_sizes` is an integer input and has no
gradient.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` so
that your `workload` returns the same gradients as this one.

The configuration is the plain one: default dimension numbers, no activation,
no quantisation, no group offset. The quantisation code of the kernel (which
needs the `qwix` package) is left out; everything else in `gmm` is as in
Tokamax. The kernel needs `M`, `K` and `N` to be at least 128 and `M` to be a
multiple of the row tile size that the heuristic picks (at most 1024).

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py  (kernel `gmm`)
  tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py  (tile heuristic, wrapper)
  tokamax/_src/mosaic_tpu.py                (`custom_buffered_pallas_call`)
  tokamax/_src/ops/ragged_dot/base.py       (forward reference `RaggedDot._fwd`)
  tokamax/_src/ops/ragged_dot/arg_specs.py  (shapes, group-size distribution)
  tokamax/_src/ops/ragged_dot/test_base.py  (value ranges)
"""

# Dimensions.
#   M   number of lhs rows (tokens routed to experts), sorted by group
#   K   contracting dimension (input features)
#   N   output features
#   G   number of groups (experts)
#   tm  tile extent of M
#   tk  tile extent of K
#   tn  tile extent of N

from collections.abc import Callable
import contextlib
import dataclasses
import functools
import json
from typing import Any, Final, Sequence

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/ragged_dot/base.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================

DEFAULT_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [1]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)


def forward(lhs, rhs, group_sizes):
  """Ragged dot forward pass: `out[m] = lhs[m] @ rhs[group of row m]`.

  Args:
    lhs: Activations, `[M, K]`, rows sorted by group.
    rhs: Per-group weights, `[G, K, N]`.
    group_sizes: Number of rows in each group, int32 `[G]`, summing to `M`.

  Returns:
    The output, `[M, N]`, in the result dtype of `lhs` and `rhs`. Products are
    accumulated in float32 before the cast.
  """
  out_dtype = jnp.result_type(lhs, rhs)
  # Tokamax's canonical form of the default matmul precision.
  precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)

  # NOTE: `preferred_element_type` changes the accumulation type when using
  # `jax.lax.Precision`.
  is_integer = jnp.issubdtype(out_dtype, jnp.integer)
  acc_dtype = jnp.int32 if is_integer else jnp.float32
  preferred_element_type = jnp.promote_types(out_dtype, acc_dtype)
  dot_out = jax.lax.ragged_dot_general(
      lhs,
      rhs,
      group_sizes=group_sizes,
      ragged_dot_dimension_numbers=DEFAULT_RAGGED_DOT_DIM_NUMS,
      precision=precision,
      preferred_element_type=preferred_element_type,
  ).astype(out_dtype)
  return dot_out


# ==============================================================================
# Pallas TPU pipeline helper
# (inlined from tokamax/_src/mosaic_tpu.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================


def custom_buffered_pallas_call(
    kernel: Callable[..., Any],
    out_shape: jax.ShapeDtypeStruct,
    grid_spec: pltpu.PrefetchScalarGridSpec,
    compiler_params: pltpu.CompilerParams,
    input_buffer_count: Sequence[int] | None = None,
    **kw,
):
  # pylint: disable=invalid-name
  """Custom PrefetchScalarGrid pallas_call using emit_pipeline."""
  num_scalar_prefetch = grid_spec.num_scalar_prefetch
  len_ = lambda x: len(x) if isinstance(x, (list, tuple)) else 1
  args_len = len_(grid_spec.in_specs) + len_(grid_spec.out_specs)

  def _augment_blockspec(bs, smem_refs):
    index_map_ = lambda *idxs: bs.index_map(*idxs, *smem_refs)
    return pl.BlockSpec(bs.block_shape, index_map_)

  grid_static = tuple(
      dim if isinstance(dim, int) else None for dim in grid_spec.grid
  )
  grid_dynamic = tuple(
      None if isinstance(dim, int) else jnp.atleast_1d(dim)
      for dim in grid_spec.grid
  )

  def _bind_pipeline(spec, count):
    if count == 2:
      return spec
    return dataclasses.replace(
        spec, pipeline_mode=pl.Buffered(buffer_count=count, use_lookahead=True)
    )

  def pallas_call(*args):
    smem_args = args[:num_scalar_prefetch]  # close over smem data

    def pipeline(*args_refs):
      # unpack the dynamic grid elements from smem
      grid = tuple(
          d[0] if d is not None else s
          for d, s in zip(args_refs[0], grid_static)
      )

      # unpack the smem prefetch values and bind them to the inspecs
      smem_refs = args_refs[1 : num_scalar_prefetch + 1]
      _bind_smem = functools.partial(_augment_blockspec, smem_refs=smem_refs)
      in_specs_ = jax.tree.map(_bind_smem, grid_spec.in_specs)
      if input_buffer_count is not None:
        if len(input_buffer_count) != len(in_specs_):
          raise ValueError(
              f"`{input_buffer_count=}` must a list[int] equal in length to"
              f" {len(in_specs_)=}."
          )

        in_specs_ = tuple(
            jax.tree.map(functools.partial(_bind_pipeline, count=c), spec)
            for spec, c in zip(in_specs_, input_buffer_count)
        )
      out_specs_ = jax.tree.map(_bind_smem, grid_spec.out_specs)

      # unpack inputs/outputs and scratch refs
      input_output_refs = args_refs[
          num_scalar_prefetch + 1 : num_scalar_prefetch + args_len + 1
      ]
      scratch_refs = args_refs[num_scalar_prefetch + args_len + 1 :]

      # bind smem and scratch to the pipeline body
      # specify dimension semantic from the scalar prefetch grid and emit
      pltpu.emit_pipeline(
          lambda step, *args: kernel(
              step, grid, *smem_refs, *args, *scratch_refs
          ),
          grid=grid,  # pyrefly: ignore[bad-argument-type]
          in_specs=in_specs_,
          out_specs=out_specs_,
          dimension_semantics=compiler_params.dimension_semantics,  # pyrefly: ignore[bad-argument-type]
          _explicit_indices=True,
      )(*input_output_refs)

    bs_smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    bs_hbm = pl.BlockSpec(memory_space=pl.ANY)

    smem_specs = (jax.tree.map(lambda _: bs_smem, grid_dynamic),)
    smem_specs += jax.tree.map(lambda _: bs_smem, smem_args)
    in_specs = jax.tree.map(lambda _: bs_hbm, tuple(grid_spec.in_specs))
    out_specs = jax.tree.map(lambda _: bs_hbm, grid_spec.out_specs)

    params = dataclasses.replace(compiler_params, dimension_semantics=())
    return pl.pallas_call(
        pipeline,
        out_shape,
        compiler_params=params,
        in_specs=smem_specs + in_specs,
        out_specs=out_specs,
        scratch_shapes=grid_spec.scratch_shapes,
        **kw,
    )(grid_dynamic, *args)

  # pylint: enable=invalid-name
  return pallas_call


# ==============================================================================
# Pallas TPU forward kernel
# (inlined from tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================

# pylint: disable=too-many-positional-arguments, unnecessary-lambda-assignment

_KERNEL_NAME_GMM: Final[str] = "pallas_tpu_ragged_dot_gmm"


def _validate_args(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    expected_rhs_dims: int = 3,
) -> jax.Array:
  """Validates the arguments for the gmm function."""
  if lhs.ndim != 2:
    raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")

  if rhs.ndim != expected_rhs_dims:
    raise ValueError(
        f"Expected {expected_rhs_dims}-tensor for 'rhs' but got"
        f" {rhs.ndim}-tensor."
    )

  if group_sizes.dtype not in (jnp.int32, jnp.uint32):
    raise ValueError(
        f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}."
    )
  return group_sizes.astype(jnp.int32)


GroupMetadata = tuple[jax.Array, jax.Array, jax.Array]


def make_group_metadata(
    *,
    group_sizes: jax.Array,
    m: int,
    tm: int,
    start_group: jax.Array,
    num_nonzero_groups: int,
    visit_empty_groups: bool,
) -> tuple[GroupMetadata, jax.Array]:
  """Create the metadata needed for grouped matmul computation.

  Args:
    group_sizes: A 1d, jax.Array with shape `[num_groups]` and `jnp.int32`
      dtype.
    m: The number of rows in lhs.
    tm: The m-dimension tile size being used.
    start_group: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_nonzero_groups: Number of groups in group sizes to compute on. Useful in
      combination with group_offset.
    visit_empty_groups: If True, do not squeeze tiles for empty groups out of
      the metadata. This is necessary for tgmm, where we at least need to zero
      the output for each group.

  Returns:
    tuple of:
      group_offsets: A 1d, jax.Array with shape [num_groups+1] and jnp.int32
        dtype. group_offsets[i] indicates the row at which group [i] starts in
        the lhs matrix and group_offsets[i-1] = m.
      group_ids: A 1d, jax.Array with shape [m_tiles + num_groups] and
        jnp.int32 dtype. group_ids[i] indicates which group grid index 'i' will
        work on.
      m_tile_ids: A 1d, jax.Array with shape [m_tiles + num_groups] and
        jnp.int32. m_tile_ids[i] indicates which m-dimension tile grid index 'i'
        will work on.
    num_tiles: The number of m-dimension tiles to execute.
  """
  num_groups = group_sizes.shape[0]
  end_group = start_group + num_nonzero_groups - 1

  # Calculate the offset of each group, starting at zero. This metadata is
  # similar to row offsets in a CSR matrix. The following properties hold:
  #
  # group_offsets.shape = [num_groups + 1]
  # group_offsets[0] = 0
  # group_offsets[num_groups] = m
  #
  # The row at which group 'i' starts is group_offsets[i].
  group_ends = jnp.cumsum(group_sizes)
  group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

  # Assign a group id to each grid index.
  #
  # If a group starts somewhere other than the start of a tile or ends somewhere
  # other than the end of a tile we need to compute that full tile. Calculate
  # the number of tiles for each group by rounding their end up to the nearest
  # 'tm' and their start down to the nearest 'tm'.

  # (1) Round the group_ends up to the nearest multiple of 'tm'.
  #
  # NOTE: This does not change group_offsets[num_groups], which is m
  # (because we enforce m is divisible by tm).
  rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)

  # (2) Round the group_starts down to the nearest multiple of 'tm'.
  group_starts = jnp.concatenate(
      [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
  )
  rounded_group_starts = group_starts // tm * tm

  # (3) Calculate the number of rows in each group.
  #
  # NOTE: Handle zero-sized groups as a special case. If the start for a
  # zero-sized group is not divisible by 'tm' its start will be rounded down and
  # its end will be rounded up such that its size will become 1 tile here.
  rounded_group_sizes = rounded_group_ends - rounded_group_starts
  rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)

  # (4) Convert the group sizes from units of rows to unit of 'tm' sized tiles.
  #
  # An m-dimension tile is 'owned' by group 'i' if the first row of the tile
  # belongs to group 'i'. In addition to owned tiles, each group can have 0 or 1
  # initial partial tiles if it's first row does not occur in the first row of a
  # tile. The '0-th' group never has a partial tile because it always starts at
  # the 0-th row.
  #
  # If no group has a partial tile, the total number of tiles is equal to
  # 'm // tm'. If every group has a partial except the 0-th group, the total
  # number of tiles is equal to 'm // tm + num_groups - 1'. Thus we know that
  #
  # tiles_m <= group_tiles.sum() <= tiles_m + num_groups - 1
  #
  # Where tiles_m = m // tm.
  #
  # NOTE: All group sizes are divisible by 'tm' because of the rounding in steps
  # (1) and (2) so this division is exact.
  group_tiles = rounded_group_sizes // tm

  if visit_empty_groups:
    # Insert one tile for empty groups.
    group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

  # Create the group ids for each grid index based on the tile counts for each
  # group.
  #
  # NOTE: This repeat(...) will pad group_ids with the final group id if
  # group_tiles.sum() < tiles_m + num_groups - 1. The kernel grid will be sized
  # such that we only execute the necessary number of tiles.
  if m % tm != 0:
    raise NotImplementedError(f"{m=} must be divisible by tile size ({tm}).")

  tiles_m = m // tm
  group_ids = jnp.repeat(
      jnp.arange(num_groups, dtype=jnp.int32),
      group_tiles,
      total_repeat_length=tiles_m + num_groups - 1,
  )

  # Assign an m-dimension tile id to each grid index.
  #
  # NOTE: Output tiles can only be re-visited consecutively. The following
  # procedure guarantees that m-dimension tile indices respect this.

  # (1) Calculate how many times each m-dimension tile will be visited.
  #
  # Each tile is guaranteed to be visited once by the group that owns the tile.
  # The remaining possible visits occur when a group starts inside of a tile at
  # a position other than the first row. We can calculate which m-dimension tile
  # each group starts in by floor-dividing its offset with `tm` and then count
  # tile visits with a histogram.
  #
  # To avoid double counting tile visits from the group that owns the tile,
  # filter these out by assigning their tile id to `tile_m` (one beyond the max)
  # such that they're ignored by the subsequent histogram. Also filter out any
  # group which is empty.
  #
  # TODO: Invert 'partial_tile_mask' predicates to be more clear.
  partial_tile_mask = ((group_offsets[:-1] % tm) == 0) | (group_sizes == 0)

  # Explicitly enable tiles for zero sized groups, if specified. This covers
  # zero sized groups that start on a tile-aligned row and those that do not.
  if visit_empty_groups:
    partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

  partial_tile_ids = jnp.where(
      partial_tile_mask, tiles_m, group_offsets[:-1] // tm
  )

  tile_visits = (
      jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0]
      + 1
  )

  # Create the m-dimension tile ids for each grid index based on the visit
  # counts for each tile.
  m_tile_ids = jnp.repeat(
      jnp.arange(tiles_m, dtype=jnp.int32),
      tile_visits.astype(jnp.int32),
      total_repeat_length=tiles_m + num_groups - 1,
  )

  # Account for sharding.
  #
  # Find the start of the groups owned by our shard and shift the group_ids and
  # m_tile_ids s.t. the metadata for our tiles are at the front of the arrays.
  #
  # TODO: Move this offset into the kernel to avoid these rolls.
  first_tile_in_shard = (group_ids < start_group).sum()
  group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
  m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

  # Calculate the number of tiles we need to compute for our shard.
  #
  # Remove tile visits that belong to a group not in our shard.
  iota = jnp.arange(num_groups, dtype=jnp.int32)
  active_group_mask = (iota <= end_group) & (iota >= start_group)
  group_tiles = jnp.where(active_group_mask, group_tiles, 0)
  num_tiles = group_tiles.sum()
  return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_store_mask(
    *,
    grid_id: jax.Array,
    group_metadata: GroupMetadata,
    tm: int,
    tn: int,
) -> jax.Array:
  """Mask for rows that belong to the current group in the current tile."""
  group_offsets, group_ids, m_tile_ids = group_metadata
  group_id = group_ids[grid_id]
  group_start = group_offsets[group_id]
  group_end = group_offsets[group_id + 1]
  m_id = m_tile_ids[grid_id] * tm
  iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
  return (iota >= group_start) & (iota < group_end)


_TilingFn = Callable[[int, int, int], tuple[int, int, int] | None]


@functools.partial(
    jax.jit,
    static_argnames=[
        "precision",
        "out_dtype",
        "tiling",
        "transpose_rhs",
        "interpret",
        "input_buffer_count",
        "activation",
        "manual_axis_type",
    ],
)
def gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    precision,
    out_dtype: jnp.dtype,
    tiling: tuple[int, int, int] | _TilingFn | None = (128, 128, 128),
    input_buffer_count: int = 2,
    group_offset: jax.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    manual_axis_type: jax.sharding.ManualAxisType | None = None,
    activation: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
  """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jax.Array with shape [m, k].
    rhs: A 3d, jax.Array with shape [num_groups, k, n].
    precision: The precision to use for the dot operation.
    group_sizes: A 1d, jax.Array with shape [num_groups] and jnp.int32 dtype.
    out_dtype: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    input_buffer_count: The buffering count to use for the lhs input buffer.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    transpose_rhs: True if the rhs needs to be transposed.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.
    manual_axis_type: Manual axis type for the operation
      https://docs.jax.dev/en/latest/notebooks/shard_map.html.
    activation: Activation function to apply to the output of the dot operation.

  Returns:
    A 2d, jax.Array with shape [m, n].
  """
  group_sizes = _validate_args(lhs, rhs, group_sizes)

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if group_offset.shape:
      raise ValueError(
          f"group_offset must be a ()-shaped array. Got: {group_offset.shape}."
      )
    group_offset = group_offset[None]

  m, k = lhs.shape
  n = rhs.shape[1 if transpose_rhs else 2]

  if callable(tiling):
    tiling = tiling(m, k, n)

  if tiling is None:
    raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

  tm, tk, tn = tiling
  tiles_k = pl.cdiv(k, tk)
  tiles_n = pl.cdiv(n, tn)

  group_metadata, num_active_tiles = make_group_metadata(
      group_sizes=group_sizes,
      m=m,
      tm=tm,
      start_group=group_offset[0],
      num_nonzero_groups=rhs.shape[0],
      visit_empty_groups=False,
  )
  group_offsets, group_ids, _ = group_metadata

  def dot(x, y, preferred_element_type, *, precision=precision):
    # TODO: Pallas-MTPU doesn't support `DotAlgorithmPreset`.
    if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      x = x.astype(jnp.bfloat16)
      y = y.astype(jnp.bfloat16)
      precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
      preferred_element_type = jnp.float32

    rhs_contracting_dim = 1 if transpose_rhs else 0
    return jax.lax.dot_general(
        x,
        y,
        dimension_numbers=(((1,), (rhs_contracting_dim,)), ((), ())),
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  def _maybe_get_subslice(x, idx: int, subslice_count: int | None, axis: int):
    """Returns a subslice of the given array along the given axis."""
    if subslice_count == 1:  # we're not quantized or full-channel
      return jax.tree.map(lambda x: x[...], x)
    assert axis in (0, 1)

    def getter(x):
      size = x.shape[axis] // subslice_count
      s, e = idx * size, (idx + 1) * size
      return x[s:e, :] if axis == 0 else x[:, s:e]

    return jax.tree.map(getter, x)

  def kernel(
      step,
      unused_grid,
      group_metadata,
      _,
      lhs_ref,
      rhs_ref,
      out_ref,
      acc_scratch,
      *,
      subchannel_iters: int,
  ):
    grid_id, k_i = step.index[1], step.index[2]

    @pl.when(k_i == 0)
    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    def accum(is_last_k_tile):
      with jax.named_scope(f"accum:{is_last_k_tile=}"):
        scales_per_k_tile = subchannel_iters
        sc_tile = tk // scales_per_k_tile

        # compute the actual number of iterations for the last tile
        iterations = scales_per_k_tile
        if is_last_k_tile:
          iterations = pl.cdiv(k - tk * (tiles_k - 1), sc_tile)

        for it in range(iterations):
          # read the value of the input buffers
          lhs = _maybe_get_subslice(lhs_ref, it, scales_per_k_tile, 1)
          rhs_axis = 1 if transpose_rhs else 0
          rhs = _maybe_get_subslice(rhs_ref, it, scales_per_k_tile, rhs_axis)

          # mask values if the current tile extends beyond reduction axis size
          is_last_subtile = (
              is_last_k_tile and (tiles_k - 1) * tk + (it + 1) * sc_tile >= k
          )
          if is_last_subtile and (k_rem := ((k % tk) % sc_tile)) != 0:
            iota = lambda x, d: lax.broadcasted_iota(jnp.int32, x.shape, d)
            lhs = jnp.where(iota(lhs, 1) < k_rem, lhs, 0)
            rhs = jnp.where(iota(rhs, rhs_axis) < k_rem, rhs, 0)

          # perform the dot operation
          is_int = lambda x: jnp.issubdtype(x.dtype, jnp.integer)
          acc_dtype = jnp.int32 if is_int(lhs) and is_int(rhs) else jnp.float32
          out = dot(lhs, rhs, acc_dtype)

          # accumulate and possibly store the output
          acc_scratch[...] += out.astype(acc_scratch.dtype)
          if is_last_subtile:
            mask = _get_store_mask(
                grid_id=grid_id, group_metadata=group_metadata, tm=tm, tn=tn
            )
            acc = acc_scratch[...]
            if activation is not None:
              acc = activation(acc)
            acc = jax.lax.select(mask, acc, out_ref[...].astype(acc.dtype))
            out_ref[...] = acc.astype(out_dtype)

    lax.cond(k_i == tiles_k - 1, lambda: accum(True), lambda: accum(False))

  def lhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, group_offset  # Unused.
    # lhs is (m, k). Load the [tm, tk] matrix for this m-tile.
    _, _, m_tile_ids = group_metadata
    return m_tile_ids[grid_id], k_i

  def rhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
    # rhs is (num_groups, k, n). Load the [tk, tn] matrix based on the group id
    # for this m-tile.
    _, group_ids, _ = group_metadata
    if transpose_rhs:
      k_i, n_i = n_i, k_i

    # NOTE: If we're working on only a shard of the rhs we need to adjust the
    # group index we load from to account for this. The group_ids are in the
    # "unsharded" domain.
    return group_ids[grid_id] - group_offset[0], k_i, n_i

  def out_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
    del k_i, group_offset  # Unused.
    # out is (m, n). Load the [tm, tn] matrix for this m-tile.
    _, _, m_tile_ids = group_metadata
    return m_tile_ids[grid_id], n_i

  lhs_block_spec = pl.BlockSpec((tm, tk), lhs_index_map)
  if transpose_rhs:
    rhs_block_spec = pl.BlockSpec((None, tn, tk), rhs_index_map)
  else:
    rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_index_map)
  out_block_spec = pl.BlockSpec((tm, tn), out_index_map)

  subchannel_iters = 1

  lhs_bytes = jax.tree.reduce(lambda acc, x: acc + x.size * x.itemsize, lhs, 0)
  rhs_bytes = k * n * rhs.itemsize

  out_bytes = (m * n) * jnp.dtype(out_dtype).itemsize
  bytes_accessed = lhs_bytes * tiles_n + rhs_bytes * group_ids.size + out_bytes
  cost_estimate = pl.CostEstimate(
      flops=2 * m * k * n, bytes_accessed=bytes_accessed, transcendentals=0
  )
  kernel_name = _KERNEL_NAME_GMM
  if transpose_rhs:
    kernel_name += "_transpose_rhs"
  metadata = dict(
      prefer_element_type=jnp.dtype(out_dtype).name,
      tiling=dict(tile_m=tm, tile_k=tk, tile_n=tn),
      transpose_rhs=transpose_rhs,
      input_buffer_count=input_buffer_count,
  )
  call_gmm = custom_buffered_pallas_call(
      functools.partial(kernel, subchannel_iters=subchannel_iters),
      out_shape=jax.ShapeDtypeStruct(
          (m, n), out_dtype, manual_axis_type=manual_axis_type
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[lhs_block_spec, rhs_block_spec],
          out_specs=out_block_spec,
          grid=(tiles_n, num_active_tiles, tiles_k),
          scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary")
      ),
      interpret=interpret,
      cost_estimate=cost_estimate,
      name=kernel_name,
      metadata=dict(xprof_metadata=json.dumps(metadata)),
      input_buffer_count=(input_buffer_count, 2),  # rhs always default buffered
  )

  with jax.named_scope(kernel_name):
    out = call_gmm(group_metadata, group_offset, lhs, rhs)

  if rhs.shape[0] < group_sizes.shape[0]:  # Need to zero uninitialized memory.
    group_start = group_offsets[group_offset[0]]
    group_end = group_offsets[group_offset[0] + rhs.shape[0]]
    row_idxs = jnp.arange(out.shape[0], dtype=jnp.int32)
    valid_mask = (row_idxs >= group_start) & (row_idxs < group_end)
    out = jnp.where(valid_mask[:, None], out, 0)
  return out


# ==============================================================================
# Tile-size heuristic and forward wrapper
# (inlined from tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Pallas Mosaic TPU Ragged Dot config."""

  tile_m: int = 128
  tile_k: int = 128
  tile_n: int = 128
  input_buffer_count: int = 2
  combine_scopes: bool = False


def _fit_within_tpu_vmem(
    input_tiles: list[tuple[int, int, Any]],
    output_tile: tuple[int, int, Any],
    input_buffer_count: int,
    utilize_ratio: float = 0.75,
) -> bool:
  """Returns whether the given tiling fits within TPU VMEM."""
  total_size = 0
  for tile in input_tiles:
    tile_size = np.prod(tile[:-1])
    dtype = tile[-1]
    # For Quantized types, we assume that the upper bound on the tile size is
    # bfloat16. That is, the QArray tile size (qvalue + scale) should be less
    # than or equal to the bfloat16 tile size.
    num_bytes = max(jnp.dtype(dtype).itemsize, 2)
    tile_size *= num_bytes
    total_size += tile_size * input_buffer_count

  # Now do the same for the output tile.
  output_tile_size = np.prod(output_tile[:-1])
  dtype = output_tile[-1]
  # For Quantized types, we assume that the upper bound on the tile size is
  # bfloat16. That is, the QArray tile size (qvalue + scale) should be less
  # than or equal to the bfloat16 tile size.
  num_bytes = max(jnp.dtype(dtype).itemsize, 2)
  output_tile_size *= num_bytes
  # To account for accumulation buffer and prefetch, otherwise we see OOM
  # errors.
  total_size += 3 * output_tile_size

  # Get the TPU scoped VMEM size. Default to half of the VMEM capacity.
  tpu_info = pltpu.get_tpu_info()
  scoped_vmem_capacity = tpu_info.vmem_capacity_bytes / 2
  if tpu_info.generation == 6:
    scoped_vmem_capacity /= 2
  elif tpu_info.generation == 5:
    scoped_vmem_capacity /= 4
  return total_size <= scoped_vmem_capacity * utilize_ratio


def _deflate_tile(current_tile: int) -> int:
  if current_tile <= 128:
    return 128
  new_tile = current_tile // 2
  # Ensure partial tiles are always multiples of 128
  return max((new_tile // 128) * 128, 128)


def _get_heuristics_config(lhs: jax.Array, rhs: jax.Array) -> Config:
  """Tile sizes for the forward gmm: the largest tiles that fit in VMEM."""
  if pltpu.get_tpu_info().generation < 5:
    return Config()
  input_buffer_count = 2

  # Extract the dimensions of the input arrays. Here m, k, n, are the
  # dimensions of the forward gmm.
  (m, k), (_, _, n) = lhs.shape, rhs.shape

  # Main heuristic: start with the largest possible tile size and "deflate" it
  # until it fits within VMEM.

  # Default m to 1024.
  tile_m = min(m, 1024)
  tile_n = n
  tile_k = k
  # Deflate m, min size for m is 128.
  while (
      not _fit_within_tpu_vmem(
          [(tile_m, tile_k, lhs.dtype), (tile_k, tile_n, rhs.dtype)],
          (tile_m, tile_n, lhs.dtype),
          input_buffer_count,
      )
      and tile_m > 128
  ):
    tile_m = _deflate_tile(tile_m)

  # Deflate n, min size for n is 128.
  while (
      not _fit_within_tpu_vmem(
          [(tile_m, tile_k, lhs.dtype), (tile_k, tile_n, rhs.dtype)],
          (tile_m, tile_n, lhs.dtype),
          input_buffer_count,
      )
      and tile_n > 128
  ):
    tile_n = _deflate_tile(tile_n)

  # Last priority: deflate k (reduction dimension), min size for k is 128.
  while (
      not _fit_within_tpu_vmem(
          [(tile_m, tile_k, lhs.dtype), (tile_k, tile_n, rhs.dtype)],
          (tile_m, tile_n, lhs.dtype),
          input_buffer_count,
      )
      and tile_k > 128
  ):
    tile_k = _deflate_tile(tile_k)

  return Config(
      tile_m=tile_m,
      tile_k=tile_k,
      tile_n=tile_n,
      input_buffer_count=input_buffer_count,
  )


# Device kind that `pltpu.get_tpu_info()` reports when this file is not run on
# a TPU (Pallas interpret mode, shape inference). On a TPU the real device is
# used.
_OFF_TPU_DEVICE_KIND = 'TPU v6e'


def _tpu_info_context():
  """Context in which `pltpu.get_tpu_info()` resolves, also off TPU."""
  if jax.default_backend() == 'tpu':
    return contextlib.nullcontext()
  device = jax.sharding.AbstractDevice(
      device_kind=_OFF_TPU_DEVICE_KIND, num_cores=1, platform='tpu'
  )
  mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
  return jax.sharding.use_abstract_mesh(mesh)


def forward_pallas(lhs, rhs, group_sizes):
  """Ragged dot forward pass on the Pallas TPU `gmm` kernel.

  Takes the same arguments as `forward`. Returns `(out, residuals)`; see the
  module docstring. Tile sizes come from the Tokamax heuristic for the device.
  Off TPU the kernel runs in Pallas interpret mode.
  """
  # Pallas-Mosaic-TPU requires all non-expert dimensions to be >= 128.
  if any(size < 128 for size in tuple(lhs.shape[-2:]) + tuple(rhs.shape[-2:])):
    raise NotImplementedError(
        f'RaggedDot inputs must be >= 128, but {lhs.shape=}, {rhs.shape=}'
    )
  # Tokamax's canonical form of the default matmul precision.
  precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
  preferred_element_type = jnp.result_type(lhs, rhs)
  with _tpu_info_context():
    config = _get_heuristics_config(lhs, rhs)
    out = gmm(
        lhs,  # [m, k]
        rhs,  # [g, k, n]
        group_sizes=group_sizes,
        precision=precision,
        out_dtype=preferred_element_type,
        tiling=(config.tile_m, config.tile_k, config.tile_n),
        interpret=jax.default_backend() != 'tpu',
        input_buffer_count=config.input_buffer_count,
    )
  residuals = out
  return out, residuals


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: 8x7b (Mixtral 8x7B expert FFN, bf16, random groups).
    'mixtral_8x7b': {
        'name': 'ragged_dot_mixtral_8x7b_bwd_pallas_fwd',
        'model': 'Mixtral-8x7B',
        'operator': 'ragged_dot_backward',
        'num_rows': 8192,
        'in_features': 4096,
        'out_features': 14336,
        'num_groups': 8,
    },
    # Tiny shapes for CPU smoke tests.
    'smoke': {
        'name': 'ragged_dot_smoke_bwd_pallas_fwd',
        'model': 'Mixtral-8x7B',
        'operator': 'ragged_dot_backward',
        'num_rows': 2048,
        'in_features': 256,
        'out_features': 128,
        'num_groups': 6,
    },
}
CONFIG = CONFIGS['mixtral_8x7b']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(lhs, rhs, group_sizes, dout)`; `dout` is the output cotangent."""
  cfg = _resolve_config(config)
  M, K = cfg['num_rows'], cfg['in_features']
  N, G = cfg['out_features'], cfg['num_groups']
  keys = jax.random.split(jax.random.key(42), 4)
  # Value ranges follow the Tokamax ragged dot tests: normal(0, 0.1) operands.
  lhs = jax.random.normal(keys[0], (M, K)) * 0.1
  rhs = jax.random.normal(keys[1], (G, K, N)) * 0.1
  # Each row is assigned to a uniformly random group, as Tokamax does for the
  # group sizes of its benchmark specs (`generate_group_sizes`). The sizes are
  # uneven and sum to M.
  group_ids = jax.random.randint(keys[2], (M,), 0, G)
  group_sizes = jnp.bincount(group_ids, length=G).astype(jnp.int32)
  # The VJP is linear in the cotangent; normal(0, 0.1) keeps the gradients
  # between about 0.1 and 1 in rms for the configs above.
  dout = jax.random.normal(keys[3], (M, N)) * 0.1
  # Sample in float32 and cast, so every dtype sees the same values.
  return lhs.astype(dtype), rhs.astype(dtype), group_sizes, dout.astype(dtype)


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 3, 'diff_argnums': (0, 1)}


def workload(lhs, rhs, group_sizes, dout):
  """Returns `(dlhs, drhs)`, the VJP of `forward` with `dout`."""
  _, vjp_fn = jax.vjp(lambda l, r: forward(l, r, group_sizes), lhs, rhs)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  The forward multiplies every row of `lhs` (`K` values) by one `K x N` weight
  matrix: `2 * M * K * N`. The backward is two products of the same size,
  `dlhs[m] = dout[m] @ rhs[g].T` and `drhs[g] = lhs[rows of g].T @ dout[rows of
  g]`, so it costs twice the forward and the total is 3x.
  """
  cfg = _resolve_config(config)
  M, K, N = cfg['num_rows'], cfg['in_features'], cfg['out_features']
  return int(3 * 2 * M * K * N)


def benchmark(num_warmup=5, num_iters=100, config=None):
  """Benchmark and return results dict."""
  import time

  cfg = _resolve_config(config)
  inputs = create_inputs(config=cfg)
  fn = jax.jit(workload)
  for _ in range(num_warmup):
    out = jax.block_until_ready(fn(*inputs))
  times = []
  for _ in range(num_iters):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*inputs))
    times.append(time.perf_counter() - t0)
  times = np.array(times) * 1000
  return {
      'name': cfg['name'],
      'model': cfg['model'],
      'operator': cfg['operator'],
      'config': {
          k: v for k, v in cfg.items() if k not in ('name', 'model', 'operator')
      },
      'time_ms': round(float(np.mean(times)), 4),
      'std_ms': round(float(np.std(times)), 4),
      'output_shape': [list(t.shape) for t in jax.tree.leaves(out)],
      'status': 'success',
  }


if __name__ == '__main__':
  print(json.dumps(benchmark()))
