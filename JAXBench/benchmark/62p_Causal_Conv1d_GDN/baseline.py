from __future__ import annotations

"""Causal Conv1D + Gated Delta Net (GDN) — Pallas TPU Kernel.

Self-contained implementation.
"""
import dataclasses
import enum
import functools
import sys
import types
from typing import Any, Optional, Tuple
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

# ==============================================================================
# Inlined causal_conv1d_gdn/config.py
# ==============================================================================
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
# ==============================================================================

import dataclasses
import enum
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_VMEM_LIMIT_FACTOR: float = 0.80


class GDNMode(enum.StrEnum):
  BATCHED = enum.auto()
  PER_SEQ = enum.auto()

  def get_seq_tile_size(self, tile_size: int) -> int:
    if self == GDNMode.BATCHED:
      return tile_size
    return 1

  def get_chunk_size(self, tile_size: int) -> int:
    if self == GDNMode.BATCHED:
      return 1
    return tile_size


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
  act_in: jnp.dtype
  act_out: jnp.dtype
  compute: jnp.dtype
  recurrent_state: jnp.dtype
  conv_state: jnp.dtype


def get_vmem_limit_bytes(
    vmem_limit_factor: float = DEFAULT_VMEM_LIMIT_FACTOR,
) -> int:
  """Returns the maximum allowable VMEM capacity budget in bytes."""
  tpu_info = pltpu.get_tpu_info()
  return int(vmem_limit_factor * tpu_info.vmem_capacity_bytes)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfig:
  mode: GDNMode
  dtypes: Dtypes
  batch_size: int
  dim_size: int
  kernel_size: int
  tile_size: int
  num_kq_heads: int
  num_v_heads: int
  kq_head_dim: int
  v_head_dim: int
  num_buffers: int = 2

  @property
  def chunk_size(self) -> int:
    return self.mode.get_chunk_size(self.tile_size)

  @property
  def seq_tile_size(self) -> int:
    return self.mode.get_seq_tile_size(self.tile_size)

  @property
  def prev_kernel_size(self) -> int:
    return self.kernel_size - 1

  @property
  def v_dim_size(self) -> int:
    return self.num_v_heads * self.v_head_dim

  @property
  def kq_dim_size(self) -> int:
    return self.num_kq_heads * self.kq_head_dim

  @property
  def v_per_kq_head(self) -> int:
    return self.num_v_heads // self.num_kq_heads

  @property
  def aligned_num_v_heads(self) -> int:
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    return pl.cdiv(self.num_v_heads, num_lanes) * num_lanes

  def get_kernel_name(self) -> str:
    return (
        f"fused_conv1d_gdn_{self.mode.value}_b{self.seq_tile_size}"
        f"_c{self.chunk_size}"
    )

  def get_metadata(self) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(self)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
      key = jax.tree_util.keystr(path, simple=True, separator=".")
      if not isinstance(val, str | int | float):
        val = str(val)
      ret[key] = val
    return ret

  def get_out_shape(self) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        (self.batch_size, self.num_v_heads, self.v_head_dim),
        self.dtypes.act_out,
    )

  def get_scratch_shape_dict(self) -> dict[str, Any]:
    conv_shape = (self.seq_tile_size, self.prev_kernel_size, 1, self.dim_size)
    recurrent_shape = (
        self.seq_tile_size,
        self.num_v_heads,
        self.kq_head_dim,
        self.v_head_dim,
    )

    carry_conv_scratch = carry_recurrent_scratch = None
    # NOTE: Currently, batched mode only supports case where 1 seq = 1 tile.
    # Therefore, inter tile carry is not needed.
    if self.mode != GDNMode.BATCHED:
      carry_conv_scratch = pltpu.VMEM(conv_shape, jnp.float32)
      carry_recurrent_scratch = pltpu.VMEM(recurrent_shape, jnp.float32)

    return dict(
        carry_conv_scratch_ref=carry_conv_scratch,
        carry_recurrent_scratch_ref=carry_recurrent_scratch,
    )


config = types.SimpleNamespace(
    GDNMode=GDNMode,
    Dtypes=Dtypes,
    get_vmem_limit_bytes=get_vmem_limit_bytes,
    GDNConfig=GDNConfig,
)

# ==============================================================================
# Inlined causal_conv1d_gdn/memory_ref.py
# ==============================================================================
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
# ==============================================================================

import dataclasses
import functools
from typing import Any
import json

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# inlined config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvWeightsRef:
  weight: Any
  bias: Any | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNWeightsRef:
  a_log: Any
  dt_bias: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightRefs:
  conv: ConvWeightsRef
  gdn: GDNWeightsRef


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
  """Maps physical 1-D data into logical N-D representation."""

  data: Any
  shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  def _get_pos(self, indices: tuple[Any, ...]) -> Any:
    strides = pl.strides_from_shape(self.shape)
    assert len(strides) == len(indices)

    pos = 0
    for stride, idx in zip(strides, indices):
      pos += stride * idx
    return pos

  def __getitem__(self, indices: tuple[Any, ...]) -> Any:
    return self.data[self._get_pos(indices)]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  num_tiles: Any
  p_id_to_s_idx: SmemWrapper
  p_id_to_r_base: SmemWrapper
  p_id_to_r_size: SmemWrapper
  p_id_is_first_tile: SmemWrapper
  p_id_is_last_tile: SmemWrapper
  s_idx_has_initial_state: Any
  s_idx_to_state_indices: Any

  @classmethod
  def create(
      cls,
      cfgs: config.GDNConfig,
      num_tiles: jax.Array,
      p_id_to_s_idx: jax.Array,
      p_id_to_r_base: jax.Array,
      p_id_to_r_size: jax.Array,
      p_id_is_first_tile: jax.Array,
      p_id_is_last_tile: jax.Array,
      s_idx_has_initial_state: jax.Array,
      s_idx_to_state_indices: jax.Array,
  ) -> "MetadataRef":
    # NOTE: First dim does not matter when it comes to calculating stride.
    shape = (1, cfgs.seq_tile_size)
    return cls(
        num_tiles=num_tiles,
        p_id_to_s_idx=SmemWrapper(p_id_to_s_idx, shape),
        p_id_to_r_base=SmemWrapper(p_id_to_r_base, shape),
        p_id_to_r_size=SmemWrapper(p_id_to_r_size, shape),
        p_id_is_first_tile=SmemWrapper(p_id_is_first_tile, shape),
        p_id_is_last_tile=SmemWrapper(p_id_is_last_tile, shape),
        s_idx_has_initial_state=s_idx_has_initial_state,
        s_idx_to_state_indices=s_idx_to_state_indices,
    )

  def __len__(self) -> int:
    return len(jax.tree_util.tree_leaves(self))


@dataclasses.dataclass(frozen=True, kw_only=True)
class BaseBufferedRef(pltpu.BufferedRef):

  cfg: config.GDNConfig = dataclasses.field(metadata=dict(static=True))
  # NOTE: Despite being ref, metadata_ref should be set to static. This is
  # because the memory will be allocated outside of kernel and metadata_ref
  # merely points to the reference.
  metadata_ref: MetadataRef = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pyrefly: ignore[bad-override]
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: jax.Array,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfg: config.GDNConfig,
      metadata_ref: MetadataRef,
  ):
    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
    )
    return cls(
        cfg=cfg,
        metadata_ref=metadata_ref,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class InBufferedRef(BaseBufferedRef):

  def copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      r_base = self.metadata_ref.p_id_to_r_base[p_id, idx]
      dma_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
      pltpu.make_async_copy(
          src_ref.at[pl.ds(r_base, dma_size)],
          vmem_ref.at[idx, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
          sem,
      ).start()

  def wait_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      dma_size += self.metadata_ref.p_id_to_r_size[p_id, idx]

    pltpu.make_async_copy(
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutBufferedRef(BaseBufferedRef):

  def copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      r_base = self.metadata_ref.p_id_to_r_base[p_id, idx]
      dma_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
      pltpu.make_async_copy(
          vmem_ref.at[idx, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
          dst_ref.at[pl.ds(r_base, dma_size)],
          sem,
      ).start()

  def wait_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      dma_size += self.metadata_ref.p_id_to_r_size[p_id, idx]

    pltpu.make_async_copy(
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[0, pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class StateBufferedRef(BaseBufferedRef):

  def copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):

      is_first_tile = self.metadata_ref.p_id_is_first_tile[p_id, idx]
      s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
      state_idx = self.metadata_ref.s_idx_to_state_indices[s_idx]
      has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
      should_read = jnp.logical_and(is_first_tile, has_initial_state)
      dma_size = jnp.where(should_read, 1, 0)

      pltpu.make_async_copy(
          src_ref.at[pl.ds(state_idx, dma_size)],
          vmem_ref.at[pl.ds(idx, dma_size)],  # pyrefly: ignore[missing-attribute]
          sem,
      ).start()

  def wait_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      is_first_tile = self.metadata_ref.p_id_is_first_tile[p_id, idx]
      s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
      has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
      should_read = jnp.logical_and(is_first_tile, has_initial_state)
      dma_size += jnp.where(should_read, 1, 0)

    pltpu.make_async_copy(
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()

  def copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    for idx in range(self.cfg.seq_tile_size):
      is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
      s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
      state_idx = self.metadata_ref.s_idx_to_state_indices[s_idx]
      dma_size = jnp.where(is_last_tile, 1, 0)

      pltpu.make_async_copy(
          vmem_ref.at[pl.ds(idx, dma_size)],  # pyrefly: ignore[missing-attribute]
          dst_ref.at[pl.ds(state_idx, dma_size)],
          sem,
      ).start()

  def wait_out(self, dst_ref: jax.Array, grid_indices: tuple[int | jax.Array]):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    p_id = grid_indices[0]

    dma_size = 0
    for idx in range(self.cfg.seq_tile_size):
      is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
      dma_size += jnp.where(is_last_tile, 1, 0)

    pltpu.make_async_copy(
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        vmem_ref.at[pl.ds(0, dma_size)],  # pyrefly: ignore[missing-attribute]
        sem,
    ).wait()


def create_allocs(
    metadata_ref: MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    out_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[
    InBufferedRef,
    InBufferedRef,
    InBufferedRef,
    StateBufferedRef,
    StateBufferedRef,
    OutBufferedRef,
]:
  qkv_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.dim_size)
  ba_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.aligned_num_v_heads)

  out_shape = (
      cfg.seq_tile_size,
      cfg.chunk_size,
      cfg.num_v_heads,
      cfg.v_head_dim,
  )
  conv_shape = (cfg.seq_tile_size, cfg.prev_kernel_size, 1, cfg.dim_size)
  recurrent_shape = (
      cfg.seq_tile_size,
      cfg.num_v_heads,
      cfg.kq_head_dim,
      cfg.v_head_dim,
  )

  pipeline_mode = pl.Buffered(buffer_count=cfg.num_buffers, use_lookahead=False)

  block_spec_partial = functools.partial(
      pl.BlockSpec,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pipeline_mode,
  )

  qkv_spec = block_spec_partial(block_shape=qkv_shape)
  ba_spec = block_spec_partial(block_shape=ba_shape)
  in_buffered_partial = functools.partial(
      InBufferedRef.input,
      buffer_count=pipeline_mode.buffer_count,
      use_lookahead=pipeline_mode.use_lookahead,
      cfg=cfg,
      metadata_ref=metadata_ref,
  )
  qkv_alloc = in_buffered_partial(spec=qkv_spec, dtype_or_type=qkv_ref)
  b_alloc = in_buffered_partial(spec=ba_spec, dtype_or_type=b_ref)
  a_alloc = in_buffered_partial(spec=ba_spec, dtype_or_type=a_ref)

  out_alloc = OutBufferedRef.output(
      spec=block_spec_partial(block_shape=out_shape),
      dtype_or_type=out_ref,
      buffer_count=pipeline_mode.buffer_count,
      use_lookahead=pipeline_mode.use_lookahead,
      cfg=cfg,
      metadata_ref=metadata_ref,
  )

  conv_spec = block_spec_partial(block_shape=conv_shape)
  recurrent_spec = block_spec_partial(block_shape=recurrent_shape)
  state_buffered_partial = functools.partial(
      StateBufferedRef.input_output,
      buffer_count=pipeline_mode.buffer_count,
      use_lookahead=pipeline_mode.use_lookahead,
      cfg=cfg,
      metadata_ref=metadata_ref,
  )
  conv_alloc = state_buffered_partial(
      spec=conv_spec, dtype_or_type=conv_state_ref
  )
  recurrent_alloc = state_buffered_partial(
      spec=recurrent_spec, dtype_or_type=recurrent_state_ref
  )

  return qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc


memory_ref = types.SimpleNamespace(
    ConvWeightsRef=ConvWeightsRef,
    GDNWeightsRef=GDNWeightsRef,
    WeightRefs=WeightRefs,
    SmemWrapper=SmemWrapper,
    MetadataRef=MetadataRef,
    BaseBufferedRef=BaseBufferedRef,
    InBufferedRef=InBufferedRef,
    OutBufferedRef=OutBufferedRef,
    StateBufferedRef=StateBufferedRef,
    create_allocs=create_allocs,
)

# ==============================================================================
# Inlined causal_conv1d_gdn/metadata.py
# ==============================================================================
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
# ==============================================================================

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

# inlined
# inlined


def compute_batched_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    end_seq: jax.Array,
) -> memory_ref.MetadataRef:
  """Metadata for computing multiple sequences per tile."""

  max_seqs = seq_lens.size
  all_seqs = jnp.arange(max_seqs)

  # NOTE: Only supports use case where query_lens[i] = 1 where i < end_seq.
  # This must be guaranteed by the function caller.
  # TODO(b/534541682): Add error handling when above condition is not met.
  query_lens = query_start_loc[1:] - query_start_loc[:-1]
  is_valid_seqs = jnp.where(all_seqs < end_seq, True, False)
  has_initial_state = (seq_lens - query_lens) > 0
  all_valid_seqs = jnp.where(is_valid_seqs, all_seqs, 0)

  return memory_ref.MetadataRef.create(
      cfgs=cfg,
      num_tiles=pl.cdiv(end_seq, cfg.tile_size),
      p_id_to_s_idx=all_valid_seqs,
      p_id_to_r_base=all_valid_seqs,
      p_id_to_r_size=jnp.where(is_valid_seqs, 1, 0),
      p_id_is_first_tile=is_valid_seqs,
      p_id_is_last_tile=is_valid_seqs,
      s_idx_has_initial_state=has_initial_state,
      s_idx_to_state_indices=state_indices,
  )


def compute_per_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    start_seq: jax.Array,
    end_seq: jax.Array,
) -> memory_ref.MetadataRef:
  """Metadata for computing single sequence per tile."""

  max_seqs = seq_lens.size
  max_tokens = cfg.batch_size
  all_seqs = jnp.arange(max_seqs)
  all_tokens = jnp.arange(max_tokens)

  # Shift to ensure first element is for start_seq.
  query_start_loc = jnp.roll(query_start_loc, shift=-start_seq)
  seq_lens = jnp.roll(seq_lens, shift=-start_seq)
  state_indices = jnp.roll(state_indices, shift=-start_seq)

  query_lens = query_start_loc[1:] - query_start_loc[:-1]
  # NOTE: query_lens is used for calculating num_tiles. Defensive programming
  # that masks out all the other values (seq_lens, state_indices) are not needed
  # since they will not be visited as long as num_tiles is correct.
  num_seqs = end_seq - start_seq
  query_lens = jnp.where(all_seqs < num_seqs, query_lens, 0)

  # Calculate number of tiles needed for each sequence.
  s_idx_to_num_tiles = pl.cdiv(query_lens, cfg.chunk_size)
  # Calculate starting p_id of each sequence.
  s_idx_to_start_p_id = jnp.cumulative_sum(
      s_idx_to_num_tiles, include_initial=True
  )
  # Map tile index to seq index.
  # Consider following case:
  # all_seqs = [0 1 2 3 4]
  # s_idx_to_num_tiles = [1 2 3 0 1]
  # jnp.repeat will return following results:
  # p_id_to_s_idx = [0 1 1 2 2 2 4]
  # This means p_id_to_s_idx[i] will point to its corresponding seq index.

  # NOTE: To make jnp.repeat jit compilable, we add total_repeat_length. This
  # introduces padding to p_id_to_s_idx[i] where i >= num_tiles. Since the
  # kernel only checks value up-to p_id_to_s_idx[num_tiles-1], padded value
  # will not impact kernel execution.
  p_id_to_s_idx = jnp.repeat(
      all_seqs, s_idx_to_num_tiles, total_repeat_length=max_tokens
  )
  # Map program id (p_id) to tile id of a sequence.
  p_id_to_t_id = all_tokens - s_idx_to_start_p_id[p_id_to_s_idx]
  # Map tile index to starting row of its activation.
  p_id_to_r_base = (
      query_start_loc[p_id_to_s_idx] + p_id_to_t_id * cfg.chunk_size
  )
  # Calculate number of rows to calculate / fetch for each tile.
  p_id_to_r_size = jnp.minimum(
      query_start_loc[p_id_to_s_idx + 1] - p_id_to_r_base,
      cfg.tile_size,
  )

  # Calculate predicate used for state DMA. State is read if program id (p_id)
  # is the first tile of a sequence and the sequence had been computed before
  # (chunked prefill, decode, etc). State is written if the program id is the
  # last tile of a sequence.
  has_initial_state = (seq_lens - query_lens) > 0
  p_id_is_first_tile = p_id_to_t_id == 0
  p_id_is_last_tile = p_id_to_t_id == (s_idx_to_num_tiles[p_id_to_s_idx] - 1)

  # NOTE: Since query_lens[i] = 0 where i >= num_seqs, s_idx_to_num_tiles[i]
  # where i >= num_seqs will also be 0. Therefore, s_idx_to_num_tiles.sum()
  # will contain number of tiles for valid sequence.
  num_tiles = s_idx_to_num_tiles.sum()

  return memory_ref.MetadataRef.create(
      cfgs=cfg,
      num_tiles=num_tiles,
      p_id_to_s_idx=p_id_to_s_idx,
      p_id_to_r_base=p_id_to_r_base,
      p_id_to_r_size=p_id_to_r_size,
      p_id_is_first_tile=p_id_is_first_tile,
      p_id_is_last_tile=p_id_is_last_tile,
      s_idx_has_initial_state=has_initial_state,
      s_idx_to_state_indices=state_indices,
  )


metadata = types.SimpleNamespace(
    compute_batched_seq_metadata=compute_batched_seq_metadata,
    compute_per_seq_metadata=compute_per_seq_metadata,
)

# ==============================================================================
# Inlined causal_conv1d_gdn/tiling.py
# ==============================================================================
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
# ==============================================================================

"""Dynamic tiling heuristics and VMEM memory estimation for Fused Conv1D-GDN."""

from jax.experimental import pallas as pl
import jax.numpy as jnp

# inlined config


def align_to(x: int, alignment: int) -> int:
  """Aligns an integer upward to the nearest multiple of alignment."""
  return pl.cdiv(x, alignment) * alignment


def get_vmem_estimate_bytes(
    tile_b: int,
    chunk_sz: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    act_in_bytes: int,
    act_out_bytes: int,
    conv_state_bytes: int,
    rec_state_bytes: int,
    num_lanes: int,
    conv_state_dim_size: int,
    is_decode: bool = False,
) -> int:
  """Estimates total on-chip VMEM footprint in bytes for a GDN tile."""
  aligned_num_v_heads = align_to(n_v, num_lanes)
  aligned_d_k = align_to(d_k, num_lanes)
  aligned_d_v = align_to(d_v, num_lanes)
  dim_size = align_to(2 * n_kq * d_k + n_v * d_v, num_lanes)
  aligned_out_dim = align_to(n_v * d_v, num_lanes)

  # 1. Double-buffered input activation buffers (QKV, A, B).
  qkv_bytes = 2 * (tile_b * chunk_sz * dim_size * act_in_bytes)
  b_bytes = 2 * (tile_b * chunk_sz * aligned_num_v_heads * act_in_bytes)
  a_bytes = 2 * (tile_b * chunk_sz * aligned_num_v_heads * act_in_bytes)

  # 2. Double-buffered state cache buffers (convolution and recurrent states).
  conv_state_buffer_bytes = 2 * (
      tile_b * max(0, kernel_size - 1) * conv_state_dim_size * conv_state_bytes
  )
  recurrent_state_buffer_bytes = 2 * (
      tile_b * n_v * d_v * d_k * rec_state_bytes
  )

  # 3. Double-buffered output activation buffer.
  out_bytes = 2 * (tile_b * chunk_sz * aligned_out_dim * act_out_bytes)

  # 4. Temporary scratch buffers and working state memory.
  # Both decode and prefill maintain working recurrent state representations
  # concurrently with double-buffered recurrent state DMA buffers.
  scratch_recurrent_bytes = 2 * (tile_b * n_v * d_v * d_k * rec_state_bytes)
  if is_decode:
    scratch_conv_bytes = 0
  else:
    scratch_conv_bytes = (
        tile_b * max(0, kernel_size - 1) * dim_size * conv_state_bytes
    )

  # 5. Static weight cache references in on-chip memory.
  weights_bytes = (kernel_size * dim_size * 4) + (aligned_num_v_heads * 8)

  # 6. Working memory for intra-chunk recurrence and projections.
  intermediate_bytes = (
      n_v
      * (5 * chunk_sz * chunk_sz + 3 * chunk_sz * (aligned_d_v + aligned_d_k))
      * 4
  )

  return (
      qkv_bytes
      + b_bytes
      + a_bytes
      + conv_state_buffer_bytes
      + recurrent_state_buffer_bytes
      + out_bytes
      + scratch_conv_bytes
      + scratch_recurrent_bytes
      + weights_bytes
      + intermediate_bytes
  )


def calculate_decode_tile_size(
    batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    conv_state_dim_size: int,
    act_in_dtype: jnp.dtype,
    act_out_dtype: jnp.dtype,
    conv_state_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_capacity_limit_bytes: int,
    kernel_size: int = 4,
) -> int:
  """Derives optimal batch tile size for decode execution.

  Searches candidate batch tile sizes within maximum VMEM capacity limits.

  Args:
    batch_size: Total batch size of the active decode sequence.
    n_kq: Number of key/query heads.
    n_v: Number of value heads.
    d_k: Key head dimension.
    d_v: Value head dimension.
    conv_state_dim_size: Feature dimension size for conv state.
    act_in_dtype: Data type for input activations.
    act_out_dtype: Data type for output activations.
    conv_state_dtype: Data type for conv state cache.
    recurrent_state_dtype: Data type for recurrent state matrix.
    num_lanes: Number of lanes for TPU vector layout alignment.
    vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
    kernel_size: 1D convolution kernel window size.

  Returns:
    Derived batch tile size fitting within VMEM capacity limits.
  """
  # Return a minimum valid tile size of 1 for empty or zero-length batches.
  if batch_size <= 0:
    return 1

  act_in_bytes = jnp.dtype(act_in_dtype).itemsize
  act_out_bytes = jnp.dtype(act_out_dtype).itemsize
  conv_state_bytes = jnp.dtype(conv_state_dtype).itemsize
  rec_state_bytes = jnp.dtype(recurrent_state_dtype).itemsize

  # Search descending power-of-2 tile candidates (up to 32) bounded by
  # batch size.
  decode_candidates = [c for c in (32, 16, 8, 4, 2, 1) if c <= batch_size]
  decode_tile_size = decode_candidates[-1]

  for cand in decode_candidates:
    vmem_est = get_vmem_estimate_bytes(
        tile_b=cand,
        chunk_sz=1,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        act_in_bytes=act_in_bytes,
        act_out_bytes=act_out_bytes,
        conv_state_bytes=conv_state_bytes,
        rec_state_bytes=rec_state_bytes,
        num_lanes=num_lanes,
        conv_state_dim_size=conv_state_dim_size,
        is_decode=True,
    )
    if vmem_est <= vmem_capacity_limit_bytes:
      return cand

  return decode_tile_size


def calculate_mixed_tile_size(
    seq_len: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    conv_state_dim_size: int,
    act_in_dtype: jnp.dtype,
    act_out_dtype: jnp.dtype,
    conv_state_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_capacity_limit_bytes: int,
    kernel_size: int = 4,
) -> int:
  """Derives optimal chunk tile size for prefill and mixed execution.

  Searches candidate tile sizes within maximum VMEM capacity limits.

  Args:
    seq_len: Sequence length of the active prefill or mixed sequence.
    n_kq: Number of key/query heads.
    n_v: Number of value heads.
    d_k: Key head dimension.
    d_v: Value head dimension.
    conv_state_dim_size: Feature dimension size for conv state.
    act_in_dtype: Data type for input activations.
    act_out_dtype: Data type for output activations.
    conv_state_dtype: Data type for conv state cache.
    recurrent_state_dtype: Data type for recurrent state matrix.
    num_lanes: Number of lanes for TPU vector layout alignment.
    vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
    kernel_size: 1D convolution kernel window size.

  Returns:
    Derived chunk tile size fitting within VMEM capacity limits.
  """
  # Return a minimum valid chunk size of 1 for empty or zero-length sequences.
  if seq_len <= 0:
    return 1

  act_in_bytes = jnp.dtype(act_in_dtype).itemsize
  act_out_bytes = jnp.dtype(act_out_dtype).itemsize
  conv_state_bytes = jnp.dtype(conv_state_dtype).itemsize
  rec_state_bytes = jnp.dtype(recurrent_state_dtype).itemsize

  # Limit chunk size to C <= 128: above 128, intra-chunk triangular
  # solve operations and vector register pressure outweigh systolic compute
  # density gains.
  # When value head count is large (n_v >= 64), intra-chunk intermediate
  # memory scales up, so cap chunk search space to C <= 64 to avoid on-chip
  # memory overflow.
  max_chunk_cap = 64 if n_v >= 64 else 128
  prefill_candidates = [
      c
      for c in (128, 64, 32, 16, 8, 4, 2, 1)
      if c <= seq_len and c <= max_chunk_cap
  ]
  mixed_tile_size = prefill_candidates[-1]
  for candidate in prefill_candidates:
    vmem_est = get_vmem_estimate_bytes(
        tile_b=1,
        chunk_sz=candidate,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        act_in_bytes=act_in_bytes,
        act_out_bytes=act_out_bytes,
        conv_state_bytes=conv_state_bytes,
        rec_state_bytes=rec_state_bytes,
        num_lanes=num_lanes,
        conv_state_dim_size=conv_state_dim_size,
        is_decode=False,
    )
    if vmem_est <= vmem_capacity_limit_bytes:
      return candidate

  return mixed_tile_size


def get_tile_sizes(
    batch_size: int,
    num_seqs: int,
    padded_batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    conv_state_dim_size: int,
    act_in_dtype: jnp.dtype,
    act_out_dtype: jnp.dtype,
    conv_state_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    decode_tile_size: int | None = None,
    mixed_tile_size: int | None = None,
) -> tuple[int, int]:
  """Derives optimal decode and mixed tile sizes fitting within VMEM limits."""
  vmem_capacity_limit_bytes = config.get_vmem_limit_bytes()

  if decode_tile_size is None or decode_tile_size <= 0:
    decode_tile_size = calculate_decode_tile_size(
        batch_size=padded_batch_size,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        conv_state_dim_size=conv_state_dim_size,
        act_in_dtype=act_in_dtype,
        act_out_dtype=act_out_dtype,
        conv_state_dtype=conv_state_dtype,
        recurrent_state_dtype=recurrent_state_dtype,
        num_lanes=num_lanes,
        vmem_capacity_limit_bytes=vmem_capacity_limit_bytes,
        kernel_size=kernel_size,
    )

  if mixed_tile_size is None or mixed_tile_size <= 0:
    if batch_size <= num_seqs:
      # When all sequences have length 1 (decode), size prefill chunks to 1.
      effective_prefill_seq_len = 1
    else:
      # Estimate maximum sequence length across uniform and mixed prefill
      # batches. In an adversarial mixed batch of B tokens with N_seq sequences,
      # the largest prefill sequence length is bounded by
      # B - (N_seq - 1) = B - N_seq + 1.
      effective_prefill_seq_len = max(
          batch_size // max(1, num_seqs),
          batch_size - num_seqs + 1,
      )

    mixed_tile_size = calculate_mixed_tile_size(
        seq_len=effective_prefill_seq_len,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        conv_state_dim_size=conv_state_dim_size,
        act_in_dtype=act_in_dtype,
        act_out_dtype=act_out_dtype,
        conv_state_dtype=conv_state_dtype,
        recurrent_state_dtype=recurrent_state_dtype,
        num_lanes=num_lanes,
        vmem_capacity_limit_bytes=vmem_capacity_limit_bytes,
        kernel_size=kernel_size,
    )

  # Guarantee strictly positive tile sizes (>= 1) for Pallas grid compilation.
  decode_tile_size = max(1, min(decode_tile_size, padded_batch_size))
  mixed_tile_size = max(1, min(mixed_tile_size, batch_size))
  return decode_tile_size, mixed_tile_size


tiling = types.SimpleNamespace(
    align_to=align_to,
    get_vmem_estimate_bytes=get_vmem_estimate_bytes,
    calculate_decode_tile_size=calculate_decode_tile_size,
    calculate_mixed_tile_size=calculate_mixed_tile_size,
    get_tile_sizes=get_tile_sizes,
)

# ==============================================================================
# Inlined causal_conv1d_gdn/vmem_ldst.py
# ==============================================================================
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
# ==============================================================================

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# inlined
# inlined


def load_as_qkv_large(
    qkv_vmem_ref: jax.Ref, cfgs: config.GDNConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Split qkv and transpose by performing 1 load per chunk for large layout.

  Args:
    qkv_vmem_ref: qkv reference in VMEM containing concatenated values of q, k,
      and v of shape [seq_tile_size, chunk_size, 1, num_kq_heads * kq_head_dim *
      2 + num_v_heads * v_head_dim].
    cfgs: GDN configuration object.

  Returns:
    q, k: [seq_tile_size, num_kq_heads, chunk_size, kq_head_dim]
    v: [seq_tile_size, num_v_heads, chunk_size, v_head_dim]
  """

  num_lanes = pltpu.get_tpu_info().num_lanes
  lanes_per_col = qkv_vmem_ref.shape[-1] // num_lanes
  kq_lanes_per_head = cfgs.kq_head_dim // num_lanes
  k_offset = cfgs.num_kq_heads * kq_lanes_per_head

  q_large_list = []
  k_large_list = []
  v_large_list = []

  qkv_slot_flat_ref = qkv_vmem_ref.reshape(-1, num_lanes)  # pyrefly: ignore[missing-attribute]
  for kq_head in range(cfgs.num_kq_heads):
    q_head_list = []
    k_head_list = []
    for lane in range(kq_lanes_per_head):
      q_lane = kq_head * kq_lanes_per_head + lane
      k_lane = k_offset + q_lane

      q_head_list.append(qkv_slot_flat_ref[q_lane::lanes_per_col])
      k_head_list.append(qkv_slot_flat_ref[k_lane::lanes_per_col])
    q_large_list.append(jnp.concat(q_head_list, axis=-1))
    k_large_list.append(jnp.concat(k_head_list, axis=-1))
  v_offset = kq_lanes_per_head * cfgs.num_kq_heads * 2
  v_lanes_per_head = cfgs.v_head_dim // num_lanes
  for v_head in range(cfgs.num_v_heads):
    v_head_list = []
    for lane in range(v_lanes_per_head):
      v_lane = v_offset + v_head * v_lanes_per_head + lane
      v_head_list.append(qkv_slot_flat_ref[v_lane::lanes_per_col])
    v_large_list.append(jnp.concat(v_head_list, axis=-1))

  q_large = jnp.stack(q_large_list, axis=0)
  k_large = jnp.stack(k_large_list, axis=0)
  v_large = jnp.stack(v_large_list, axis=0)

  return q_large, k_large, v_large


def load_as_qkv_compact(
    qkv_vmem_ref: jax.Ref, cfg: config.GDNConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Split qkv and transpose by performing 1 load per head for compact layout.

  Args:
    qkv_vmem_ref: qkv reference in VMEM containing concatenated values of q, k,
      and v of shape [seq_tile_size, chunk_size, 1, num_kq_heads * kq_head_dim *
      2 + num_v_heads * v_head_dim].
    cfg: GDN configuration object.

  Returns:
    q, k: [seq_tile_size, num_kq_heads, chunk_size, 1, kq_head_dim]
    v: [seq_tile_size, num_v_heads, chunk_size, 1, v_head_dim]
  """

  k_offset = cfg.num_kq_heads * cfg.kq_head_dim
  v_offset = cfg.num_kq_heads * 2 * cfg.kq_head_dim

  q_compact_list = []
  k_compact_list = []
  v_compact_list = []

  for kq_head in range(cfg.num_kq_heads):
    q_start = kq_head * cfg.kq_head_dim
    q_end = q_start + cfg.kq_head_dim
    k_start = k_offset + q_start
    k_end = k_start + cfg.kq_head_dim
    q_compact_list.append(qkv_vmem_ref[..., q_start:q_end])
    k_compact_list.append(qkv_vmem_ref[..., k_start:k_end])
  for v_head in range(cfg.num_v_heads):
    v_start = v_offset + v_head * cfg.v_head_dim
    v_end = v_start + cfg.v_head_dim
    v_compact_list.append(qkv_vmem_ref[..., v_start:v_end])

  q_compact = jnp.stack(q_compact_list, axis=1)
  k_compact = jnp.stack(k_compact_list, axis=1)
  v_compact = jnp.stack(v_compact_list, axis=1)

  return q_compact, k_compact, v_compact


def load_compact_to_large(vmem_ref: jax.Ref) -> jax.Array:
  """Use strided load to convert compact to large layout without transpose."""

  # NOTE: Only support 32-bits for now.
  assert vmem_ref.dtype.itemsize == 4
  assert vmem_ref.shape[-2] == 1
  col_size = vmem_ref.shape[-1]
  new_shape = vmem_ref.shape[:-2] + (col_size,)
  tpu_info = pltpu.get_tpu_info()
  num_lanes = tpu_info.num_lanes

  vreg_list = []
  vmem_ref = vmem_ref.reshape(-1, col_size)  # pyrefly: ignore[missing-attribute]
  for col_start in range(0, col_size, num_lanes):
    col_end = min(col_start + num_lanes, col_size)
    vreg = vmem_ref[..., col_start:col_end]
    vreg_list.append(vreg)
  return jnp.concat(vreg_list, axis=-1).reshape(new_shape)


def load_and_select_states(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    conv_state_slot_ref: jax.Ref,
    recurrent_slot_ref: jax.Ref,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Load correct states from HBM or prior tile, and masks invalid states.

  Reference metadata to select the appropriate prior states. If `is_first_tile`
  is True, it selects states read from HBM. If it is False, it selects
  carry states from previous tile. If `has_initial_state` is False, states are
  zero initialized.

  Args:
    metadata_ref: Metadata reference containing grid and sequence mappings.
    p_id: Current Pallas program ID.
    conv_state_slot_ref: Convolution state read from HBM of shape
      [seq_tile_size, prev_kernel_size, 1, dim_size].
    recurrent_slot_ref: Recurrent state read from HBM of shape [seq_tile_size,
      num_v_heads, kq_head_dim, v_head_dim].
    carry_conv_scratch_ref: Optional inter-tile convolution carry of shape
      [seq_tile_size, prev_kernel_size, 1, dim_size].
    carry_recurrent_scratch_ref: Optional inter-tile recurrent state carry of
      shape [seq_tile_size, num_v_heads, kq_head_dim, v_head_dim].
    cfg: GDN configuration object.

  Returns:
    real_sizes: Valid token count per sequence tile of shape [seq_tile_size].
    prev_conv_state: Selected convolution state of shape [seq_tile_size,
      prev_kernel_size, 1, dim_size] in float32.
    prev_recurrent_state: Selected recurrent state of shape [seq_tile_size,
      num_v_heads, kq_head_dim, v_head_dim].
  """

  real_sizes_list = []
  prev_conv_state_list = []
  prev_recurrent_state_list = []

  for idx in range(cfg.seq_tile_size):
    s_idx = metadata_ref.p_id_to_s_idx[p_id, idx]
    real_sizes = metadata_ref.p_id_to_r_size[p_id, idx]
    is_first_tile = metadata_ref.p_id_is_first_tile[p_id, idx]
    has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

    # NOTE: Conv1D mandates fp32 due to its usage of compact layout.
    hbm_conv_state = conv_state_slot_ref[idx].astype(jnp.float32)
    prev_conv_state = jnp.where(has_initial_state, hbm_conv_state, 0)

    if carry_conv_scratch_ref is not None:
      prev_tile_conv = carry_conv_scratch_ref[idx]
      prev_conv_state = jnp.where(
          is_first_tile, prev_conv_state, prev_tile_conv
      )

    hbm_recurrent_state = recurrent_slot_ref[idx]
    prev_recurrent_state = jnp.where(has_initial_state, hbm_recurrent_state, 0)

    if carry_recurrent_scratch_ref is not None:
      prev_tile_recurrent_scratch = carry_recurrent_scratch_ref[idx]
      prev_recurrent_state = jnp.where(
          is_first_tile, prev_recurrent_state, prev_tile_recurrent_scratch
      )

    real_sizes_list.append(real_sizes)
    prev_conv_state_list.append(prev_conv_state)
    prev_recurrent_state_list.append(prev_recurrent_state)

  real_sizes = jnp.stack(real_sizes_list, axis=0)
  prev_conv_state = jnp.stack(prev_conv_state_list, axis=0)
  prev_recurrent_state = jnp.stack(prev_recurrent_state_list, axis=0)

  return real_sizes, prev_conv_state, prev_recurrent_state


def load_activation_as_compact(
    qkv_vreg: jax.Array,
    qkv_vmem_ref: jax.Ref,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Load activations from VMEM as a compact layout."""

  qkv_vmem_ref[...] = qkv_vreg
  q_compact, k_compact, v_compact = load_as_qkv_compact(qkv_vmem_ref, cfgs)
  b_compact = jnp.expand_dims(b_vmem_ref[...], axis=1)
  a_compact = jnp.expand_dims(a_vmem_ref[...], axis=1)
  return q_compact, k_compact, v_compact, b_compact, a_compact


def load_activation_as_large(
    qkv_vreg: jax.Array,
    qkv_vmem_ref: jax.Ref,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Load activations from VMEM as a large layout."""

  qkv_vmem_ref[...] = qkv_vreg

  q_large_list = []
  k_large_list = []
  v_large_list = []
  for idx in range(cfgs.seq_tile_size):
    q_large, k_large, v_large = load_as_qkv_large(qkv_vmem_ref.at[idx], cfgs)
    q_large_list.append(q_large)
    k_large_list.append(k_large)
    v_large_list.append(v_large)

  q_large = jnp.stack(q_large_list, axis=0)
  k_large = jnp.stack(k_large_list, axis=0)
  v_large = jnp.stack(v_large_list, axis=0)
  b_large = load_compact_to_large(b_vmem_ref)
  a_large = load_compact_to_large(a_vmem_ref)
  b_large = jnp.expand_dims(b_large, axis=1)
  a_large = jnp.expand_dims(a_large, axis=1)

  return q_large, k_large, v_large, b_large, a_large


vmem_ldst = types.SimpleNamespace(
    load_as_qkv_large=load_as_qkv_large,
    load_as_qkv_compact=load_as_qkv_compact,
    load_compact_to_large=load_compact_to_large,
    load_and_select_states=load_and_select_states,
    load_activation_as_compact=load_activation_as_compact,
    load_activation_as_large=load_activation_as_large,
)

# ==============================================================================
# Inlined causal_conv1d_gdn/compute_conv1d.py
# ==============================================================================
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
# ==============================================================================

import jax
import jax.numpy as jnp

# inlined config


def causal_conv1d(
    real_sizes: jax.Array,  # [seq]
    lhs: jax.Array,  # [seq, chunk, q, dim_size]
    conv_weight: jax.Array,  # [prev_kernel_size, 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  """Perform causal Conv1D. Returns Conv1D output and convolution states."""

  assert lhs.ndim == 4

  out_list = []

  for c_idx in range(cfg.chunk_size):
    out = jnp.zeros((cfg.seq_tile_size, 1, cfg.dim_size), jnp.float32)

    end_idx = c_idx + cfg.prev_kernel_size
    start_idx = 1 + end_idx - cfg.kernel_size
    for k in range(cfg.kernel_size):
      lhs_curr = lhs[:, start_idx + k]
      out += lhs_curr * conv_weight[k : k + 1]

    if conv_bias is not None:
      out += conv_bias.reshape(1, 1, -1)

    out_list.append(out)

  # Last prev_kernel_size elements needs to be returned as conv_state. However,
  # real_sizes may be smaller than chunk_size. Therefore, slicing last
  # prev_kernel_size elements does not gurantee numeric correctness. Instead,
  # kernel iterate each rows and perform masking to fetch correct values.
  # NOTE: lhs[:, : prev_kernel_size] can be skipped since they were loaded from
  # previous conv states.
  new_conv_state = lhs[:, 1 : cfg.kernel_size]
  real_sizes = real_sizes.reshape(-1, 1, 1, 1)
  # NOTE: Even though for loop is invoked twice, since they are static loops,
  # compiler will perform loop fusion.
  for c_idx in range(2, cfg.chunk_size + 1):
    row_end = c_idx + cfg.prev_kernel_size
    new_conv_state = jnp.where(
        c_idx == real_sizes,
        lhs[:, c_idx:row_end],
        new_conv_state,
    )

  return jnp.stack(out_list, axis=1), new_conv_state


compute_conv1d = types.SimpleNamespace(causal_conv1d=causal_conv1d)

# ==============================================================================
# Inlined causal_conv1d_gdn/compute_gdn.py
# ==============================================================================
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
# ==============================================================================

import jax
import jax.numpy as jnp

# inlined config


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
  norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps)
  return x / norm


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
  match jnp.dtype(dtype).itemsize:
    case 4:
      return jnp.int32
    case 2:
      return jnp.int16
    case _:
      raise ValueError(f"Unsupported dtype: {dtype}")


# NOTE: Fork of recurrent_scan_v2.py but applied various optimizations.
def invert_triangular_matrix(t: jax.Array, block_size: int = 16) -> jax.Array:
  """Compute invert matrix of a given triauglar matrix."""

  # NOTE: if chunk_size=1, compiler will perform DCE.
  out_dtype = t.dtype
  chunk = t.shape[-1]
  block_size = min(block_size, chunk)
  num_blocks = chunk // block_size

  def local_forward_sub(t_mat: jax.Array, b_mat: jax.Array) -> jax.Array:
    x_list = []
    for i in range(block_size):
      b_i = b_mat[:, i, :]
      if i == 0:
        x_i = b_i
      else:
        stacked_x = jnp.stack(x_list, axis=1)
        all_prev_t = t_mat[:, i, :i]
        prev_sum = jnp.sum(all_prev_t[..., None] * stacked_x, axis=1)
        x_i = b_i - prev_sum
      x_list.append(x_i)
    return jnp.stack(x_list, axis=1)

  x_blocks = []
  iota_r = jax.lax.broadcasted_iota(jnp.int32, t.shape, 1)
  iota_c = jax.lax.broadcasted_iota(jnp.int32, t.shape, 2)
  identity_mask = jnp.where(iota_r == iota_c, 1.0, 0.0)
  for i in range(num_blocks):
    start, end = i * block_size, (i + 1) * block_size
    e_block = identity_mask[:, start:end, :]

    if i == 0:
      target_b = e_block
    else:
      interaction_t = t[:, start:end, :start]
      solved_x = jnp.concatenate(x_blocks, axis=1)
      prev_sum = jax.lax.dot(
          interaction_t,
          solved_x,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      )
      target_b = e_block - prev_sum

    # NOTE: Utilize fp32 to minimize cost of sublane rolling.
    local_t = t[:, start:end, start:end].astype(jnp.float32)
    x_block = local_forward_sub(local_t, target_b)
    x_blocks.append(x_block.astype(out_dtype))

  return jnp.concatenate(x_blocks, axis=1)


def fused_transpose_broadcast(
    x: jax.Array, src_dim: int, dst_dim: int
) -> jax.Array:
  """Perform 1D transpose where results are broadcasted along src_dim."""
  assert x.shape[dst_dim] == 1

  dtype = x.dtype
  mask_dtype = get_mask_dtype(dtype)
  mask_shape = list(x.shape)
  mask_size = mask_shape[src_dim]
  mask_shape[dst_dim] = mask_size
  src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
  dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
  mask = src_mask == dst_mask
  return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


def chunked_gdn_per_seq(
    q_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [num_v_heads, chunk, v_head_dim]
    gating_log: jax.Array,  # [1, 1, num_v_heads]
    beta: jax.Array,  # [1, 1, num_v_heads]
    state_prev: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  """Perform chunked GDN over input [num_heads, chunk, head_dim]."""

  # NOTE: Repeat along non lane/sublane dim is free.
  q_repeat = jnp.repeat(q_large, cfg.v_per_kq_head, axis=0)
  k_repeat = jnp.repeat(k_large, cfg.v_per_kq_head, axis=0)

  # Compute cumulative sum of decay.
  # [1, 1, num_v_heads]
  g_cum_sum_list = [gating_log[:, :1]]
  for row in range(1, cfg.chunk_size):
    g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[:, row : row + 1])
  # [1, chunk, num_v_heads]
  g_cum_sum_log = jnp.concat(g_cum_sum_list, axis=1)

  # [num_v_heads, chunk, 1]
  g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log, src_dim=2, dst_dim=0)
  g_cum_sum_log = g_cum_sum_log[: cfg.num_v_heads]
  beta = fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)
  beta_large = beta[: cfg.num_v_heads]

  # [num_v_heads, 1, chunk]
  g_cum_sum_log_t = fused_transpose_broadcast(
      g_cum_sum_log, src_dim=1, dst_dim=2
  )
  # [num_v_heads, chunk, chunk]
  g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
  gating_map = jnp.exp(g_cum_sum_diff_log)
  # [num_v_heads, chunk, 1]
  gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
  # [num_v_heads, chunk, 1]
  gating_forward = jnp.exp(g_cum_sum_log)
  # [num_v_heads, 1, 1]
  gating_last = gating_forward[:, -1:]

  mask_dtype = get_mask_dtype(cfg.dtypes.compute)
  iota_r = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 1)
  iota_c = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 2)
  identity_mask = iota_r == iota_c
  strictly_lower_mask = iota_r > iota_c
  lower_mask = iota_r >= iota_c
  # [num_v_heads, chunk, chunk]
  gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)

  # [num_v_heads, chunk, kq_head_dim]
  k_beta_repeat = k_repeat * beta_large

  # [num_v_heads, chunk, chunk]
  beta_k_k_t = jax.lax.dot(
      k_beta_repeat,
      k_repeat,
      dimension_numbers=(((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  ).astype(cfg.dtypes.compute)
  gating_beta_k_k_t = gating_map_masked * beta_k_k_t
  t = jnp.where(identity_mask, 1, gating_beta_k_k_t)

  # [num_v_heads, chunk, chunk]
  t_inv = invert_triangular_matrix(t)

  # [num_v_heads, chunk, v_head_dim]
  v_beta_large = v_large * beta_large
  # [num_v_heads, chunk, kv_head_dim]
  k_beta_gating = k_beta_repeat * gating_forward
  # NOTE: If v_head_dim < mxu size, concatenating them will help increase mxu
  # utilization. Also, if v_head_dim is multiple of lane size, concat / split
  # along lane dim is free - making this optimization strictly beneficial.
  # [num_v_heads, chunk, v_head_dim + kq_head_dim]
  merged_v_k = jnp.concat([v_beta_large, k_beta_gating], axis=-1)
  merged_uw = jax.lax.dot(
      t_inv,
      merged_v_k,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  ).astype(cfg.dtypes.compute)

  # [num_v_heads, chunk, v_head_dim]
  u, w = jnp.split(merged_uw, [cfg.v_head_dim], axis=-1)

  # [num_v_heads, chunk, kq_head_dim]
  q_large_gating = q_repeat * gating_forward
  # NOTE: Concatenate lhs with same rhs to leverage weight
  # stationary architecture.
  # [num_v_heads, 2 * chunk, kq_head_dim]
  merged_w_q = jnp.concat([w, q_large_gating], axis=1)
  # [num_v_heads, 2 * chunk, v_head_dim]
  merged_ws_out_updated = jax.lax.dot(
      merged_w_q,
      state_prev,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )

  # NOTE: Splitting along non sublane/lane dim is free.
  ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
  ws = ws.astype(cfg.dtypes.compute)

  # [num_v_heads, chunk, v_head_dim]
  u_ws = u - ws

  # [num_v_heads, chunk, kq_head_dim]
  k_repeat_gating = k_repeat * gating_backward

  # [num_v_heads, kq_head_dim, v_head_dim]
  state_new = jax.lax.dot(
      k_repeat_gating,
      u_ws,
      dimension_numbers=(((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )

  # [num_v_heads, kq_head_dim, v_head_dim]
  state_updated = state_prev * gating_last
  state = state_updated + state_new

  # [num_kq_heads, chunk, chunk]
  out_qk = jax.lax.dot(
      q_large,
      k_large,
      dimension_numbers=(((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  ).astype(cfg.dtypes.compute)
  # NOTE: must perform repeat after matmul to reduce required compute.
  # [num_v_heads, chunk, chunk]
  out_qk = jnp.repeat(out_qk, cfg.v_per_kq_head, axis=0)
  out_qk *= gating_map
  out_qk = jnp.where(lower_mask, out_qk, 0)

  # [num_v_heads, chunk, v_head_dim]
  out_new = jax.lax.dot(
      out_qk,
      u_ws,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  out = out_updated + out_new

  return out, state


def chunked_gdn(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  """Perform chunked GDN over input [seq, num_heads, chunk, head_dim]."""

  mask_dtype = get_mask_dtype(cfg.dtypes.compute)
  iota = jax.lax.broadcasted_iota(
      mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1), 2
  )
  mask = iota < real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)

  # [seqs, num_kq_heads, chunk, kq_head_dim]
  q_large = jnp.where(mask, q_large.astype(cfg.dtypes.compute), 0)
  k_large = jnp.where(mask, k_large.astype(cfg.dtypes.compute), 0)
  # [seqs, num_v_heads, chunk, v_head_dim]
  v_large = jnp.where(mask, v_large.astype(cfg.dtypes.compute), 0)

  b_large = b_large.astype(cfg.dtypes.compute)
  a_large = a_large.astype(cfg.dtypes.compute)

  a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
  dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)

  # NOTE: Any element-wise computations should occur before repeat.
  q_large = l2_norm(q_large)
  q_scale = cfg.kq_head_dim**-0.5
  q_large *= q_scale
  k_large = l2_norm(k_large)

  # [seqs, 1, chunk, num_v_heads]
  beta = jax.nn.sigmoid(b_large)
  gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_large + dt_bias)

  beta = jnp.where(mask, beta, 0)
  # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
  # be masked to 1 since it signifies that strength of state from previous row
  # will be 1 (i.e., no decay) if current row is invalid.
  gating_log = jnp.where(mask, gating_log, 0)

  out_list = []
  state_list = []
  for idx in range(cfg.seq_tile_size):
    out, state = chunked_gdn_per_seq(
        q_large[idx],
        k_large[idx],
        v_large[idx],
        gating_log[idx],
        beta[idx],
        state_prev[idx],
        cfg,
    )
    out_list.append(out.swapaxes(0, 1))
    state_list.append(state)
  out = jnp.stack(out_list, axis=0)
  state = jnp.stack(state_list, axis=0)
  return out, state


def recurrent_gdn_per_seq(
    q_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact_t: jax.Array,  # [num_kq_heads, chunk, kq_head_dim, 1]
    v_compact: jax.Array,  # [num_v_heads, chunk, 1, v_head_dim]
    gating_log: jax.Array,  # [num_v_heads, chunk, 1, 1]
    beta: jax.Array,  # [num_v_heads, chunk, 1, 1]
    state: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  """Perform recurrent GDN over input [num_heads, chunk, 1, head_dim]."""

  out_list = []
  for c_idx in range(cfgs.chunk_size):
    # [num_v_heads, 1, kq_head_dim]
    q_curr = q_compact[:, c_idx]
    q_curr = jnp.repeat(q_curr, cfgs.v_per_kq_head, axis=0)
    k_curr = k_compact[:, c_idx]
    k_curr = jnp.repeat(k_curr, cfgs.v_per_kq_head, axis=0)

    # [num_v_heads, 1, v_head_dim]
    v_curr = v_compact[:, c_idx]

    # [num_v_heads, kq_head_dim, 1]
    k_curr_t = k_compact_t[:, c_idx]
    k_curr_t = jnp.repeat(k_curr_t, cfgs.v_per_kq_head, axis=0)

    # [num_v_heads, 1, 1]
    beta_curr = beta[:, c_idx]
    gating_curr = gating_log[:, c_idx]

    # [num_v_heads, kq_head_dim, v_head_dim]
    state_updated = state * gating_curr

    # [num_v_heads, 1, v_head_dim]
    v_updated = jax.lax.dot(
        k_curr,
        state_updated,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)

    # [num_v_heads, 1, v_head_dim]
    v_diff = v_curr - v_updated
    v_new = beta_curr * v_diff

    # [num_v_heads, kq_head_dim, v_head_dim]
    # NOTE: Multiplication with k_curr_t needs to be deferred as much as
    # possible as it expands the dimension size by kq_head_dim.
    state_new = k_curr_t * v_new
    # [num_v_heads, kq_head_dim, v_head_dim]
    state = state_updated + state_new

    # [num_v_heads, 1, v_head_dim]
    out = jax.lax.dot(
        q_curr,
        state,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)

    out_list.append(out[:, 0, :])

  return jnp.stack(out_list, axis=0), state


def recurrent_gdn(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  """Perform recurrent GDN over input [seq, num_heads, chunk, 1, head_dim]."""

  mask_dtype = get_mask_dtype(cfg.dtypes.compute)
  iota = jax.lax.broadcasted_iota(
      mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2
  )
  mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

  # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
  q_compact = jnp.where(mask, q_compact.astype(cfg.dtypes.compute), 0)
  k_compact = jnp.where(mask, k_compact.astype(cfg.dtypes.compute), 0)
  # [seqs, num_v_heads, chunk, 1, v_head_dim]
  v_compact = jnp.where(mask, v_compact.astype(cfg.dtypes.compute), 0)

  b_compact = b_compact.astype(cfg.dtypes.compute)
  a_compact = a_compact.astype(cfg.dtypes.compute)

  a_log = a_log.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)
  dt_bias = dt_bias.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)

  # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
  q_compact = l2_norm(q_compact)
  q_scale = cfg.kq_head_dim**-0.5
  q_compact *= q_scale
  k_compact = l2_norm(k_compact)
  k_compact_t = fused_transpose_broadcast(k_compact, src_dim=4, dst_dim=3)

  beta = jax.nn.sigmoid(b_compact)
  gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_compact + dt_bias)

  beta = jnp.where(mask, beta, 0)
  # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
  # be masked to 1 since it signifies that strength of state from previous row
  # will be 1 (i.e., no decay) if current row is invalid.
  gating_log = jnp.where(mask, gating_log, 0)
  gating_log = jnp.exp(gating_log)

  beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
  beta = beta[:, : cfg.num_v_heads]
  gating_log = fused_transpose_broadcast(gating_log, src_dim=4, dst_dim=1)
  gating_log = gating_log[:, : cfg.num_v_heads]

  # Shapes:
  #   q_compact:    [B, n_kq, 1, 1, d_k] -> q_curr:   [B, n_v, 1, d_k]
  #   k_compact:    [B, n_kq, 1, 1, d_k] -> k_curr:   [B, n_v, 1, d_k]
  #   k_compact_t:  [B, n_kq, 1, d_k, 1] -> k_curr_t: [B, n_v, d_k, 1]
  #   v_compact:    [B, n_v, 1, 1, d_v]  -> v_curr:   [B, n_v, 1, d_v]
  #   beta:         [B, n_v, 1, 1, 1]    -> beta_curr: [B, n_v, 1, 1]
  #   gating_log:   [B, n_v, 1, 1, 1]    -> gating_curr: [B, n_v, 1, 1]
  q_curr = jnp.repeat(q_compact[:, :, 0, 0, :], cfg.v_per_kq_head, axis=1)
  q_curr = jnp.expand_dims(q_curr, axis=2)

  k_curr = jnp.repeat(k_compact[:, :, 0, 0, :], cfg.v_per_kq_head, axis=1)
  k_curr = jnp.expand_dims(k_curr, axis=2)

  k_curr_t = jnp.repeat(k_compact_t[:, :, 0, :, 0], cfg.v_per_kq_head, axis=1)
  k_curr_t = jnp.expand_dims(k_curr_t, axis=3)

  v_curr = jnp.expand_dims(v_compact[:, :, 0, 0, :], axis=2)
  beta_curr = beta[:, :, 0, 0, 0].reshape(
      cfg.seq_tile_size, cfg.num_v_heads, 1, 1
  )
  gating_curr = gating_log[:, :, 0, 0, 0].reshape(
      cfg.seq_tile_size, cfg.num_v_heads, 1, 1
  )

  # 1. State decay update
  state_updated = state_prev * gating_curr

  # 2. Fused [k, q] projection against state
  b_size, n_v_heads = cfg.seq_tile_size, cfg.num_v_heads
  kq_merged = jnp.concat([k_curr, q_curr], axis=2)  # [B, n_v, 2, d_k]
  kq_merged_flat = kq_merged.reshape(-1, 2, cfg.kq_head_dim)
  state_updated_flat = state_updated.reshape(
      -1, cfg.kq_head_dim, cfg.v_head_dim
  )

  kq_proj_flat = jax.lax.dot_general(
      kq_merged_flat,
      state_updated_flat,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  ).astype(
      cfg.dtypes.compute
  )  # [B * n_v, 2, d_v]
  kq_proj = kq_proj_flat.reshape(b_size, n_v_heads, 2, cfg.v_head_dim)

  v_updated = kq_proj[:, :, :1, :]
  out_decayed = kq_proj[:, :, 1:, :]

  # 3. Output difference
  v_diff = v_curr - v_updated
  v_new = beta_curr * v_diff

  qk_dot = jnp.sum(q_curr * k_curr, axis=-1, keepdims=True)  # [B, n_v, 1, 1]
  out = out_decayed + qk_dot * v_new  # [B, n_v, 1, d_v]

  # 4. State rank-1 update
  state_new = k_curr_t * v_new  # [B, n_v, d_k, d_v]
  new_recurrent_state = state_updated + state_new

  # Match Pallas output shape [B, 1, n_v, d_v]:
  out = out.swapaxes(1, 2)

  return out, new_recurrent_state


compute_gdn = types.SimpleNamespace(
    l2_norm=l2_norm,
    get_mask_dtype=get_mask_dtype,
    invert_triangular_matrix=invert_triangular_matrix,
    fused_transpose_broadcast=fused_transpose_broadcast,
    chunked_gdn_per_seq=chunked_gdn_per_seq,
    chunked_gdn=chunked_gdn,
    recurrent_gdn_per_seq=recurrent_gdn_per_seq,
    recurrent_gdn=recurrent_gdn,
)

# ==============================================================================
# Inlined causal_conv1d_gdn/wrapper.py
# ==============================================================================
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
# ==============================================================================

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# inlined namespaces


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size]
    b_slot_ref: jax.Ref,  # [seq, chunk, 1, num_v_heads]
    a_slot_ref: jax.Ref,  # [seq, chunk, 1, num_v_heads]
    conv_state_slot_ref: jax.Ref,  # [seq, prev_kernel_size, 1, dim_size]
    recurrent_slot_ref: jax.Ref,  # [seq, num_v_heads, kq_head, v_head]
    # Outputs.
    out_slot_ref: jax.Array,  # [seq * chunk, num_v_heads, v_head]
    # Scratches.
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    *,
    cfg: config.GDNConfig,
) -> None:
  """Orchestrates computation of Conv1D and GDN for a single tile.

  This kernel acts as a facade adhering to strict separation of concerns. It
  operates VMEM reference without knowledge on DMA logic. Furthermore, the
  kernel invokes vmem_ldst to pre-processes data needed for compute and
  invokes compute_conv1d and compute_gdn for actual compute.

  Args:
    qkv_slot_ref: qkv VMEM ref that stores data loaded from HBM.
    b_slot_ref: b VMEM ref that stores data loaded from HBM.
    a_slot_ref: a VMEM ref that stores data loaded from HBM.
    conv_state_slot_ref: Convolution state VMEM ref that stores data loaded from
      HBM. Data written into this VMEM ref will be used for VMEM to HBM write.
    recurrent_slot_ref: Recurrent state VMEM ref that stores data loaded from
      HBM. Data written into this VMEM ref will be used for VMEM to HBM write.
    out_slot_ref: Output VMEM ref that will be used for VMEM to HBM write.
    metadata_ref: Metadata reference containing grid and sequence mappings.
    weights_ref: Weight references for Conv1D and GDN in VMEM.
    carry_conv_scratch_ref: Optional VMEM scratch reference for inter-tile
      convolution carry.
    carry_recurrent_scratch_ref: Optional VMEM scratch reference for inter-tile
      recurrent state carry.
    cfg: GDN configuration object.
  """

  p_id = pl.program_id(0)

  # Prepare states.
  real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
      metadata_ref=metadata_ref,
      p_id=p_id,
      conv_state_slot_ref=conv_state_slot_ref,
      recurrent_slot_ref=recurrent_slot_ref,
      carry_conv_scratch_ref=carry_conv_scratch_ref,
      carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
      cfg=cfg,
  )

  # Step 1: Conv1D.
  # NOTE: Conv1D requires performing sliding window where inputs are slided
  # across rows. If typical 2D layout was used, multiple rows are stored in a
  # single register which necessitate costly shuffling for every sliding.
  # Therefore, it is extremely important to leverage compact layout that
  # ensures 1 register only stores data from 1 row.
  qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
  qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

  # Prepare conv1d weights.
  conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
  conv_bias = None
  if weights_ref.conv.bias is not None:
    conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

  qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
      real_sizes=real_sizes,
      lhs=qkv_in_compact,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      cfg=cfg,
  )

  conv_state_slot_ref[...] = new_conv_state
  if carry_conv_scratch_ref is not None:
    carry_conv_scratch_ref[...] = new_conv_state

  # Apply activation function.
  qkv_out_compact = jax.nn.silu(qkv_out_compact)

  # Step 2: GDN.

  # Prepare gdn weights.
  padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
  a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
  dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

  # NOTE: Ideally, we want to move this branching logic into gdn.py. However,
  # load_activation_as_compact and load_activation_as_large leverages vmem ldst.
  # Passing refs into gdn.py breaks strict separation of concerns.
  if cfg.chunk_size == 1:
    q_compact, k_compact, v_compact, b_compact, a_compact = (
        vmem_ldst.load_activation_as_compact(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )

    out, new_recurrent_state = compute_gdn.recurrent_gdn(
        q_compact=q_compact,
        k_compact=k_compact,
        v_compact=v_compact,
        b_compact=b_compact,
        a_compact=a_compact,
        state_prev=prev_recurrent,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
        real_sizes=real_sizes,
    )

  else:
    q_large, k_large, v_large, b_large, a_large = (
        vmem_ldst.load_activation_as_large(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )

    out, new_recurrent_state = compute_gdn.chunked_gdn(
        q_large=q_large,
        k_large=k_large,
        v_large=v_large,
        b_large=b_large,
        a_large=a_large,
        state_prev=prev_recurrent,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
        real_sizes=real_sizes,
    )

  # Store output and recurrent to vmem.
  out_slot_ref[...] = out.astype(out_slot_ref.dtype)
  recurrent_slot_ref[...] = new_recurrent_state.astype(recurrent_slot_ref.dtype)

  if carry_recurrent_scratch_ref is not None:
    carry_recurrent_scratch_ref[...] = new_recurrent_state


def outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    _: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
) -> None:
  """Setup memory allocations and emit pipeline for running inner_kernel."""
  del conv_state_out_ref, recurrent_state_out_ref

  qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc = (
      memory_ref.create_allocs(
          metadata_ref=metadata_ref,
          qkv_ref=qkv_ref,
          b_ref=b_ref,
          a_ref=a_ref,
          out_ref=out_ref,
          conv_state_ref=conv_state_ref,
          recurrent_state_ref=recurrent_state_ref,
          cfg=cfg,
      )
  )

  num_tiles = metadata_ref.num_tiles[...]

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(
          inner_kernel,
          cfg=cfg,
      ),
      grid=(num_tiles,),
      in_specs=(
          qkv_alloc.spec,
          b_alloc.spec,
          a_alloc.spec,
          conv_alloc.spec,
          recurrent_alloc.spec,
      ),
      out_specs=(out_alloc.spec,),
  )

  @pl.with_scoped(
      allocations=(
          qkv_alloc,
          b_alloc,
          a_alloc,
          conv_alloc,
          recurrent_alloc,
          out_alloc,
      ),
  )
  def _run(allocations):
    pipeline_func(
        qkv_ref,
        b_ref,
        a_ref,
        conv_state_ref,
        recurrent_state_ref,
        out_ref,
        scratches=(
            metadata_ref,
            weights_ref,
            carry_conv_scratch_ref,
            carry_recurrent_scratch_ref,
        ),
        allocations=allocations,
    )

  _run()


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "decode_tile_size",
        "mixed_tile_size",
        "zero_initialize_out",
        "compute_precision",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,  # [batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size]
    b: jax.Array,  # [batch_size, n_v]
    a: jax.Array,  # [batch_size, n_v]
    conv_state: jax.Array,  # [num_seqs + 1, kernel_size - 1, dim_size]
    recurrent_state: jax.Array,  # [num_seqs + 1, nv, dk, dv]
    conv_weight: jax.Array,  # [kernel_size - 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v]
    query_start_loc: jax.Array,  # [num_seqs + 1]
    state_indices: jax.Array,  # [num_seqs]
    distribution: jax.Array,  # [3]
    seq_lens: jax.Array,  # [num_seqs]
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    decode_tile_size: int | None = None,
    mixed_tile_size: int | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
  """Perform conv1d and gdn in a single fused kernel.

  Args:
    qkv: Mixed query, key, value input tensor of shape [batch_size, dim_size],
      where `dim_size = n_kq * d_k * 2 + n_v * d_v`.
    b: b tensor (for beta) of shape [batch_size, n_v].
    a: a tensor (for g) of shape [batch_size, n_v].
    conv_state: Convolution state cache tensor of shape [num_seqs + 1,
      kernel_size - 1, dim_size] containing the last (kernel_size - 1) tokens
      from the last sequence invocation. The first slot is a null block used for
      padded or invalid tokens. It may contain garbage data if it is a first
      invocation of a sequence.
    recurrent_state: Recurrent state cache tensor of shape [num_seqs + 1, n_v,
      d_k, d_v]. The first slot is a null block used for padded or invalid
      tokens. It may contain garbage data if it is a first invocation of a
      sequence.
    conv_weight: Convolution weight tensor of shape [kernel_size - 1, dim_size].
    conv_bias: Optional convolution bias tensor of shape [dim_size].
    a_log: a_log tensor of shape [n_v].
    dt_bias: dt_bias tensor of shape [n_v].
    query_start_loc: Start locations of sequences of shape [num_seqs + 1].
    state_indices: Indices mapping sequences to state cache slots of shape
      [num_seqs].
    distribution: Tensor of shape [3] int32 — [decode_end, prefill_end,
      mixed_end].
    seq_lens: Sequence lengths for each sequence of shape [num_seqs].
    n_kq: Number of key/query heads.
    n_v: Number of value heads.
    d_k: Key/query dimension.
    d_v: Value dimension.
    kernel_size: Convolution kernel size.
    zero_initialize_out: Whether to zero-initialize the output buffer before
      executing non-batched sequences.
    compute_precision: Computation precision dtype.
    decode_tile_size: Tile size along sequence dimension for decode sequences.
    mixed_tile_size: Tile size along token/chunk dimension for prefill/mixed
      sequences.

  Returns:
    (new_conv_state, new_recurrent_state): Updated convolution state cache and
      recurrent state cache tensors.
    out: Fused output tensor.
  """
  # TODO(b/534537779): Support bf16
  act_in_dtype = qkv.dtype
  act_out_dtype = qkv.dtype
  conv_out_dtype = conv_state.dtype
  recurrent_out_dtype = recurrent_state.dtype
  assert a.dtype == b.dtype == qkv.dtype == act_in_dtype

  with jax.named_scope("fused_conv1d_gdn_input_preprocessing"):
    qkv = qkv.astype(jnp.float32)
    b = b.astype(jnp.float32)
    a = a.astype(jnp.float32)
    conv_state = conv_state.astype(jnp.float32)

    # Step 1: Validate inputs.
    num_seqs = state_indices.size
    batch_size, dim = qkv.shape
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
      assert conv_bias.shape == (dim,)
    assert query_start_loc.shape == (num_seqs + 1,)
    assert state_indices.shape == (num_seqs,)
    assert distribution.shape == (3,)

    # Step 2: Compute tile sizes and pad/reshape activations.
    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    conv_state_dim_size = conv_state.shape[-1]

    decode_tile_size, mixed_tile_size = tiling.get_tile_sizes(
        batch_size=batch_size,
        num_seqs=num_seqs,
        padded_batch_size=padded_batch_size,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        conv_state_dim_size=conv_state_dim_size,
        act_in_dtype=act_in_dtype,
        act_out_dtype=act_out_dtype,
        conv_state_dtype=conv_state.dtype,
        recurrent_state_dtype=recurrent_state.dtype,
        num_lanes=num_lanes,
        decode_tile_size=decode_tile_size,
        mixed_tile_size=mixed_tile_size,
    )

    batch_padding_size = padded_batch_size - batch_size
    aligned_num_v_heads = tiling.align_to(n_v, num_lanes)
    num_v_padding_size = aligned_num_v_heads - n_v
    qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size)))

    qkv = qkv.reshape(padded_batch_size, 1, -1)
    b = b.reshape(padded_batch_size, 1, -1)
    a = a.reshape(padded_batch_size, 1, -1)

    # Step 3: States and weights pre-processing.
    # To eliminate runtime cost, this logic can be moved into model loading
    conv_state_shape = conv_state.shape
    conv_state = conv_state.reshape(-1, kernel_size - 1, 1, dim)
    conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = conv_bias.astype(jnp.float32) if conv_bias is not None else None

    # Step 4: Wrap inputs for the kernel.
    conv_weights = memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias)
    gdn_weights = memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
    weights = memory_ref.WeightRefs(conv=conv_weights, gdn=gdn_weights)

    # Step 5: Create specs.
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

  def call_kernel(
      in_conv_state: jax.Array,
      in_recurrent_state: jax.Array,
      in_act: jax.Array | None,
      mode: config.GDNMode,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    if mode == config.GDNMode.BATCHED:
      tile_size = decode_tile_size
    else:
      tile_size = mixed_tile_size

    cfg = config.GDNConfig(
        mode=mode,
        batch_size=padded_batch_size,
        kernel_size=kernel_size,
        tile_size=tile_size,
        dim_size=dim,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        dtypes=config.Dtypes(
            act_in=act_in_dtype,
            act_out=act_out_dtype,
            compute=compute_precision,
            recurrent_state=in_recurrent_state.dtype,
            conv_state=in_conv_state.dtype,
        ),
    )

    # Step 6: Metadata preprocessing. Will be executed multiple times per-layer
    # but will be CSEed by compiler.
    if mode == config.GDNMode.BATCHED:
      metadata_obj = metadata.compute_batched_seq_metadata(
          cfg=cfg,
          seq_lens=seq_lens,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          end_seq=distribution[0],
      )
    else:
      metadata_obj = metadata.compute_per_seq_metadata(
          cfg=cfg,
          seq_lens=seq_lens,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          start_seq=distribution[0],
          end_seq=distribution[-1],
      )

    metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)

    # Step 7: Handle case where write needs to be done in existing out.
    in_out_spec = None
    input_output_aliases = {len(metadata_obj) + 3: 1, len(metadata_obj) + 4: 2}
    out_shape = cfg.get_out_shape()

    if in_act is None and zero_initialize_out:
      in_act = jnp.zeros_like(out_shape)
    if in_act is not None:
      out_shape = in_act
      in_out_spec = hbm_spec
      input_output_aliases[len(metadata_obj) + 5] = 0

    return pl.pallas_call(
        functools.partial(outer_kernel, cfg=cfg),
        out_shape=(out_shape, in_conv_state, in_recurrent_state),
        in_specs=(
            metadata_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            in_out_spec,
            weights_spec,
        ),
        out_specs=(hbm_spec, hbm_spec, hbm_spec),
        scratch_shapes=cfg.get_scratch_shape_dict(),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=config.get_vmem_limit_bytes(),
        ),
        name=cfg.get_kernel_name(),
        metadata=cfg.get_metadata(),
    )(
        metadata_obj,
        qkv,
        b,
        a,
        in_conv_state,
        in_recurrent_state,
        in_act,
        weights,
    )

  # Pre-allocate zero output buffer when zero_initialize_out is enabled.
  out_act_init = (
      jnp.zeros((padded_batch_size, n_v, d_v), dtype=act_out_dtype)
      if zero_initialize_out
      else None
  )

  def call_kernel_if_active(
      in_conv_state: jax.Array,
      in_recurrent_state: jax.Array,
      in_act: jax.Array | None,
      mode: config.GDNMode,
      is_active: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Conditionally launches Pallas kernel if the pass has active sequences."""
    mode_name = (
        "batched_decode" if mode == config.GDNMode.BATCHED else "per_seq_mixed"
    )

    def _execute():
      with jax.named_scope(f"execute_{mode_name}"):
        return call_kernel(in_conv_state, in_recurrent_state, in_act, mode)

    def _skip():
      act = in_act if in_act is not None else out_act_init
      return act, in_conv_state, in_recurrent_state

    return jax.lax.cond(is_active, _execute, _skip)

  # Phase 1: Decode pass over sequences [0, distribution[0])
  out_act, out_conv_state, out_recurrent_state = call_kernel_if_active(
      conv_state,
      recurrent_state,
      None,
      config.GDNMode.BATCHED,
      distribution[0] > 0,
  )

  # Phase 2: Prefill/mixed pass over sequences
  # [distribution[0], distribution[-1])
  out_act, out_conv_state, out_recurrent_state = call_kernel_if_active(
      out_conv_state,
      out_recurrent_state,
      out_act,
      config.GDNMode.PER_SEQ,
      distribution[-1] > distribution[0],
  )

  out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
  out_conv_state = out_conv_state.astype(conv_out_dtype)
  out_conv_state = out_conv_state.reshape(conv_state_shape)
  out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

  return (out_conv_state, out_recurrent_state), out_act


CONFIGS = {
    "gdn_small": {
        "name": "gdn_small",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 256,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 2,
        "prefill_length": 128,
        "max_reqs": 32,
    },
    "gdn_prefill_medium": {
        "name": "gdn_prefill_medium",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 2048,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 4,
        "prefill_length": 512,
        "max_reqs": 64,
    },
    "gdn_prefill_large": {
        "name": "gdn_prefill_large",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 8192,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 8,
        "prefill_length": 1024,
        "max_reqs": 64,
    },
    "gdn_decode_b64": {
        "name": "gdn_decode_b64",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 64,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 0,
        "prefill_length": 0,
        "num_decode_reqs": 64,
        "max_reqs": 64,
    },
    "gdn_mixed_decode_prefill": {
        "name": "gdn_mixed_decode_prefill",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 1056,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 2,
        "prefill_length": 512,
        "num_decode_reqs": 32,
        "max_reqs": 64,
    },
}
CONFIG = CONFIGS["gdn_small"]


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias, a_log, dt_bias, query_start_loc, state_indices, distribution, seq_lens)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jax.random.split(key, 10)

  num_tokens = cfg["num_tokens"]
  n_kq = cfg["n_kq"]
  n_v = cfg["n_v"]
  d_k = cfg["d_k"]
  d_v = cfg["d_v"]
  kernel_size = cfg.get("kernel_size", 4)
  num_prefill_reqs = cfg.get("num_prefill_reqs", 0)
  prefill_length = cfg.get("prefill_length", 0)
  num_decode_reqs = cfg.get("num_decode_reqs", 0)
  max_reqs = cfg["max_reqs"]

  key_dim = n_kq * d_k
  val_dim = n_v * d_v
  mixed_dim = 2 * key_dim + val_dim

  qkv = jax.random.normal(k1, (num_tokens, mixed_dim), dtype=dtype)
  b = jax.random.normal(k2, (num_tokens, n_v), dtype=dtype)
  a = jax.random.normal(k3, (num_tokens, n_v), dtype=dtype)

  state_size = max_reqs + 1
  conv_state = jax.random.normal(
      k9, (state_size, kernel_size - 1, mixed_dim), dtype=dtype
  )
  recurrent_state = jax.random.normal(
      k10, (state_size, n_v, d_k, d_v), dtype=jnp.float32
  )

  conv_weight = jax.random.normal(k4, (mixed_dim, 1, kernel_size), dtype=dtype)
  conv_bias = jax.random.normal(k5, (mixed_dim,), dtype=dtype)

  a_log = jax.random.normal(k6, (n_v,), dtype=jnp.float32)
  dt_bias = jax.random.normal(k7, (n_v,), dtype=jnp.float32)

  lengths = [1] * num_decode_reqs + [prefill_length] * num_prefill_reqs
  num_seqs = len(lengths)
  lengths_with_past = [1 + 128] * num_decode_reqs + [
      prefill_length
  ] * num_prefill_reqs
  query_start_loc = jnp.cumsum(jnp.array([0] + lengths, dtype=jnp.int32))
  last_val = query_start_loc[-1]
  query_start_loc = jnp.pad(
      query_start_loc,
      (0, max_reqs + 1 - len(query_start_loc)),
      constant_values=last_val,
  )

  available_indices = jax.random.permutation(
      k8, jnp.arange(1, max_reqs + 1, dtype=jnp.int32)
  )
  valid_state_indices = available_indices[:num_seqs]
  state_indices = jnp.pad(valid_state_indices, (0, max_reqs - num_seqs))

  distribution = jnp.array(
      [num_decode_reqs, num_seqs, num_seqs], dtype=jnp.int32
  )
  seq_lens = jnp.pad(
      jnp.array(lengths_with_past, dtype=jnp.int32), (0, max_reqs - num_seqs)
  )

  return (
      qkv,
      b,
      a,
      conv_state,
      recurrent_state,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      query_start_loc,
      state_indices,
      distribution,
      seq_lens,
  )


def get_inputs(dtype=jnp.bfloat16):
  """Returns list of inputs across all defined configurations."""
  return [
      (list(create_inputs(dtype=dtype, config=cfg)), [])
      for cfg in CONFIGS.values()
  ]


def workload(
    qkv,
    b,
    a,
    conv_state,
    recurrent_state,
    conv_weight,
    conv_bias,
    a_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    seq_lens,
):
  """Optimized Pallas TPU Fused Causal Conv1D Gated Delta Net implementation."""
  n_v = b.shape[1]
  n_kq = 2
  d_k = 128
  d_v = 128
  kernel_size = 4
  return fused_conv1d_gdn(
      qkv=qkv,
      b=b,
      a=a,
      conv_state=conv_state,
      recurrent_state=recurrent_state,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      a_log=a_log,
      dt_bias=dt_bias,
      query_start_loc=query_start_loc,
      state_indices=state_indices,
      distribution=distribution,
      seq_lens=seq_lens,
      n_kq=n_kq,
      n_v=n_v,
      d_k=d_k,
      d_v=d_v,
      kernel_size=kernel_size,
  )


def get_flops(config=None):
  """Total FLOPs for the fused Conv1D and recurrent delta rule updates."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )
  num_tokens = cfg["num_tokens"]
  n_kq = cfg["n_kq"]
  n_v = cfg["n_v"]
  d_k = cfg["d_k"]
  d_v = cfg["d_v"]
  kernel_size = cfg.get("kernel_size", 4)
  mixed_dim = 2 * n_kq * d_k + n_v * d_v
  gdn_flops = num_tokens * 4 * n_v * d_k * d_v
  conv_flops = num_tokens * 2 * kernel_size * mixed_dim
  return int(gdn_flops + conv_flops)


def benchmark(num_warmup=5, num_iters=100, config=None):
  """Benchmark and return results dict."""
  import time

  if config is None:
    cfg = CONFIG
  elif isinstance(config, str):
    cfg = CONFIGS[config]
  else:
    cfg = config
  inputs = create_inputs(config=cfg)
  fn = jax.jit(workload)
  for _ in range(num_warmup):
    out = fn(*inputs)
    jax.tree.map(
        lambda x: x.block_until_ready()
        if hasattr(x, "block_until_ready")
        else None,
        out,
    )
  times = []
  for _ in range(num_iters):
    t0 = time.perf_counter()
    out = fn(*inputs)
    jax.tree.map(
        lambda x: x.block_until_ready()
        if hasattr(x, "block_until_ready")
        else None,
        out,
    )
    times.append(time.perf_counter() - t0)
  times = np.array(times) * 1000
  avg = float(np.mean(times))
  flops = get_flops(cfg)
  tflops = round((flops / (avg / 1000)) / 1e12, 2) if avg > 0 else 0.0
  if hasattr(out, "shape"):
    output_shape = list(out.shape)
  elif isinstance(out, (list, tuple)):
    output_shape = [
        list(x.shape) for x in jax.tree.leaves(out) if hasattr(x, "shape")
    ]
  else:
    output_shape = []
  return {
      "name": cfg["name"],
      "model": cfg["model"],
      "operator": cfg["operator"],
      "config": {
          k: v for k, v in cfg.items() if k not in ("name", "model", "operator")
      },
      "time_ms": round(avg, 4),
      "std_ms": round(float(np.std(times)), 4),
      "tflops": tflops,
      "output_shape": output_shape,
      "status": "success",
  }


if __name__ == "__main__":
  print(json.dumps(benchmark()))
