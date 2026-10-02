"""Batched Ragged Paged Attention (Batched RPA) — Pallas TPU Kernel.

Self-contained implementation.
"""
import math
import sys
import types
from typing import Any
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

# ==============================================================================
# Inlined batched_rpa/utils.py
# ==============================================================================
# Source: https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/experimental/batched_rpa/utils.py
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

"""Utility functions for Ragged Paged Attention."""

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


def align_to(a, b):
  """Returns 'a' aligned to 'b'."""
  return pl.cdiv(a, b) * b


def broadcast_minor(src, shape):
  """Broadcasts 'src' to 'shape' in the minor dimension."""
  if src.shape == shape:
    return src
  num_lanes = pltpu.get_tpu_info().num_lanes
  assert src.shape[:-1] == shape[:-1]
  assert src.shape[-1] % num_lanes == 0
  target_minor = align_to(shape[-1], src.shape[-1])
  # no-op concatenation.
  broadcasted = jnp.tile(src, (target_minor // src.shape[-1],))
  return broadcasted[..., : shape[-1]]


def get_dtype_packing(dtype):
  return 32 // jax.dtypes.itemsize_bits(dtype)


def strided_load(ref, start_row, num_rows, step, *, dtype=None):
  """Loads data from HBM with strided access, handling 128-lane alignment."""
  _, row_width = ref.shape
  num_lanes = pltpu.get_tpu_info().num_lanes
  num_sub_lanes = row_width // num_lanes
  ref_flat = ref.reshape(-1, num_lanes)

  # scale indices to match flattened arraw.
  v_start = start_row * num_sub_lanes
  v_num = num_rows * num_sub_lanes
  v_step = step * num_sub_lanes

  # Gather the chunks into the original head dimension.
  chunks = [
      ref_flat[pl.ds(v_start + i, v_num // v_step, v_step)]
      for i in range(num_sub_lanes)
  ]
  vec = jnp.concat(chunks, axis=1)

  return pltpu.bitcast(vec, dtype) if dtype is not None else vec


def strided_store(ref, start, sz, step, val):
  """Stores data to HBM with strided access, handling 128-lane alignment."""
  assert get_dtype_packing(ref.dtype) == 1
  assert ref.dtype == val.dtype
  assert ref.shape == val.shape
  assert ref.ndim == 2
  rows, cols = ref.shape
  num_lanes = pltpu.get_tpu_info().num_lanes
  assert cols % num_lanes == 0
  folds = cols // num_lanes
  ref = ref.reshape(rows * folds, num_lanes)
  start *= folds
  sz *= folds
  step *= folds
  assert sz % step == 0
  for i in range(folds):
    val_slice = val[:, i * num_lanes : (i + 1) * num_lanes]
    ref[pl.ds(start + i, sz // step, step)] = val_slice


def convert_to_target_bitwidth(val, target_bitwidth: int, kv_dtype: jnp.dtype):
  """Converts a value to a target bitwidth."""
  # If we want to convert 32-bits into 32//N number of N-bits value, naive
  # approach would be to perform 32//N number of 32-bits to N-bits conversion.
  # However, we can reduce number of instructions by utilizing binary tree.
  # 0: [32]
  # 1: [16, 16]
  # ...
  # log2(32//N): [N, N, ... N]

  curr_dtype = val.dtype
  curr_bitwidth = jax.dtypes.itemsize_bits(curr_dtype)
  assert target_bitwidth != curr_bitwidth, "No conversion is needed."

  # We split val into two vals (left and right) where each have half of the
  # original bitwidth.
  next_bitwidth = curr_bitwidth // 2
  next_dtype = jnp.dtype(f"uint{next_bitwidth}")

  left = val.astype(next_dtype)

  # Bitwise shift is only supported in uint32.
  val_u32 = pltpu.bitcast(val, jnp.uint32)
  val_u32_shifted = val_u32 >> next_bitwidth
  # Convert back to original dtype.
  val_shifted = pltpu.bitcast(val_u32_shifted, curr_dtype)
  right = val_shifted.astype(next_dtype)

  if next_bitwidth == target_bitwidth:
    k = pltpu.bitcast(left, kv_dtype)
    v = pltpu.bitcast(right, kv_dtype)
    return [(k, v)]
  else:
    left_out = convert_to_target_bitwidth(
        left, target_bitwidth=target_bitwidth, kv_dtype=kv_dtype
    )
    right_out = convert_to_target_bitwidth(
        right, target_bitwidth=target_bitwidth, kv_dtype=kv_dtype
    )
    return left_out + right_out


def has_bank_conflicts(stride: int, distance=24, num_banks=32) -> bool:
  banks = set()
  for i in range(distance):
    bank = (i * stride) % num_banks
    if bank in banks:
      return True
    banks.add(bank)
  return False

cdiv = pl.cdiv
utils = types.SimpleNamespace(
    align_to=align_to,
    cdiv=cdiv,
    get_dtype_packing=get_dtype_packing,
    broadcast_minor=broadcast_minor,
    strided_load=strided_load,
    strided_store=strided_store,
    convert_to_target_bitwidth=convert_to_target_bitwidth,
    has_bank_conflicts=has_bank_conflicts,
)

# ==============================================================================
# Inlined batched_rpa/configs.py
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

import dataclasses
import enum

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
# inlined utils


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tuning parameters for the RPA kernel."""

  bq_sz: int
  bq_c_sz: int
  bkv_sz: int
  batch_size: int
  n_buffer: int


@dataclasses.dataclass(frozen=True)
class ModelConfigs:
  """Model config that will always stay constant."""

  num_q_heads: int
  num_kv_heads: int
  head_dim: int
  mask_value: float
  sm_scale: float = 1.0
  soft_cap: float | None = None
  sliding_window: int | None = None

  @property
  def num_q_heads_per_kv_head(self) -> int:
    return self.num_q_heads // self.num_kv_heads


@dataclasses.dataclass(frozen=True)
class ServingConfigs:
  """Serving config that can change depending on use cases."""

  num_seqs: int
  page_size: int
  total_q_tokens: int
  num_page_indices: int
  dtype_q: jnp.dtype
  dtype_kv: jnp.dtype
  dtype_out: jnp.dtype
  scale_q: int | None = None
  scale_k: int | None = None
  scale_v: int | None = None

  @property
  def pages_per_seq(self) -> int:
    return self.num_page_indices // self.num_seqs

  @property
  def page_size_log2(self) -> int:
    return (self.page_size - 1).bit_length()

  @property
  def page_size_mask(self) -> int:
    return self.page_size - 1

  @property
  def int_ty(self) -> jnp.dtype:
    return jnp.int32

  @property
  def packing_q(self) -> int:
    return utils.get_dtype_packing(self.dtype_q)

  @property
  def packing_kv(self) -> int:
    return utils.get_dtype_packing(self.dtype_kv)


class RpaCase(enum.StrEnum):
  """Represents the different cases for Ragged Paged Attention.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """

  DECODE = enum.auto()
  PREFILL = enum.auto()
  MIXED = enum.auto()

  @property
  def symbol(self):
    return {
        RpaCase.DECODE: "d",
        RpaCase.PREFILL: "p",
        RpaCase.MIXED: "m",
    }[self]

  def get_range(
      self, distribution: jax.Array
  ) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
    assert distribution.shape == (3,)
    match self:
      case RpaCase.DECODE:
        return 0, distribution[0]
      case RpaCase.PREFILL:
        return distribution[0], distribution[1]
      case RpaCase.MIXED:
        return distribution[1], distribution[2]


@dataclasses.dataclass(frozen=True, eq=True)
class RpaConfigs:
  block: BlockSizes
  model: ModelConfigs
  serve: ServingConfigs
  mode: RpaCase
  vmem_limit_bytes: int

  # Expose block sizes for ease of use.

  @property
  def bq_sz(self) -> int:
    return self.block.bq_sz

  @property
  def bq_c_sz(self) -> int:
    return self.block.bq_c_sz

  @property
  def bkv_sz(self) -> int:
    return self.block.bkv_sz

  @property
  def batch_size(self) -> int:
    return self.block.batch_size

  @property
  def n_buffer(self) -> int:
    return self.block.n_buffer

  # Define derived values.

  @property
  def max_steps_ub(self) -> int:
    """Get maximum upper bound of kernel steps based on SMEM limit."""

    fixed_bytes = 0
    fixed_bytes += self.serve.num_seqs  # kv_lens
    fixed_bytes += self.serve.num_seqs + 1  # cu_q_lens
    fixed_bytes += (
        self.serve.num_seqs * self.serve.pages_per_seq
    )  # page_indices
    fixed_bytes += 3  # distribution
    fixed_bytes += self.block.batch_size  # lane_lengths
    fixed_bytes += 1  # actual_steps

    word_size_bytes = 4
    fixed_bytes *= word_size_bytes

    smem_limit_bytes = pltpu.get_tpu_info().smem_capacity_bytes - 32 * 1024
    available_bytes = smem_limit_bytes - fixed_bytes

    # Per step per batch item:
    # s_idx, q_idx, k_idx, is_last_k, do_writeback: 5 * 4 = 20
    # dma_q: 2 * 4 = 8
    # dma_kv_cache: bkv_p_cache * 3 * 4 = 12 * bkv_p_cache
    # dma_kv_new: bkv_p_new * 4 * 4 = 16 * bkv_p_new
    bytes_per_step = 28 + 12 * self.bkv_p_cache + 16 * self.bkv_p_new
    bytes_per_step *= self.block.batch_size

    max_steps_ub = available_bytes // bytes_per_step

    num_lanes = pltpu.get_tpu_info().num_lanes
    max_steps_ub = max(1, max_steps_ub // num_lanes) * num_lanes
    return max_steps_ub

  @property
  def bkv_p(self) -> int:
    return self.block.bkv_sz // self.serve.page_size

  @property
  def bkv_p_cache(self) -> int:
    if self.mode == RpaCase.PREFILL:
      return 0
    return self.bkv_p

  @property
  def bkv_p_new(self) -> int:
    if self.mode == RpaCase.DECODE:
      return 1
    return self.bkv_p

  @property
  def bkv_stride(self) -> int:
    bkv_stride = pl.cdiv(self.model.num_kv_heads * 2, self.serve.packing_kv)

    if utils.has_bank_conflicts(bkv_stride):
      bkv_stride += 1
    return bkv_stride

  @property
  def aligned_head_dim(self) -> int:
    num_lanes = pltpu.get_tpu_info().num_lanes
    return utils.align_to(self.model.head_dim, num_lanes)

  @property
  def aligned_q_head_dim(self) -> int:
    return self.aligned_head_dim

  @property
  def aligned_kv_head_dim(self) -> int:
    return self.aligned_head_dim

  @property
  def aligned_num_kv_heads_x2(self) -> int:
    packing_kv = self.serve.packing_kv
    return utils.align_to(self.model.num_kv_heads * 2, packing_kv)

  @property
  def aligned_num_q_heads_per_kv_head(self) -> int:
    packing_q = self.serve.packing_q
    return utils.align_to(self.model.num_q_heads_per_kv_head, packing_q)

  @property
  def kv_hbm_stride(self) -> int:
    kv_packing = utils.get_dtype_packing(self.serve.dtype_kv)
    return utils.align_to(self.model.num_kv_heads * 2, kv_packing) // kv_packing

  @property
  def fuse_accum(self) -> bool:
    return self.mode == RpaCase.DECODE

  @property
  def q_vmem_shape(self):
    q_per_kv_packing = (
        self.aligned_num_q_heads_per_kv_head // self.serve.packing_q
    )
    return (
        self.block.batch_size,
        self.model.num_kv_heads,
        self.block.bq_sz,
        q_per_kv_packing,
        self.serve.packing_q,
        self.aligned_q_head_dim,
    )

  @property
  def kv_vmem_shape(self):
    return (
        self.block.batch_size,
        self.block.bkv_sz,
        self.bkv_stride,
        self.serve.packing_kv,
        self.aligned_kv_head_dim,
    )

  @property
  def lm_scratch_shape(self):
    num_lanes = pltpu.get_tpu_info().num_lanes
    return (
        self.block.batch_size,
        self.model.num_kv_heads,
        self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
        num_lanes,
    )

  @property
  def acc_scratch_shape(self):
    return (
        self.block.batch_size,
        self.model.num_kv_heads,
        self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
        self.aligned_kv_head_dim,
    )

  def validate_inputs(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      kv_cache: jax.Array,
      kv_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
  ):
    """Validate inputs to the RPA kernel statically."""

    if not q.ndim == k.ndim == v.ndim == 3:
      raise ValueError(
          f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}"
      )
    if k.shape != v.shape:
      raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
      raise ValueError(
          "Expected number of sequences in Q, K, and V to be the same, but got"
          f" {q.shape[0]=}, {k.shape[0]=}, and {v.shape[0]=}"
      )
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
      raise ValueError(
          "Expected number of head dimensions in Q, K, and V to be the same,"
          f" but got {q.shape[2]=}, {k.shape[2]=}, and {v.shape[2]=}"
      )

    expected_kv_cache_shape = (
        kv_cache.shape[0],
        self.serve.page_size,
        self.aligned_num_kv_heads_x2 // self.serve.packing_kv,
        self.serve.packing_kv,
        self.aligned_head_dim,
    )

    if kv_cache.shape != expected_kv_cache_shape:
      raise ValueError(
          f"Expected {kv_cache.shape=} to be equal to"
          f" {expected_kv_cache_shape=}"
      )

    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
      raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")
    if not (kv_cache.dtype == k.dtype == v.dtype):
      raise ValueError(
          "Expected KV cache dtype and K/V dtype to be the same, but got"
          f" {kv_cache.dtype=}, {k.dtype=}, and {v.dtype=}"
      )

    if not (
        jnp.int32
        == kv_lens.dtype
        == page_indices.dtype
        == cu_q_lens.dtype
        == distribution.dtype
    ):
      raise ValueError(
          f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
          f" {cu_q_lens.dtype=}, {distribution.dtype=}"
      )

    if not (kv_lens.ndim == page_indices.ndim == cu_q_lens.ndim == 1):
      raise ValueError(
          f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
          f" {cu_q_lens.shape=}"
      )

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
      raise ValueError(
          f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
      )
    if cu_q_lens.shape != (max_num_seqs + 1,):
      raise ValueError(
          f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)."
      )
    if distribution.shape != (3,):
      raise ValueError(f"Expected {distribution.shape=} to be (3,).")

configs = types.SimpleNamespace(
    BlockSizes=BlockSizes, ModelConfigs=ModelConfigs, ServingConfigs=ServingConfigs,
    RpaCase=RpaCase, RpaConfigs=RpaConfigs
)

# ==============================================================================
# Inlined batched_rpa/schedule.py
# ==============================================================================
# Source: https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/experimental/batched_rpa/schedule.py
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

import dataclasses
import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
# inlined configs
# inlined utils


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
  """Maps physical 1-D data into logical N-D representation."""

  data: Any
  shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create_shape_dtype(cls, shape):
    return cls(
        data=jax.ShapeDtypeStruct((np.prod(shape),), jnp.int32), shape=shape
    )

  def _get_pos(self, indices):
    strides = pl.strides_from_shape(self.shape)
    assert len(strides) == len(indices)

    pos = 0
    for stride, idx in zip(strides, indices):
      pos += stride * idx
    return pos

  def __getitem__(self, indices):
    return self.data[self._get_pos(indices)]

  def __setitem__(self, indices, value):
    self.data[self._get_pos(indices)] = value


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RpaSchedule:
  """Container for metadata arrays with integrated shape/spec logic."""

  s_idx: SmemWrapper  # [steps, batch]
  q_idx: SmemWrapper  # [steps, batch]
  k_idx: SmemWrapper  # [steps, batch]
  is_last_k: SmemWrapper  # [steps, batch]
  do_writeback: SmemWrapper  # [steps, batch]
  dma_q: SmemWrapper  # [steps, batch, 2]
  dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 3]
  dma_kv_new: SmemWrapper  # [steps, batch, bkv_p_new, 4]
  actual_steps: Any  # [1]

  cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create_shape_dtype(cls, cfgs: configs.RpaConfigs):

    idx_wrapper = SmemWrapper.create_shape_dtype(
        (cfgs.max_steps_ub, cfgs.batch_size)
    )

    return cls(
        s_idx=idx_wrapper,
        q_idx=idx_wrapper,
        k_idx=idx_wrapper,
        is_last_k=idx_wrapper,
        do_writeback=idx_wrapper,
        dma_q=SmemWrapper.create_shape_dtype(
            (cfgs.max_steps_ub, cfgs.batch_size, 2)
        ),
        dma_kv_cache=SmemWrapper.create_shape_dtype(
            (cfgs.max_steps_ub, cfgs.batch_size, cfgs.bkv_p_cache, 3)
        ),
        dma_kv_new=SmemWrapper.create_shape_dtype(
            (cfgs.max_steps_ub, cfgs.batch_size, cfgs.bkv_p_new, 4),
        ),
        actual_steps=jax.ShapeDtypeStruct((1,), jnp.int32),
        cfgs=cfgs,
    )

  def get_dma_kv_cache(
      self,
      step: jax.typing.ArrayLike,
      batch_idx: jax.typing.ArrayLike,
      page_idx: jax.typing.ArrayLike,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    # 0: src_hbm, 1: dst_vmem, 2: size
    src_off = self.dma_kv_cache[step, batch_idx, page_idx, 0]
    dst_off = self.dma_kv_cache[step, batch_idx, page_idx, 1]
    sz = self.dma_kv_cache[step, batch_idx, page_idx, 2]
    return src_off, dst_off, sz

  def get_dma_kv_new(
      self,
      step: jax.typing.ArrayLike,
      batch_idx: jax.typing.ArrayLike,
      page_idx: jax.typing.ArrayLike,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # 0: dst_hbm, 1: src_hbm, 2: dst_vmem, 3: size
    dst_hbm = self.dma_kv_new[step, batch_idx, page_idx, 0]
    src_hbm = self.dma_kv_new[step, batch_idx, page_idx, 1]
    dst_vmem = self.dma_kv_new[step, batch_idx, page_idx, 2]
    sz = self.dma_kv_new[step, batch_idx, page_idx, 3]
    return dst_hbm, src_hbm, dst_vmem, sz

  def get_dma_q(
      self, step: jax.typing.ArrayLike, batch_idx: jax.typing.ArrayLike
  ) -> tuple[jax.Array, jax.Array]:
    # 0: src_hbm, 1: size
    src_hbm = self.dma_q[step, batch_idx, 0]
    sz = self.dma_q[step, batch_idx, 1]
    return src_hbm, sz

  def scratch_shapes(self):
    """Returns a Pytree of SMEM scratch memory."""

    return jax.tree.map(
        lambda x: pltpu.SMEM(x.shape, x.dtype),
        self,
    )

  def in_specs(self):
    """Returns a Pytree of input BlockSpecs."""

    def wrapper(x):
      if x.size == 1:
        return pl.BlockSpec(memory_space=pltpu.SMEM)
      else:
        # Since we use maximum upper bound when allocating scheduler data,
        # it is not feasible to use scalar prefetch and fetch entire scheduler
        # data into the kernel. Instead, we stored it to HBM first and perform
        # dynamic sized DMA inside the kernel using actual number of steps.
        return pl.BlockSpec(memory_space=pltpu.HBM)

    return jax.tree.map(wrapper, self)

  def out_specs(self):
    """Returns a Pytree of output BlockSpecs."""

    return jax.tree.map(
        lambda x: pl.BlockSpec(memory_space=pltpu.HBM),
        self,
    )


def compute_metadata(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    schedule: RpaSchedule,
    lane_lengths_ref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
  """Fill metadata using triple nested loop of seq->q->k loop."""

  @jax.named_scope("k_loop")
  def k_loop(
      k_idx,
      step,
      *,
      target_lane,
      s_idx,
      q_idx,
      q_end,
      q_src,
      q_sz_task,
      k_len,
      q_len,
      end_k_idx,
  ):

    schedule.s_idx[step, target_lane] = s_idx
    schedule.q_idx[step, target_lane] = q_idx
    schedule.k_idx[step, target_lane] = k_idx

    is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
    schedule.is_last_k[step, target_lane] = is_last_k

    schedule.dma_q[step, target_lane, 0] = q_src
    schedule.dma_q[step, target_lane, 1] = q_sz_task

    kv_len_start = k_idx * cfgs.bkv_sz
    kv_p_start = k_idx * cfgs.bkv_p
    kv_left = k_len - kv_len_start
    kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
    p_offset = s_idx * cfgs.serve.pages_per_seq + kv_p_start

    for i in range(cfgs.bkv_p_cache):
      dst_vmem = i << cfgs.serve.page_size_log2
      dma_sz = kv_left_frm_cache - dst_vmem
      dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

      src_hbm = jnp.minimum(p_offset + i, cfgs.serve.num_page_indices - 1)

      schedule.dma_kv_cache[step, target_lane, i, 0] = src_hbm
      schedule.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
      schedule.dma_kv_cache[step, target_lane, i, 2] = dma_sz

    kv_left_frm_new = kv_left - kv_left_frm_cache
    bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
    new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

    # Writeback logic: each new k block is written back by the first q block
    # that attends to it.
    q_wb = jnp.maximum(0, (kv_len_start - (k_len - q_len))) // cfgs.bq_sz

    do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
    schedule.do_writeback[step, target_lane] = do_writeback
    src_hbm = q_end - kv_left_frm_new

    if cfgs.bkv_p_new < cfgs.bkv_p:
      # Special case where we only need to write back one page.
      assert cfgs.bkv_p_new == 1
      dst_vmem = bkv_sz_cache
      dma_sz = new_sz

      p_idx = (kv_len_start + dst_vmem) >> cfgs.serve.page_size_log2
      p_idx = jnp.minimum(p_idx, cfgs.serve.pages_per_seq - 1)
      p_off = (kv_len_start + dst_vmem) & cfgs.serve.page_size_mask
      global_p_idx = s_idx * cfgs.serve.pages_per_seq + p_idx

      dst_hbm = (global_p_idx << cfgs.serve.page_size_log2) | p_off
      schedule.dma_kv_new[step, target_lane, 0, 0] = dst_hbm
      schedule.dma_kv_new[step, target_lane, 0, 1] = src_hbm
      schedule.dma_kv_new[step, target_lane, 0, 2] = dst_vmem
      schedule.dma_kv_new[step, target_lane, 0, 3] = dma_sz
    else:
      for i in range(cfgs.bkv_p):
        slot_start = i << cfgs.serve.page_size_log2
        slot_end = (i + 1) << cfgs.serve.page_size_log2

        dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
        end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
        dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)

        p_idx = (kv_len_start + dst_vmem) >> cfgs.serve.page_size_log2
        p_idx = jnp.minimum(p_idx, cfgs.serve.pages_per_seq - 1)
        p_off = (kv_len_start + dst_vmem) & cfgs.serve.page_size_mask
        global_p_idx = s_idx * cfgs.serve.pages_per_seq + p_idx

        dst_hbm = (global_p_idx << cfgs.serve.page_size_log2) | p_off
        schedule.dma_kv_new[step, target_lane, i, 0] = dst_hbm
        schedule.dma_kv_new[step, target_lane, i, 1] = src_hbm
        schedule.dma_kv_new[step, target_lane, i, 2] = dst_vmem
        schedule.dma_kv_new[step, target_lane, i, 3] = dma_sz

    return step + 1

  @jax.named_scope("q_loop")
  def q_loop(q_idx, _, *, s_idx, q_start, q_end, k_len, q_len, num_k):
    target_lane = 0
    min_len = lane_lengths_ref[0]
    for b in range(1, cfgs.batch_size):
      is_better = lane_lengths_ref[b] < min_len
      target_lane = jnp.where(is_better, b, target_lane)
      min_len = jnp.where(is_better, lane_lengths_ref[b], min_len)

    curr_ptr = lane_lengths_ref[target_lane]
    q_src = q_start + q_idx * cfgs.bq_sz
    q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

    start_k_idx = 0
    if (sliding_window := cfgs.model.sliding_window) is not None:
      sw_start_idx = k_len - q_len + q_idx * cfgs.bq_sz - sliding_window + 1
      start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

    end_k_idx_causal = (
        k_len - q_len + q_idx * cfgs.bq_sz + q_sz_task - 1
    ) // cfgs.bkv_sz + 1
    end_k_idx = jnp.minimum(num_k, end_k_idx_causal)

    k_loop_fn = functools.partial(
        k_loop,
        target_lane=target_lane,
        s_idx=s_idx,
        q_idx=q_idx,
        q_end=q_end,
        q_src=q_src,
        q_sz_task=q_sz_task,
        k_len=k_len,
        q_len=q_len,
        end_k_idx=end_k_idx,
    )
    lane_lengths_ref[target_lane] = jax.lax.fori_loop(
        start_k_idx, end_k_idx, k_loop_fn, curr_ptr
    )

  @jax.named_scope("seq_loop")
  def seq_loop(s_idx, _):
    q_start = cu_q_lens_ref[s_idx]
    q_end = cu_q_lens_ref[s_idx + 1]
    k_len = kv_lens_ref[s_idx]
    q_len = q_end - q_start

    num_q = pl.cdiv(q_len, cfgs.bq_sz)
    num_k = pl.cdiv(k_len, cfgs.bkv_sz)

    q_loop_fn = functools.partial(
        q_loop,
        s_idx=s_idx,
        q_start=q_start,
        q_end=q_end,
        k_len=k_len,
        q_len=q_len,
        num_k=num_k,
    )

    jax.lax.fori_loop(0, num_q, q_loop_fn, None)

  start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)  # pyrefly: ignore[bad-argument-type]
  jax.lax.fori_loop(start_seq_idx, end_seq_idx, seq_loop, None)


def rpa_metadata_schedule_kernel(
    ## Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    # Outputs.
    schedule_hbm_ref: RpaSchedule,
    # Scratch.
    schedule_ref: RpaSchedule,
    lane_lengths_ref: jax.Ref,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
  """Generates the HBM-to-VMEM DMA schedule.

  This kernel:
  1. Iterates through each (potentially ragged) sequence
  2. Breaks Queries (Q) and Key-Values (KV) into blocks (bq_sz, bkv_sz).
  3. Assigns tasks to 'lanes' (TPU batch items) based on current lane occupancy
    to ensure balanced execution across the batch dimension.
  4. Encodes DMA offsets:
    - dma_q: HBM start index and size for Query blocks.
    - dma_kv_cache: Paged indices for existing KV tokens.
    - dma_kv_new: offsets for new tokens being added to the cache.
    - do_writeback: boolean flag indicating if a block should be flushed to
      HBM (ie does this block contain new tokens to add to KV cache).

  Args:
    cu_q_lens_ref: [max_num_seqs + 1]. Cumulative sum of each sequence's query
      length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
      b=cu_q_lens[i+1] represents q/k/v of sequence i.
    kv_lens_ref: [max_num_seqs]. Existing kv cache length of each sequence.
    distribution_ref: [3]. Cumulative sum of number of decode, prefill, and
      mixed
    schedule_hbm_ref: HBM memory that will store output of the kernel.
    schedule_ref: Scratch memory where schedule results gets written.
    lane_lengths_ref: Scratch memory that keeps track of number of steps for
      each batch lane.
    dma_sem: Semaphore used for writing scheduler output to HBM.
    cfgs: Configuration of the kernel.
  """

  for b_idx in range(cfgs.batch_size):
    lane_lengths_ref[b_idx] = 0

  # Step 1: Compute and fill scheduler metadata.
  compute_metadata(
      cu_q_lens_ref,
      kv_lens_ref,
      distribution_ref,
      schedule_ref,
      lane_lengths_ref,
      cfgs=cfgs,
  )

  # Step 2: Compute actual number of steps.
  max_steps = 0
  for b_idx in range(cfgs.batch_size):
    max_steps = jnp.maximum(max_steps, lane_lengths_ref[b_idx])
  schedule_ref.actual_steps[0] = max_steps

  safe_max_steps = jnp.minimum(max_steps + cfgs.n_buffer + 1, cfgs.max_steps_ub)

  # Step 3: Mask out unvisited steps.
  @jax.named_scope("mask_out_steps")
  def mask_out_steps(step, _, *, b_idx):
    schedule_ref.s_idx[step, b_idx] = -1
    schedule_ref.q_idx[step, b_idx] = 0
    schedule_ref.k_idx[step, b_idx] = 0
    schedule_ref.is_last_k[step, b_idx] = 0
    schedule_ref.do_writeback[step, b_idx] = 0

    schedule_ref.dma_q[step, b_idx, 0] = 0
    schedule_ref.dma_q[step, b_idx, 1] = 0

    for i in range(cfgs.bkv_p_cache):
      schedule_ref.dma_kv_cache[step, b_idx, i, 0] = 0
      schedule_ref.dma_kv_cache[step, b_idx, i, 1] = 0
      schedule_ref.dma_kv_cache[step, b_idx, i, 2] = 0

    for i in range(cfgs.bkv_p_new):
      schedule_ref.dma_kv_new[step, b_idx, i, 0] = 0
      schedule_ref.dma_kv_new[step, b_idx, i, 1] = 0
      schedule_ref.dma_kv_new[step, b_idx, i, 2] = 0
      schedule_ref.dma_kv_new[step, b_idx, i, 3] = 0

  for b_idx in range(cfgs.batch_size):
    start_step = lane_lengths_ref[b_idx]
    mask_step_fn = functools.partial(mask_out_steps, b_idx=b_idx)
    jax.lax.fori_loop(start_step, safe_max_steps, mask_step_fn, None)

  # Ste 4: Write back results to HBM.
  flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
  flat_smem = jax.tree_util.tree_leaves(schedule_ref)
  dma_list = []
  for h, s in zip(flat_hbm, flat_smem):
    write_size = h.shape[0]
    if write_size > 1:
      write_size = (write_size // cfgs.max_steps_ub) * safe_max_steps
      write_size = utils.align_to(write_size, 1024)

    copy = pltpu.make_async_copy(
        s.at[pl.ds(0, write_size)],
        h.at[pl.ds(0, write_size)],
        dma_sem.at[0],
    )
    dma_list.append(copy)

  jax.tree.map(lambda x: x.start(), dma_list)
  jax.tree.map(lambda x: x.wait(), dma_list)


def generate_rpa_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    distribution: jax.Array,
    cfgs: configs.RpaConfigs,
    *,
    interpret=False,
) -> RpaSchedule:
  schedule_shaped_dtype = RpaSchedule.create_shape_dtype(cfgs)

  return pl.pallas_call(
      functools.partial(rpa_metadata_schedule_kernel, cfgs=cfgs),
      out_shape=schedule_shaped_dtype,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[],
          out_specs=schedule_shaped_dtype.out_specs(),
          scratch_shapes=[
              schedule_shaped_dtype.scratch_shapes(),
              pltpu.SMEM((cfgs.batch_size,), jnp.int32),
              pltpu.SemaphoreType.DMA((1,)),
          ],
      ),
      interpret=interpret,
      name="rpa_metadata_schedule",
  )(cu_q_lens, kv_lens, distribution)

schedule = types.SimpleNamespace(
    SmemWrapper=SmemWrapper, RpaSchedule=RpaSchedule, compute_metadata=compute_metadata,
    generate_rpa_metadata=generate_rpa_metadata
)

# ==============================================================================
# Inlined batched_rpa/flash_attention.py
# ==============================================================================
# Source: https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/experimental/batched_rpa/flash_attention.py
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

import jax
from jax import lax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
# inlined configs, utils


def flash_attention_qk_softmax(
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H]
    m_prev: jax.Array,  # [B, KV, TQ, 128]
    l_prev: jax.Array,  # [B, KV, TQ, 128]
    *,
    processed_q_len: list[jax.Array],  # [B]
    processed_kv_len: list[jax.Array],  # [B]
    effective_kv_len: list[jax.Array],  # [B]
    cfgs: configs.RpaConfigs,
    bq_start: int,
):
  """Flash attention kernel."""
  b, k_heads, tq, _ = q.shape
  s = k.shape[2]

  if cfgs.serve.scale_q is not None:
    q = q / cfgs.serve.scale_q
    if jnp.issubdtype(k.dtype, jnp.floating):
      dtype_info = jnp.finfo(k.dtype)
      minval = float(dtype_info.min)
      maxval = float(dtype_info.max)
      q = jnp.clip(q, min=minval, max=maxval)
    q = q.astype(k.dtype)

  qk = lax.dot_general(
      pltpu.einshape("bkth->(bk)th", q, True),
      pltpu.einshape("bksh->(bk)sh", k, True),
      dimension_numbers=(([2], [2]), ([0], [0])),
      preferred_element_type=jnp.float32,
  ).astype(cfgs.serve.dtype_out)
  qk = pltpu.einshape("(bk)ts->bkts", qk, True, b=b)

  qk *= cfgs.model.sm_scale
  if cfgs.serve.scale_k is not None:
    qk *= cfgs.serve.scale_k
  if cfgs.serve.scale_q is not None:
    qk *= cfgs.serve.scale_q

  if cfgs.model.soft_cap is not None:
    qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

  qk_masked = []

  int_ty = cfgs.serve.int_ty

  for b_idx in range(cfgs.block.batch_size):
    kv_idx_b = (
        lax.broadcasted_iota(int_ty, (k_heads, tq, s), 2)
        + processed_kv_len[b_idx]
    )
    q_idx_b = (
        lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1)
        // cfgs.aligned_num_q_heads_per_kv_head
        + bq_start
    ).astype(int_ty) + processed_q_len[b_idx]

    eff_kv_len_b = effective_kv_len[b_idx]
    mask_b = q_idx_b < eff_kv_len_b
    mask_b = jnp.logical_and(mask_b, q_idx_b >= kv_idx_b)

    if (sliding_window := cfgs.model.sliding_window) is not None:
      mask_b = jnp.logical_and(mask_b, q_idx_b < kv_idx_b + sliding_window)

    qk_masked.append(jnp.where(mask_b, qk[b_idx], cfgs.model.mask_value))
  qk = jnp.stack(qk_masked, axis=0)

  m_curr = jnp.max(qk, axis=-1, keepdims=True)
  m_next = jnp.maximum(m_prev, m_curr)
  p = jnp.exp(qk - utils.broadcast_minor(m_next, qk.shape))
  p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=cfgs.serve.dtype_out)

  alpha = jnp.exp(m_prev - m_next)
  l_next = alpha * l_prev + p_rowsum

  return p, alpha, m_next, l_next


def flash_attention_pv(
    p: jax.Array,  # [B, KV, TQ, S]
    v: jax.Array,  # [B, KV, S, H]
    alpha: jax.Array,  # [B, KV, TQ, 128]
    o_prev: jax.Array,  # [B, KV, TQ, H]
    cfgs: configs.RpaConfigs,
):
  """Flash attention kernel."""
  b = p.shape[0]
  pv = lax.dot_general(
      pltpu.einshape("bkts->(bk)ts", p, True),
      pltpu.einshape("bksh->(bk)sh", v, True),
      dimension_numbers=(([2], [1]), ([0], [0])),
      preferred_element_type=jnp.float32,
  ).astype(cfgs.serve.dtype_out)
  pv = pltpu.einshape("(bk)th->bkth", pv, True, b=b)

  if cfgs.serve.scale_v is not None:
    pv *= cfgs.serve.scale_v

  o_next = utils.broadcast_minor(alpha, o_prev.shape) * o_prev + pv

  return o_next

flash_attention = types.SimpleNamespace(
    flash_attention_qk_softmax=flash_attention_qk_softmax, flash_attention_pv=flash_attention_pv
)

# ==============================================================================
# Inlined batched_rpa/bref_override.py
# ==============================================================================
# Source: https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/experimental/batched_rpa/bref_override.py
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

import dataclasses

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
# inlined


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _BypassRef(pltpu.BufferedRef):
  """Helper class to safely bypass buffer_count checks during creation."""

  def __post_init__(self):
    # pallas doesn't allow you to set n_buffer > 2 for output refs, so
    # we override to bypass this check.
    pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KVBufferedRef(_BypassRef):
  """Handles fetching and updating KV cache using precomputed metadata."""

  cfgs: configs.RpaConfigs = dataclasses.field(
      default=None, metadata=dict(static=True)
  )

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: jax.Array,
      buffer_type,  # pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.RpaConfigs,
  ):
    # TODO(kyuyeunk): Uncomment this out after jax version update.
    # assert buffer_type == pltpu.BufferType.INPUT_OUTPUT

    standard_ref = _BypassRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
    )
    return cls(
        cfgs=cfgs,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # src_ref: (kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref)
    kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref = src_ref
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot, :, :, : self.cfgs.kv_hbm_stride]
    block_idx = jnp.maximum(grid_indices[0], 0)

    kv_cache_hbm_flat = kv_cache_hbm.reshape(-1, *kv_cache_hbm.shape[2:])

    dma_list_cache = []
    dma_list_new = []

    for b in range(self.cfgs.batch_size):
      for i in range(self.cfgs.bkv_p_cache):
        p_idx, dst_off, sz = schedule_ref.get_dma_kv_cache(block_idx, b, i)
        src_off = page_indices_ref[p_idx] * self.cfgs.serve.page_size
        dma_list_cache.append((src_off, dst_off, sz, b))

      # Contiguous fetch for new KV
      _, src_new_off, dst_vmem_off, _ = schedule_ref.get_dma_kv_new(
          block_idx, b, 0
      )
      total_new_sz = 0
      for i in range(self.cfgs.bkv_p_new):
        _, _, _, sz = schedule_ref.get_dma_kv_new(block_idx, b, i)
        total_new_sz += sz
      dma_list_new.append((src_new_off, dst_vmem_off, total_new_sz, b))

    for i in range(len(dma_list_cache)):
      src_off, dst_off, sz, b = dma_list_cache[i]
      pltpu.make_async_copy(
          kv_cache_hbm_flat.at[pl.ds(src_off, sz)],
          vmem_dst.at[b, pl.ds(dst_off, sz)],
          sem,
      ).start()

    for i in range(len(dma_list_new)):
      src_off, dst_off, sz, b = dma_list_new[i]
      pltpu.make_async_copy(
          new_kv_hbm.at[pl.ds(src_off, sz)],
          vmem_dst.at[b, pl.ds(dst_off, sz)],
          sem,
      ).start()

  def copy_out(
      self,
      dst_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    kv_out_ref, _, schedule_ref, page_indices_ref = dst_ref
    kv_out_ref_flat = kv_out_ref.reshape(-1, *kv_out_ref.shape[2:])
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_src = self.window_ref.at[slot, :, :, : self.cfgs.kv_hbm_stride]
    block_idx = grid_indices[0]

    for b in range(self.cfgs.batch_size):
      do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
      for i in range(self.cfgs.bkv_p_new):
        encoded_dst_hbm_off, _, src_vmem_off, new_sz = (
            schedule_ref.get_dma_kv_new(block_idx, b, i)
        )
        global_p_idx = encoded_dst_hbm_off >> self.cfgs.serve.page_size_log2
        p_off = encoded_dst_hbm_off & self.cfgs.serve.page_size_mask
        dst_hbm_off = (
            page_indices_ref[global_p_idx] << self.cfgs.serve.page_size_log2
        ) | p_off
        sz = jnp.where(do_writeback, new_sz, 0)
        pltpu.make_async_copy(
            vmem_src.at[b, pl.ds(src_vmem_off, sz)],
            kv_out_ref_flat.at[pl.ds(dst_hbm_off, sz)],
            sem,
        ).start()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    _, _, schedule_ref, _ = src_ref
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    total_sz = 0
    for b in range(self.cfgs.batch_size):
      for i in range(self.cfgs.bkv_p_cache):
        _, _, sz = schedule_ref.get_dma_kv_cache(block_idx, b, i)
        total_sz += sz

      # Contiguous wait for new KV
      for i in range(self.cfgs.bkv_p_new):
        _, _, _, sz = schedule_ref.get_dma_kv_new(block_idx, b, i)
        total_sz += sz

    # Flatten the first two dimensions (Batch, Seq) to create a 1D view.
    flat_vmem = vmem_dst.reshape((-1, *vmem_dst.shape[2:]))
    pltpu.make_async_copy(
        flat_vmem.at[pl.ds(0, total_sz), : self.cfgs.kv_hbm_stride],
        flat_vmem.at[pl.ds(0, total_sz), : self.cfgs.kv_hbm_stride],
        sem,
    ).wait()

  def wait_out(
      self,
      dst_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    kv_out_ref, _, schedule_ref, _ = dst_ref
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    block_idx = grid_indices[0]

    total_sz = 0
    for b in range(self.cfgs.batch_size):
      do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
      for i in range(self.cfgs.bkv_p_new):
        _, _, _, new_sz = schedule_ref.get_dma_kv_new(block_idx, b, i)
        sz = jnp.where(do_writeback, new_sz, 0)
        total_sz += sz

    # Flatten to 2D: (Total_Rows, Head_Dim)
    flat_ref = kv_out_ref.reshape((-1, *kv_out_ref.shape[2:]))
    pltpu.make_async_copy(
        flat_ref.at[pl.ds(0, total_sz)],
        flat_ref.at[pl.ds(0, total_sz)],
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingORef(pltpu.BufferedRef):
  """Handles normalizing and storing the final attention output."""

  cfgs: configs.RpaConfigs = dataclasses.field(
      default=None, metadata=dict(static=True)
  )

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: jax.Array,
      buffer_type,  # pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.RpaConfigs,
  ):
    # TODO(kyuyeunk): Uncomment this out after jax version update.
    # assert buffer_type == pltpu.BufferType.OUTPUT

    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
    )
    return cls(
        cfgs=cfgs,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  def copy_out(
      self,
      dst_ref: tuple[jax.Ref, schedule.RpaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # dst_ref: (o_hbm, schedule_ref)
    o_hbm, schedule_ref = dst_ref
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_src = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    # is_last_k stride: batch size
    dma_list = []
    for b in range(self.cfgs.batch_size):
      is_last_k = schedule_ref.is_last_k[block_idx, b] == 1
      q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
      q_sz = jnp.where(is_last_k, q_sz, 0)
      dma_list.append((q_src, q_sz, b))

    for i in range(len(dma_list)):
      q_src, q_sz, b = dma_list[i]
      pltpu.make_async_copy(
          vmem_src.at[b, :, pl.ds(0, q_sz)],
          o_hbm.at[:, pl.ds(q_src, q_sz)],
          sem,
      ).start()

  def wait_out(
      self,
      dst_ref: tuple[jax.Ref, schedule.RpaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # dst_ref: (o_hbm, schedule_ref)
    o_hbm, schedule_ref = dst_ref
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    block_idx = grid_indices[0]

    total_sz = 0
    for b in range(self.cfgs.batch_size):
      is_last_k = schedule_ref.is_last_k[block_idx, b] == 1
      _, q_sz = schedule_ref.get_dma_q(block_idx, b)
      q_sz = jnp.where(is_last_k, q_sz, 0)
      total_sz += q_sz

    flat_ref = o_hbm.reshape((-1, *o_hbm.shape[2:]))
    pltpu.make_async_copy(
        flat_ref.at[pl.ds(0, total_sz * o_hbm.shape[0])],
        flat_ref.at[pl.ds(0, total_sz * o_hbm.shape[0])],
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingQRef(pltpu.BufferedRef):
  """Handles fetching Q blocks using precomputed metadata."""

  cfgs: configs.RpaConfigs = dataclasses.field(
      default=None, metadata=dict(static=True)
  )

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: jax.Array,
      buffer_type,  # pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.RpaConfigs,
  ):
    # TODO(kyuyeunk): Uncomment this out after jax version update.
    # assert buffer_type == pltpu.BufferType.INPUT

    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
    )
    return cls(
        cfgs=cfgs,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, schedule.RpaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # src_ref: (q_hbm, schedule_ref)
    q_hbm, schedule_ref = src_ref
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    dma_list = []
    for b in range(self.cfgs.batch_size):
      q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
      dma_list.append((q_src, q_sz, b))

    for i in range(len(dma_list)):
      q_src, q_sz, b = dma_list[i]
      pltpu.make_async_copy(
          q_hbm.at[:, pl.ds(q_src, q_sz)],
          vmem_dst.at[b, :, pl.ds(0, q_sz)],
          sem,
      ).start()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, schedule.RpaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    _, schedule_ref = src_ref
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    total_sz = 0
    for b in range(self.cfgs.batch_size):
      _, q_sz = schedule_ref.get_dma_q(block_idx, b)
      total_sz += q_sz

    # Flatten to 2D: (Total_Rows, Head_Dim)
    # vmem_dst is (Batch, Heads, Q, Head_Dim). We copy Heads * q_sz rows.
    flat_vmem = vmem_dst.reshape((-1, *vmem_dst.shape[3:]))
    pltpu.make_async_copy(
        flat_vmem.at[pl.ds(0, total_sz * vmem_dst.shape[1])],
        flat_vmem.at[pl.ds(0, total_sz * vmem_dst.shape[1])],
        sem,
    ).wait()

bref_override = types.SimpleNamespace(
    BatchingQRef=BatchingQRef, KVBufferedRef=KVBufferedRef, BatchingORef=BatchingORef
)

# ==============================================================================
# Inlined batched_rpa/kernel.py
# ==============================================================================
# Source: https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/experimental/batched_rpa/kernel.py
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

import dataclasses
import functools

import jax
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
# inlined namespaces


# Define inner kernel.


def strided_load_bkv(
    kv_in_vref: jax.Ref,
    b_idx: int,
    start: int,
    *,
    cfgs: configs.RpaConfigs,
) -> list[tuple[jax.Array, jax.Array]]:
  assert start % cfgs.serve.packing_kv == 0
  start //= cfgs.serve.packing_kv
  kv_u32_ref = kv_in_vref.at[b_idx].bitcast(jnp.uint32)
  kv_ref = kv_u32_ref.reshape(-1, cfgs.model.head_dim)

  if cfgs.serve.packing_kv == 1:
    k = utils.strided_load(
        kv_ref,
        start,
        cfgs.bkv_sz * cfgs.bkv_stride,
        cfgs.bkv_stride,
        dtype=cfgs.serve.dtype_kv,
    )
    v = utils.strided_load(
        kv_ref,
        start + 1,
        cfgs.bkv_sz * cfgs.bkv_stride,
        cfgs.bkv_stride,
        dtype=cfgs.serve.dtype_kv,
    )
    return [(k, v)]

  kv = utils.strided_load(
      kv_ref, start, cfgs.bkv_sz * cfgs.bkv_stride, cfgs.bkv_stride
  )
  bitwidth = jax.dtypes.itemsize_bits(cfgs.serve.dtype_kv)

  return utils.convert_to_target_bitwidth(
      kv, target_bitwidth=bitwidth, kv_dtype=cfgs.serve.dtype_kv
  )


def calculate_and_store_out(
    step_idx: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    acc_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    o_vref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):

  def _accum(b_idx: int):
    batch_acc = acc_scratch_ref[b_idx]
    batch_l = l_scratch_ref[b_idx]
    batch_l = utils.broadcast_minor(batch_l, batch_acc.shape)

    if (
        cfgs.serve.dtype_out == jnp.float32
        or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16
    ):
      result = lax.div(batch_acc, batch_l)
    else:
      result = batch_acc * pl.reciprocal(batch_l, approx=True)
    out = result.astype(cfgs.serve.dtype_out)

    o_u32_vref = o_vref.at[b_idx].bitcast(jnp.uint32)
    out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
    if cfgs.aligned_q_head_dim != cfgs.aligned_kv_head_dim:
      out = jnp.pad(
          out,
          (
              (0, 0),
              (0, 0),
              (0, cfgs.aligned_q_head_dim - cfgs.aligned_kv_head_dim),
          ),
          constant_values=0,
      )
    out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
    utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out)

  for b in range(cfgs.batch_size):
    # Adding a conditional causes a scheduling barrier. In prefill, we often
    # use small block sizes, so it's not worth executing the accumulation
    # on every block. In decode, because of the large block sizes / and or
    # batch sizes, we almost always use accumulation on every block. Please
    # tune `fuse_accum` for your use case.
    if not cfgs.fuse_accum:
      is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
      jax.lax.cond(
          is_last_k, jax.named_call(_accum, name="accum"), lambda _: None, b
      )
    else:
      _accum(b)


def rpa_body(
    # Inputs.
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    # Outputs
    o_vref: jax.Ref,
    # Scratches.
    schedule_ref: schedule.RpaSchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    *,
    # Passed refs
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    # Configs.
    cfgs: configs.RpaConfigs,
):
  step = pl.program_id(0)

  # Step 1: Fetch metadata.
  processed_q_len = []
  processed_kv_len = []
  effective_kv_len = []
  int_ty = cfgs.serve.int_ty
  for b_idx in range(cfgs.batch_size):
    s_idx = schedule_ref.s_idx[step, b_idx]
    is_valid = s_idx != -1
    q_idx = schedule_ref.q_idx[step, b_idx]
    k_idx = schedule_ref.k_idx[step, b_idx]
    k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
    kv_len = jnp.where(is_valid, kv_lens_ref[s_idx], 0)
    q_start = jnp.where(is_valid, cu_q_lens_ref[s_idx], 0)
    q_end = jnp.where(is_valid, cu_q_lens_ref[s_idx + 1], 0)
    q_len = q_end - q_start
    offset = kv_len - q_len

    processed_q_len.append((q_idx * cfgs.bq_sz + offset).astype(int_ty))
    processed_kv_len.append(k_id.astype(int_ty))
    effective_kv_len.append(kv_len.astype(int_ty))

    start_k_idx = 0
    if (sliding_window := cfgs.model.sliding_window) is not None:
      sw_start_idx = kv_len - q_len + q_idx * cfgs.bq_sz - sliding_window + 1
      start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

    is_first_k_block = k_idx == start_k_idx
    reset_cond = jnp.logical_and(is_valid, is_first_k_block)
    m_scratch_ref[b_idx] = jnp.where(reset_cond, -jnp.inf, m_scratch_ref[b_idx])
    l_scratch_ref[b_idx] = jnp.where(reset_cond, 0.0, l_scratch_ref[b_idx])
    acc_scratch_ref[b_idx] = jnp.where(reset_cond, 0.0, acc_scratch_ref[b_idx])

  # Step 2: Fetch inputs.
  q_p = cfgs.aligned_num_q_heads_per_kv_head // cfgs.serve.packing_q
  q_ref = q_vref.bitcast(jnp.uint32).reshape(-1, cfgs.aligned_q_head_dim)
  q_loaded = utils.strided_load(
      q_ref,
      0,
      cfgs.batch_size * cfgs.model.num_kv_heads * cfgs.bq_sz * q_p,
      1,
      dtype=cfgs.serve.dtype_q,
  )
  q = q_loaded.reshape(
      cfgs.batch_size,
      cfgs.model.num_kv_heads,
      cfgs.bq_sz * cfgs.aligned_num_q_heads_per_kv_head,
      cfgs.aligned_q_head_dim,
  )
  if cfgs.aligned_q_head_dim != cfgs.aligned_kv_head_dim:
    q = q[..., :cfgs.aligned_kv_head_dim]

  # We want to load k, v from (batch, bkv_sz, bkv_stride, kv_packing, d)
  # where bkv_stride ~= num_kv_heads * 2 // kv_packing
  # to 2x (batch, num_kv_heads, bkv_sz, d)
  # We use strided_load to avoid the expensive transpose.
  k_b = []
  v_b = []
  for b_idx in range(cfgs.batch_size):
    heads_per_load = pl.cdiv(cfgs.serve.packing_kv, 2)
    ks = []
    vs = []
    for kv_head_start in range(0, cfgs.model.num_kv_heads, heads_per_load):
      bkv_lst = strided_load_bkv(
          kv_in_vref,
          b_idx,
          kv_head_start * 2,
          cfgs=cfgs,
      )
      ks.append(jnp.stack([k for k, _ in bkv_lst], axis=0))
      vs.append(jnp.stack([v for _, v in bkv_lst], axis=0))
    k, v = jnp.concat(ks, axis=0), jnp.concat(vs, axis=0)
    k = k.reshape(-1, cfgs.bkv_sz, cfgs.model.head_dim)
    v = v.reshape(-1, cfgs.bkv_sz, cfgs.model.head_dim)

    k = k[: cfgs.model.num_kv_heads]
    v = v[: cfgs.model.num_kv_heads]
    k_b.append(k)
    v_b.append(v)
  # Stack to (batch, num_heads, bkv_sz, num_lanes)
  k = jnp.stack(k_b, axis=0)
  v = jnp.stack(v_b, axis=0)

  # Step 3: Perform compute.
  m_val = m_scratch_ref[...]
  l_val = l_scratch_ref[...]
  acc_val = acc_scratch_ref[...]

  prev_p = prev_alpha = prev_q_slice = None
  for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
    bq_end = min(bq_start + cfgs.bq_c_sz, cfgs.bq_sz)
    q_start = bq_start * cfgs.aligned_num_q_heads_per_kv_head
    q_end = bq_end * cfgs.aligned_num_q_heads_per_kv_head
    q_slice = slice(q_start, q_end)

    p, alpha, m_next, l_next = flash_attention.flash_attention_qk_softmax(
        q[:, :, q_slice],
        k,
        m_val[:, :, q_slice],
        l_val[:, :, q_slice],
        processed_q_len=processed_q_len,
        processed_kv_len=processed_kv_len,
        effective_kv_len=effective_kv_len,
        cfgs=cfgs,
        bq_start=bq_start,
    )
    m_scratch_ref[:, :, q_slice] = m_next
    l_scratch_ref[:, :, q_slice] = l_next

    if prev_p is not None:
      o_next = flash_attention.flash_attention_pv(
          prev_p,
          v,
          prev_alpha,
          acc_val[:, :, prev_q_slice],
          cfgs=cfgs,
      )
      acc_scratch_ref[:, :, prev_q_slice] = o_next

    prev_p = p
    prev_alpha = alpha
    prev_q_slice = q_slice

  assert prev_p is not None
  o_next = flash_attention.flash_attention_pv(
      prev_p,
      v,
      prev_alpha,
      acc_val[:, :, prev_q_slice],
      cfgs=cfgs,
  )
  acc_scratch_ref[:, :, prev_q_slice] = o_next

  # Step 4: Write back outputs.
  calculate_and_store_out(
      step,
      schedule_ref,
      acc_scratch_ref,
      l_scratch_ref,
      o_vref,
      cfgs=cfgs,
  )


# Define main kernel.


def create_allocs(
    kv_cache_hbm_ref: jax.Ref, o_hbm_ref: jax.Ref, cfgs: configs.RpaConfigs
) -> tuple[
    bref_override.BatchingQRef,
    bref_override.KVBufferedRef,
    bref_override.BatchingORef,
]:
  kv_cache_spec = pl.BlockSpec(
      block_shape=cfgs.kv_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer, use_lookahead=True),
  )
  q_spec = pl.BlockSpec(
      block_shape=cfgs.q_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer, use_lookahead=True),
  )
  o_spec = pl.BlockSpec(
      block_shape=cfgs.q_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
  )

  kv_cache_alloc = bref_override.KVBufferedRef.input_output(
      spec=kv_cache_spec,
      dtype_or_type=kv_cache_hbm_ref,
      buffer_count=cfgs.n_buffer,
      use_lookahead=True,
      cfgs=cfgs,
  )
  q_alloc = bref_override.BatchingQRef.input(
      spec=q_spec,
      dtype_or_type=o_hbm_ref,
      buffer_count=cfgs.n_buffer,
      use_lookahead=True,
      cfgs=cfgs,
  )
  o_alloc = bref_override.BatchingORef.output(
      spec=o_spec,
      dtype_or_type=o_hbm_ref,
      buffer_count=2,
      use_lookahead=False,
      cfgs=cfgs,
  )

  return q_alloc, kv_cache_alloc, o_alloc


def get_kernel_name(cfgs: configs.RpaConfigs) -> str:
  name = f"RPA{cfgs.mode.symbol}-p{cfgs.serve.page_size}"
  name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
  if cfgs.model.sliding_window:
    name += f"-sw{cfgs.model.sliding_window}"
  return name


def get_kernel_metadata(
    cfgs: configs.RpaConfigs,
) -> dict[str, str | int | float]:
  cfgs_dict = dataclasses.asdict(cfgs)
  ret = {}
  for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
    key = jax.tree_util.keystr(path, simple=True, separator=".")
    if not isinstance(val, str | int | float):
      val = str(val)
    ret[key] = val
  return ret


def rpa_kernel(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
) -> tuple[jax.Array, jax.Array]:
  """Perform batched ragged paged attention with scheduler data.

  Args:
    cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
      length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
      b=cu_q_lens[i+1] represents q/k/v of sequence i.
    kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
    page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
      sequence.
    schedule_hbm: Output of scheduler kernel. It informs which: 1. seqs 2. q
      block 3. kv block that should be processed at a given step.
    q_hbm: [max_num_tokens, num_q_heads_per_kv_heads, cdiv(num_kv_heads,
      q_packing), q_packing, head_dim]. Output of q projection that has been
      pre-processed to align with existing kv cache data layout.
    new_kv_hbm: [max_num_tokens, cdiv(num_kv_heads * 2, kv_packing), kv_packing,
      head_dim]. Output of k & v projection that has been pre-processed to align
      with existing kv cache data layout.
    kv_cache_hbm: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
      kv_packing, head_dim]. Stores existing kv cache data where k & vs are
      concatenated along num kv heads dim.
    cfgs: Configuration of the kernel.

  Returns:
    out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
    new_kv_cache: [num_pages, page_size, num_kv_heads // kv_packing, kv_packing,
      head_dim]. Result of new kv cache.
  """

  def ragged_paged_attention_pipeline(
      # Scalar prefetch.
      cu_q_lens_ref: jax.Ref,
      kv_lens_ref: jax.Ref,
      page_indices_ref: jax.Ref,
      # Inputs.
      schedule_hbm_ref: schedule.RpaSchedule,
      q_hbm_ref: jax.Ref,
      new_kv_hbm_ref: jax.Ref,
      kv_cache_hbm_ref: jax.Ref,
      # Outputs.
      o_hbm_ref: jax.Ref,
      o_kv_cache_hbm_ref: jax.Ref,
  ):

    del o_kv_cache_hbm_ref

    q_alloc, kv_cache_alloc, o_alloc = create_allocs(
        kv_cache_hbm_ref, q_hbm_ref, cfgs
    )

    actual_steps = schedule_hbm_ref.actual_steps[0]
    safe_steps = jnp.minimum(actual_steps, cfgs.max_steps_ub)
    pipeline_func = pltpu.emit_pipeline(
        body=functools.partial(
            rpa_body,
            cfgs=cfgs,
            cu_q_lens_ref=cu_q_lens_ref,
            kv_lens_ref=kv_lens_ref,
        ),
        grid=(safe_steps,),
        in_specs=(q_alloc.spec, kv_cache_alloc.spec),
        out_specs=(o_alloc.spec,),
    )

    @pl.with_scoped(
        final_allocs=(q_alloc, kv_cache_alloc, o_alloc),
        schedule_ref=schedule_hbm_ref.scratch_shapes(),
        dma_sem=pltpu.SemaphoreType.DMA((1,)),
        scratches=(
            pltpu.VMEM(
                cfgs.lm_scratch_shape,
                dtype=cfgs.serve.dtype_out,
            ),  # m
            pltpu.VMEM(
                cfgs.lm_scratch_shape,
                dtype=cfgs.serve.dtype_out,
            ),  # l
            pltpu.VMEM(
                cfgs.acc_scratch_shape,
                dtype=cfgs.serve.dtype_out,
            ),  # acc
        ),
    )
    def _run(final_allocs, schedule_ref, dma_sem, scratches):

      # Transfer schedule from HBM to SMEM --- we only copy what we need. Since
      # we almost always over-allocate schedule size, we only want to copy a
      # small portion of it from HBM to SMEM.
      flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
      flat_smem = jax.tree_util.tree_leaves(schedule_ref)
      dma_list = []
      for h, s in zip(flat_hbm, flat_smem):
        if h.memory_space == pltpu.HBM:
          read_size = (h.shape[0] // cfgs.max_steps_ub) * safe_steps
          read_size = utils.align_to(read_size, 1024)

          copy = pltpu.make_async_copy(
              h.at[pl.ds(0, read_size)],
              s.at[pl.ds(0, read_size)],
              dma_sem.at[0],
          )
          copy.start()
          dma_list.append(copy)

      # Initialize KV cache to zeros.
      # When perfomring p * v, we perform causal masking on lhs (p) by zeroing
      # out columns that should not be processed for a given row. Even if we
      # don't perform masking on rows of rhs (v), the output is still correct
      # since reuslt of multiplication will be zero thanks zero on lhs. However,
      # this assumption does not hold if a row of rhs has NaNs. To avoid this,
      # we initiallize scratch memory with non-zero values. Even if the scratch
      # memory is storing kv cache from previous step, as long as the data is
      # not NaNs, there will be no numeric concerns.
      num_lanes = pltpu.get_tpu_info().num_lanes
      kv_alloc = final_allocs[1]
      kv_ref_flat = kv_alloc.window_ref.bitcast(jnp.uint32).reshape(
          -1, num_lanes
      )
      kv_ref_flat[...] = jnp.zeros_like(kv_ref_flat)

      jax.tree.map(lambda x: x.wait(), dma_list)

      pipeline_func(
          (q_hbm_ref, schedule_ref),
          (kv_cache_hbm_ref, new_kv_hbm_ref, schedule_ref, page_indices_ref),
          (o_hbm_ref, schedule_ref),
          scratches=(schedule_ref,) + scratches,
          allocations=final_allocs,
      )

    _run()

  return pl.pallas_call(
      ragged_paged_attention_pipeline,
      out_shape=[q_hbm, kv_cache_hbm],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
              schedule_hbm.in_specs(),
              pl.BlockSpec(memory_space=pltpu.HBM),  # q_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_hbm_ref
          ],
          out_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_o_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_kv_cache_hbm_ref
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfgs.vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      input_output_aliases={12: 0, 14: 1},
      name=get_kernel_name(cfgs),
      metadata=get_kernel_metadata(cfgs),
  )(
      cu_q_lens,
      kv_lens,
      page_indices,
      schedule_hbm,
      q_hbm,
      new_kv_hbm,
      kv_cache_hbm,
  )

kernel = types.SimpleNamespace(
    rpa_kernel=rpa_kernel
)

# ==============================================================================
# Inlined batched_rpa/wrapper.py
# ==============================================================================
# Source: https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/experimental/batched_rpa/wrapper.py
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
"""Wrapper for RPA kernel to match expected interface.

NOTE: all of the code in this directory is experimental and not fully tested!
To enable usage of this kernel in full run, you can pass the USE_BATCHED_RPA_KERNEL=1
environment variable.

Compared to the default RPA kernel, this kernel does the following:

1. Batches multiple sequences together to replace per-request flash_attention loops. 

2. Enables triple-buffering via Pallas emit_pipeline

3. Precomputes expensive metadata upfront (e.g., page locations and bounds clipping) via 
scheduler.py kernel. Kernel is calculated once and ammortized across different layers in a model. 

Note: batched_rpa is build on top / derived from RPA3. 
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
# inlined
# inlined
# inlined
# inlined


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:

  total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
  _, actual_num_kv_heads, _ = k.shape
  num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

  q_packing = utils.get_dtype_packing(q_dtype)
  kv_packing = utils.get_dtype_packing(kv_dtype)

  aligned_num_q_heads_per_kv_head = utils.align_to(
      num_q_heads_per_kv_head, q_packing
  )
  num_lanes = pltpu.get_tpu_info().num_lanes
  aligned_head_dim = utils.align_to(actual_head_dim, num_lanes)

  # queries: (T, H, D) -> (T, H_kv, G, D)
  o_hbm_alias_q_hbm = (
      jnp.pad(
          q.reshape(
              total_q_tokens,
              actual_num_kv_heads,
              num_q_heads_per_kv_head,
              actual_head_dim,
          ),
          (
              (0, 0),
              (0, 0),
              (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
              (0, aligned_head_dim - actual_head_dim),
          ),
          constant_values=0,
      )
      .reshape(
          total_q_tokens,
          actual_num_kv_heads,
          aligned_num_q_heads_per_kv_head // q_packing,
          q_packing,
          aligned_head_dim,
      )
      .swapaxes(0, 1)
  )

  # Pad keys and values head_dim
  actual_num_kv_heads_x2 = actual_num_kv_heads * 2
  num_kv_heads_x2_aligned = utils.align_to(actual_num_kv_heads_x2, kv_packing)
  new_kv_hbm = jnp.pad(
      jnp.concatenate([k, v], axis=-1).reshape(
          total_q_tokens, actual_num_kv_heads_x2, actual_head_dim
      ),
      (
          (0, 0),
          (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
          (0, aligned_head_dim - actual_head_dim),
      ),
      constant_values=0,
  ).reshape(
      total_q_tokens,
      num_kv_heads_x2_aligned // kv_packing,
      kv_packing,
      aligned_head_dim,
  )
  return o_hbm_alias_q_hbm, new_kv_hbm


def prepare_outputs(out: jax.Array) -> jax.Array:
  kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
  return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
  num_lanes = pltpu.get_tpu_info().num_lanes
  kv_packing = utils.get_dtype_packing(kv_dtype)
  return (
      total_num_pages,
      page_size,
      utils.align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
      kv_packing,
      utils.align_to(actual_head_dim, num_lanes),
  )


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
  """Calculate optimal block size for decode and prefill."""

  tpu_info = pltpu.get_tpu_info()
  num_lanes = tpu_info.num_lanes
  mxu_column_size = tpu_info.mxu_column_size

  # Calculate aligned model dimensions.
  aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)
  aligned_num_q_heads_per_kv_head = utils.align_to(
      model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
  )
  aligned_num_q_heads = (
      aligned_num_q_heads_per_kv_head * model_cfgs.num_kv_heads
  )

  bkv_stride = pl.cdiv(model_cfgs.num_kv_heads * 2, serve_cfgs.packing_kv)
  if utils.has_bank_conflicts(bkv_stride):
    bkv_stride += 1
  aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

  q_bytes = jnp.dtype(serve_cfgs.dtype_q).itemsize
  kv_bytes = jnp.dtype(serve_cfgs.dtype_kv).itemsize
  out_bytes = jnp.dtype(serve_cfgs.dtype_out).itemsize

  def calculate_vmem_usage(
      batch_size: int, n_buffer: int, bq_sz: int, bkv_sz: int
  ) -> int:
    """Given tile size, calculate VMEM usage of the kernel."""

    # Step 1: Calculate buffer sizes.

    # Calculate size bq & bkv arrays for a single buffer.
    bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
    bkv_array_size = bkv_sz * aligned_num_kv_heads_x2 * aligned_head_dim

    # Get output buffer size as well - which has same size as query size.
    bo_array_size = bq_array_size

    # Convert to bytes.
    bq_bytes = bq_array_size * q_bytes
    bkv_bytes = bkv_array_size * kv_bytes
    bo_bytes = bo_array_size * out_bytes

    # Account for multiple buffers. For output, we always use double buffer.
    bq_bytes *= n_buffer
    bkv_bytes *= n_buffer
    bo_bytes *= 2

    # Sum up all buffer memory usage.
    buffer_bytes = bq_bytes + bkv_bytes + bo_bytes

    # Step 2: Calculate worst case memory usage during computation.

    # Calculate the size of loaded bq and bkv size.
    loaded_bq_size = bq_sz * model_cfgs.num_q_heads * aligned_head_dim
    loaded_bkv_size = bkv_sz * model_cfgs.num_kv_heads * aligned_head_dim

    # Calculate peak memory requirement of otuput - which is attention weight.
    qk_size = bq_sz * bkv_sz * model_cfgs.num_q_heads

    # Convert to bytes.
    loaded_bq_bytes = loaded_bq_size * q_bytes
    loaded_bkv_bytes = loaded_bkv_size * kv_bytes
    qk_bytes = qk_size * out_bytes

    # Sum up all compute memory usage.
    compute_bytes = loaded_bq_bytes + loaded_bkv_bytes + qk_bytes

    # Step 3: Sum up all memory usage.
    total_bytes = buffer_bytes + compute_bytes

    # Account for batch size.
    total_bytes *= batch_size

    return total_bytes

  def calculate_compute_buffer_time(
      batch_size: int, bq_c_sz: int, bkv_sz: int
  ) -> int:
    """Calculate computational complexity of a single compute block."""

    num_k_rows = pl.cdiv(bkv_sz, mxu_column_size)
    num_k_cols = pl.cdiv(model_cfgs.head_dim, mxu_column_size)
    num_k = num_k_rows * num_k_cols
    num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads

    return batch_size * num_muls

  def find_best_block_sizes(
      max_batch_size: int, max_n_buffer: int, fixed_bq_sz: int | None = None
  ) -> configs.BlockSizes:
    """Loop through different block sizes to find the most optimal one."""

    # Even if we loose some potential performance, we want to avoid OOM at all
    # costs. Therefore, we conservatively only use 80% of the VMEM budget.
    capped_vmem_limit_bytes = vmem_limit_bytes * 0.8

    bkv_sz = bkv_stride = mxu_column_size
    if fixed_bq_sz is None:
      bq_sz = bq_stride = bkv_sz
    else:
      bq_sz = fixed_bq_sz
      bq_stride = 0
    batch_size = max_batch_size
    n_buffer = max_n_buffer

    # Step 1: Lower batch_size and/or n_buffer if even the smallest bq and bkv
    # size can trigger OOM.

    # If current batch size triggers OOM, decrease batch size until the kernel
    # fits within VMEM limit.
    while (
        calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
        > capped_vmem_limit_bytes
    ):
      batch_size -= 1

    # As a last resort, attempt to decrease number of buffers to avoid OOM.
    while (
        calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
        > capped_vmem_limit_bytes
    ):
      n_buffer -= 1

    # Indicates OOM was triggered even when batch_size=1 or n_buffer=1.
    # NOTE: If the function does not exit at this point even when either values
    # are zero, it will trigger infinite loop at the next while loop.
    if batch_size == 0 or n_buffer == 0:
      raise ValueError("Cannot find batch size that fits within VMEM limit.")

    # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
    while (
        calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
        < capped_vmem_limit_bytes
    ):
      # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
      # size. When using causal masking, if bq size is larger than bkv size,
      # entire kv tile can be masked out for some query tokens. Similarly, if
      # bkv size is larger than bq size, entire query tile can be masked out for
      # some kv tokens.
      bkv_sz += bkv_stride
      bq_sz += bq_stride

    # Rollback one step since the last attempted value triggered OOM.
    bkv_sz -= bkv_stride
    bq_sz -= bq_stride

    # Indicates OOM was triggered from the starting bkv size.
    if bkv_sz == 0:
      raise ValueError("Cannot find block sizes that fit within VMEM limit.")

    # Step 3: Given current tile size, calculate compute tile size.

    # Fixed threshold value based on hardware spec.
    # TODO(kyuyeunk): Use different threshold based on hardware and precision.
    threshold = 1500

    num_bq_c = 1
    last_valid_bq_c_sz = bq_c_sz = bq_sz
    bq_c_rem = 0

    while (
        calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz) > threshold
        or bq_c_rem != 0
    ) and num_bq_c < bq_sz:
      if bq_c_rem == 0:
        last_valid_bq_c_sz = bq_c_sz
      num_bq_c += 1
      bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)

    return configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=last_valid_bq_c_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        n_buffer=n_buffer,
    )

  # Default to triple buffer as its almost always beneficial.
  n_buffer = 3
  # Fixed value based on experimental results.
  decode_batch_size = 8
  prefill_batch_size = 2

  decode_block_sizes = find_best_block_sizes(decode_batch_size, n_buffer, 1)
  prefill_block_sizes = find_best_block_sizes(prefill_batch_size, n_buffer)

  return decode_block_sizes, prefill_block_sizes


def get_vmem_estimate_bytes(
    bq_sz,
    bkv_sz,
    batch_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    q_dtype,
    kv_dtype,
    n_buffer=3,
):
  q_packing = utils.get_dtype_packing(q_dtype)
  kv_packing = utils.get_dtype_packing(kv_dtype)
  actual_num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_heads_per_kv_head = utils.align_to(
      actual_num_q_heads_per_kv_head, q_packing
  )
  num_kv_heads_x2 = utils.align_to(num_kv_heads * 2, kv_packing)
  aligned_head_dim = utils.align_to(head_dim, 128)

  total_bits = (
      # bkv_x2_ref
      (n_buffer * bkv_sz * num_kv_heads_x2 * aligned_head_dim)
      * (32 // kv_packing)
      +
      # bq_x2_ref + bo_x2_ref
      n_buffer
      * (2 * num_kv_heads * bq_sz * num_q_heads_per_kv_head * aligned_head_dim)
      * (32 // q_packing)
      +
      # l_ref + m_ref
      2 * (num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * 32
      +
      # acc_ref
      (num_kv_heads * bq_sz * num_q_heads_per_kv_head * aligned_head_dim) * 32
  )
  return pl.cdiv(total_bits, 8) * batch_size


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Perform batched ragged paged attention.

  Args:
    queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
    keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
    values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
    kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
      kv_packing, head_dim]. Stores existing kv cache data where k & vs are
      concatenated along num kv heads dim.
    kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
    page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
      sequence.
    cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
      length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
      b=cu_q_lens[i+1] represents q/k/v of sequence i.
    distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
      sequences. distribution[2] represents total number of sequences.
    sm_scale: Softmax scale value.
    sliding_window: Size of sliding window (also known as local attention). kvs
      outside of the window is not fetched from hbm and masked out during
      computation.
    soft_cap: Cap values of softmax inputs.
    mask_value: Value to use for causal masking. Defaults to smallest
      representable value of the activation dtype.
    q_scale: Quantization scale value of queries.
    k_scale: Quantization scale value of keys.
    v_scale: Quantization scale value of values.
    chunk_prefill_size: Not used.
    decode_block_sizes: Kernel block size to use during decode.
    prefill_block_sizes: Kernel block size to use during prefill.
    vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
      size of the hardware.
    debug_mode: Not used.
    out_dtype: Dtype of output. Defaults to dtype of queries.
    use_causal_mask: Not used.

  Returns:
    out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
    new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
      kv_packing, head_dim]. Result of new kv cache where k & vs are
      concatenated along num kv heads dim.
  """

  if not use_causal_mask:
    raise ValueError("Only causal attention is supported.")
  if chunk_prefill_size is not None:
    raise ValueError("Specifying chunk prefill size is not supported.")
  if debug_mode:
    raise ValueError("Debug mode is not supported.")

  if out_dtype is None:
    out_dtype = queries.dtype
  if mask_value is None:
    mask_value = jnp.finfo(out_dtype).min  # pyrefly: ignore[bad-assignment]
  if vmem_limit_bytes is None:
    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

  max_num_seqs = kv_lens.shape[0]
  page_size = kv_cache.shape[1]

  num_q_heads = queries.shape[1]
  head_dim = queries.shape[2]
  num_kv_heads = keys.shape[1]
  num_page_indices = page_indices.shape[0]

  model_cfgs = configs.ModelConfigs(
      num_q_heads=num_q_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      sliding_window=sliding_window,
      sm_scale=sm_scale,
      soft_cap=soft_cap,
      mask_value=mask_value,  # pyrefly: ignore[bad-argument-type]
  )
  serve_cfgs = configs.ServingConfigs(
      num_seqs=max_num_seqs,
      num_page_indices=num_page_indices,
      total_q_tokens=queries.shape[0],
      dtype_q=queries.dtype,
      dtype_kv=kv_cache.dtype,
      dtype_out=out_dtype,
      page_size=page_size,
      scale_q=q_scale,  # pyrefly: ignore[bad-argument-type]
      scale_k=k_scale,  # pyrefly: ignore[bad-argument-type]
      scale_v=v_scale,  # pyrefly: ignore[bad-argument-type]
  )

  q_hbm, new_kv_hbm = prepare_inputs(
      queries, keys, values, queries.dtype, kv_cache.dtype
  )

  default_decode, default_prefill = calculate_block_sizes(
      model_cfgs, serve_cfgs, vmem_limit_bytes
  )

  def run_rpa_kernel(
      mode: configs.RpaCase,
      o_hbm_alias_q_hbm: jax.Array,
      kv_cache: jax.Array,
  ):
    if mode == configs.RpaCase.DECODE:
      effective_blocks = decode_block_sizes or default_decode
    else:
      effective_blocks = prefill_block_sizes or default_prefill

    cfgs = configs.RpaConfigs(
        block=effective_blocks,
        model=model_cfgs,
        serve=serve_cfgs,
        vmem_limit_bytes=vmem_limit_bytes,
        mode=mode,
    )
    cfgs.validate_inputs(
        q=queries,
        k=keys,
        v=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )

    schedule_hbm = schedule.generate_rpa_metadata(
        cu_q_lens, kv_lens, distribution, cfgs=cfgs
    )
    return kernel.rpa_kernel(
        cu_q_lens,
        kv_lens,
        page_indices,
        schedule_hbm,
        o_hbm_alias_q_hbm,
        new_kv_hbm,
        kv_cache,
        cfgs=cfgs,
    )

  o_hbm_alias_q_hbm, kv_cache = run_rpa_kernel(
      configs.RpaCase.DECODE, q_hbm, kv_cache
  )
  o_hbm_alias_q_hbm, kv_cache = run_rpa_kernel(
      configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache
  )

  # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
  o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
  # after: [kv_heads, max_tokens, q_per_kv, d]

  # slice back to original shape if padded
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
  o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

  return o_hbm, kv_cache



batched_ragged_paged_attention = ragged_paged_attention

CONFIGS = {
    'llama3_8b_decode': {
        'name': 'batched_rpa_llama3_8b_decode',
        'model': 'Llama-3.1-8B',
        'operator': 'batched_ragged_paged_attention',
        'max_num_batched_tokens': 64,
        'max_num_seqs': 64,
        'num_q_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'page_size': 16,
        'pages_per_seq': 256,
        'active_q_lens': (1,) * 64,
        'active_kv_lens': tuple(
            128 + ((i * 37) % 120) * 16 for i in range(64)
        ),
        'distribution': (64, 64, 64),
    },
    'llama3_70b_decode': {
        'name': 'batched_rpa_llama3_70b_decode',
        'model': 'Llama-3.1-70B',
        'operator': 'batched_ragged_paged_attention',
        'max_num_batched_tokens': 64,
        'max_num_seqs': 64,
        'num_q_heads': 64,
        'num_kv_heads': 8,
        'head_dim': 128,
        'page_size': 16,
        'pages_per_seq': 256,
        'active_q_lens': (1,) * 64,
        'active_kv_lens': tuple(
            256 + ((i * 47) % 200) * 16 for i in range(64)
        ),
        'distribution': (64, 64, 64),
    },
    'qwen3_32b_decode': {
        'name': 'batched_rpa_qwen3_32b_decode',
        'model': 'Qwen3-32B',
        'operator': 'batched_ragged_paged_attention',
        'max_num_batched_tokens': 64,
        'max_num_seqs': 64,
        'num_q_heads': 40,
        'num_kv_heads': 8,
        'head_dim': 128,
        'page_size': 16,
        'pages_per_seq': 256,
        'active_q_lens': (1,) * 64,
        'active_kv_lens': tuple(
            128 + ((i * 53) % 150) * 16 for i in range(64)
        ),
        'distribution': (64, 64, 64),
    },
}

CONFIG = CONFIGS['llama3_8b_decode']
ACTIVE_Q_LENS = CONFIG['active_q_lens']
ACTIVE_KV_LENS = CONFIG['active_kv_lens']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Create inputs matching the TPU kernel specifications for Batched RPA."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )
  key = jax.random.key(42)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)

  max_tokens = cfg['max_num_batched_tokens']
  max_seqs = cfg['max_num_seqs']
  h_q = cfg['num_q_heads']
  h_kv = cfg['num_kv_heads']
  d = cfg['head_dim']
  page_size = cfg['page_size']
  pages_per_seq = cfg['pages_per_seq']
  total_pages = max_seqs * pages_per_seq
  kv_packing = get_dtype_packing(dtype)

  queries = jax.random.normal(k1, (max_tokens, h_q, d), dtype=dtype)
  keys = jax.random.normal(k2, (max_tokens, h_kv, d), dtype=dtype)
  values = jax.random.normal(k3, (max_tokens, h_kv, d), dtype=dtype)

  num_kv_heads_x2 = align_to(h_kv * 2, kv_packing)
  cache_shape = (
      total_pages,
      page_size,
      num_kv_heads_x2 // kv_packing,
      kv_packing,
      align_to(d, 128),
  )
  kv_cache = jax.random.normal(k5, cache_shape, dtype=dtype)

  active_q = cfg['active_q_lens']
  active_kv = cfg['active_kv_lens']
  num_active = len(active_q)

  kv_lens = jnp.pad(
      jnp.array(active_kv, dtype=jnp.int32), (0, max_seqs - num_active)
  )
  q_lens = jnp.array(active_q, dtype=jnp.int32)
  active_cu_q = jnp.concatenate(
      [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(q_lens)]
  )
  cu_q_lens = jnp.pad(
      active_cu_q,
      (0, max_seqs + 1 - len(active_cu_q)),
      constant_values=active_cu_q[-1],
  )

  page_indices = jax.random.permutation(
      k4, total_pages, independent=True
  ).astype(jnp.int32)
  distribution = jnp.array(cfg['distribution'], dtype=jnp.int32)

  return (
      queries,
      keys,
      values,
      kv_cache,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
  )


def get_inputs(dtype=jnp.bfloat16):
  """Returns list of (dynamic_args, static_args) for all configs."""
  return [
      (list(create_inputs(dtype=dtype, config=cfg)), [])
      for cfg in CONFIGS.values()
  ]


def get_flops(config=None):
  """Estimate attention FLOPs: 4 * H_q * D * sum(q_len * kv_len)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )
  h_q = cfg['num_q_heads']
  d = cfg['head_dim']
  active_q = cfg.get('active_q_lens', ACTIVE_Q_LENS)
  active_kv = cfg.get('active_kv_lens', ACTIVE_KV_LENS)
  return int(h_q * 4 * d * sum(q * k for q, k in zip(active_q, active_kv)))


batched_ragged_paged_attention = ragged_paged_attention


def workload(
    queries,
    keys,
    values,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
):
  """Optimized batched ragged paged attention with fused KV cache update."""
  actual_head_dim = queries.shape[2]
  sm_scale = 1.0 / math.sqrt(actual_head_dim)
  return batched_ragged_paged_attention(
      queries,
      keys,
      values,
      kv_cache,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      sm_scale=sm_scale,
  )


def benchmark(num_warmup=5, num_iters=100, config=None):
  """Benchmark optimized kernel and return results dict."""
  import time

  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )
  inputs = create_inputs(config=cfg)
  fn = jax.jit(workload)
  for _ in range(num_warmup):
    out = fn(*inputs)
    jax.block_until_ready(out)
  times = []
  for _ in range(num_iters):
    t0 = time.perf_counter()
    out = fn(*inputs)
    jax.block_until_ready(out)
    times.append(time.perf_counter() - t0)
  times = np.array(times) * 1000
  avg = float(np.mean(times))
  flops = get_flops(cfg)
  attn_out, updated_cache = out
  return {
      'name': cfg['name'],
      'model': cfg['model'],
      'operator': cfg['operator'],
      'config': {
          k: v
          for k, v in cfg.items()
          if k
          not in (
              'name',
              'model',
              'operator',
              'active_q_lens',
              'active_kv_lens',
          )
      },
      'time_ms': round(avg, 4),
      'std_ms': round(float(np.std(times)), 4),
      'tflops': round(flops / (avg / 1000) / 1e12, 2) if avg > 0 else 0.0,
      'output_shape': list(attn_out.shape),
      'status': 'success',
  }


if __name__ == '__main__':
  import json

  print(json.dumps(benchmark()))

