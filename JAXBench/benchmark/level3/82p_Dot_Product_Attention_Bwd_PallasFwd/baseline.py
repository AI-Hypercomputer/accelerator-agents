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
"""Causal dot-product attention backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

`forward_pallas(q, k, v)` is the Tokamax Pallas TPU forward pass: the Splash
attention kernel, applied to each batch element. It returns `(out, residuals)`,
where `residuals = (max_logits, logsumexp)` are the softmax statistics the
kernel saves for a backward pass, both over the keys each query can see and in
natural-log units:
  max_logits  `[B, H, T]` float32  maximum of the logits `(q * D**-0.5) @ k.T`
  logsumexp   `[B, H, T]` float32  log of the sum of the exponentiated logits
The kernel rescales `q` in the input dtype before the matmul, so for bfloat16
inputs both statistics include that rounding.

`workload(q, k, v, dout)` returns the gradients `(dq, dk, dv)` of causal,
grouped-query scaled dot-product attention for the output cotangent `dout`. It
is the correctness oracle: JAX autodiff (`jax.vjp`) of the pure-JAX forward
reference `forward`. This file contains no backward kernel and no hand-derived
backward pass.

`forward` is Tokamax's XLA attention with its options fixed: causal mask,
logits scaled by `D ** -0.5`, default dot precision, and no bias, logit soft
cap, dropout or paging. In Tokamax the softmax carries a custom JVP rule; here
it is the plain function, differentiated by JAX.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` (it
may consume the saved residuals) so that your `workload` returns the same
gradients as this one.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/splash_attention/splash_attention_kernel.py     (kernel)
  tokamax/_src/ops/splash_attention/splash_attention_mask.py       (masks)
  tokamax/_src/ops/splash_attention/splash_attention_mask_info.py  (schedule)
  tokamax/_src/ops/splash_attention/reference.py           (mask value)
  tokamax/_src/ops/attention/pallas_mosaic_tpu.py          (kernel wrapper)
  tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py   (block sizes)
  tokamax/_src/ops/attention/base.py       (forward: `_fwd`, `_softmax`)
  tokamax/_src/precision.py                (default dot precision)
  tokamax/_src/shape.py                    (float32 upcast for repeated heads)
  tokamax/_src/ops/attention/arg_specs.py  (shapes)
  tokamax/_src/ops/attention/test_base.py  (value ranges)
"""

# Dimensions.
#   B   batch size
#   T   query sequence length
#   t   key/value sequence length (equal to T here)
#   H   number of query heads
#   h   number of key/value heads (divides H)
#   D   query/key head dimension
#   d   value head dimension
#   bq           query block size of the kernel (`block_q`)
#   bkv          key/value block size of the kernel (`block_kv`)
#   bkv_compute  key/value tile size of one matmul inside a block

import collections
from collections.abc import Callable
import dataclasses
import enum
import functools
import json
import math
from typing import Any, NamedTuple

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference
# (from tokamax/_src/ops/attention/base.py, with helpers from
#  tokamax/_src/precision.py and tokamax/_src/shape.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================


def _dot_preset(a_dtype, b_dtype):
  """Dot algorithm Tokamax selects for `jax.lax.Precision.DEFAULT`.

  `to_dot_algorithm_preset` from tokamax/_src/precision.py with the precision
  fixed to `DEFAULT` and the dtypes limited to bfloat16 and float32.
  """
  dtype = jnp.result_type(a_dtype, b_dtype)
  backend = jax.default_backend()
  if backend == 'gpu':
    device = jax.devices()[0]
    compute_capability = getattr(device, 'compute_capability', None)
    if compute_capability is not None and float(compute_capability) < 8.0:
      backend = 'gpu_old'

  if dtype == jnp.bfloat16:
    if backend == 'gpu_old':
      return jax.lax.DotAlgorithmPreset.F32_F32_F32
    return jax.lax.DotAlgorithmPreset.BF16_BF16_F32
  if dtype == jnp.float32:
    return {
        'tpu': jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        'gpu_old': jax.lax.DotAlgorithmPreset.F32_F32_F32,
        'gpu': jax.lax.DotAlgorithmPreset.TF32_TF32_F32,
        'cpu': jax.lax.DotAlgorithmPreset.F32_F32_F32,
    }[backend]
  raise ValueError(f'Unsupported dtype: {dtype}')


def _repeat_kv_heads(x, repeats):
  """Repeats each key/value head `repeats` times along the head axis."""
  # Tokamax calls `jnp.repeat` under `shape.upcast_broadcast()`, which patches
  # the broadcast inside it to upcast to float32 first and cast back after, so
  # that the VJP sums over the repeated heads in float32. The casts are written
  # out here instead; the forward values are unchanged.
  dtype = x.dtype
  x = x.astype(jnp.promote_types(dtype, jnp.float32))
  return jnp.repeat(x, repeats, axis=-2).astype(dtype)


def _softmax(x):
  """Softmax over the last axis; rows that are fully masked give zeros."""
  # Always perform reductions in at least f32 precision.
  dtype = jnp.promote_types(x.dtype, jnp.float32)
  x_max = jnp.max(x.astype(dtype), axis=-1, keepdims=True)
  safe_x_max = jnp.where(jnp.isneginf(x_max), 0.0, x_max)
  unnormalized = jnp.exp(x - safe_x_max)
  denom = jnp.sum(unnormalized, axis=-1, keepdims=True)
  safe_denom = jnp.where(denom == 0.0, 1.0, denom)
  return unnormalized / safe_denom


def forward(q, k, v):
  """Causal scaled dot-product attention with grouped key/value heads.

  Computes `softmax(q @ k.T / sqrt(D)) @ v` per head, where query `i` attends
  to keys `j <= i`. Query heads `[g * H // h, (g + 1) * H // h)` share
  key/value head `g`.

  Args:
    q: Queries, `[B, T, H, D]`.
    k: Keys, `[B, t, h, D]`; `h` divides `H`.
    v: Values, `[B, t, h, d]`.

  Returns:
    The output, `[B, T, H, d]`, in the dtype of `q`.
  """
  if k.shape[-2] not in (1, q.shape[-2]):
    repeats = q.shape[-2] // k.shape[-2]
    k = _repeat_kv_heads(k, repeats)
    v = _repeat_kv_heads(v, repeats)

  q_k_dot_precision = _dot_preset(q.dtype, k.dtype)
  weights_v_dot_precision = _dot_preset(v.dtype, v.dtype)
  logits_dtype = jnp.dtype(q_k_dot_precision.accumulation_type)
  logits_scale = 1 / math.sqrt(q.shape[-1])

  logits = jnp.einsum(
      '...qhd,...khd->...hqk',
      q,
      k,
      precision=q_k_dot_precision,
      preferred_element_type=q_k_dot_precision.accumulation_type,
  ).astype(logits_dtype)

  logits *= logits_scale

  q_indices = jnp.arange(q.shape[-3])[..., None]
  k_indices = jnp.arange(k.shape[-3])[..., None, :]
  mask = q_indices >= k_indices
  logits = jnp.where(mask, logits, -float('inf'))

  weights = _softmax(logits)

  out = jnp.einsum(
      '...hqk,...khd->...qhd',
      weights,
      v,
      precision=weights_v_dot_precision,
      preferred_element_type=weights_v_dot_precision.accumulation_type,
  ).astype(q.dtype)
  return out


# ==============================================================================
# Pallas TPU forward kernel (inlined from Tokamax, Apache-2.0)
# ==============================================================================

# ------------------------------------------------------------------------------
# tokamax/_src/ops/splash_attention/reference.py
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------------

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype('float32')).max)

SplashCustomReturnType = jax.Array | tuple[jax.Array, dict[str, jax.Array]]


# ------------------------------------------------------------------------------
# tokamax/_src/ops/splash_attention/splash_attention_mask.py
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------------


class Mask:
  """A base class for splash attention masks."""

  @property
  def shape(self) -> tuple[int, ...]:
    raise NotImplementedError

  def __getitem__(self, idx) -> np.ndarray:
    raise NotImplementedError

  def __bool__(self) -> bool:
    raise NotImplementedError(
        'Conversion to bool is unsupported. Could be caused by using logical'
        ' instead of bitwise operations on masks.'
    )

  def __or__(self, other: 'Mask') -> 'Mask':
    if self.shape != other.shape:
      raise ValueError(
          f'Invalid shape for other: {other.shape}, expected: {self.shape}'
      )
    return LogicalOr(self, other)

  def __and__(self, other: 'Mask') -> 'Mask':
    if self.shape != other.shape:
      raise ValueError(
          f'Invalid shape for other: {other.shape}, expected: {self.shape}'
      )
    return LogicalAnd(self, other)


@dataclasses.dataclass(slots=True)
class LogicalOr(Mask):
  left: Mask
  right: Mask

  def __init__(self, left: Mask, right: Mask):
    if left.shape != right.shape:
      raise ValueError('Masks must have the same shape')
    self.left = left
    self.right = right

  @property
  def shape(self) -> tuple[int, ...]:
    return self.left.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.left[idx] | self.right[idx]

  def __hash__(self):
    return hash((type(self),) + (self.left, self.right))


@dataclasses.dataclass(slots=True)
class LogicalAnd(Mask):
  left: Mask
  right: Mask

  def __init__(self, left: Mask, right: Mask):
    if left.shape != right.shape:
      raise ValueError('Masks must have the same shape')
    self.left = left
    self.right = right

  @property
  def shape(self) -> tuple[int, ...]:
    return self.left.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.left[idx] & self.right[idx]

  def __hash__(self):
    return hash((type(self),) + (self.left, self.right))


class _ComputableMask(Mask):
  """Superclass for all masks that can be computed inside the kernel using a callable object.

  This subclass is designed to be used with Splash Attention.
  It allows the mask logic to be computed on-the-fly or fused into the attention
  kernel, avoiding the memory cost of materializing the full
  (sequence_length, sequence_length) boolean mask array, which can be excessive
  for long sequences.

  Attributes:
    _shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
    q_sequence: Indices of Q sequence. q_sequence is reused across __getitem__
      calls which is important for compile-time performance.
    mask_function: Function used by the SplashAttention kernel to compute the
      mask rather than loading it.
  """

  _shape: tuple[int, int]
  q_sequence: np.ndarray
  mask_function: Callable[..., Any]

  def __init__(
      self,
      shape: tuple[int, int],
      mask_function: Callable[..., Any],
      shard_count: int = 1,
  ):
    self._shape = shape
    self.mask_function = mask_function
    q_seq_len = self.shape[0]

    if q_seq_len % (shard_count * shard_count) != 0:
      raise ValueError(
          f'Shard count squared ({shard_count * shard_count}) must'
          f' divide Q seq_len ({self.shape[0]}) evenly.'
      )

    self.q_sequence = np.arange(q_seq_len, dtype=np.int32)

  @property
  def shape(self) -> tuple[int, ...]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')

    q_slice, kv_slice = idx
    if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')

    q_slice = _fill_slice(q_slice, self.shape[0])
    kv_slice = _fill_slice(kv_slice, self.shape[1])

    rows = self.q_sequence[q_slice]
    cols = np.arange(kv_slice.start, kv_slice.stop)

    return self.mask_function(rows[:, None], cols[None, :])

  def __eq__(self, other: object):
    raise NotImplementedError()

  def __hash__(self):
    raise NotImplementedError()


class CausalMask(_ComputableMask):
  """Lazy causal mask, prevents the model from attending to future tokens.

  Attributes:
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  offset: int

  def __init__(
      self,
      shape: tuple[int, int],
      offset: int = 0,
      shard_count: int = 1,
  ):
    self.offset = offset

    def causal_mask_function(q_ids, kv_ids):
      # When evaluating the mask in _process_mask we typically work with numpy
      # array views.
      # Avoid the addition when possible to avoid instantiating an actual array.
      if self.offset == 0:
        return q_ids >= kv_ids
      else:
        return q_ids + self.offset >= kv_ids

    mask_function = causal_mask_function

    super().__init__(
        shape=shape,
        mask_function=mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
        self.shape == other.shape
        and self.offset == other.offset
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.offset,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


class ChunkedCausalMask(_ComputableMask):
  """Lazy chunked causal mask.

  Attention is causal within each chunk (0, K), (K, 2K), (2K, 3K), ... tokens
  attend to each other but not across chunks.
  Llama4 models use interleaved chunk attention along with global attention.


  Attributes:
    chunk_size: The size of each attention chunk.
  """

  chunk_size: int

  def __init__(
      self,
      shape: tuple[int, int],
      chunk_size: int,
      shard_count: int = 1,
  ):
    if chunk_size <= 0:
      raise ValueError('chunk_size must be positive')
    self.chunk_size = chunk_size

    # Define the mask function for chunk attention
    def chunked_causal_mask_function(q_ids, kv_ids):
      """Computes the mask logic for the given slice indices."""
      # Condition 1: Same chunk
      same_chunk = (q_ids // self.chunk_size) == (kv_ids // self.chunk_size)

      # Condition 2: Causal
      causal = q_ids >= kv_ids

      return same_chunk & causal

    super().__init__(
        shape=shape,
        mask_function=chunked_causal_mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
        self.shape == other.shape
        and self.chunk_size == other.chunk_size
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.chunk_size,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


class LocalMask(_ComputableMask):
  """Lazy local mask, prevents model from attending to tokens outside window.

  Attributes:
    window_size: Size of the two sides of the local window (None identifies no
      limit for the given side).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  window_size: tuple[int | None, int | None]
  offset: int

  def __init__(
      self,
      shape: tuple[int, int],
      window_size: tuple[int | None, int | None],
      offset: int,
      shard_count: int = 1,
  ):
    self.window_size = window_size
    self.offset = offset

    def local_mask_function(q_ids, kv_ids):
      """Computes the local attention mask for the given slice indices."""
      left_size, right_size = self.window_size

      assert q_ids.ndim == 2
      assert kv_ids.ndim == 2

      if left_size is None and right_size is None:
        return np.ones((q_ids.shape[0], kv_ids.shape[1]), dtype=np.bool_)

      # Avoid the addition when possible to avoid instantiating an actual array.
      if offset != 0:
        shifted_q_ids = q_ids + self.offset
      else:
        shifted_q_ids = q_ids

      mask = None
      if left_size is not None:
        mask = shifted_q_ids - left_size <= kv_ids
      if right_size is not None:
        if mask is None:
          mask = shifted_q_ids + right_size >= kv_ids
        else:
          mask &= shifted_q_ids + right_size >= kv_ids
      return mask

    super().__init__(
        shape=shape,
        mask_function=local_mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return False

    return (
        self.shape == other.shape
        and self.window_size == other.window_size
        and self.offset == other.offset
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.window_size,
        self.offset,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


@dataclasses.dataclass(slots=True)
class NumpyMask(Mask):
  """A mask backed by a dense numpy array."""

  array: np.ndarray

  def __post_init__(self):
    if self.array.ndim != 2:
      raise ValueError('Expected a 2-dim array')

    if self.array.dtype != np.bool_:
      raise ValueError('Mask must be a boolean array')

  @property
  def shape(self) -> tuple[int, ...]:
    return self.array.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.array[idx]

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return np.array_equal(self.array, other.array, equal_nan=True)

  def __hash__(self):
    return hash((type(self), self.array.tobytes()))


def _fill_slice(inp_slice: slice, size: int) -> slice:
  assert inp_slice.step is None or inp_slice.step == 1
  start = 0 if inp_slice.start is None else inp_slice.start
  stop = size if inp_slice.stop is None else inp_slice.stop
  assert start >= 0
  assert stop <= size
  return slice(start, stop, None)


@dataclasses.dataclass(frozen=True, slots=True)
class FullMask(Mask):
  """Lazy full mask, allows all tokens to attend to all other tokens."""

  # TODO: Transform FullMask into a _ComputableMask.

  _shape: tuple[int, int]

  def __post_init__(self):
    if not isinstance(self.shape, tuple):
      raise ValueError(f'Unsupported shape type: {type(self.shape)}')

  @property
  def shape(self) -> tuple[int, ...]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')
    i, j = idx
    if not isinstance(i, slice) or not isinstance(j, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')
    i = _fill_slice(i, self.shape[0])
    j = _fill_slice(j, self.shape[1])
    return np.ones((i.stop - i.start, j.stop - j.start), dtype=np.bool_)

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return self.shape == other.shape

  def __hash__(self):
    return hash((type(self), self.shape))


# ------------------------------------------------------------------------------
# tokamax/_src/ops/splash_attention/splash_attention_mask_info.py
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------------


MaskCallable = Any


def find_bounds(
    arr: jax.Array | np.ndarray,
) -> tuple[jax.Array | np.ndarray | None, jax.Array | np.ndarray | None]:
  # Find the first and last block of a row to determine when to initialize/store
  # the output.

  if arr is None:
    return None, None

  bounds_start = (arr != jnp.roll(arr, shift=1, axis=-1)).astype(jnp.int32)
  bounds_end = (arr != jnp.roll(arr, shift=-1, axis=-1)).astype(jnp.int32)
  bounds_start = bounds_start.at[0].set(1)
  bounds_end = bounds_end.at[-1].set(1)

  return bounds_start, bounds_end


def dkv_q_blocks_are_contiguous(
    mask_info: 'MaskInfo', q_blocks_per_shard: int
) -> bool:
  """Returns whether the dkv grid visits each q block in one contiguous run.

  In every shard each q block must appear in a single run of consecutive steps.
  Trivially true without a dynamic grid.

  Args:
    mask_info: dKV `MaskInfo` with concrete (non-traced) arrays.
    q_blocks_per_shard: Number of q blocks in each q sequence shard.
  """
  if mask_info.active_cols is None:
    return True
  num_active = np.asarray(mask_info.num_active_blocks).reshape(-1)
  shard_cols = np.asarray(mask_info.active_cols).reshape(num_active.size, -1)
  for cols, n in zip(shard_cols, num_active):
    cols = cols[:n].astype(np.int32)
    run_starts = cols[np.diff(cols, prepend=-1) != 0]
    if not np.array_equal(np.sort(run_starts), np.arange(q_blocks_per_shard)):
      return False
  return True


# Logic for processing NumPy masks for kernels
class MaskInfo(NamedTuple):
  """Contains runtime masking information for the Splash attention kernel.

  The arrays, mask_next and block_mask are placed in TPU
  scalar-memory. This is a scarse resource so the mask creation logic attempts
  to shrink the data-type of these arrays to the smallest possible one.
  This can be: np.int32, np.int16 or np.int8.

  Attributes:
    mask_next: An integer[num_active_blocks] NumPy array where each entry
      contains the next mask block index in `partial_mask_blocks` to prefetch.
    active_rows: An integer[num_active_blocks] NumPy array where each entry
      contains the row index of the corresponding active block in the original
      mask.
    active_cols: An integer[num_active_blocks] NumPy array where each entry
      contains the column index of the corresponding active block in the
      original mask.
    block_mask: An integer[num_active_blocks] NumPy array where each entry is 0,
      1 or 2. 1 means the corresponding block is partially masked (and so the
      mask in `partial_mask_blocks` must be applied), and 2 means the
      corresponding block is full (no mask needs to be applied). 0 means the
      block is entirely masked out; it only appears in a dense (uncompacted)
      `block_mask`, or as a block that is scheduled purely so that its row's
      output gets written.
    num_active_blocks: An integer[1] NumPy array containing the number of
      leading entries of the compacted arrays above that are valid. The
      remaining entries are padding.
    partial_mask_blocks: An int8[num_partial_blocks, block_q, block_kv] NumPy
      array that contains the blocks of the original mask that contained both
      zeros and ones. The entries in `mask_next` point to indices in the first
      axis of this array.
    q_sequence: A i32[q_sequence_length] NumPy array. When using causal masking,
      this contains the list of indices that correspond to q tokens. For plain
      causal this is just np.arange(q_sequence_length).
  """

  mask_next: np.ndarray | jax.Array | None
  active_rows: np.ndarray | jax.Array | None
  active_cols: np.ndarray | jax.Array | None
  block_mask: np.ndarray | jax.Array | None
  num_active_blocks: np.ndarray | jax.Array | None
  partial_mask_blocks: np.ndarray | jax.Array | None
  q_sequence: np.ndarray | None


def _downcast_to_small_type(array: np.ndarray) -> np.ndarray:
  """Downcast numpy array.

  If possible, downcast the data-type of the input array to the smallest numpy
  type (among np.int16 and np.int8) that fits the content of the array.

  Args:
    array: the array to downcast

  Returns:
    The downcasted array.

  Raises:
    ValueError: if the input array is not np.int32 or if its elements are not
    all positive.
  """
  if array.dtype != np.int32:
    raise ValueError(f'Expected int32 input, but got {array.dtype}.')

  if not np.all(array >= -1):
    # Allow -1 for padding.
    raise ValueError('Expected non-negative array.')

  if array.size == 0:
    return array

  max_value = np.max(array)

  if max_value <= np.iinfo(np.int8).max:
    return array.astype(np.int8)
  elif max_value <= np.iinfo(np.int16).max:
    return array.astype(np.int16)
  else:
    return array.astype(np.int32)


class _HashableNDArray:
  """Helper to make a numpy array hashable: can be added associative containers.

  Attributes:
    array: The underlying numpy array.
  """

  __slots__ = ('array', '_hash')
  array: np.ndarray

  def __init__(self, array: np.ndarray):
    self.array = array
    self._hash = hash(array.tobytes())

  def __hash__(self):
    return self._hash

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, _HashableNDArray):
      return NotImplemented
    return np.array_equal(self.array, other.array, equal_nan=True)


def _q_block_stats(
    q_sequence: np.ndarray, q_blocks_count: int, q_block_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes per-block min, max, and step-1 contiguity for `q_sequence`."""
  q_seq = np.asarray(q_sequence).reshape(q_blocks_count, q_block_size)
  q_min = q_seq.min(axis=1).astype(np.int64)
  q_max = q_seq.max(axis=1).astype(np.int64)
  if q_block_size == 1:
    q_contig = np.ones(q_blocks_count, dtype=np.bool_)
  else:
    q_contig = (q_seq[:, 1:] == q_seq[:, :-1] + 1).all(axis=1)
  return q_min, q_max, q_contig


def _classify_mask_blocks(
    mask: Mask,
    q_blocks_count: int,
    kv_blocks_count: int,
    q_block_size: int,
    kv_block_size: int,
) -> np.ndarray:
  """Classifies mask blocks as 0 (empty), 1 (partial), 2 (full), or -1."""
  # pylint: disable=unidiomatic-typecheck
  blocked_shape = (q_blocks_count, kv_blocks_count)

  if type(mask) is FullMask:
    return np.full(blocked_shape, 2, dtype=np.int32)

  if type(mask) is NumpyMask and mask.array.flags['C_CONTIGUOUS']:
    blocks = mask.array.reshape(
        q_blocks_count, q_block_size, kv_blocks_count, kv_block_size
    ).swapaxes(1, 2)
    return blocks.any(axis=(2, 3)).astype(np.int32) + blocks.all(
        axis=(2, 3)
    ).astype(np.int32)

  if type(mask) is LogicalOr:
    left = _classify_mask_blocks(
        mask.left, q_blocks_count, kv_blocks_count, q_block_size, kv_block_size
    )
    right = _classify_mask_blocks(
        mask.right,
        q_blocks_count,
        kv_blocks_count,
        q_block_size,
        kv_block_size,
    )
    grid = np.full(blocked_shape, -1, dtype=np.int32)
    grid[(left == 0) & (right == 0)] = 0
    grid[((left == 1) & (right == 0)) | ((left == 0) & (right == 1))] = 1
    grid[(left == 2) | (right == 2)] = 2
    return grid

  if type(mask) is LogicalAnd:
    left = _classify_mask_blocks(
        mask.left, q_blocks_count, kv_blocks_count, q_block_size, kv_block_size
    )
    right = _classify_mask_blocks(
        mask.right,
        q_blocks_count,
        kv_blocks_count,
        q_block_size,
        kv_block_size,
    )
    grid = np.full(blocked_shape, -1, dtype=np.int32)
    grid[(left == 0) | (right == 0)] = 0
    grid[((left == 1) & (right == 2)) | ((left == 2) & (right == 1))] = 1
    grid[(left == 2) & (right == 2)] = 2
    return grid

  kv_min = np.arange(kv_blocks_count, dtype=np.int64) * kv_block_size
  kv_max = kv_min + (kv_block_size - 1)

  if type(mask) is CausalMask:
    q_min, q_max, _ = _q_block_stats(
        mask.q_sequence, q_blocks_count, q_block_size
    )
    offset = int(mask.offset)
    grid = np.ones(blocked_shape, dtype=np.int32)
    grid[(q_max + offset)[:, None] < kv_min[None, :]] = 0
    grid[(q_min + offset)[:, None] >= kv_max[None, :]] = 2
    return grid

  if type(mask) is LocalMask:
    left_w, right_w = mask.window_size
    if left_w is None and right_w is None:
      return np.full(blocked_shape, 2, dtype=np.int32)
    if left_w is not None and right_w is not None and left_w + right_w < 0:
      return np.zeros(blocked_shape, dtype=np.int32)

    q_min, q_max, q_contig = _q_block_stats(
        mask.q_sequence, q_blocks_count, q_block_size
    )
    offset = int(mask.offset)

    if left_w is None:
      all_left = True
      empty_left = False
    else:
      all_left = (q_max + offset - left_w)[:, None] <= kv_min[None, :]
      empty_left = (q_min + offset - left_w)[:, None] > kv_max[None, :]

    if right_w is None:
      all_right = True
      empty_right = False
    else:
      all_right = (q_min + offset + right_w)[:, None] >= kv_max[None, :]
      empty_right = (q_max + offset + right_w)[:, None] < kv_min[None, :]

    full = all_left & all_right
    empty = empty_left | empty_right

    grid = np.full(blocked_shape, 1, dtype=np.int32)
    if left_w is not None and right_w is not None and not q_contig.all():
      grid[~q_contig, :] = -1
    grid[empty] = 0
    grid[full] = 2
    return grid

  if type(mask) is ChunkedCausalMask:
    q_min, q_max, q_contig = _q_block_stats(
        mask.q_sequence, q_blocks_count, q_block_size
    )
    chunk_size = int(mask.chunk_size)
    q_chunk_min = q_min // chunk_size
    q_chunk_max = q_max // chunk_size
    kv_chunk_min = kv_min // chunk_size
    kv_chunk_max = kv_max // chunk_size

    full = (
        (q_chunk_min == q_chunk_max)[:, None]
        & (kv_chunk_min == kv_chunk_max)[None, :]
        & (q_chunk_min[:, None] == kv_chunk_min[None, :])
        & (q_min[:, None] >= kv_max[None, :])
    )
    empty = (
        (q_max[:, None] < kv_min[None, :])
        | (q_chunk_max[:, None] < kv_chunk_min[None, :])
        | (q_chunk_min[:, None] > kv_chunk_max[None, :])
    )

    grid = np.full(blocked_shape, 1, dtype=np.int32)
    if not q_contig.all():
      grid[~q_contig, :] = -1
    grid[empty] = 0
    grid[full] = 2
    return grid

  return np.full(blocked_shape, -1, dtype=np.int32)


def _generate_shard_metadata(
    block_mask: np.ndarray,
    partial_blocks: np.ndarray,
    is_dkv: bool,
    return_dynamic_grid: bool,
    compute_mask_next: bool = True,
):
  if is_dkv:
    block_mask = block_mask.mT
    partial_blocks = partial_blocks.mT

  width = block_mask.shape[1]

  if return_dynamic_grid:
    active_mask = block_mask > 0
    if is_dkv:
      # If an entire row is masked then that kv output tile won't be visited.
      # We extend the grid to visit these tiles to initialize them.
      active_mask[:, 0] |= ~active_mask.any(axis=1)
    active_indices = np.argwhere(active_mask)
    active_rows = active_indices[:, 0].astype(np.int32)
    active_cols = active_indices[:, 1].astype(np.int32)
    block_mask = block_mask[active_mask]
    grid_size = active_rows.size
  else:
    active_indices = None
    active_rows = active_cols = grid_size = None

  if not compute_mask_next:
    mask_next = None
  else:
    partial_coords = np.argwhere(partial_blocks != -1)
    if partial_coords.size > 0:
      partial_flat_indices = (
          partial_coords[:, 0].astype(np.int64) * width + partial_coords[:, 1]
      )
      if active_indices is not None:
        active_flat_indices = (
            active_indices[:, 0].astype(np.int64) * width
            + active_indices[:, 1]
        )
      else:
        active_flat_indices = np.arange(block_mask.size, dtype=np.int64)
      next_idx = (
          np.searchsorted(
              partial_flat_indices, active_flat_indices, side='left'
          )
          % partial_flat_indices.size
      )
      mask_next = partial_blocks[
          partial_coords[next_idx, 0], partial_coords[next_idx, 1]
      ].astype(np.int32)
    else:
      mask_next = np.full(block_mask.size, -1, dtype=np.int32)

  flat_block_mask = block_mask.flatten()

  return active_rows, active_cols, mask_next, flat_block_mask, grid_size


# When used in a transformer network with multiple layers, the SplashAttention
# kernel is created several times with the same mask. Cache MaskInfo to avoid
# blowing up compile times. Ideally the size of the cache should be determined
# by the client.
@functools.lru_cache(maxsize=12)
def _process_mask(
    mask: Mask,  # [q_seq_len, kv_seq_len]
    block_shape: tuple[int, int],
    is_dkv: bool,
    *,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int = 1,
    kv_seq_shards: int = 1,
    return_dynamic_grid: bool = True,
) -> tuple[MaskInfo, MaskCallable | None]:
  """Transform a dense mask into a sparse representation.

  The number Q sequence shards are needed to create a MaskInfo
  object that is partitionable (with shard_map) along that dimension.
  Args:
    mask: Dense mask to process.
    block_shape: Shape of the Pallas grid block.
    is_dkv: True if we are processing the dKV mask
    downcast_smem_data: If True, downcast the SMEM data of MaskInfo to a data
      type smaller if possible.
    q_seq_shards: Number of Q sequence shards of the mesh in which the kernel is
      launched.

  Returns:
    `MaskInfo`, a sparse representation of the dense mask.
    `MaskCallable`: a callable that, given Q and KV indices, returns
      the value of the mask at those coordinates.

  Raises:
    ValueError: if the input mask is invalid or the block sizes are not
    compatible with the mask sizes.
  """

  if len(mask.shape) != 2:
    raise ValueError(f'Expected a 2-dim mask, instead got: {mask.shape=}')

  q_seq_len, kv_seq_len = mask.shape
  q_block_size, kv_block_size = block_shape
  q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
  kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)

  if q_mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len=}.')
  if kv_mod != 0:
    raise ValueError(f'{kv_block_size=} should divide {kv_seq_len=}.')

  q_seq_len_per_shard, mod = divmod(q_seq_len, q_seq_shards)
  if mod != 0:
    raise ValueError(f'{q_seq_shards=} should divide {q_seq_len=}.')

  q_blocks_per_shard, mod = divmod(q_seq_len_per_shard, q_block_size)
  if mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len_per_shard=}.')

  kv_seq_len_per_shard, mod = divmod(kv_seq_len, kv_seq_shards)
  if mod != 0:
    raise ValueError(f'{kv_seq_shards=} should divide {kv_seq_len=}.')

  kv_blocks_per_shard, mod = divmod(kv_seq_len_per_shard, kv_block_size)
  if mod != 0:
    raise ValueError(f'{kv_block_size=} should divide {kv_seq_len_per_shard=}.')

  # TODO: checking the validity of the masks is slow for large masks.
  # Disable it for now, reevaluate in the future.

  # The mask object either define q_sequence and mask_function or none of
  # them.
  assert hasattr(mask, 'q_sequence') == hasattr(mask, 'mask_function')

  # If the mask object defines a q_sequence and a mask_function, then make use
  # of these in the kernel rather. This is preferable over loading the mask
  # from memory. When using a mask_function, then mask_next and
  # partial_mask_blocks are left undefined and not used in the kernel.
  if hasattr(mask, 'q_sequence') and hasattr(mask, 'mask_function'):
    q_sequence = mask.q_sequence
    mask_function = mask.mask_function
  else:
    q_sequence = mask_function = None

  # Identify the partial mask blocks and the value of the block mask for each
  # block.
  # Partial mask blocks are uniquified. When partitioning, all partial mask
  # blocks are replicated across shards.
  blocked_shape = (q_blocks_count, kv_blocks_count)
  state_grid = _classify_mask_blocks(
      mask, q_blocks_count, kv_blocks_count, q_block_size, kv_block_size
  )
  partial_id_grid = np.full(blocked_shape, -1, dtype=np.int32)

  # Partition the dense mask into blocks and categorize them:
  # 0 = Empty, 1 = Partial (mixed 0s and 1s), 2 = Full (all 1s).
  # Partial blocks are deduplicated and stored in unique_chunks to save memory.
  if mask_function is not None:
    for q_idx, kv_idx in np.argwhere(state_grid == -1):
      q_idx, kv_idx = int(q_idx), int(kv_idx)
      chunk = mask[(
          slice(q_idx * q_block_size, (q_idx + 1) * q_block_size),
          slice(kv_idx * kv_block_size, (kv_idx + 1) * kv_block_size),
      )]
      if chunk.any():
        state_grid[q_idx, kv_idx] = 2 if chunk.all() else 1
      else:
        state_grid[q_idx, kv_idx] = 0
    partial_mask_blocks = None
  else:
    partial_blocks_map = collections.defaultdict(
        lambda: len(partial_blocks_map)
    )
    unique_chunks = []
    for q_idx, kv_idx in np.argwhere((state_grid == -1) | (state_grid == 1)):
      q_idx, kv_idx = int(q_idx), int(kv_idx)
      chunk = mask[(
          slice(q_idx * q_block_size, (q_idx + 1) * q_block_size),
          slice(kv_idx * kv_block_size, (kv_idx + 1) * kv_block_size),
      )]
      if state_grid[q_idx, kv_idx] == -1:
        if not chunk.any():
          state_grid[q_idx, kv_idx] = 0
          continue
        if chunk.all():
          state_grid[q_idx, kv_idx] = 2
          continue
        state_grid[q_idx, kv_idx] = 1
      chunk_id = partial_blocks_map[_HashableNDArray(chunk)]
      partial_id_grid[q_idx, kv_idx] = chunk_id
      if chunk_id == len(unique_chunks):
        unique_chunks.append(chunk)

    if unique_chunks:
      partial_mask_blocks = np.stack(unique_chunks).astype(
          partial_mask_blocks_dtype
      )
      if is_dkv:
        partial_mask_blocks = partial_mask_blocks.mT
    else:
      partial_mask_blocks = None

  full_mask = (state_grid == 2).all()
  if full_mask:
    return MaskInfo(
        mask_next=None,
        active_rows=None,
        active_cols=None,
        block_mask=None,
        num_active_blocks=None,
        partial_mask_blocks=None,
        q_sequence=q_sequence,
    ), None

  need_mask_next = mask_function is None and partial_mask_blocks is not None

  # Work on a fraction of the mask at the time to compute the mask. This is
  # needed to compute the correct data indices, which are relative to the
  # current slice of the mask.
  all_shards_metadata = []
  for q_shard_idx in range(q_seq_shards):
    for kv_shard_idx in range(kv_seq_shards):
      q_slice = slice(
          q_shard_idx * q_blocks_per_shard,
          (q_shard_idx + 1) * q_blocks_per_shard,
      )
      kv_slice = slice(
          kv_shard_idx * kv_blocks_per_shard,
          (kv_shard_idx + 1) * kv_blocks_per_shard,
      )
      metadata = _generate_shard_metadata(
          state_grid[q_slice, kv_slice],
          partial_id_grid[q_slice, kv_slice],
          is_dkv,
          return_dynamic_grid,
          compute_mask_next=need_mask_next,
      )
      all_shards_metadata.append(metadata)

  (
      active_rows_slices,
      active_cols_slices,
      mask_next_slices,
      block_mask_slices,
      num_active_blocks,
  ) = zip(*all_shards_metadata)

  if return_dynamic_grid:
    # Pad each slice to the largest number of active blocks in any shard.
    max_size = max(num_active_blocks)
    pad_slice = lambda arr: np.pad(
        arr, (0, max_size - arr.shape[0]), mode='constant', constant_values=-1
    )
    active_rows_slices = list(map(pad_slice, active_rows_slices))
    active_cols_slices = list(map(pad_slice, active_cols_slices))
    if need_mask_next:
      mask_next_slices = list(map(pad_slice, mask_next_slices))
    block_mask_slices = list(map(pad_slice, block_mask_slices))

    # Concatenate the sequence shards.
    active_rows = np.concatenate(active_rows_slices, axis=0)
    active_cols = np.concatenate(active_cols_slices, axis=0)
    num_active_blocks = np.array(num_active_blocks, dtype=np.int32)

    if downcast_smem_data:
      active_rows = _downcast_to_small_type(active_rows)
      active_cols = _downcast_to_small_type(active_cols)
  else:
    active_rows = active_cols = num_active_blocks = None

  block_mask = np.concatenate(block_mask_slices, axis=0)
  if downcast_smem_data:
    block_mask = _downcast_to_small_type(block_mask)

  if need_mask_next:
    mask_next = np.concatenate(mask_next_slices, axis=0)
    if downcast_smem_data:
      mask_next = _downcast_to_small_type(mask_next)
  else:
    mask_next = None

  assert (mask_function is not None) == (q_sequence is not None)
  # When the mask can be computed inside the kernel with a mask_function,
  # there is no need to load it from memory. So mask_next and
  # partial_mask_blocks are unused.
  return (
      MaskInfo(
          mask_next=mask_next,
          active_rows=active_rows,
          active_cols=active_cols,
          block_mask=block_mask,
          num_active_blocks=num_active_blocks,
          partial_mask_blocks=partial_mask_blocks,
          q_sequence=q_sequence,
      ),
      mask_function,
  )


process_mask = functools.partial(_process_mask, is_dkv=False)


# ------------------------------------------------------------------------------
# tokamax/_src/ops/splash_attention/splash_attention_kernel.py
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
# Forward path only: the backward kernels and the `jax.custom_vjp` wiring are
# left out, so `_splash_attention_custom` is a plain function here.
# ------------------------------------------------------------------------------


partial = functools.partial
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

LOG2E = math.log2(math.e)


def _is_fp8(x) -> bool:
  return x.dtype == jnp.float8_e4m3fn


def _unquantized_dtype(x):
  """bf16 for fp8 inputs; otherwise the input dtype (pre-cl/978108475 behavior)."""
  return jnp.bfloat16 if _is_fp8(x) else x.dtype


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array  # [q_seq_len]
  kv: jax.Array  # [kv_seq_len]


MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
    is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str
) -> str:
  """Returns a unique name for all SplashAttention kernel variants."""
  assert phase in ["dq", "dkv", "fwd"]
  # Saving residuals is supported only for the fwd phase.
  assert not save_residuals or phase == "fwd"
  residuals = "_residuals" if save_residuals else "_no_residuals"
  attention_type = "mqa" if is_mqa else "mha"
  segments = "_segmented" if is_segmented else ""
  return f"splash_{attention_type}_{phase}{segments}{residuals}"


# Splash attention implementation


# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
  SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])


def _check_dropout_args(
    config: "SplashConfig",
    prng_key: jax.Array | None,
) -> jax.Array | None:
  """Validates the dropout arguments and converts the key for Pallas."""

  if not config.dropout_rate:
    return None
  if prng_key is None:
    raise ValueError(
        "A prng_key is required when config.dropout_rate > 0; got None."
    )
  # A key passed as a regular operand fails to lower ("AssertionError:
  # key<pl>"), so it travels through scalar prefetch as a Pallas key instead.
  return pltpu.to_pallas_key(prng_key)


def _above_diag(q_band: int, q_size: int, kv_band: int, kv_size: int) -> bool:
  """Whether a diagonal-block sub-tile lies entirely in kv > q (masked)."""
  return kv_band * kv_size >= (q_band + 1) * q_size


def _diag_grid_pair(grid: int | tuple[int, int]) -> tuple[int, int]:
  """(q splits, kv splits) of a diagonal-skip grid."""
  return (grid, grid) if isinstance(grid, int) else grid


@dataclasses.dataclass(frozen=True, slots=True)
class SplashConfig:
  """Tile sizes parameterizing SplashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.

  Note that changing the layouts only influences the physical layout that the
  kernel will enforce. The logical interface to splash attention always takes
  the head dimension as the minormost one.
  """

  block_q: int
  block_kv: int
  block_kv_compute: int | None = None

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_kv_dkv_compute: int | None = None

  # TODO: Remove these 3 params, they're only kept for backwards compatibility.
  block_q_dq: int | None = None
  block_kv_dq: int | None = None
  use_fused_bwd_kernel: bool = True
  num_stacked_q_heads: int = 1
  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  fwd_cost_estimate: pl.CostEstimate | None = None
  bwd_cost_estimate: pl.CostEstimate | None = None

  residual_checkpoint_name: str | None = None  # whether to checkpoint outputs
  attn_logits_soft_cap: float | None = None
  fuse_reciprocal: bool = True  # whether to compute o / lse inside the kernel
  use_base2_exp: bool = True
  max_logit_const: float | None = None
  interpret: bool = False
  # The fused bwd kernel accumulates dq at every grid step. To safely avoid
  # read/write conflicts we conservatively avoid *any* in-kernel reductions.
  # This parameter allows to override this behavior and specifies the number of
  # reduction steps. Only 1, 3, or all kv steps (None) are supported: 1
  # coalesces consecutive steps for the same q block in fp32 VMEM and reduces
  # non-contiguous revisits into a single pl.Buffered(1) q.shape buffer in HBM,
  # trading unpipelined dq DMAs for 3x lower dq HBM usage than 3.
  dq_reduction_steps: int | None = None
  # An experimental scheduler that sometimes produces better softmax overlap.
  use_experimental_scheduler: bool = False
  # Skip the wasted causal-diagonal work. Bit-exact: the output, dQ, dK and dV
  # are identical to qk_diag_skip=False.
  # Forward: requires block_q == block_kv (block_kv_compute may be smaller). On
  # a diagonal block, compute tile c covers kv [c * bkc, (c + 1) * bkc) of the
  # block (bkc = block_kv_compute). q rows [0, c * bkc) see no kv in it, so
  # their QK, softmax and SV are skipped (an exact no-op for the online
  # softmax). Rows [c * bkc, (c + 1) * bkc) form the causal triangle: the QK
  # matmul is split into a qk_diag_grid sub-grid and sub-tiles fully above the
  # diagonal are filled with mask_value instead of being computed, and the SV
  # matmul runs on the visible kv prefix of each row band. Rows below the
  # triangle are one dense matmul.
  # Backward (dkv): requires square tiles (block_q_dkv == block_kv_dkv ==
  # block_kv_dkv_compute); the QK^T sub-tiles of the qk_diag_grid_dkv sub-grid
  # fully above the diagonal are skipped the same way.
  # PRECONDITION (enforced): pure CausalMask with offset 0, and the block
  # conditions above. Any other config raises; it never silently corrupts.
  qk_diag_skip: bool = False
  # Sub-grid of the diagonal skip, as (q splits, kv splits) of the causal
  # triangle; an int g means (g, g). A finer grid skips more of the triangle
  # but issues smaller matmuls. Every entry must be a power of 2 (1 = no split
  # along that dimension). Used by the forward (also by sv_diag_skip) and, by
  # default, by the backward dkv kernel.
  qk_diag_grid: int | tuple[int, int] = 2
  # Sub-grid of the backward dkv diagonal skip (and of diag_band_skip), in the
  # same format. None (the default) uses qk_diag_grid. It is resolved where it
  # is used (`active_qk_diag_grid_dkv`) rather than in __post_init__, so it
  # keeps following qk_diag_grid through `dataclasses.replace`.
  qk_diag_grid_dkv: int | tuple[int, int] | None = None
  # Skip the wasted causal-diagonal SV matmul. On a partial-mask (diagonal)
  # block, split the SV matmul into the qk_diag_grid sub-grid and skip
  # sub-tiles fully above the diagonal during dot_general. Redundant with
  # qk_diag_skip, which already skips SV in the forward.
  sv_diag_skip: bool = False
  # With qk_diag_skip, run aligned square causal diagonal blocks as row-bands so
  # softmax and matmuls skip strictly-above-diagonal tiles:
  # - Forward (when block_q == block_kv == block_kv_compute): splits into
  #   qk_diag_grid query row-bands, each attending only to its visible kv
  #   prefix.
  # - Backward dkv: splits into qk_diag_grid_dkv kv row-bands, each over only
  #   the queries it attends to. Accumulates dq in VMEM scratch when present
  #   (dq_reduction_steps=1), or per query band in f32 registers otherwise.
  diag_band_skip: bool = False
  # Attention dropout probability, applied to the softmax weights. The mask is
  # generated inside the kernel per (head, q block, kv block) tile from a
  # `prng_key` passed at call time, so it costs no HBM traffic and is never
  # saved for the backward pass. Requires a `prng_key`; 0.0 disables it and
  # leaves the kernel bit-identical to the no-dropout version.
  dropout_rate: float = 0.0
  # Granularity of the grid the dropout mask is drawn on. Keying the mask on
  # absolute canonical coordinates (instead of kernel tiles) allows fwd and
  # bwd to use different block sizes. Every tile size must be a multiple of the
  # canonical size. Recommended to set to the smaller of the fwd/bwd block sizes (coarsest
  # common grid). Set smaller to decouple block sizes that aren't multiples.
  dropout_block_q: int | None = None
  dropout_block_kv: int | None = None
  # [experimental] In the fused dkv kernel, reuse the
  # LOG2E-scaled q for the dK matmul and undo the scale with dk /= LOG2E
  # (exact grad w.r.t. the rounded fp8 q; one fewer q tile live).
  # False = previous behavior.
  dkv_reuse_scaled_q: bool = False
  # [experimental] q arrives already multiplied by LOG2E (the caller folds it
  # into the softmax scale before the fp8 cast, so q is rounded to fp8 once).
  # The kernel skips its own q scaling and returns dq, dk w.r.t. the
  # pre-scaled input (both divided by LOG2E). Subsumes dkv_reuse_scaled_q.
  q_prescaled_log2e: bool = False
  # Set by _make_splash_attention when the static dkv schedule visits every q
  # block in a single contiguous run, allowing dq_reduction_steps=1 to write dq
  # directly from VMEM without HBM read-modify-write aliasing.
  dkv_q_blocks_are_contiguous: bool = False

  @property
  def active_dropout_block_q(self) -> int:
    assert self.dropout_block_q is not None
    return self.dropout_block_q

  @property
  def active_dropout_block_kv(self) -> int:
    assert self.dropout_block_kv is not None
    return self.dropout_block_kv

  def __post_init__(self):
    if not 0.0 <= self.dropout_rate < 1.0:
      raise ValueError(
          f"dropout_rate must be in [0, 1), got {self.dropout_rate}."
      )
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)

    if self.dq_reduction_steps not in (None, 1, 3):
      raise ValueError(
          f"Invalid dq_reduction_steps: {self.dq_reduction_steps}, only 1, 3 or"
          " None are supported."
      )
    if not self.use_fused_bwd_kernel:
      raise ValueError("Only the fused bwd kernel is supported.")
    if self.q_prescaled_log2e and not self.use_base2_exp:
      raise ValueError("q_prescaled_log2e requires use_base2_exp=True.")

    if self.dropout_rate:
      # The backward regenerates the mask rather than reloading it, so the two
      # passes have to agree on which weights were dropped. They are keyed on a
      # canonical grid of absolute coordinates instead of on their own tiles,
      # which makes agreement independent of the tiling; all that is left to
      # check is that each tile covers a whole number of canonical blocks.
      if self.dropout_block_q is None or self.dropout_block_kv is None:
        raise ValueError(
            "dropout_block_q and dropout_block_kv must be set if dropout_rate"
            " > 0."
        )
      if self.dropout_block_q <= 0:
        raise ValueError(
            f"dropout_block_q must be positive, got {self.dropout_block_q}."
        )
      if self.dropout_block_kv <= 0:
        raise ValueError(
            f"dropout_block_kv must be positive, got {self.dropout_block_kv}."
        )

      if self.block_q % self.dropout_block_q != 0:
        raise ValueError(
            f"block_q={self.block_q} must be a multiple of"
            f" dropout_block_q={self.dropout_block_q}."
        )
      if (
          self.block_q_dkv is not None
          and self.block_q_dkv % self.dropout_block_q != 0
      ):
        raise ValueError(
            f"block_q_dkv={self.block_q_dkv} must be a multiple of"
            f" dropout_block_q={self.dropout_block_q}."
        )

      assert self.block_kv_compute is not None
      if self.block_kv_compute % self.dropout_block_kv != 0:
        raise ValueError(
            f"block_kv_compute={self.block_kv_compute} must be a multiple of"
            f" dropout_block_kv={self.dropout_block_kv}."
        )
      if (
          self.block_kv_dkv_compute is not None
          and self.block_kv_dkv_compute % self.dropout_block_kv != 0
      ):
        raise ValueError(
            f"block_kv_dkv_compute={self.block_kv_dkv_compute} must be a"
            f" multiple of dropout_block_kv={self.dropout_block_kv}."
        )

    fwd_grid = _diag_grid_pair(self.qk_diag_grid)
    dkv_grid = _diag_grid_pair(self.active_qk_diag_grid_dkv)
    dkv_grid_name = (
        "qk_diag_grid" if self.qk_diag_grid_dkv is None else "qk_diag_grid_dkv"
    )
    if self.qk_diag_skip or self.sv_diag_skip:
      for name, grid in (
          ("qk_diag_grid", fwd_grid),
          (dkv_grid_name, dkv_grid),
      ):
        # 1 is allowed: no split along that dimension.
        if any(g < 1 or g & (g - 1) for g in grid):
          raise ValueError(
              f"{name} entries must each be a power of 2; got {grid}."
          )
      assert self.block_kv_compute is not None
      if any(self.block_kv_compute % g for g in fwd_grid):
        raise ValueError(
            f"qk_diag_grid {fwd_grid} must divide"
            f" block_kv_compute={self.block_kv_compute}."
        )

    if self.qk_diag_skip:
      # The skip fills mask_value / skips work exactly where kv > q, so it needs
      # the diagonal blocks to be aligned (block_q == block_kv); causality and
      # offset 0 are checked in _make_splash_attention.
      assert self.block_kv_compute is not None
      if self.block_q != self.block_kv:
        raise ValueError(
            "qk_diag_skip requires block_q == block_kv in the forward; got"
            f" {self.block_q}/{self.block_kv}."
        )
      if self.block_kv % self.block_kv_compute:
        raise ValueError(
            f"qk_diag_skip requires block_kv_compute={self.block_kv_compute}"
            f" to divide block_kv={self.block_kv}."
        )
      if self.has_backward_blocks:
        assert self.block_q_dkv is not None
        assert self.block_kv_dkv_compute is not None
        if not (
            self.block_q_dkv == self.block_kv_dkv == self.block_kv_dkv_compute
        ):
          raise ValueError(
              "qk_diag_skip requires square backward blocks "
              "(block_q_dkv == block_kv_dkv == block_kv_dkv_compute); got "
              f"{self.block_q_dkv}/{self.block_kv_dkv}/"
              f"{self.block_kv_dkv_compute}."
          )
        if (
            self.block_q_dkv % dkv_grid[0]
            or self.block_kv_dkv_compute % dkv_grid[1]
        ):
          raise ValueError(
              f"{dkv_grid_name} {dkv_grid} must divide the dkv blocks"
              f" {self.block_q_dkv}/{self.block_kv_dkv_compute}."
          )

    if self.sv_diag_skip:
      # The skip assumes the kv > q region is masked. That holds only for aligned
      # SQUARE blocks (kv-band > q-band <=> fully above the causal line, single
      # compute tile per block) — enforce it or the skip silently corrupts.
      # Causality is checked in _make_splash_attention.
      if not (self.block_q == self.block_kv == self.block_kv_compute):
        raise ValueError(
            "sv_diag_skip requires square forward blocks "
            "(block_q == block_kv == block_kv_compute); got "
            f"{self.block_q}/{self.block_kv}/{self.block_kv_compute}."
        )

  @property
  def active_qk_diag_grid_dkv(self) -> int | tuple[int, int]:
    """The dkv diagonal-skip grid: qk_diag_grid_dkv, else qk_diag_grid."""
    if self.qk_diag_grid_dkv is None:
      return self.qk_diag_grid
    return self.qk_diag_grid_dkv

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_dkv,
        self.block_kv_dkv,
        self.block_kv_dkv_compute,
    )
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls):
    # TODO: Select better parameters based on a heuristic.
    return SplashConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
        fuse_reciprocal=True,
    )


to_i32 = lambda x: x.astype(jnp.int32)


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float | None,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
    has_partial_mask: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if has_partial_mask:
    if mask_ref is not None:
      mask = mask_ref[:, k_slice] if k_in_lanes else mask_ref[k_slice, :]
      masks.append(mask.astype(jnp.bool_))
    elif mask_function is not None:
      # Compute the mask using the given q_sequence indices.
      # KV indices are computed on the fly. This works because we only support Q
      # sequence sharding. If we wanted to compute Q indices too, then we would
      # need to keep into account the current shard along Q sequence.

      if k_in_lanes:
        assert q_sequence_ref.shape == (bq, NUM_LANES)  # pyrefly: ignore[missing-attribute]

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (bq, k_slice.size), 1
        )

        repeats, rem = divmod(k_slice.size, NUM_LANES)
        assert rem == 0
        q_sequence = jnp.tile(
            q_sequence_ref[...], (1, repeats)  # pyrefly: ignore[unsupported-operation]
        )  # [bq, k_slice.size]
      else:
        assert q_sequence_ref.shape == (NUM_SUBLANES, bq)  # pyrefly: ignore[missing-attribute]

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (k_slice.size, bq), 0
        )
        q_sequence = q_sequence_ref[:1, :]  # [1, bq]  # pyrefly: ignore[unsupported-operation]
        q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

      assert q_sequence.shape == k_sequence.shape
      computed_mask = mask_function(q_sequence, k_sequence)
      if computed_mask.dtype != jnp.dtype(jnp.bool_):
        raise ValueError(
            "Mask function must return a boolean-valued array, but got:"
            f" {computed_mask.dtype}"
        )
      masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = jnp.tile(q_segment_ids_ref[:], (1, repeats))  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
      kv_ids = jnp.tile(
          kv_segment_ids_ref[k_slice, :], (1, repeats)
      )  # [k_slice, bq]
      q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits):
    if attn_logits_soft_cap is not None:
      logits = jnp.tanh(qk / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    masks = [m.astype(jnp.bool_) for m in masks]
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    if mask.ndim == 2 and qk.ndim == 3:
      mask = jnp.expand_dims(mask, axis=0)

    qk = jnp.where(mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def flash_attention_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    prng_key_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    mask_ref,
    q_sequence_ref,
    max_logit_value_ref,
    # Outputs
    o_ref,
    logsumexp_ref,
    l_linear_ref,
    max_logits_ref,
    # Scratch
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    mask_value: float,
    q_steps: int,
    kv_steps: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    num_stacked_q_heads: int,
    mask_function: MaskFunctionType | None,
    fuse_reciprocal: bool,  # config.fuse_reciprocal or not save_residuals
    config: SplashConfig,
):
  del mask_next_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  dropout_rate = config.dropout_rate
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  # If the head_dim_v is not a multiple of the number of lanes, it will be
  # padded to that multiple with zeros.
  head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)

  grid_idx = pl.program_id(1)
  h = pl.program_id(0)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) != 0
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    j = active_cols_ref[grid_idx].astype(jnp.int32)
    # Dropout needs the q block index too; it is only tracked on the dynamic
    # grid, where `grid_idx` is a position in the sparse (i, j) work list.
    i = active_rows_ref[grid_idx].astype(jnp.int32) if dropout_rate else 0
  else:
    should_not_mask = False
    should_run = True
    j = grid_idx % kv_steps
    should_initialize = j == 0
    should_write = j == kv_steps - 1
    i = grid_idx // kv_steps

  max_logit_estimate = config.max_logit_const  # potentially None
  if max_logit_value_ref is not None:  # already ensures max_logit_const is None
    assert num_stacked_q_heads == 1
    max_logit_estimate = max_logit_value_ref[0, h]

  if config.use_base2_exp and max_logit_estimate is not None:
    max_logit_estimate *= LOG2E

  @pl.when(should_initialize)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    sink = None
    if sinks_ref is not None:
      sink = sinks_ref[0, h].astype(m_scratch_ref.dtype)
      if config.use_base2_exp:
        sink *= LOG2E

    if sinks_ref is None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is None and max_logit_estimate is not None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is not None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, sink)  # pyrefly: ignore[bad-argument-type]
      l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
    else:  # sinks_ref is not None and max_logit_estimate is not None
      exp = jnp.exp2 if config.use_base2_exp else jnp.exp
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)  # pyrefly: ignore[bad-argument-type]
      l_scratch_ref[...] = exp(
          sink - jnp.full_like(l_scratch_ref, max_logit_estimate)  # pyrefly: ignore[bad-argument-type, unsupported-operation]
      )

  def qk_matmul(q_flat, k, qk_dims, *, diag):
    """Returns q_flat @ k.T in f32.

    On a diagonal tile (see `body`), rows [0, bkv_compute) of `q_flat` are the
    causal triangle: only the sub-tiles of the qk_diag_grid sub-grid on or
    below the diagonal are computed; the rest are filled with mask_value, which
    the mask would write there anyway. Rows below the triangle see the whole
    tile.

    Args:
      q_flat: The (scaled) queries, (rows, head_dim_qk).
      k: The keys of the compute tile, in `config.k_layout`.
      qk_dims: Dimension numbers of the q @ k.T dot.
      diag: Whether this is a diagonal tile of `body`.
    """
    if not diag:
      return lax.dot_general(q_flat, k, qk_dims, preferred_element_type=float32)
    gq, gk = _diag_grid_pair(config.qk_diag_grid)
    sq, sk = bkv_compute // gq, bkv_compute // gk
    if config.k_layout == HEAD_DIM_MINOR:
      k_parts = [k[t * sk : (t + 1) * sk, :] for t in range(gk)]
    else:
      k_parts = [k[:, t * sk : (t + 1) * sk] for t in range(gk)]
    qk_rows = []
    for qi in range(gq):
      q_part = q_flat[qi * sq : (qi + 1) * sq, :]
      cols = []
      for kj in range(gk):
        if _above_diag(qi, sq, kj, sk):
          cols.append(jnp.full((sq, sk), mask_value, dtype=float32))
        else:
          cols.append(
              lax.dot_general(
                  q_part, k_parts[kj], qk_dims, preferred_element_type=float32
              )
          )
      qk_rows.append(jnp.concatenate(cols, axis=1))
    if q_flat.shape[0] > bkv_compute:  # Rows below the triangle.
      qk_rows.append(
          lax.dot_general(
              q_flat[bkv_compute:, :],
              k,
              qk_dims,
              preferred_element_type=float32,
          )
      )
    return jnp.concatenate(qk_rows, axis=0)

  def sv_matmul(s_flat, v, sv_dims, *, has_partial_mask, diag):
    """Returns s_flat @ v in f32, skipping the part of s that is exactly 0.

    Args:
      s_flat: The softmax numerators, (rows, bkv_compute).
      v: The values of the compute tile, in `config.v_layout`.
      sv_dims: Dimension numbers of the s @ v dot.
      has_partial_mask: Whether the block is partially masked.
      diag: Whether this is a diagonal tile of `body`.
    """
    gq, gk = _diag_grid_pair(config.qk_diag_grid)
    if diag:
      # Each row band of the triangle only multiplies the kv prefix it can see
      # (the rest of its s is exactly 0). One dot per band over the prefix, not
      # a sum of per-sub-tile dots, keeps the f32 accumulation identical to the
      # dense matmul. Bands are at least 256 rows: on TPU7x an M=128 dot rounds
      # differently from the dense one (o off by 1 bf16 ulp).
      sv_bands = 1  # Largest power of 2 <= gq that keeps bands >= 256 rows.
      while sv_bands * 2 <= gq and bkv_compute // (sv_bands * 2) >= 256:
        sv_bands *= 2
      band = bkv_compute // sv_bands
      o_rows = []
      for b in range(sv_bands):
        kv_end = (b + 1) * band
        if config.v_layout == HEAD_DIM_MINOR:
          v_pre = v[:kv_end, :]
        else:
          v_pre = v[:, :kv_end]
        o_rows.append(
            lax.dot_general(
                s_flat[b * band : (b + 1) * band, :kv_end],
                v_pre,
                sv_dims,
                preferred_element_type=float32,
            )
        )
      if s_flat.shape[0] > bkv_compute:  # Rows below the triangle.
        o_rows.append(
            lax.dot_general(
                s_flat[bkv_compute:, :],
                v,
                sv_dims,
                preferred_element_type=float32,
            )
        )
      return jnp.concatenate(o_rows, axis=0)
    if (
        config.sv_diag_skip
        and has_partial_mask
        and num_stacked_q_heads == 1
        and config.v_layout == HEAD_DIM_MINOR
        and bq % gq == 0
        and bkv_compute % gk == 0
    ):
      # sv_diag_skip: skip the sub-tiles of the qk_diag_grid sub-grid that lie
      # entirely above the diagonal of the (square) diagonal block.
      sq = bq // gq
      sk = bkv_compute // gk
      v_parts = [v[t * sk : (t + 1) * sk, :] for t in range(gk)]
      o_rows = []
      for qi in range(gq):
        o_row = None
        for kj in range(gk):
          if _above_diag(qi, sq, kj, sk):
            continue
          part = lax.dot_general(
              s_flat[qi * sq : (qi + 1) * sq, kj * sk : (kj + 1) * sk],
              v_parts[kj],
              sv_dims,
              preferred_element_type=float32,
          )
          o_row = part if o_row is None else o_row + part
        o_rows.append(o_row)
      return jnp.concatenate(o_rows, axis=0)
    return lax.dot_general(s_flat, v, sv_dims, preferred_element_type=float32)

  def body(
      kv_compute_index,
      _,
      has_partial_mask=False,
      diag=False,
      band: tuple[int, int, int] | None = None,
  ):
    """Runs the online softmax update for one compute tile of the kv block.

    Args:
      kv_compute_index: Index of the compute tile within the kv block; a
        static int when `diag` or `band` is set.
      _: Unused loop carry.
      has_partial_mask: Whether the block is partially masked.
      diag: `qk_diag_skip` on an aligned causal diagonal block (bq == bkv,
        offset 0). The tile covers kv [r0, r0 + bkv_compute) of the block with
        r0 = kv_compute_index * bkv_compute, so q rows [0, r0) see none of it:
        their m and l stay unchanged and their o gets nothing added, exactly
        what the online softmax would do with an all-masked row (alpha = 1,
        s = 0). Only rows [r0, bq) are processed, and `qk_matmul` and
        `sv_matmul` skip the sub-tiles above the diagonal.
      band: `(q0, q1, kv_end)` for `diag_band_skip` on an aligned square causal
        diagonal block (`bq == bkv == bkv_compute`), running q rows `[q0, q1)`
        against only the visible kv prefix `[0, kv_end)`.
    """
    if band is not None:
      r0, r1, nk = band
      k0 = 0
    else:
      r0 = kv_compute_index * bkv_compute if diag else 0
      r1 = bq
      k0 = kv_compute_index * bkv_compute
      nk = bkv_compute
    nq = r1 - r0
    sliced_q = diag or band is not None
    # Index of the processed rows [r0, r1) in the (heads, bq, ...) refs.
    rows = (slice(None), slice(r0, r1), slice(None)) if sliced_q else ...
    row_view = (
        lambda ref: ref.at[r0:r1] if sliced_q and ref is not None else ref
    )
    slice_k = pl.ds(k0, nk)
    m_prev, l_prev = m_scratch_ref[rows], l_scratch_ref[rows]
    assert m_prev.shape == (num_stacked_q_heads, nq, NUM_LANES)
    assert l_prev.shape == (num_stacked_q_heads, nq, NUM_LANES)

    if config.q_layout == HEAD_DIM_MINOR:
      q = q_ref[rows]
    else:
      q = (
          q_ref[(slice(None), slice(None), slice(r0, r1)) if sliced_q else ...]
          .mT
      )
    if config.use_base2_exp and not config.q_prescaled_log2e:
      if q.dtype == jnp.float8_e4m3fn:
        q = (q.astype(jnp.bfloat16) * LOG2E).astype(jnp.float8_e4m3fn)
      else:
        q *= LOG2E

    head_dim_qk = q.shape[-1]
    # Collapse the head and sequence dimensions for a larger matmul.
    q_flat = q.reshape((num_stacked_q_heads * nq, head_dim_qk))

    if config.k_layout == HEAD_DIM_MINOR:
      k = k_ref[slice_k, :]
      qk_dims = NT_DIM_NUMBERS
    else:
      k = k_ref[:, slice_k]
      qk_dims = NN_DIM_NUMBERS

    qk_flat = qk_matmul(q_flat, k, qk_dims, diag=diag)
    qk = qk_flat.reshape((num_stacked_q_heads, nq, nk))

    qk = _apply_mask_and_soft_cap(
        qk,
        mask_value,
        row_view(mask_ref),
        row_view(q_sequence_ref),
        row_view(q_segment_ids_ref),
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=j * bkv + k0,
        bq=nq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )

    if max_logit_estimate is None:
      m_curr = qk.max(axis=-1)[..., None]  # pyrefly: ignore[missing-attribute]
      assert m_curr.shape == (num_stacked_q_heads, nq, 1)
      m_next = jnp.maximum(m_prev, m_curr)
      assert m_next.shape == (num_stacked_q_heads, nq, NUM_LANES)
    else:
      m_next = None

    bkv_repeats, rem = divmod(nk, NUM_LANES)
    if rem != 0:
      raise NotImplementedError(f"{nk=} should be a multiple of {NUM_LANES}")

    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    if max_logit_estimate is None:
      s_curr = exp(qk - jnp.tile(m_next, (1, 1, bkv_repeats)))  # pyrefly: ignore[bad-argument-type, unsupported-operation]
    else:
      s_curr = exp(qk - max_logit_estimate)  # pyrefly: ignore[unsupported-operation]
    assert s_curr.shape == (num_stacked_q_heads, nq, nk)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0, 1))
    assert l_curr.shape == (num_stacked_q_heads, nq, NUM_LANES)

    if max_logit_estimate is None:
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      m_scratch_ref[rows], l_scratch_ref[rows] = m_next, l_next
    else:
      alpha = None
      l_scratch_ref[rows] = l_curr + l_prev

    # Dropout is applied *after* the running softmax denominator has been
    # accumulated from the undropped weights, so it does not renormalize over
    # the survivors: only the numerator (the s @ v product) sees the mask.
    if dropout_rate:
      global_kv_block_idx = j * (bkv // bkv_compute) + kv_compute_index
      canonical_q = config.active_dropout_block_q
      canonical_kv = config.active_dropout_block_kv
      n_q_blocks = q_steps * (bq // canonical_q)
      n_kv_blocks = kv_steps * (bkv // canonical_kv)
      # One mask per stacked head, assembled into the (heads, bq, bkv) tile:
      # writing them in with `.at[head].set` would lower to a scatter, which
      # Mosaic does not implement. Built with an explicit loop rather than a
      # comprehension: `i` is also a comprehension target above, and an inlined
      # comprehension (PEP 709) that reads it would bind the shadowed local.
      head_masks = []
      for head in range(num_stacked_q_heads):
        head_mask = _dropout_mask_tile(
            prng_key_ref,
            head_idx=h * num_stacked_q_heads + head,
            q_block_idx=i,
            kv_block_idx=global_kv_block_idx,
            q_block_size=bq,
            kv_block_size=bkv_compute,
            canonical_q=canonical_q,
            canonical_kv=canonical_kv,
            dropout_rate=dropout_rate,
            n_q_blocks=n_q_blocks,
            n_kv_blocks=n_kv_blocks,
        )
        head_masks.append(head_mask[r0:r1, :nk] if sliced_q else head_mask)
      dropout_scale = jnp.stack(head_masks, axis=0)
      s_curr = s_curr * dropout_scale

    s_curr_flat = s_curr.reshape((num_stacked_q_heads * nq, nk))

    if config.v_layout == HEAD_DIM_MINOR:
      v = v_ref[slice_k, :]
      sv_dims = NN_DIM_NUMBERS
    else:
      v = v_ref[:, slice_k]
      sv_dims = NT_DIM_NUMBERS

    o_curr_flat = sv_matmul(
        s_curr_flat,
        v,
        sv_dims,
        has_partial_mask=has_partial_mask and band is None,
        diag=diag,
    )
    o_curr = o_curr_flat.reshape((num_stacked_q_heads, nq, head_dim_v))

    if max_logit_estimate is None:
      alpha_o = jnp.tile(alpha, (1, 1, head_dim_v_repeats))  # pyrefly: ignore[bad-argument-type]
      alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
      o_scratch_ref[rows] = alpha_o * o_scratch_ref[rows] + o_curr
    else:
      o_scratch_ref[rows] = o_scratch_ref[rows] + o_curr

  assert bkv % bkv_compute == 0
  num_iters = (
      k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
  )
  # The diagonal path may only run on diagonal blocks, so the partial blocks
  # must be known: that needs the block mask (without it every block is
  # "partial").
  use_diag = (
      config.qk_diag_skip
      and block_mask_ref is not None
      and num_stacked_q_heads == 1
      and bq == bkv
  )
  gq, gk = _diag_grid_pair(config.qk_diag_grid)
  use_diag_band = (
      use_diag
      and config.diag_band_skip
      and not dropout_rate
      and bq == bkv_compute
      and gq == gk
      and bq % gq == 0
      and (bq // gq) % NUM_LANES == 0
  )

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_and(jnp.logical_not(should_not_mask), should_run))
  def _():
    if use_diag_band:
      sq = bq // gq
      for qi in range(gq):
        body(
            0,
            None,
            has_partial_mask=True,
            band=(qi * sq, (qi + 1) * sq, (qi + 1) * sq),
        )
    elif use_diag:
      for c in range(num_iters):  # Static c: the diagonal path slices by it.
        body(c, None, has_partial_mask=True, diag=True)
    else:
      lax.fori_loop(
          0, num_iters, partial(body, has_partial_mask=True), None, unroll=True
      )

  @pl.when(should_write)
  def end():
    l = l_scratch_ref[...]
    m = m_scratch_ref[...]
    if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
      l_inv = jnp.tile(
          jnp.where(l == 0.0, 0.0, 1.0 / l), (1, 1, head_dim_v_repeats)
      )
      l_inv = l_inv[..., : o_scratch_ref.shape[-1]]
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    else:
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
    if logsumexp_ref is not None:
      assert logsumexp_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = m + log(l)
      logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)
    if l_linear_ref is not None:
      assert l_linear_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      l_linear_ref[...] = l.astype(l_linear_ref.dtype)
    if max_logits_ref is not None:
      assert max_logits_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      max_logits_ref[...] = m.astype(max_logits_ref.dtype)


def _div(dividend: int, divisor: int):
  if divisor == 1:
    return dividend

  return lax.div(dividend, divisor)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct | None) -> int:
  if x is None:
    return 0

  if jnp.issubdtype(x.dtype, jnp.floating):
    info = jnp.finfo
  elif jnp.issubdtype(x.dtype, jnp.integer):
    info = jnp.iinfo
  else:
    raise ValueError(f"Unsupported dtype: {x.dtype}")
  return math.ceil(math.prod(x.shape) * info(x.dtype).bits / 8)


def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
    prng_key: jax.Array | None = None,
) -> SplashCustomReturnType:
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = config.block_q, config.block_kv
  bkv_compute = config.block_kv_compute
  fuse_reciprocal = config.fuse_reciprocal or not save_residuals
  num_stacked_q_heads = config.num_stacked_q_heads
  prng_key = _check_dropout_args(config, prng_key)

  if num_stacked_q_heads > 1 and (
      sinks is not None or max_logit_value is not None
  ):
    raise ValueError(
        "Stacked heads are not supported with sinks or max_logit_value."
    )

  if is_mqa:
    expected_kv_rank = 2
    num_kv_heads = 1
  else:
    expected_kv_rank = 3
    num_kv_heads = k.shape[0]

  if len(k.shape) != expected_kv_rank:
    raise ValueError(
        f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a"
        f" {len(k.shape)}-dim one."
    )

  if k.shape[-1] != head_dim_qk:
    raise ValueError(
        f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:"
        f" {k.shape[-1]}."
    )

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if num_q_heads % num_stacked_q_heads != 0:
    raise ValueError(
        f"{num_q_heads=} must be a multiple of {num_stacked_q_heads=}"
    )

  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if q_heads_per_kv_head % num_stacked_q_heads != 0:
    raise ValueError(
        f"{q_heads_per_kv_head=} must be a multiple of {num_stacked_q_heads=}"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
        "leading dimensions."
    )

  if bkv % bkv_compute:  # pyrefly: ignore[unsupported-operation]
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:  # pyrefly: ignore[unsupported-operation]
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  kv_seq_len = k.shape[-2]
  kv_steps = kv_seq_len // bkv

  if segment_ids is not None:
    assert isinstance(segment_ids.q, jax.Array)  # for pytype
    assert isinstance(segment_ids.kv, jax.Array)  # for pytype
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError(
          "Invalid shape for q segment_ids: "
          f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}"
      )
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError(
          "Invalid shape for kv segment_ids: "
          f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
      )
    mask_info, bounds_start, bounds_end = _refine_mask_info_with_segments(
        mask_info,
        segment_ids,
        q_blocks=q_seq_len // bq,
        kv_blocks=kv_steps,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
    )
  else:
    bounds_start, bounds_end = find_bounds(mask_info.active_rows)  # pyrefly: ignore[bad-argument-type]
  dynamic_grid = mask_info.active_rows is not None

  if config.max_logit_const is not None and max_logit_value is not None:
    raise ValueError(
        f"Only one of {config.max_logit_const=} and"
        f" {max_logit_value=} can be set."
    )
  if max_logit_value is not None:
    if max_logit_value.shape not in ((), (1,), (num_q_heads,)):
      raise ValueError(
          "max_logit_value should be a 0,1-dim jax.Array of shape (), (1,) or"
          f" ({num_q_heads=},) but got {jax.typeof(max_logit_value)}"
      )
    max_logit_value = jnp.broadcast_to(
        jnp.atleast_1d(max_logit_value), (num_q_heads,)
    )

  q_layout = config.q_layout
  k_layout = config.k_layout
  v_layout = config.v_layout

  def unravel(f):
    def index_map(h_block, grid_idx, rows_ref, cols_ref, *_):
      if dynamic_grid:
        i = to_i32(rows_ref[grid_idx])
        j = to_i32(cols_ref[grid_idx])
      else:
        i = grid_idx // kv_steps
        j = grid_idx % kv_steps
      return f(h_block, i, j)

    return index_map

  def create_kv_index_map(layout):
    def index_map(h_block, i, j):
      del i  # Unused.
      first_h_in_block = h_block * num_stacked_q_heads
      prefix = () if is_mqa else (_div(first_h_in_block, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  q_index_map = unravel(
      lambda h_block, i, j: from_head_minor((h_block, i, 0), q_layout)
  )
  out_index_map = unravel(lambda h_block, i, j: (h_block, i, 0))
  k_index_map = unravel(create_kv_index_map(k_layout))
  v_index_map = unravel(create_kv_index_map(v_layout))

  def mask_index_map(
      h_block, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_
  ):
    del h_block, rows_ref, cols_ref  # Unused.
    next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
    return next_m, 0, 0

  q_segment_ids_index_map = unravel(lambda h_block, i, j: (i, 0))
  kv_segment_ids_index_map = unravel(lambda h_block, i, j: (0, j))

  # Convert the logical shape from head-minor to sequence-minor.
  in_specs = [
      pl.BlockSpec(
          from_head_minor((num_stacked_q_heads, bq, head_dim_qk), q_layout),
          q_index_map,
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
              k_layout,
          ),
          k_index_map,
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout
          ),
          v_index_map,
      ),
  ]
  if segment_ids is not None:
    in_specs += [
        pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
        pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
    ]
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)  # pyrefly: ignore[bad-argument-type]
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)  # pyrefly: ignore[bad-argument-type]
    )
  else:
    in_specs += [None, None]
    q_segment_ids = kv_segment_ids = None

  if sinks is not None:
    assert sinks.shape == (num_q_heads,), f"{sinks.shape=} != {num_q_heads=}"
    # align sinks to sublanes to allow vmap and shard_map over the kernel
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda h, i, j, *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
    sinks = jnp.broadcast_to(
        sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads)
    )
  else:
    in_specs += [None]

  if mask_info.partial_mask_blocks is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
  else:
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  if max_logit_value is not None:
    # reshape to allow sublane selection for vmap-ping and shard_map-ping
    max_logit_value = jnp.broadcast_to(
        max_logit_value.astype(jnp.float32)[None, :],
        (NUM_SUBLANES, num_q_heads),
    )
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
  else:
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  out_shapes = [
      jax.ShapeDtypeStruct(
          (num_q_heads, q_seq_len, head_dim_v), _unquantized_dtype(q)
      ),
  ]
  out_specs = [
      pl.BlockSpec((num_stacked_q_heads, bq, head_dim_v), out_index_map),
  ]
  if save_residuals:
    logsumexp_index_map = unravel(lambda h_block, i, j, *_: (h_block, i, 0))

    out_shapes += [
        # logsumexp
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
        if fuse_reciprocal
        else None,
        # l_linear
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
        if not fuse_reciprocal
        else None,
        # max_logits
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),
    ]
    out_specs += [
        pl.BlockSpec((num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map)
        if fuse_reciprocal
        else None,
        pl.BlockSpec((num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map)
        if not fuse_reciprocal
        else None,
        pl.BlockSpec((num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map),
    ]
  else:
    out_shapes += [None, None, None]
    out_specs += [None, None, None]

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      is_segmented=segment_ids is not None,
      phase="fwd",
  )
  metadata = {"xprof_metadata": json.dumps(dataclasses.asdict(config))}

  def _fwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      partial_mask_blocks: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    matmul_flops = (
        2 * q_seq_len * kv_seq_len * head_dim_qk
        + 2 * q_seq_len * kv_seq_len * head_dim_v
    )

    # This is an upper bound because `mask_sparsity` is actually the mean
    # sparsity of the non-fully masked **blocks**.
    total_flops = num_q_heads * matmul_flops * mask_sparsity

    # Count expensive exp() calls
    transcendentals = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity

    inputs_ = [q, k, v, q_segment_ids, kv_segment_ids, partial_mask_blocks]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))
    return pl.CostEstimate(
        flops=int(total_flops),
        transcendentals=int(transcendentals),
        bytes_accessed=int(input_bytes + output_bytes),
    )

  vmem_inputs = [
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      mask_info.partial_mask_blocks,
  ]
  cost_estimate = config.fwd_cost_estimate or _fwd_cost_estimate(
      *vmem_inputs, out_shapes, fwd_mask_sparsity  # pyrefly: ignore[bad-argument-count, bad-argument-type]
  )

  grid_size_h = num_q_heads // num_stacked_q_heads
  if dynamic_grid:
    num_active_blocks = mask_info.num_active_blocks[0]  # pyrefly: ignore[unsupported-operation]
    grid = (grid_size_h, num_active_blocks)
    is_empty_attention_block = num_active_blocks == 0
  else:
    grid = (grid_size_h, kv_steps * (q_seq_len // bq))
    is_empty_attention_block = False

  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
        partial(
            flash_attention_kernel,
            mask_value=mask_value,
            q_steps=q_seq_len // bq,
            kv_steps=kv_steps,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,  # pyrefly: ignore[bad-argument-type]
            head_dim_v=head_dim_v,
            num_stacked_q_heads=num_stacked_q_heads,
            # note: fuse_reciprocal can only be False if save_residuals is True
            # fuse_reciprocal = (config.fuse_reciprocal or not save_residuals)
            fuse_reciprocal=fuse_reciprocal,
            config=config,
            mask_function=mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=7,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=[
                pltpu.VMEM(
                    (num_stacked_q_heads, bq, NUM_LANES), jnp.float32
                ),  # m_scratch
                pltpu.VMEM(
                    (num_stacked_q_heads, bq, NUM_LANES), jnp.float32
                ),  # l_scratch
                pltpu.VMEM(
                    (num_stacked_q_heads, bq, head_dim_v), jnp.float32
                ),  # o_scratch
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            flags={
                "XLA_TPU_FORCE_LP_LLO_SCHEDULER": (
                    config.use_experimental_scheduler
                )
            },
        ),
        out_shape=out_shapes,
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(
        mask_info.active_rows,
        mask_info.active_cols,
        mask_info.mask_next,
        bounds_start,
        bounds_end,
        mask_info.block_mask,
        prng_key,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
        q_segment_ids,
        kv_segment_ids,
        sinks,
        mask_info.partial_mask_blocks,
        q_sequence,
        max_logit_value,
    )
  out, logsumexp, l_linear, max_logits = all_out

  # If there is no compute to do within an attention block, then we want to
  # initialize the output and residuals to default values. Otherwise, we will
  # read uninitialized memory. This is a common case in ring attention.
  def init_if_empty(x: jax.Array, value: float) -> jax.Array:
    if not dynamic_grid:
      return x

    return jnp.where(is_empty_attention_block, value, x)

  out = init_if_empty(out, 0.0)

  if save_residuals:
    assert max_logits is not None
    max_logits = init_if_empty(max_logits[..., 0], mask_value)

    if fuse_reciprocal:
      assert logsumexp is not None
      logsumexp = init_if_empty(logsumexp[..., 0], mask_value)
    else:
      assert l_linear is not None
      log = jnp.log2 if config.use_base2_exp else jnp.log

      l = l_linear[..., 0]
      logsumexp = max_logits + log(l)
      out = (out / l[..., None]).astype(out.dtype)
  else:
    # If we're not saving residuals, then we can't fuse the reciprocal
    # out of the kernel.
    assert fuse_reciprocal

  if config.residual_checkpoint_name is not None:
    out = ad_checkpoint.checkpoint_name(
        out, name=config.residual_checkpoint_name
    )
    if logsumexp is not None:
      logsumexp = ad_checkpoint.checkpoint_name(
          logsumexp, name=config.residual_checkpoint_name
      )
  if save_residuals:
    stats = {"logsumexp": logsumexp, "max_logits": max_logits}
    stats = jax.tree.map(jax.lax.stop_gradient, stats)
    return out, stats
  return out


def _splash_attention_custom(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
    prng_key: jax.Array | None = None,
) -> SplashCustomReturnType:
  # The forward function does not use the dq and dkv MaskInfos, it just forwards
  # them to the backward function as residuals. This is a way to communicate
  # arbitrary Arrays to the backward function. Since the three MaskInfos are
  # constants there is no overhead in passing them to the backward function as
  # residuals. When sharding computation MaskInfos are partitioned so both the
  # forward and the backward kernels need to work on the relevant slice. If we
  # recomputed the backward MaskInfos in the backward function from the numpy
  # mask then we would not work with the MaskInfo slice relevant to the current
  # device.
  del dkv_mask_info

  ret = _splash_attention_forward(
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=save_residuals,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
      prng_key=prng_key,
  )
  if save_residuals:
    out, stats = ret
    if config.use_base2_exp:  # for user, output values in natural base
      stats["logsumexp"] = stats["logsumexp"] / LOG2E
      stats["max_logits"] = stats["max_logits"] / LOG2E
    return out, stats  # pyrefly: ignore[bad-return]
  else:
    return ret


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "save_residuals",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
    ],
)
def _splash_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig | None,
    save_residuals: bool,
    mask_value: float,
    max_logit_value: jax.Array | None = None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    prng_key: jax.Array | None = None,
) -> SplashCustomReturnType:
  return _splash_attention_custom(
      fwd_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      config=config,
      max_logit_value=max_logit_value,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
      prng_key=prng_key,
  )


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:

  def __init__(
      self,
      fwd_mask_info: MaskInfo,
      dkv_mask_info: MaskInfo | None,
      **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dkv_mask_info = dkv_mask_info

  def __call__(self, *args, **kwargs) -> SplashCustomReturnType:
    return _splash_attention(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(self.kwargs, **kwargs),
    )

  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    if self.fwd_mask_info.block_mask is not None:
      block_mask_shape = self.fwd_mask_info.block_mask.shape
      try:
        sharding.shard_shape(block_mask_shape)
      except ValueError as exc:
        raise ValueError(
            "The sharding must divide the mask blocks evenly between devices"
        ) from exc

    if len(sharding.spec) != 1:
      raise ValueError("Only q sequence sharding is supported.")

    _resolve_spec = lambda x: sharding.spec if x is not None else None
    mask_info_specs = MaskInfo(
        mask_next=_resolve_spec(self.fwd_mask_info.mask_next),  # pyrefly: ignore[bad-argument-type]
        active_rows=_resolve_spec(self.fwd_mask_info.active_rows),  # pyrefly: ignore[bad-argument-type]
        active_cols=_resolve_spec(self.fwd_mask_info.active_cols),  # pyrefly: ignore[bad-argument-type]
        num_active_blocks=_resolve_spec(self.fwd_mask_info.num_active_blocks),  # pyrefly: ignore[bad-argument-type]
        block_mask=_resolve_spec(self.fwd_mask_info.block_mask),  # pyrefly: ignore[bad-argument-type]
        partial_mask_blocks=jax.sharding.PartitionSpec()  # replicated  # pyrefly: ignore[bad-argument-type]
        if self.fwd_mask_info.partial_mask_blocks is not None
        else None,
        q_sequence=_resolve_spec(self.fwd_mask_info.q_sequence),  # pyrefly: ignore[bad-argument-type]
    )
    return SplashAttentionKernel(
        mask_info_specs,
        mask_info_specs if self.dkv_mask_info is not None else None,
        **self.kwargs,
    )

  def tree_flatten(self):
    return ((self.fwd_mask_info, self.dkv_mask_info), self.kwargs)

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dkv_mask_info = values
    # NamedTuples are not preserved during pytree serialization.
    dkv_mask_info = (
        MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
    )
    return SplashAttentionKernel(
        MaskInfo(*fwd_mask_info), dkv_mask_info, **kwargs
    )


def _make_splash_attention(
    mask: np.ndarray | Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int,
):
  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if isinstance(mask, np.ndarray):
    mask = NumpyMask(mask)

  if config is None:
    config = SplashConfig.get_default()

  if (config.qk_diag_skip or config.sv_diag_skip) and not (
      isinstance(mask, CausalMask) and mask.offset == 0
  ):
    # The skip assumes kv > q is ALWAYS masked and kv <= q never is — a property
    # of the pure causal mask with offset 0. Any mask that admits a valid kv > q
    # entry (bidirectional, local/sliding window, positive offset, custom) would
    # be silently corrupted, so fail loud. (Block-shape preconditions are
    # enforced in SplashConfig.__post_init__.)
    param_name = "sv_diag_skip" if config.sv_diag_skip else "qk_diag_skip"
    # The mask classes define no useful repr, so describe the mask by hand.
    mask_desc = type(mask).__name__
    if isinstance(mask, CausalMask):
      mask_desc += f" with offset {mask.offset}"
    raise ValueError(
        f"{param_name}=True requires a pure CausalMask with offset 0 (the skip"
        f" assumes the kv > q region is masked); got {mask_desc}. Disable"
        f" {param_name} for non-causal masks."
    )

  process_fn = partial(
      process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
  )

  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
  )
  fwd_mask_sparsity = float(np.mean(fwd_mask_info.block_mask != 0))
  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dkv_mask_info = None
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps in (1, 3),
    )
    assert bq_dkv is not None
    q_blocks_per_shard = mask.shape[0] // q_seq_shards // bq_dkv
    config = dataclasses.replace(
        config,
        dkv_q_blocks_are_contiguous=dkv_q_blocks_are_contiguous(
            dkv_mask_info, q_blocks_per_shard
        ),
    )
    assert (mask_function_fwd is None) == (mask_function_dkv is None)

    dkv_mask_sparsity = float(np.mean(dkv_mask_info.block_mask != 0))
    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)
  else:
    dkv_mask_sparsity = 1.0

  return SplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )


make_splash_mha = partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = partial(_make_splash_attention, is_mqa=True)


# Tokamax's `_dropout_mask_tile` and `_refine_mask_info_with_segments` are not
# inlined. The kernel code above calls them only when attention dropout or
# segment ids are enabled, and `forward_pallas` enables neither.


def _dropout_mask_tile(*args, **kwargs):
  raise NotImplementedError('Attention dropout is not included in this file.')


def _refine_mask_info_with_segments(*args, **kwargs):
  raise NotImplementedError('Segment ids are not included in this file.')


# ------------------------------------------------------------------------------
# tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------------


def largest_divisor_tile(seq_len: int, max_tile: int) -> int:
  """Returns the largest power-of-two tile <= `max_tile` dividing `seq_len`."""
  for tile in (1024, 512, 256):
    if tile <= max_tile and seq_len % tile == 0:
      return tile
  return NUM_LANES


def _padded_row_bytes(x: Any) -> int:
  padded_dim = math.ceil(x.shape[-1] / NUM_LANES) * NUM_LANES
  return padded_dim * jnp.dtype(x.dtype).itemsize


# ------------------------------------------------------------------------------
# tokamax/_src/ops/attention/pallas_mosaic_tpu.py and
# tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
# `PallasMosaicTpuFlashAttention._fwd`, `_get_heuristics_config`,
# `max_vmem_tile` and `build_splash_kernel` without the `Op` class, for a
# causal mask and no bias, soft cap, dropout or paging.
# ------------------------------------------------------------------------------


def _heuristic_block_sizes(q, k, v):
  """Returns `(block_q, block_kv, block_kv_compute)` chosen by Tokamax."""
  # The max tile size fitting inputs within 16 MiB TPU VMEM.
  row_bytes = max(_padded_row_bytes(q), _padded_row_bytes(v))
  max_tile = 1024 if row_bytes <= 256 else 512
  block_q = largest_divisor_tile(q.shape[-3], max_tile)
  block_kv = largest_divisor_tile(k.shape[-3], max_tile)
  return block_q, block_kv, min(block_kv, 512)


def forward_pallas(q, k, v):
  """Causal attention forward pass on the Pallas TPU kernel.

  Takes the same arguments as `forward`. Returns `(out, residuals)` with
  `residuals = (max_logits, logsumexp)`; see the module docstring. The sequence
  lengths must be multiples of 128. Off TPU the kernel runs in Pallas interpret
  mode.
  """
  supported_dtypes = (jnp.bfloat16, jnp.float32)
  if any(arg.dtype not in supported_dtypes for arg in (q, k, v)):
    raise NotImplementedError('Only bfloat16 and float32 inputs are supported.')

  block_q, block_kv, block_kv_compute = _heuristic_block_sizes(q, k, v)
  logits_scale = 1 / math.sqrt(q.shape[-1])

  q *= logits_scale

  orig_q_shape = q.shape
  as_4d = lambda x: jax.lax.collapse(jax.lax.broadcast_to_rank(x, 4), 0, -3)
  q, k, v = map(as_4d, (q, k, v))

  _, q_seq_len, num_q_heads, _ = q.shape
  _, kv_seq_len, num_kv_heads, head_dim_out = v.shape

  # The SplashAttention kernel expects the sequence dimension to be after the
  # num_heads dimension.
  q, k, v = map(lambda x: jnp.swapaxes(x, 1, 2), (q, k, v))

  is_mqa = num_kv_heads == 1
  if num_q_heads % num_kv_heads:
    raise ValueError(f'{num_q_heads=} must be divisible by {num_kv_heads=}')
  splash_config = dataclasses.replace(
      SplashConfig.get_default(),
      attn_logits_soft_cap=None,
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      q_layout=QKVLayout.HEAD_DIM_MINOR,
      k_layout=QKVLayout.HEAD_DIM_MINOR,
      v_layout=QKVLayout.HEAD_DIM_MINOR,
      use_experimental_scheduler=True,
      use_base2_exp=True,
      interpret=jax.default_backend() != 'tpu',
  )

  shard_count = 1
  splash_mask = CausalMask(
      shape=(q_seq_len, kv_seq_len), shard_count=shard_count
  )
  splash_maker = make_splash_mqa if is_mqa else make_splash_mha
  splash_maker = functools.partial(
      splash_maker,
      q_seq_shards=shard_count,
      config=splash_config,
      save_residuals=True,
      mask_value=float(jnp.finfo(jnp.float32).min),
      downcast_smem_data=False,
  )
  if is_mqa:
    k = jnp.squeeze(k, axis=1)
    v = jnp.squeeze(v, axis=1)

  def splash_fn(q, k, v):
    return splash_maker(mask=splash_mask)(q, k, v)

  out, stats = jax.vmap(splash_fn)(q, k, v)

  residuals = (stats['max_logits'], stats['logsumexp'])
  # Restore the original batch dimensions (*B) and head/sequence structure
  # (H, T).
  # Input q is [*B, T, H, D]. Target residuals are [*B, H, T].
  res_target_shape = (
      *orig_q_shape[:-3],
      orig_q_shape[-2],
      orig_q_shape[-3],
  )
  residuals = jax.tree.map(lambda x: x.reshape(res_target_shape), residuals)
  out = jnp.swapaxes(out, 1, 2)
  # Reshape back to original shape.
  out = out.reshape(*orig_q_shape[:-1], out.shape[-1])[..., :head_dim_out]
  return out, residuals


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # Default: the Mixtral shape below at batch 1 with half the heads; the
    # sequence length, the head dimensions and the number of query heads per
    # key/value head are unchanged. It is reduced so that the pure-JAX
    # `workload`, which holds `[B, H, T, t]` float32 logits, fits in memory:
    # XLA's CPU memory plan for `jit(workload)` is 5.1 GiB of temporaries here
    # and 326 GiB for the full shape. Both figures are XLA estimates on CPU,
    # not TPU measurements.
    'mixtral_8x7b_b1_h16': {
        'name': 'mixtral_8x7b_b1_h16_attention_bwd_pallas_fwd',
        'model': 'Mixtral-8x7B',
        'operator': 'dot_product_attention_backward',
        'batch': 1,
        'seq_len': 4096,
        'num_q_heads': 16,
        'num_kv_heads': 4,
        'head_dim_qk': 128,
        'head_dim_v': 128,
    },
    # tokamax arg_specs: project 'mixtral', name '8x7b_bf16' (tagged primary).
    'mixtral_8x7b': {
        'name': 'mixtral_8x7b_attention_bwd_pallas_fwd',
        'model': 'Mixtral-8x7B',
        'operator': 'dot_product_attention_backward',
        'batch': 32,
        'seq_len': 4096,
        'num_q_heads': 32,
        'num_kv_heads': 8,
        'head_dim_qk': 128,
        'head_dim_v': 128,
    },
    # Small shapes for CPU smoke tests. The sequence spans two 1024-token
    # blocks of the Tokamax TPU kernel, and the number of query heads per
    # key/value head is that of the default config.
    'smoke': {
        'name': 'attention_smoke_bwd_pallas_fwd',
        'model': 'Mixtral-8x7B',
        'operator': 'dot_product_attention_backward',
        'batch': 2,
        'seq_len': 2048,
        'num_q_heads': 8,
        'num_kv_heads': 2,
        'head_dim_qk': 128,
        'head_dim_v': 128,
    },
}
CONFIG = CONFIGS['mixtral_8x7b_b1_h16']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(q, k, v, dout)`; `dout` is the output cotangent."""
  cfg = _resolve_config(config)
  B, T = cfg['batch'], cfg['seq_len']
  H, h = cfg['num_q_heads'], cfg['num_kv_heads']
  D, d = cfg['head_dim_qk'], cfg['head_dim_v']
  keys = jax.random.split(jax.random.key(42), 4)
  # Standard-normal values, as in the Tokamax attention tests; `forward` scales
  # the logits by `D ** -0.5`. Sample in float32 and cast, so every dtype sees
  # the same values.
  q = jax.random.normal(keys[0], (B, T, H, D)).astype(dtype)
  k = jax.random.normal(keys[1], (B, T, h, D)).astype(dtype)
  v = jax.random.normal(keys[2], (B, T, h, d)).astype(dtype)
  dout = jax.random.normal(keys[3], (B, T, H, d)).astype(dtype)
  return q, k, v, dout


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 3, 'diff_argnums': (0, 1, 2)}


def workload(q, k, v, dout):
  """Returns `(dq, dk, dv)`, the VJP of `forward` with `dout`."""
  _, vjp_fn = jax.vjp(forward, q, k, v)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  Per batch element and query head, `forward` computes the `T x t` logits with
  a `T x D` by `D x t` matmul (2*T*t*D) and the output with a `T x t` by `t x d`
  matmul (2*T*t*d). Both are dense: the causal mask is applied to the logits
  afterwards. The VJP runs the forward once and a backward that costs about
  twice the forward, so 3x in total.
  """
  cfg = _resolve_config(config)
  B, T = cfg['batch'], cfg['seq_len']
  H = cfg['num_q_heads']
  D, d = cfg['head_dim_qk'], cfg['head_dim_v']
  return int(3 * B * H * (2 * T * T * D + 2 * T * T * d))


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
