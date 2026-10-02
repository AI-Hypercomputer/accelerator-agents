from __future__ import annotations
"""Gated Delta Net (GDN) — Pallas TPU Kernel.

Self-contained implementation.
"""
import functools
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
# Inlined compute_schedule_v2.py
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

import jax
import jax.numpy as jnp


def compute_schedule_table_v2(
    query_start_loc: jax.Array,
    decode_tokens: int | jax.Array,
    num_valid_seqs: int | jax.Array,
    max_tokens: int,
    chunk_size: int,
    BT: int | None = None,
    alignment: int = 8,
) -> tuple[jax.Array, jax.Array]:
  """Compute number of iterations in grid and work each iteration will do

  At high level
    - each iteration of grid is either prefill and or decode
    - grid moves in size of bt decode tokens (sequence) backwards starting from
    boundary
    - and prefill moves in chunk sized tokens forward from boundary to end
  Input characteristics
    - each sequence start and end may not be sublane aligned,
    boundary between decode and prefill maybe in shared sublane
    - sequence may not divide chunk size

  hardware req
    - offset for each block has to be sublane aligned

  So for this we have transition blocks at boundaries between prefill sequences,
  including first one with decode, token by token math is done here instead of
  chunk wise

  TODO: optimize table ,
    remove metadata which can be derived from other metadata or loop indices,
    like
        block offset can be derived from block idx and sequence start,
        block count can be derived from block idx and sequence start/end.
        also some metadata is only used for prefill or decode and can be stored
        in separate tables or encoded in same table with fewer bits.
        dtype of some metadata can be reduced to save space, for example
        block_is_first and block_is_last can be stored in 2 bits together,
        Sublane token by token metadata can be optimized by only storing
        boundaries
  """
  if BT is None:
    BT = chunk_size

  num_decode_batches = (decode_tokens + BT - 1) // BT
  num_seqs = query_start_loc.shape[0] - 1

  max_blocks = (max_tokens + chunk_size - 1) // chunk_size
  safe_max_blocks = int(max_blocks + num_seqs * 2)

  # =========================================================================
  # 1. Get each prefill sequence's effective start for chunkwise math
  # =========================================================================
  r_idx = jnp.arange(num_seqs)
  is_last_seq = r_idx == num_seqs - 1
  seq_start = query_start_loc[:-1]
  seq_end = query_start_loc[1:]
  num_tokens = query_start_loc[num_valid_seqs]

  # create vector of sequence ends
  prev_seq_end = jnp.pad(seq_end[:-1], (1, 0), constant_values=0)
  effective_start = jnp.where(
      prev_seq_end % alignment != 0,
      (prev_seq_end // alignment) * alignment + alignment,
      prev_seq_end,
  )

  # if seq_len < sublane size
  is_decode_boundary = prev_seq_end == decode_tokens
  is_swallowed = (effective_start >= seq_end) & (~is_decode_boundary)

  # compute the effective end of the rounded up to nearest sublane
  next_aligned_start = (seq_end // alignment) * alignment
  needs_transition = (
      (seq_end % alignment != 0) & (~is_last_seq) & (~is_swallowed)
  )

  is_decode_boundary = prev_seq_end == decode_tokens

  needs_start_transition = (
      (prev_seq_end % alignment != 0) & (~is_swallowed) & is_decode_boundary
  )

  effective_end = jnp.where(needs_transition, next_aligned_start, seq_end)
  effective_end = jnp.maximum(effective_start, effective_end)

  # Block counts per sequence
  num_regular_blocks = (
      effective_end - effective_start + chunk_size - 1
  ) // chunk_size
  total_blocks_per_seq = (
      num_regular_blocks
      + needs_transition.astype(jnp.int32)
      + needs_start_transition.astype(jnp.int32)
  )
  total_blocks_per_seq = jnp.where(is_swallowed, 0, total_blocks_per_seq)

  # Calculate the last perfectly aligned decode boundary
  is_pure_decode = seq_end <= decode_tokens
  total_blocks_per_seq = jnp.where(is_pure_decode, 0, total_blocks_per_seq)

  # Starting block index for each sequence
  base_idx = jnp.cumsum(total_blocks_per_seq) - total_blocks_per_seq
  total_prefill_blocks = jnp.sum(total_blocks_per_seq)

  # =========================================================================
  # 2. shows up as gathers
  # create block table
  # =========================================================================
  b_idx = jnp.arange(safe_max_blocks)
  prefill_valid_mask = b_idx < total_prefill_blocks

  # map grid index to sequence/request,
  # key for previous metadata arrays constructed to gather by sequence
  r_for_block = jnp.sum(b_idx[:, None] >= base_idx[None, :], axis=-1) - 1
  r_for_block = jnp.minimum(jnp.maximum(r_for_block, 0), num_seqs - 1)

  # index of block within blocks for a sequence
  local_b = b_idx - base_idx[r_for_block]

  start_trans_offset = (seq_start[r_for_block] // alignment) * alignment

  is_start_trans = needs_start_transition[r_for_block] & (local_b == 0)

  # Adjust local_b for regular blocks if there was a start transition
  adj_local_b = jnp.where(
      needs_start_transition[r_for_block], local_b - 1, local_b
  )

  is_end_trans = needs_transition[r_for_block] & (
      adj_local_b == num_regular_blocks[r_for_block]
  )

  reg_offset = effective_start[r_for_block] + adj_local_b * chunk_size
  reg_count = jnp.minimum(chunk_size, effective_end[r_for_block] - reg_offset)
  #   reg_is_last = reg_offset + reg_count >= seq_end[r_for_block]
  #   reg_is_first = reg_offset == seq_start[r_for_block]

  trans_offset = next_aligned_start[r_for_block]

  # Apply predication
  block_offset = jnp.where(
      is_start_trans,
      start_trans_offset,
      jnp.where(is_end_trans, trans_offset, reg_offset),
  )

  block_count = jnp.where(
      is_start_trans,
      effective_start[r_for_block] - seq_start[r_for_block],
      jnp.where(is_end_trans, alignment, reg_count),
  )

  is_trans_block = is_start_trans | is_end_trans

  # =========================================================================
  # 3. Metadata for shared sublane tiles
  # =========================================================================
  last_valid_loc = query_start_loc[num_valid_seqs]
  valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
  fixed_query_start_loc = jnp.where(
      valid_loc_mask, query_start_loc, last_valid_loc
  )
  glob_idxs = block_offset[:, None] + jnp.arange(alignment)[None, :]

  # [safe_max_blocks, sublane size, num_seqs]
  valid_mask = glob_idxs < num_tokens
  t_reqs = (
      jnp.sum(
          glob_idxs[:, :, None] >= fixed_query_start_loc[None, None, :], axis=-1
      )
      - 1
  )
  # there could be padding in query_start_loc
  last_valid_seq = jnp.max(
      jnp.where(total_blocks_per_seq > 0, jnp.arange(num_seqs), -1)
  )
  t_reqs = jnp.where(valid_mask, t_reqs, last_valid_seq)
  t_reqs = jnp.minimum(jnp.maximum(t_reqs, 0), num_seqs - 1)

  is_first_tok = (glob_idxs == query_start_loc[t_reqs]).astype(jnp.int32)
  is_last_tok = (glob_idxs == query_start_loc[t_reqs + 1] - 1).astype(jnp.int32)

  # =========================================================================
  # 4. Decode blocks metadata
  # =========================================================================
  decode_valid_mask = b_idx < num_decode_batches
  decode_batch_idx = jnp.where(
      decode_valid_mask, (num_decode_batches - 1) - b_idx, 0
  )
  decode_offsets = decode_batch_idx * BT
  decode_req_ids = decode_batch_idx * BT
  decode_counts = jnp.where(
      decode_valid_mask, jnp.minimum(BT, decode_tokens - decode_offsets), 0
  )

  # Mask out invalid prefill
  prefill_valid_ints = prefill_valid_mask.astype(jnp.int32)
  block_offset = jnp.where(prefill_valid_mask, block_offset, 0)
  r_for_block = jnp.where(prefill_valid_mask, r_for_block, 0)
  block_count = jnp.where(prefill_valid_mask, block_count, 0)
  block_is_first = block_offset <= seq_start[r_for_block]
  block_is_last = (block_offset + block_count) >= seq_end[r_for_block]
  block_is_first = jnp.where(prefill_valid_mask, block_is_first, False)
  block_is_last = jnp.where(prefill_valid_mask, block_is_last, False)
  is_trans_block = jnp.where(prefill_valid_mask, is_trans_block, False)
  t_reqs = jnp.where(prefill_valid_mask[:, None], t_reqs, 0)
  is_first_tok = jnp.where(prefill_valid_mask[:, None], is_first_tok, 0)
  is_last_tok = jnp.where(prefill_valid_mask[:, None], is_last_tok, 0)

  # =========================================================================
  # 5. Merge all
  # =========================================================================
  # Columns mapping:
  # 0: prefill_valid_ints - 1 if this grid block has valid prefill work,
  # .                  0 otherwise
  # 1: block_offset - start index of prefill start in tile, usually 0
  #                     but in shared sublane case its not
  # 2: r_for_block - request ID (sequence index) this prefill block belongs to
  # 3: block_count - number of valid tokens in this prefill block
  # 4: decode_valid_mask - 1 if this step has valid decode work, 0 otherwise
  # 5: decode_offsets - start index for the decode batch
  # 6: decode_req_ids - starting request ID in decode batch
  # 7: decode_counts - number of valid decode requests in this batch
  # 8: block_is_last - 1 if this is the last block for the request, 0 otherwise
  # 9: block_is_first - 1 if first block for request, 0 otherwise
  # 10: is_trans_block - 1 if this is a transition block, 0 otherwise
  cols = [
      prefill_valid_ints,  # 0
      block_offset,  # 1
      r_for_block,  # 2
      block_count,  # 3
      decode_valid_mask.astype(jnp.int32),  # 4
      decode_offsets,  # 5
      decode_req_ids,  # 6
      decode_counts,  # 7
      block_is_last.astype(jnp.int32),  # 8
      block_is_first.astype(jnp.int32),  # 9
      is_trans_block.astype(jnp.int32),  # 10
  ]

  # 11 to 11 + alignment - 1: Request ID for each token in the sublane tile
  for i in range(alignment):
    cols.append(t_reqs[:, i])  # e.g., 11-18 if alignment=8
  # 11 + alignment to 11 + 2*alignment - 1: 1 if token is first in request
  for i in range(alignment):
    cols.append(is_first_tok[:, i])  # e.g., 19-26
  # 11 + 2*alignment to 11 + 3*alignment - 1: 1 if token is last in request
  for i in range(alignment):
    cols.append(is_last_tok[:, i])  # e.g., 27-34

  final_table = jnp.stack(cols, axis=1)
  total_blocks = jnp.maximum(total_prefill_blocks, num_decode_batches)

  return final_table, total_blocks

compute_schedule_table_v2 = types.SimpleNamespace(
    compute_schedule_table_v2=compute_schedule_table_v2
)

# ==============================================================================
# Inlined triangle_solver.py
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

import enum
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


# Implementation of inverse of triangle based on Newton-Schulz iteration.
def newton_schulz_inverse_ref(A, n=None):
  """Inverse of unit lower triangular matrix using Newton-Schulz iteration.

  Args:
    A: Tensor with last two dimensions representing a square lower triangular
      matrix with unit diagonal.
    n: Number of iterations to run.

  Newton Schulz iteration:
  https://en.wikipedia.org/wiki/Matrix_sign_function#Newton%E2%80%93Schulz_iteration
  S_{k+1} = S_k @ (2 * I - A @ S_k)

  Let L = A - I
  Starting with S_0 = I, this is equivalent mathematically to
  S_k = (I - L) @ (I + L^2) @ (I + L^4)....(I + (L^(2^k))), k > 0

  If L is strictly lower (or upper) triangular, L ^ n == 0.
  So this series converges after log(n) steps.

  We don't directly compute S_k as above to reduce precision loss.
  We run the last step in higher precision to improve the overall estimate.
  Initial steps are kept in lower precision for speed.

  Returns:
    Inverse of A.
  """
  if n is None:
    n = A.shape[-1]
  eye = jnp.broadcast_to(jnp.eye(n, dtype=A.dtype), A.shape)
  S = 2 * eye - A
  k = 1
  while k < n:
    precision = jax.lax.Precision.HIGHEST
    k *= 2
    I_plus_error = 2 * eye - jnp.matmul(A, S, precision=precision)
    S = jnp.matmul(S, I_plus_error, precision=precision)
  return S


# Pallas implementation of Newton-Schulz iteration for unit lower triangular
# matrices.
def newton_schulz_inverse_pallas_kernel(A_ref, x_ref):
  x_ref[...] = newton_schulz_inverse_ref(A_ref[...])


def newton_schulz_inverse_pallas(A, *, block_size=64):
  """Newton-Schulz iteration for unit lower triangular matrices on Pallas."""

  A_shape = A.shape
  # Squash all the leading dimensions
  A = A.reshape(-1, *A.shape[-2:])
  N = A.shape[0]
  grid_size = pl.cdiv(N, block_size)
  x = pl.pallas_call(
      newton_schulz_inverse_pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
      grid=(grid_size,),
      in_specs=[
          pl.BlockSpec(
              (block_size, A.shape[-2], A.shape[-1]), lambda idx: (idx, 0, 0)
          ),
      ],
      out_specs=pl.BlockSpec(
          (block_size, A.shape[-2], A.shape[-1]), lambda idx: (idx, 0, 0)
      ),
      name="newton_schulz_inverse_kernel",
  )(A)
  return x.reshape(A_shape)


def local_forward_substitution(A, b):
  """Solves A X = B for unit lower triangular matrix A using forward substitution.

  Args:
    A: A tensor of shape (B, N, N) representing a batch of unit lower triangular
      matrices.
    b: A tensor of shape (B, N, K) representing the right-hand side.

  Returns:
    A tensor of shape (B, N, K) representing the solution X.
  """
  B, N, K = b.shape
  x_list = []
  for i in range(N):
    b_i = b[:, i, :]
    if i == 0:
      x_i = b_i
    else:
      stacked_x = jnp.stack(x_list, axis=1)  # (B, i, K)
      all_prev_A = A[:, i, :i]  # (B, i)
      prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)  # (B, K)
      x_i = b_i - prev_sum  # (B, K) for the row i
    x_list.append(x_i)
  x = jnp.stack(x_list, axis=1)  # (B, N, K)
  return x


def decompose_triangular_matrix_inverse_pallas_kernel(
    A_ref, x_ref, *, block_size=16
):
  A = A_ref[...]
  # Matrix dimension
  B, N, _ = A.shape
  num_blocks = N // block_size

  # same as lower_triangle_solver_pallas_kernel but 2d block wise
  # AX = I, solve for X block wise. X = I - sum(AX_prev)
  for i in range(num_blocks):
    start, end = i * block_size, (i + 1) * block_size
    e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
    e_block = jnp.broadcast_to(e_block, (B, block_size, N))
    if i == 0:
      target_b = e_block
    else:
      interaction_A = A[:, start:end, :start]
      solved_x = x_ref[:, :start, :]
      prev_sum = jnp.matmul(
          interaction_A, solved_x, precision=jax.lax.Precision.HIGHEST
      )
      target_b = e_block - prev_sum

    local_A = A[:, start:end, start:end]
    x_block = local_forward_substitution(local_A, target_b)
    x_ref[..., start:end, :] = x_block


def decompose_triangular_matrix_inverse_pallas(
    A, *, n_block_size=64, block_size=16
):
  """Inverts unit lower triangular matrices using a block-wise approach in Pallas.

  This function solves A X = I for X, where A is a unit lower triangular matrix.
  It uses a block-wise Gaussian elimination approach to improve performance.

  Args:
    A: A tensor of shape (batch_size, chunks, heads, head_dim, head_dim) where
      the last two dimensions represent unit lower triangular matrices.
    n_block_size: The block size for Pallas grid execution.
    block_size: The block size for the block-wise inversion algorithm.

  Returns:
    A tensor of the same shape as A, representing the inverse of A.
  """

  # Squash all the leading dimensions
  A_reshaped = A.reshape(-1, *A.shape[-2:])
  A_shape = A_reshaped.shape
  x_shape = A_shape

  N = A_reshaped.shape[0]
  grid_size = pl.cdiv(N, n_block_size)

  head_dim = A_shape[-1]
  kernel = functools.partial(
      decompose_triangular_matrix_inverse_pallas_kernel, block_size=block_size
  )
  x = pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(x_shape, A.dtype),
      grid=(grid_size,),
      in_specs=[
          pl.BlockSpec(
              (n_block_size, head_dim, head_dim), lambda idx: (idx, 0, 0)
          ),
      ],
      out_specs=pl.BlockSpec(
          (n_block_size, head_dim, head_dim), lambda idx: (idx, 0, 0)
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=67108864),
      name=f"decompose_triangular_matrix_inverse_pallas_kernel_{n_block_size}_{block_size}",
  )(A_reshaped)

  return x.reshape(A.shape)


class TriangleSolverImpl(str, enum.Enum):
  GAUSSIAN = "gaussian"
  NEWTON_SCHULZ = "newton_schulz"

  # TODO: Choose based on Chunk size and vmem constraints. Newton-schulz is unsatable, it needs S to be nilpotent to converge and also with small values to avoid NaNs
  def __call__(self, A):
    if self == TriangleSolverImpl.GAUSSIAN:
      return decompose_triangular_matrix_inverse_pallas(
          A, n_block_size=min(64, A.shape[-1])
      )
    elif self == TriangleSolverImpl.NEWTON_SCHULZ:
      return newton_schulz_inverse_pallas(A)
    else:
      print(
          f"Unknown solver: {self.value} Using default solver."
          f" {TriangleSolverImpl.GAUSSIAN.value}"
      )
      return decompose_triangular_matrix_inverse_pallas(
          A, n_block_size=min(64, A.shape[-1])
      )


# ==============================================================================
# Inlined recurrent_scan_v2.py
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

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# inlined compute_schedule_v2


def invert_triangular_matrix(A, block_size=16):
  """Inverts a unit lower triangular matrix A block-wise.

  Args:
    A: Unit lower triangular matrix of shape (B, N, N).
    block_size: Size of the blocks for Gaussian elimination.

  Returns:
    Inverse of A, of shape (B, N, N).
  """
  B, N, _ = A.shape
  num_blocks = N // block_size

  def local_forward_sub(A_mat, b_mat):
    x_list = []
    for i in range(block_size):
      b_i = b_mat[:, i, :]
      if i == 0:
        x_i = b_i
      else:
        stacked_x = jnp.stack(x_list, axis=1)
        all_prev_A = A_mat[:, i, :i]
        prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)
        x_i = b_i - prev_sum
      x_list.append(x_i)
    return jnp.stack(x_list, axis=1)

  x_blocks = []
  for i in range(num_blocks):
    start, end = i * block_size, (i + 1) * block_size
    e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
    e_block = jnp.broadcast_to(e_block, (B, block_size, N))

    if i == 0:
      target_b = e_block
    else:
      interaction_A = A[:, start:end, :start]
      solved_x = jnp.concatenate(x_blocks, axis=1)
      prev_sum = jnp.matmul(
          interaction_A, solved_x, precision=jax.lax.Precision.HIGHEST
      )
      target_b = e_block - prev_sum

    local_A = A[:, start:end, start:end]
    x_block = local_forward_sub(local_A, target_b)
    x_blocks.append(x_block)

  return jnp.concatenate(x_blocks, axis=1)


def inner_kernel(
    # VMEM: (C, D) where D = 2*n_kq*d_k + n_v*d_v. QKV tokens for Prefill chunk
    prefill_qkv_ref,
    # VMEM: (C, D) where D = 2*n_kq*d_k + n_v*d_v. QKV tokens for Decode batch
    decode_qkv_ref,
    # VMEM: (C, 128). Raw a values for Prefill chunk
    prefill_a_raw_ref,
    # VMEM: (BT, 128). Raw a values for Decode batch
    decode_a_raw_ref,
    # VMEM: (C, 128). Raw b values for Prefill chunk
    prefill_b_raw_ref,
    # VMEM: (BT, 128). Raw b values for Decode batch
    decode_b_raw_ref,
    # VMEM: (n_v,). A_log for gate computation
    a_log_ref,
    # VMEM: (n_v,). dt_bias for gate computation
    dt_bias_ref,
    # VMEM: (C, n_v * d_v). Scanned outputs for prefill
    prefill_output_ref,
    # VMEM: (BT, n_v * d_v). Scanned outputs for decode
    decode_output_ref,
    # SMEM: (max_blocks, 8). Schedule table
    schedule_table,
    # SMEM: (max_reqs,). State indices
    state_indices,
    # SMEM: (max_reqs,). Whether each request has prior recurrent state
    has_initial_state,
    *,
    # HBM: (B, n_v, d_k, d_v). All recurrent states
    recurrent_state_in,
    recurrent_state_out,
    # Chunk size for prefill
    C: int,
    # Batch size for decode
    BT: int,
    #  Number of key/query heads
    n_kq: int,
    #  Number of value heads
    n_v: int,
    #  Key dimension
    d_k: int,
    #  Value dimension
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
    # VMEM scratchpad: (2, n_v, d_k, d_v). To carry state across chunks
    # (double buffered)
    prefill_scratch,
    # VMEM scratchpad: (1, 2, n_v, d_k, d_v). TODO: double
    # buffer or x buffer to to loop over BT in decode without overwriting state and using async copy for state load/store)
    decode_state_scratch,
    # VMEM scratchpad: (1, n_v, d_k, d_v). dtype = recurrent_state dtype
    # TODO: if output dtype of state is always f32 then this can be removed.
    state_commit_scratch,
    # VMEM scratchpad: (BT, n_v * d_v). To hold decode outputs before DMA
    decode_output_scratch,
    # Array of C semaphores for decode state loads
    decode_read_semaphores,
    # 1 semaphore for decode state stores
    decode_write_semaphore,
    # 1 semaphore for prefill DMA (stores only)
    prefill_semaphore,
    # Number of decode tokens (requests) in the batch
    decode_tokens,
):
  """Inner kernel for recurrent scan processing both prefill and decode.

  This function is called for each step in the schedule table and dispatches
  work to either `process_decode` or
  `process_regular_prefill`/`process_transition_prefill`.
  """
  step = pl.program_id(0)

  # READ table

  prefill_valid = schedule_table[step, 0][...]
  prefill_req_id = schedule_table[step, 2][...]

  decode_valid = schedule_table[step, 4][...]
  decode_offset = schedule_table[step, 5][...]
  decode_req_id = schedule_table[step, 6][...]
  decode_count = schedule_table[step, 7][...]

  prefill_offset = schedule_table[step, 1][...]
  is_transition = schedule_table[step, 10][...]

  is_last_chunk = schedule_table[step, 8][...]
  is_first_chunk = schedule_table[step, 9][...]

  def l2_normalize(x, eps=1e-6):
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm

  # 2. Decode Branch
  # check current iteration had decode work
  @pl.when(decode_valid > 0)
  def decode_wrapper():

    def get_target_idx(b):
      safe_req_id = jnp.minimum(decode_req_id + b, state_indices.shape[0] - 1)
      return state_indices[safe_req_id][...]

    def process_decode(b, _):
      # token by token check if decode token or not
      is_valid = b < decode_count

      @pl.when(is_valid)
      def do_work():
        target_idx = get_target_idx(b)

        # Load state TODO: make async
        copy_op = pltpu.make_async_copy(
            src_ref=recurrent_state_in.at[pl.ds(target_idx, 1)],
            dst_ref=state_commit_scratch,
            sem=decode_read_semaphores.at[0],
        )
        copy_op.start()
        copy_op.wait()
        decode_state_scratch[pl.ds(0, 1)] = state_commit_scratch[...].astype(
            jnp.float32
        )

        key_dim = n_kq * d_k
        b_aligned = (b // sublanesize) * sublanesize
        # Workaround: Upcast to fp32 to avoid NaNs
        qkv_block_data = decode_qkv_ref[
            pl.ds(b_aligned, sublanesize), :
        ].astype(jnp.float32)
        mask = (jnp.arange(sublanesize) == (b % sublanesize)).astype(
            qkv_block_data.dtype
        )[:, None]
        qkv_row = jnp.sum(qkv_block_data * mask, axis=0, keepdims=True)
        # Fused SiLU
        qkv_row = jax.nn.silu(qkv_row)
        q = qkv_row[:, :key_dim].reshape(n_kq, d_k)
        k = qkv_row[:, key_dim : 2 * key_dim].reshape(n_kq, d_k)
        v = qkv_row[:, 2 * key_dim :].reshape(n_v, d_v)

        if use_qk_norm_in_gdn:
          q = l2_normalize(q)
          k = l2_normalize(k)

        # Head repetition
        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
          q = jnp.repeat(q, repeat_factor, axis=0)
          k = jnp.repeat(k, repeat_factor, axis=0)

        scale = d_k**-0.5
        q = q * scale

        b_aligned = (b // sublanesize) * sublanesize

        g_block_new = decode_a_raw_ref[pl.ds(b_aligned, sublanesize), :]
        beta_block_new = decode_b_raw_ref[pl.ds(b_aligned, sublanesize), :]

        mask_new = (jnp.arange(sublanesize) == (b % sublanesize)).astype(
            g_block_new.dtype
        )[:, None]

        curr_g_slice_new = jnp.sum(
            g_block_new * mask_new, axis=0, keepdims=True
        )
        curr_beta_slice_new = jnp.sum(
            beta_block_new * mask_new, axis=0, keepdims=True
        )

        a_raw_new = curr_g_slice_new[:, :n_v].reshape(n_v).astype(jnp.float32)
        b_raw_new = (
            curr_beta_slice_new[:, :n_v].reshape(n_v).astype(jnp.float32)
        )

        # Compute gate
        curr_beta = jax.nn.sigmoid(b_raw_new)
        curr_g = -jnp.exp(a_log_ref[...].astype(jnp.float32)) * jax.nn.softplus(
            a_raw_new + dt_bias_ref[...].astype(jnp.float32)
        )
        curr_g = jnp.maximum(curr_g, -100.0)
        decay = jnp.exp(curr_g)

        current_state = decode_state_scratch[0]

        # TODO: compare MXU vs VPU, MXU doesn't support FP32, VPU does
        # (n_v, d_k, 1) * (n_v, 1, d_v) -> (n_v, d_k, d_v)
        out_list = []
        new_state_list = []
        for h in range(n_v):
          q_h = q[h : h + 1, :]  # (1, d_k)
          k_h = k[h : h + 1, :]  # (1, d_k)
          v_h = v[h : h + 1, :]  # (1, d_v)

          state_h = current_state[h]  # (d_k, d_v)

          k_state_h = pl.dot(
              k_h, state_h, precision=jax.lax.Precision.HIGHEST
          )  # (1, d_v)

          # v_diff_h = v_h - decay[h].astype(jnp.float32) * k_state_h
          decay_k_state = jnp.where(
              jnp.isinf(k_state_h),
              0.0,
              decay[h].astype(jnp.float32) * k_state_h,
          )
          v_diff_h = v_h - decay_k_state
          v_new_h = curr_beta[h].astype(jnp.float32) * v_diff_h

          q_state_h = pl.dot(
              q_h, state_h, precision=jax.lax.Precision.HIGHEST
          )  # (1, d_v)

          q_k_h = jnp.sum(q_h * k_h, axis=-1, keepdims=True)  # (1, 1)

          # Defensive code to handle NaNs and infs in state,
          # Saw similar issue while trying newton schulz
          # which can happen due to large decay or long sequences.
          # TODO: analyze perf impact and risk of removing this.
          decay_q_state = jnp.where(
              jnp.isinf(q_state_h), 0.0, decay[h] * q_state_h
          )
          out_h = decay_q_state + q_k_h * v_new_h
          out_list.append(out_h)

          k_v_new_h = pl.dot(
              k_h, v_new_h, trans_a=True, precision=jax.lax.Precision.HIGHEST
          )  # (d_k, 1) @ (1, d_v) -> (d_k, d_v)
          # Defensive code to handle NaNs and infs in state,
          # which can happen due to large decay or long sequences.
          # In such cases, we reset the state contribution to zero and rely solely on the new value
          # TODO: analyze perf impact and risk of removing this.
          decay_state = jnp.where(jnp.isinf(state_h), 0.0, state_h * decay[h])
          new_state_h = decay_state + k_v_new_h
          new_state_list.append(new_state_h)

        out = jnp.concatenate(out_list, axis=0)  # (n_v, d_v)
        new_state = jnp.stack(new_state_list, axis=0)  # (n_v, d_k, d_v)

        # TODO: remove VPU path if MXU is certified path
        # decay_exp = decay[..., None]  # (n_v, 1)

        # k_state = jnp.sum(k[..., None] * current_state, axis=1)  # (n_v, d_v)
        # v_diff = v - decay_exp * k_state
        # v_new = curr_beta[..., None] * v_diff  # (n_v, d_v)

        # q_state = jnp.sum(q[..., None] * current_state, axis=1)  # (n_v, d_v)
        # q_k = jnp.sum(q * k, axis=-1, keepdims=True)  # (n_v, 1)

        # out = decay_exp * q_state + q_k * v_new  # (n_v, d_v)
        # k_v_new = k[..., None] * v_new[:, None, :]
        # new_state = current_state * decay_exp[..., None] + k_v_new

        decode_state_scratch[pl.ds(0, 1)] = new_state[None, ...].astype(
            current_state.dtype
        )

        # Accumulate output in scratchpad
        current_output = decode_output_scratch[...]
        mask = (jnp.arange(BT) == b).astype(current_output.dtype)[:, None]
        new_output = jnp.where(
            mask,
            out.reshape(1, n_v * d_v),
            current_output,
        )
        decode_output_scratch[...] = new_output.astype(current_output.dtype)

        # Store state (Synchronous)
        state_commit_scratch[0] = decode_state_scratch[0].astype(
            state_commit_scratch.dtype
        )
        copy_op = pltpu.make_async_copy(
            src_ref=state_commit_scratch,
            dst_ref=recurrent_state_out.at[pl.ds(target_idx, 1)],
            sem=decode_write_semaphore.at[0],
        )
        copy_op.start()
        copy_op.wait()

        return None

      return None

    # loop over bt, could be for loop, BT is static anyway, unroll
    jax.lax.fori_loop(0, BT, process_decode, None)

    # Mask and write accumulated outputs to HBM
    mask = (jnp.arange(BT) < decode_count).astype(decode_output_scratch.dtype)[
        :, None
    ]
    decode_output_scratch_masked = decode_output_scratch[...] * mask
    decode_output_ref[...] = decode_output_scratch_masked

    return None

  # Prefill Branch
  # Process prefill if there is valid prefill work in this step
  @pl.when(prefill_valid > 0)
  def process_prefill():
    # TODO: eliminate k.transpose in matmuls by directly slicing in the right shape above

    # not used meaningfully, because dma is sync.
    # intention is to index into scratch for storing state and not overwrite each other
    prefill_slot = prefill_req_id % 2

    def process_regular_prefill():
      # 1. Initialize state if first chunk of the request in this step
      @pl.when(is_first_chunk > 0)
      def init_state():
        has_init = has_initial_state[prefill_req_id][...]

        def load_from_hbm():
          state_idx = state_indices[prefill_req_id][...]
          copy_op = pltpu.make_async_copy(
              src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
              dst_ref=state_commit_scratch,
              sem=prefill_semaphore.at[prefill_slot],
          )
          copy_op.start()
          copy_op.wait()
          prefill_scratch[prefill_slot] = state_commit_scratch[0].astype(
              prefill_scratch.dtype
          )

        def zero_state():
          prefill_scratch[prefill_slot] = jnp.zeros(
              (n_v, d_k, d_v), dtype=prefill_scratch.dtype
          )

        jax.lax.cond(has_init > 0, load_from_hbm, zero_state)
        return None

      ### Preparataion for chunk wise math,
      ### this kernel design could be optimized lot by not doing this every chunk
      # 1. Extract Q, K, V, g, beta for the chunk
      key_dim = n_kq * d_k

      # Workaround: Upcast to fp32 to avoid NaNs in long sequences
      qkv_chunk = prefill_qkv_ref[...].astype(jnp.float32)  # (C, d)
      # Fused SiLU
      qkv_chunk = jax.nn.silu(qkv_chunk)
      q = qkv_chunk[:, :key_dim]
      k = qkv_chunk[:, key_dim : 2 * key_dim]
      v = qkv_chunk[:, 2 * key_dim :]

      # Load a, b
      a_raw_chunk = prefill_a_raw_ref[...]  # (C, 128)
      b_raw_chunk = prefill_b_raw_ref[...]  # (C, 128)

      # Slice and transpose to match expected shape (n_v, C),
      # TODO: this transpose can be eliminated
      a_raw_processed = a_raw_chunk[:, :n_v].T
      b_raw_processed = b_raw_chunk[:, :n_v].T

      # Compute gates in VMEM in full fp32, not sure if needed.
      beta = jax.nn.sigmoid(b_raw_processed)
      g = -jnp.exp(
          a_log_ref[...][:, None].astype(jnp.float32)
      ) * jax.nn.softplus(
          a_raw_processed + dt_bias_ref[...][:, None].astype(jnp.float32)
      )
      # Workaround: Clamp g to avoid underflow to negative inf
      # g is always negative, from above line
      # for long prefill sequence this negative value will get more negative
      # pow(e,-100) is close to 0.
      g = jnp.maximum(g, -100.0)
      prefill_count = schedule_table[step, 3][...]
      mask_float = (jnp.arange(C) < prefill_count).astype(q.dtype)
      q = jnp.where(mask_float[:, None] > 0, q, 0.0)
      k = jnp.where(mask_float[:, None] > 0, k, 0.0)
      g = jnp.where(mask_float[None, :] > 0, g, 0.0)
      v = jnp.where(mask_float[:, None] > 0, v, 0.0)
      beta = jnp.where(mask_float[None, :] > 0, beta, 0.0)

      q = q.reshape(C, n_kq, d_k)
      k = k.reshape(C, n_kq, d_k)
      v = v.reshape(C, n_v, d_v)

      q = q.transpose(1, 0, 2)
      k = k.transpose(1, 0, 2)
      v = v.transpose(1, 0, 2)

      if use_qk_norm_in_gdn:
        q = l2_normalize(q)
        k = l2_normalize(k)

      repeat_factor = n_v // n_kq
      if repeat_factor > 1:
        q = jnp.repeat(q, repeat_factor, axis=0)
        k = jnp.repeat(k, repeat_factor, axis=0)

      scale = d_k**-0.5
      q = q * scale

      g_cumsum_list = []
      current_sum = jnp.zeros((n_v,), dtype=jnp.float32)
      # cumsum not implemented in pallas
      for i in range(C):
        current_sum = current_sum + g[:, i].astype(jnp.float32)
        g_cumsum_list.append(current_sum)
      g_cumsum = jnp.stack(g_cumsum_list, axis=-1)
      k_beta = k * beta[..., None]

      S = jnp.matmul(
          k_beta.astype(jnp.float32),
          k.transpose(0, 2, 1).astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )

      g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
      i = jnp.arange(C)[:, None]
      j = jnp.arange(C)[None, :]
      mask_float = (i > j).astype(jnp.float32)

      # Defensive code to handle large positive g_diff which can cause
      # overflow in exp,
      # TODO: analyze if this is a common case and if we can remove this or do
      # by other means (like clipping g values before cumsum or using a
      # different data type for g/g_cumsum)
      g_diff_safe = jnp.minimum(g_diff, 0.0)
      S = jnp.where(mask_float[None, :, :] > 0, S * jnp.exp(g_diff_safe), 0.0)

      S_q = jnp.matmul(
          q.astype(jnp.float32),
          k.transpose(0, 2, 1).astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      mask_float_q = (i >= j).astype(jnp.float32)
      g_diff_Sq = g_diff_safe * mask_float_q[None, ...] + (
          1.0 - mask_float_q[None, ...]
      ) * (-1e30)
      S_q = S_q * jnp.exp(g_diff_Sq)
      S_q = S_q * mask_float_q[None, ...]

      I_plus_S = jnp.eye(C, dtype=jnp.float32)[None, ...] + S
      # TODO: call the function in kernels file
      A_inv = invert_triangular_matrix(I_plus_S, block_size=16)

      # UW
      v_beta = v * beta[..., None]
      u = jnp.matmul(
          A_inv, v_beta.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
      )

      k_beta_g = k_beta * jnp.exp(g_cumsum)[..., None]
      w = jnp.matmul(
          A_inv,
          k_beta_g.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )

      q_g = q * jnp.exp(g_cumsum)[..., None]
      current_state = prefill_scratch[prefill_slot]
      attn_inter = jnp.matmul(
          q_g.astype(jnp.float32),
          current_state.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      v_prime = jnp.matmul(
          w,
          current_state.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      v_new = u - v_prime
      term2 = jnp.matmul(S_q, v_new, precision=jax.lax.Precision.HIGHEST)
      o_c = attn_inter + term2

      g_i_last_exp = jnp.exp(g_cumsum[..., -1, None, None])
      g_diff_exp_state = jnp.exp(g_cumsum[..., -1, None] - g_cumsum)[..., None]
      k_i_g_diff = k * g_diff_exp_state

      update_term = jnp.matmul(
          k_i_g_diff.transpose(0, 2, 1).astype(jnp.float32),
          v_new,
          precision=jax.lax.Precision.HIGHEST,
      )
      h_new = current_state * g_i_last_exp + update_term

      prefill_scratch[prefill_slot] = h_new.astype(prefill_scratch.dtype)

      # Store state only if it's the last chunk of the request
      @pl.when(is_last_chunk > 0)
      def store_state():
        # TODO: if dtype of state in HBM is always f32,
        # then we can eliminate this copy and directly write from scratch to HBM
        state_commit_scratch[0] = prefill_scratch[prefill_slot].astype(
            state_commit_scratch.dtype
        )
        state_idx = state_indices[prefill_req_id][...]
        copy_op = pltpu.make_async_copy(
            src_ref=state_commit_scratch,
            dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
            sem=prefill_semaphore.at[prefill_slot],
        )
        copy_op.start()
        copy_op.wait()
        return None

      # TODO: eliminate this transpose and reshape by directly writing in the right shape above
      o_c_tr = o_c.transpose(1, 0, 2)
      o_c_flat = o_c_tr.reshape(C, n_v * d_v)

      prefill_count = schedule_table[step, 3][...]
      mask_float = (jnp.arange(C) < prefill_count).astype(o_c_flat.dtype)
      o_c_flat_masked = o_c_flat * mask_float[:, None]
      prefill_output_ref[...] = o_c_flat_masked.astype(prefill_output_ref.dtype)
      return None

    def process_transition_prefill():
      # this is processing prefill sequences in a sublane that has multiple sequences
      C_trans = sublanesize
      key_dim = n_kq * d_k

      # Workaround: Upcast to fp32 to avoid NaNs
      qkv_chunk = prefill_qkv_ref[:C_trans, :].astype(jnp.float32)
      # Fused SiLU TODO: maybe 'SiLU' needs to be parametrized,
      qkv_chunk = jax.nn.silu(qkv_chunk)
      q = qkv_chunk[:, :key_dim]
      k = qkv_chunk[:, key_dim : 2 * key_dim]
      v = qkv_chunk[:, 2 * key_dim :]

      # Load untransposed a and b
      a_raw_chunk = prefill_a_raw_ref[...]  # (C, 128)
      b_raw_chunk = prefill_b_raw_ref[...]  # (C, 128)

      # Slice and transpose to match expected shape (n_v, C_trans)
      a_raw_processed = a_raw_chunk[:C_trans, :n_v].T
      b_raw_processed = b_raw_chunk[:C_trans, :n_v].T

      # NOTE: b is upcasted to f32 in ref before sigmoid, beta is bf16
      beta_chunk = jax.nn.sigmoid(b_raw_processed)
      # NOTE: a is upcasted to f32 before add to dt_bias
      g_chunk = -jnp.exp(
          a_log_ref[...][:, None].astype(jnp.float32)
      ) * jax.nn.softplus(
          a_raw_processed + dt_bias_ref[...][:, None].astype(jnp.float32)
      )
      g_chunk = jnp.maximum(g_chunk, -100.0)
      q = q.reshape(C_trans, n_kq, d_k)
      k = k.reshape(C_trans, n_kq, d_k)
      v = v.reshape(C_trans, n_v, d_v)

      # TODO: eliminate these transposes by directly slicing in the right shape above,
      q = q.transpose(1, 0, 2)
      k = k.transpose(1, 0, 2)
      v = v.transpose(1, 0, 2)

      if use_qk_norm_in_gdn:
        q = l2_normalize(q)
        k = l2_normalize(k)

      repeat_factor = n_v // n_kq
      if repeat_factor > 1:
        q = jnp.repeat(q, repeat_factor, axis=0)
        k = jnp.repeat(k, repeat_factor, axis=0)

      scale = d_k**-0.5
      q = q * scale

      # state indice for req
      first_req_id = schedule_table[step, 11][...]
      first_is_first = schedule_table[step, 11 + C_trans][...]
      first_slot = first_req_id % 2
      first_has_init = has_initial_state[first_req_id][...]

      @pl.when((first_is_first > 0) & (first_has_init > 0))
      def load_first_state():
        state_idx = state_indices[first_req_id][...]
        copy_op = pltpu.make_async_copy(
            src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
            dst_ref=state_commit_scratch,
            sem=prefill_semaphore.at[first_slot],
        )
        copy_op.start()
        copy_op.wait()
        prefill_scratch[first_slot] = state_commit_scratch[0].astype(
            prefill_scratch.dtype
        )

      h = prefill_scratch[first_slot]
      h = jnp.where(
          (first_is_first > 0) & (first_has_init == 0), jnp.zeros_like(h), h
      )

      current_r = first_req_id
      sequence_valid = True

      # loop over token by token
      for i in range(sublanesize):
        # read transition token metadata
        t_req = schedule_table[step, 11 + i][...]
        # get sequence index for token i in sublane
        t_is_first = schedule_table[step, 11 + C_trans + i][...]
        t_is_last = schedule_table[step, 11 + 2 * C_trans + i][...]

        is_new_seq = t_req != current_r
        sequence_valid = jnp.where(is_new_seq, True, sequence_valid)

        # Ignore tokens that belong to decode requests,
        # (assumes decode tokens are at packed at head)
        is_decode_token = t_req < decode_tokens
        sequence_valid = jnp.where(is_decode_token, False, sequence_valid)

        c_slot = current_r % 2

        h0 = prefill_scratch[0]
        h1 = prefill_scratch[1]
        prefill_scratch[0] = jnp.where(c_slot == 0, h, h0)
        prefill_scratch[1] = jnp.where(c_slot == 1, h, h1)

        # prefill_scratch in f32, state_commit might be in bf16
        state_commit_scratch[0] = prefill_scratch[c_slot].astype(
            state_commit_scratch.dtype
        )

        def do_write():
          # TODO: Make async
          state_idx = state_indices[current_r][...]
          copy_op = pltpu.make_async_copy(
              src_ref=state_commit_scratch,
              dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
              sem=prefill_semaphore.at[c_slot],
          )
          copy_op.start()
          copy_op.wait()
          return None

        is_current_r_prefill = current_r >= decode_tokens
        should_write = is_current_r_prefill & is_new_seq
        jax.lax.cond(should_write, do_write, lambda: None)

        t_slot = t_req % 2
        t_has_init = has_initial_state[t_req][...]

        def load_t_state():
          state_idx = state_indices[t_req][...]
          copy_op = pltpu.make_async_copy(
              src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
              dst_ref=state_commit_scratch,
              sem=prefill_semaphore.at[t_slot],
          )
          copy_op.start()
          copy_op.wait()
          prefill_scratch[t_slot] = state_commit_scratch[0].astype(
              prefill_scratch.dtype
          )

        should_load_t = (t_is_first > 0) & (t_has_init > 0)
        jax.lax.cond(should_load_t, load_t_state, lambda: None)

        h0_new = prefill_scratch[0]
        h1_new = prefill_scratch[1]
        new_h = jnp.where(t_slot == 0, h0_new, h1_new)

        new_h = jnp.where(
            (t_is_first > 0) & (t_has_init == 0), jnp.zeros_like(new_h), new_h
        )
        h = new_h

        current_r = t_req

        k_i = k[:, i, :]
        v_i = v[:, i, :]
        g_i = g_chunk[:, i]
        beta_i = beta_chunk[:, i]
        q_i = q[:, i, :]

        decay = jnp.exp(g_i)[..., None]

        k_state = jnp.sum(k_i[..., None] * h, axis=1)
        v_diff = v_i - decay * k_state
        v_new = beta_i[:, None] * v_diff

        q_state = jnp.sum(q_i[..., None] * h, axis=1)
        q_k = jnp.sum(q_i * k_i, axis=-1, keepdims=True)

        out_i = decay * q_state + q_k * v_new

        k_v_new = k_i[..., None] * v_new[:, None, :]
        h_new = h * decay[..., None] + k_v_new

        h = jnp.where(sequence_valid, h_new, h)

        # Mask output BEFORE invalidating the sequence for the next token
        out_i = jnp.where(sequence_valid, out_i, 0.0)

        sequence_valid = jnp.where(t_is_last > 0, False, sequence_valid)

        prefill_output_ref[i, :] = out_i.reshape(n_v * d_v).astype(
            prefill_output_ref.dtype
        )

      final_slot = current_r % 2
      prefill_scratch[final_slot] = h
      state_commit_scratch[0] = h.astype(state_commit_scratch.dtype)

      is_current_r_prefill = current_r >= decode_tokens

      # Store state if the current request is a prefill
      @pl.when(is_current_r_prefill)
      def do_final_write():
        # TODO: make async
        state_idx = state_indices[current_r][...]
        copy_op = pltpu.make_async_copy(
            src_ref=state_commit_scratch,
            dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
            sem=prefill_semaphore.at[final_slot],
        )
        copy_op.start()
        copy_op.wait()
        return None

      return None

    is_transition = schedule_table[step, 10][...]

    def process_prefill_dispatch():
      return jax.lax.cond(
          is_transition > 0,
          lambda _: process_transition_prefill(),
          lambda _: process_regular_prefill(),
          operand=None,
      )

    process_prefill_dispatch()
    return None

  # For transition block at boundary of decode and prefill we will have overlap
  # decode block BT contains prefill tokens
  # sublane size transition prefill block contains some decode tokens in the sublane
  # so we need to stitch the outputs so they don't overwrite each other in global index
  # we exchange decode and prefill outputs so
  # prefill output ref has decode token outputs at decode token indexes in its out ref
  # decode output ref has prefill token outputs have prefill token indexes in its out ref
  def do_stitch():
    local_start = prefill_offset - decode_offset
    local_split = decode_tokens - prefill_offset

    # Need to hint compiler, or it complains in DMA added by emit pipeline
    safe_local_start = pl.multiple_of(local_start, sublanesize)

    decode_overlap = decode_output_ref[pl.ds(safe_local_start, sublanesize), :]
    prefill_arr = prefill_output_ref[pl.ds(0, sublanesize), :]

    # 3. Build sublane size mask
    iota = jax.lax.broadcasted_iota(jnp.int32, (sublanesize,), 0)
    is_decode_mask = (iota < local_split).astype(jnp.int32)[:, None]

    # 4. Merge the tensors
    merged_overlap = jnp.where(is_decode_mask, decode_overlap, prefill_arr)

    decode_output_ref[pl.ds(safe_local_start, sublanesize), :] = merged_overlap
    prefill_output_ref[pl.ds(0, sublanesize), :] = merged_overlap

    return None

  is_first_block = pl.program_id(0) == 0
  needs_stitching = (is_transition > 0) & is_first_block & (decode_valid > 0)
  jax.lax.cond(needs_stitching, do_stitch, lambda: None)


def get_qkv_index_map_v2(
    step,
    schedule_table,
    valid_col,
    offset_col,
    count_col,
    alignment=16,
    block_size=64,
    sink_offset=0,
):
  valid = schedule_table[step, valid_col][...]
  offset = schedule_table[step, offset_col][...]
  offset = pl.multiple_of(offset, alignment)

  safe_offset = jnp.where(valid > 0, offset, sink_offset)
  safe_offset = pl.multiple_of(safe_offset, alignment)

  return (pl.ds(safe_offset, block_size), 0)


def create_block_specs(
    schedule_table,
    chunk_size,
    BT,
    d,
    n_v,
    d_v,
    alignment=16,
    sink_offset=0,
):
  """Creates block specs for recurrent scan kernel."""

  prefill_qkv_index_map = functools.partial(
      get_qkv_index_map_v2,
      schedule_table=schedule_table,
      valid_col=0,
      offset_col=1,
      count_col=3,
      alignment=alignment,
      block_size=chunk_size,
      sink_offset=sink_offset,
  )

  decode_qkv_index_map = functools.partial(
      get_qkv_index_map_v2,
      schedule_table=schedule_table,
      valid_col=4,
      offset_col=5,
      count_col=7,
      alignment=alignment,
      block_size=BT,
      sink_offset=sink_offset,
  )

  prefill_qkv_spec = pl.BlockSpec(
      block_shape=(pl.BoundedSlice(chunk_size), d),
      index_map=prefill_qkv_index_map,
  )
  decode_qkv_spec = pl.BlockSpec(
      block_shape=(pl.BoundedSlice(BT), d),
      index_map=decode_qkv_index_map,
  )

  prefill_output_spec = pl.BlockSpec(
      block_shape=(pl.BoundedSlice(chunk_size), n_v * d_v),
      index_map=prefill_qkv_index_map,
  )
  decode_output_spec = pl.BlockSpec(
      block_shape=(pl.BoundedSlice(BT), n_v * d_v),
      index_map=decode_qkv_index_map,
  )

  a_log_spec = pl.BlockSpec(block_shape=(n_v,), index_map=lambda _: (0,))
  dt_bias_spec = pl.BlockSpec(block_shape=(n_v,), index_map=lambda _: (0,))
  prefill_a_raw_spec = pl.BlockSpec(
      block_shape=(pl.BoundedSlice(chunk_size), 128),
      index_map=prefill_qkv_index_map,
  )
  decode_a_raw_spec = pl.BlockSpec(
      block_shape=(pl.BoundedSlice(BT), 128),
      index_map=decode_qkv_index_map,
  )
  prefill_b_raw_spec = pl.BlockSpec(
      block_shape=(pl.BoundedSlice(chunk_size), 128),
      index_map=prefill_qkv_index_map,
  )
  decode_b_raw_spec = pl.BlockSpec(
      block_shape=(pl.BoundedSlice(BT), 128),
      index_map=decode_qkv_index_map,
  )

  return [
      prefill_qkv_spec,
      decode_qkv_spec,
      prefill_a_raw_spec,
      decode_a_raw_spec,
      prefill_b_raw_spec,
      decode_b_raw_spec,
      a_log_spec,
      dt_bias_spec,
  ], [prefill_output_spec, decode_output_spec]


def fused_kernel(
    mixed_qkv_ref,
    aliased_recurrent_state_ref,
    state_indices_ref,
    has_initial_state_ref,
    a_raw_ref,
    b_raw_ref,
    a_log_ref,
    dt_bias_ref,
    schedule_table_ref,
    decode_tokens_ref,
    total_blocks_ref,
    recurrent_state_ref,
    output_ref,
    *,
    C: int,
    BT: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
):
  """Fused kernel for recurrent scan."""
  decode_tokens = decode_tokens_ref[0]
  total_blocks = total_blocks_ref[0]

  d = mixed_qkv_ref.shape[-1]
  pad_size = max(C, BT)
  sink_offset = mixed_qkv_ref.shape[0] - pad_size

  in_specs, out_specs = create_block_specs(
      schedule_table_ref,
      C,
      BT,
      d,
      n_v,
      d_v,
      alignment=sublanesize,
      sink_offset=sink_offset,
  )

  def _run_with_scratch(
      scratch_ref,
      decode_state_scratch_ref,
      state_commit_scratch_ref,
      decode_output_scratch_ref,
      decode_read_sems,
      decode_write_sem,
      prefill_sem,
  ):

    pipeline_func = pltpu.emit_pipeline(
        body=functools.partial(
            inner_kernel,
            C=C,
            BT=BT,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            sublanesize=sublanesize,
            prefill_scratch=scratch_ref,
            decode_state_scratch=decode_state_scratch_ref,
            decode_output_scratch=decode_output_scratch_ref,
            state_commit_scratch=state_commit_scratch_ref,
            decode_read_semaphores=decode_read_sems,
            decode_write_semaphore=decode_write_sem,
            prefill_semaphore=prefill_sem,
            decode_tokens=decode_tokens,
            recurrent_state_in=aliased_recurrent_state_ref,
            recurrent_state_out=recurrent_state_ref,
        ),
        grid=(total_blocks,),
        in_specs=in_specs,
        out_specs=out_specs,
    )

    pipeline_func(
        mixed_qkv_ref,
        mixed_qkv_ref,
        a_raw_ref,
        a_raw_ref,
        b_raw_ref,
        b_raw_ref,
        a_log_ref,
        dt_bias_ref,
        output_ref,
        output_ref,
        scratches=[
            schedule_table_ref,
            state_indices_ref,
            has_initial_state_ref,
        ],
    )

  pl.run_scoped(
      # TODO: Move this to outer pallas call and get rid of run_scoped
      _run_with_scratch,
      pltpu.VMEM(
          (2, n_v, d_k, d_v), jnp.float32
      ),  # prefill_scratch (double buffered)
      pltpu.VMEM((1, n_v, d_k, d_v), jnp.float32),  # decode_state_scratch
      pltpu.VMEM(
          (1, n_v, d_k, d_v), recurrent_state_ref.dtype
      ),  # state_commit_scratch
      pltpu.VMEM((BT, n_v * d_v), mixed_qkv_ref.dtype),  # decode_output_scratch
      pltpu.SemaphoreType.DMA((1,)),  # decode_read_semaphores
      pltpu.SemaphoreType.DMA((1,)),  # decode_write_semaphore
      pltpu.SemaphoreType.DMA((2,)),  # prefill_semaphore
  )


@functools.partial(
    jax.jit,
    static_argnames=[
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "chunk_size",
        "BT",
        "use_qk_norm_in_gdn",
    ],
)
def recurrent_scan(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 128,
    BT: int = 128,
    use_qk_norm_in_gdn: bool = True,
    has_initial_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Fused recurrent scan kernel for GDN on TPU v7.

  Args:
    mixed_qkv: jax.Array of shape [num_tokens, 2 * n_kq * d_k + n_v * d_v].
      Packed Query, Key, and Value tokens.
    b: jax.Array of shape [num_tokens, n_v]. Input for beta gate.
    a: jax.Array of shape [num_tokens, n_v]. Input for g gate.
    recurrent_state: jax.Array of shape [max_reqs, n_v, d_k, d_v]. Current
      recurrent states.
    A_log: jax.Array of shape [n_v]. Log of parameter A.
    dt_bias: jax.Array of shape [n_v]. Bias for dt.
    query_start_loc: jax.Array of shape [num_requests + 1]. Start indices of
      each request in mixed_qkv.
    state_indices: jax.Array of shape [num_requests] or larger. Mapping from
      request ID to state index.
    distribution: jax.Array of shape [2]. Contains [decode_tokens,
      total_tokens].
    n_kq: Number of query/key heads.
    n_v: Number of value heads.
    d_k: Dimension of query/key features.
    d_v: Dimension of value features.
    chunk_size: Block size for processing (default 128).
    BT: Block size for decode requests (default 128).
    use_qk_norm_in_gdn: Whether to use QK normalization.

  Returns:
    A tuple containing:
      - Updated recurrent state of shape [max_reqs, n_v, d_k, d_v].
      - The mixed_qkv array of shape [num_tokens, 2 * n_kq * d_k + n_v * d_v].
  """
  if has_initial_state is None:
    has_initial_state = jnp.zeros(state_indices.shape[0], dtype=jnp.int32)
  else:
    has_initial_state = has_initial_state.astype(jnp.int32)

  num_tokens = mixed_qkv.shape[0]
  tpu_info = pltpu.get_tpu_info()
  sublanesize = 4 // mixed_qkv.itemsize * tpu_info.num_sublanes

  # Pad token dimension so invalid pipeline steps DMA into a safe sink area.
  # Sink offset must be aligned to sublanesize for Mosaic tile compatibility.
  block_size = max(chunk_size, BT)
  sink_offset = ((num_tokens + sublanesize - 1) // sublanesize) * sublanesize
  pad_rows = sink_offset + block_size - num_tokens
  mixed_qkv = jnp.pad(mixed_qkv, ((0, pad_rows), (0, 0)))

  # Pad raw a and b to (num_tokens + pad_rows, 128) for sublanes
  a_padded = jnp.pad(a, ((0, pad_rows), (0, 128 - n_v)))
  b_padded = jnp.pad(b, ((0, pad_rows), (0, 128 - n_v)))

  # decode_tokens: scalar, number of decode tokens.
  # Assuming length 1 per decode request, this is also the number of decode
  # requests.
  decode_tokens = distribution[0]
  schedule_table, total_blocks = (
      compute_schedule_table_v2.compute_schedule_table_v2(
          query_start_loc,
          decode_tokens,
          distribution[2],
          num_tokens,
          chunk_size,
          BT,
          alignment=sublanesize,
      )
  )

  # sublane,128
  decode_tokens_arr = jnp.expand_dims(decode_tokens, 0)
  total_blocks_arr = jnp.expand_dims(total_blocks, 0)

  grid_spec = pl.GridSpec(
      grid=(1,),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(block_shape=(1,), index_map=lambda _: (0,)),
          pl.BlockSpec(block_shape=(1,), index_map=lambda _: (0,)),
      ],
      out_specs=[
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
  )

  updated_recurrent_state, output_padded = pl.pallas_call(
      functools.partial(
          fused_kernel,
          C=chunk_size,
          BT=BT,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
          use_qk_norm_in_gdn=use_qk_norm_in_gdn,
          sublanesize=sublanesize,
      ),
      out_shape=(
          jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
          jax.ShapeDtypeStruct(
              (sink_offset + block_size, n_v * d_v), mixed_qkv.dtype
          ),
      ),
      grid_spec=grid_spec,
      input_output_aliases={1: 0},
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
  )(
      mixed_qkv,
      recurrent_state,
      state_indices,
      has_initial_state,
      a_padded,
      b_padded,
      A_log,
      dt_bias,
      schedule_table,
      decode_tokens_arr,
      total_blocks_arr,
  )
  return updated_recurrent_state, output_padded[:num_tokens]


# ==============================================================================
# Inlined fused_gdn_kernel_common.py
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
"""Shared input validation for fused GDN kernels."""



from jax._src import dtypes
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def validate_gdn_inputs(
    q,
    k,
    v,
    g,
    initial_state,
    state_indices,
    *,
    b=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
):
  """Validate shapes, dtypes, and TPU alignment for fused GDN kernels.

  Args:
      q: ``[T, H_qk, K]``.
      k: ``[T, H_qk, K]``.
      v: ``[T, H_v, V]``.
      g: ``[T, H_v, K]`` float32.
      initial_state: ``[num_states, H_v, K, V]`` float32.
      state_indices: ``[max_num_req]`` int32.
      b: ``[T, H_v, num_lanes]`` or ``None``.
      use_gate_in_kernel: Whether gate transformation is applied inside kernel.
      A_log: ``[H_v, num_lanes]`` float32 or ``None``.
      dt_bias: ``[H_v, num_lanes]`` float32 or ``None``.

  Returns:
      ``(T, H_qk, H_v, K, V, dtype, num_states, num_lanes, packing)``.
  """
  T, H_qk, K = q.shape
  H_v = v.shape[1]
  V = v.shape[2]
  dtype = q.dtype
  num_states = initial_state.shape[0]
  num_lanes = pltpu.get_tpu_info().num_lanes
  packing = 32 // dtypes.itemsize_bits(dtype)

  # Shape checks
  if k.shape != (T, H_qk, K):
    raise ValueError(f"k shape {k.shape} != q shape {q.shape}")
  if H_v % H_qk != 0:
    raise ValueError(f"H_v={H_v} must be a multiple of H_qk={H_qk}")
  if v.shape != (T, H_v, V):
    raise ValueError(f"v shape {v.shape} must be [{T}, {H_v}, {V}]")
  if g.shape != (T, H_v, K):
    raise ValueError(f"g shape {g.shape} must be [{T}, {H_v}, {K}]")
  if initial_state.shape[1:] != (H_v, K, V):
    raise ValueError(
        f"initial_state trailing dims {initial_state.shape[1:]} "
        f"must be ({H_v}, {K}, {V})"
    )
  if b is not None and (b.ndim != 3 or b.shape[0] != T or b.shape[1] != H_v):
    raise ValueError(f"b shape {b.shape} must be [{T}, {H_v}, ...]")

  # TPU alignment
  if K % num_lanes != 0 or V % num_lanes != 0:
    raise ValueError(f"K={K}, V={V} must be multiples of {num_lanes}")
  if H_qk % packing != 0:
    raise ValueError(f"H_qk={H_qk} must be a multiple of packing={packing}")
  if H_v % packing != 0:
    raise ValueError(f"H_v={H_v} must be a multiple of packing={packing}")

  # Dtype checks
  if k.dtype != dtype or v.dtype != dtype:
    raise ValueError(
        f"q/k/v must share the same dtype, got q={dtype}, "
        f"k={k.dtype}, v={v.dtype}"
    )
  if g.dtype != jnp.float32:
    raise ValueError(f"g must be float32, got {g.dtype}")
  if initial_state.dtype not in (jnp.float32, jnp.bfloat16, jnp.float16):
    raise ValueError(
        "initial_state must be float32, bfloat16, or float16, "
        f"got {initial_state.dtype}"
    )
  if state_indices.dtype != jnp.int32:
    raise ValueError(f"state_indices must be int32, got {state_indices.dtype}")

  # Gate-in-kernel checks
  if use_gate_in_kernel:
    if A_log is None:
      raise ValueError("A_log is required when use_gate_in_kernel=True")
    if dt_bias is not None and (dt_bias.ndim != 2 or dt_bias.shape[0] != H_v):
      raise ValueError(f"dt_bias shape {dt_bias.shape} must be [{H_v}, ...]")
    if dt_bias is not None and dt_bias.dtype != jnp.float32:
      raise ValueError(f"dt_bias must be float32, got {dt_bias.dtype}")

  return T, H_qk, H_v, K, V, dtype, num_states, num_lanes, packing


# ==============================================================================
# Inlined fused_gdn_recurrent_kernel.py
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
"""Fused recurrent GDN forward kernel for TPU."""



import dataclasses
import functools

import jax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# inlined validate_gdn_inputs


def get_default_recurrent_block_sizes(
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    dtype,
    use_gate_in_kernel: bool,
    has_dt_bias: bool,
    vmem_bytes_limit: int,
    state_dtype=jnp.float32,
) -> int:
  """Choose bt to maximize VMEM utilization.

  The recurrent kernel uses a fixed ``(2, H_v, K, V)`` state double
  buffer of ``state_dtype`` regardless of bt.  Only pipeline tiles scale with
  bt.
  """
  ibits = dtypes.itemsize_bits(dtype)
  sbits = dtypes.itemsize_bits(state_dtype)

  # Fixed (in bits): h_bufs (2, H_v, K, V) of state_dtype — always 2 buffers
  num_lanes = pltpu.get_tpu_info().num_lanes
  fixed_bits = 2 * H_v * K * V * sbits
  if use_gate_in_kernel:
    fixed_bits += 2 * H_v * num_lanes * 32  # a_log: (H_v, num_lanes) f32
  if has_dt_bias:
    fixed_bits += 2 * H_v * num_lanes * 32  # dt_bias: (H_v, num_lanes) f32

  # bt-proportional (in bits): pipeline tiles (×2 for emit_pipeline double buffering)
  #   q(bt,H_qk,K) + k(bt,H_qk,K)           -> 2·H_qk·K·ibits
  #   g(bt,H_v,K) float32                     -> H_v·K·32
  #   v(bt,H_v,V) + o(bt,H_v,V)              -> 2·H_v·V·ibits
  #   b(bt,H_v,num_lanes)                     -> H_v·num_lanes·ibits
  per_bt_bits = 2 * (
      2 * H_qk * K * ibits
      + H_v * K * 32
      + 2 * H_v * V * ibits
      + H_v * num_lanes * ibits
  )

  bt = max(1, (vmem_bytes_limit * 8 - fixed_bits) // per_bt_bits)
  # Round down to nearest power of 2
  return 1 << (bt.bit_length() - 1)


# Backwards compatibility alias
get_default_block_sizes = get_default_recurrent_block_sizes


# ── Metadata ──


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNChunkIndices:
  num_blocks: jax.Array  # [1] int32
  block_id_to_seq_idx: jax.Array  # [max_num_blocks + 1] int32 (sentinel at end)
  block_id_to_t_offset: jax.Array  # [max_num_blocks + 1] int32


@jax.named_scope("calculate_chunk_indices")
def calculate_chunk_indices(cu_seqlens, distribution, max_num_blocks, bt: int):
  """Pre-compute per-block metadata as a standalone Pallas kernel.

  Iterates over sequences and splits each into BT-sized work
  items.  A sequence boundary that falls mid-block creates two work
  items for that block (one per sequence).

  Returns a GDNChunkIndices with num_blocks, block_id_to_seq_idx,
  block_id_to_t_offset.
  """

  def _kernel(
      cu_seqlens_ref,
      distribution_ref,
      meta_out,
      *,
      bt: int,
  ):
    seq_start = distribution_ref[0]
    seq_end = distribution_ref[1]
    n_seqs = seq_end - seq_start

    @jax.named_scope("inner_block_loop")
    def inner_block_loop(blk_rel, carry, *, seq_idx, eos):
      num_blocks, t_cursor = carry
      block_id = num_blocks + blk_rel

      t_start = t_cursor + blk_rel * bt
      t_end = jnp.minimum(t_start + bt, eos)

      meta_out.block_id_to_seq_idx[block_id] = seq_idx
      meta_out.block_id_to_t_offset[block_id] = t_start
      meta_out.block_id_to_t_offset[block_id + 1] = t_end

      return num_blocks, t_cursor

    @jax.named_scope("outer_seq_loop")
    def outer_seq_loop(seq_rel, carry):
      num_blocks, t_cursor = carry
      seq_idx = seq_start + seq_rel
      eos = cu_seqlens_ref[seq_idx + 1]

      seq_len_from_cursor = eos - t_cursor
      n_seq_blocks = pl.cdiv(seq_len_from_cursor, bt)

      loop_fn = functools.partial(
          inner_block_loop,
          seq_idx=seq_idx,
          eos=eos,
      )
      jax.lax.fori_loop(0, n_seq_blocks, loop_fn, (num_blocks, t_cursor))

      return num_blocks + n_seq_blocks, eos

    first_token = cu_seqlens_ref[seq_start]
    num_blocks, _ = jax.lax.fori_loop(
        0,
        n_seqs,
        outer_seq_loop,
        (jnp.int32(0), first_token),
    )
    # Sentinel for look-ahead: block_id+1 reads -1 past the last block
    meta_out.block_id_to_seq_idx[num_blocks] = jnp.int32(-1)
    meta_out.num_blocks[0] = num_blocks

  smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
  meta = pl.pallas_call(
      functools.partial(_kernel, bt=bt),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          out_specs=GDNChunkIndices(
              num_blocks=smem_spec,
              block_id_to_seq_idx=smem_spec,
              block_id_to_t_offset=smem_spec,
          ),
          grid=(1,),
      ),
      out_shape=GDNChunkIndices(
          num_blocks=jax.ShapeDtypeStruct((1,), jnp.int32),
          block_id_to_seq_idx=jax.ShapeDtypeStruct(
              (max_num_blocks + 1,), jnp.int32
          ),
          block_id_to_t_offset=jax.ShapeDtypeStruct(
              (max_num_blocks + 1,), jnp.int32
          ),
      ),
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
  )(cu_seqlens, distribution)

  return meta


# ── Index maps ──


class _MetadataIndexMaps:
  """Index maps driven by pre-computed metadata arrays."""

  def __init__(self, meta: GDNChunkIndices):
    self.meta = meta

  def token_map(self, block_id):
    t_start = self.meta.block_id_to_t_offset[block_id]
    t_end = self.meta.block_id_to_t_offset[block_id + 1]
    t_size = t_end - t_start
    return (pl.ds(t_start, t_size), 0, 0)


# ── Outer kernel ──


def _recurrent_gdn_main(
    meta,  # GDNChunkIndices (SMEM)
    q_hbm,  # [T, H_qk, K]
    k_hbm,  # [T, H_qk, K]
    v_hbm,  # [T, H_v, V]
    g_hbm,  # [T, H_v, K]
    b_hbm,  # [T, H_v, num_lanes]
    state_indices_ref,  # [max_num_req] int32 (SMEM)
    a_log_hbm,  # [H_v, num_lanes] or None
    dt_bias_hbm,  # [H_v, num_lanes] or None
    _state_init_ref,  # [num_states, H_v, K, V] aliased to state_hbm
    has_initial_state_ref,  # [max_num_req] int32 (SMEM); 0 = zero out h0
    o_hbm,  # [T, H_v, V]
    state_hbm,  # [num_states, H_v, K, V]
    h_bufs,  # [2, H_v, K, V] VMEM scratch (double buffer)
    h_load_sems,  # [2] DMA semaphores
    h_store_sems,  # [2] DMA semaphores
    *,
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    scale: float,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    apply_silu: bool,
    lower_bound: float | None,
    bt: int,
):
  num_blocks = meta.num_blocks[0]
  repeat_factor = H_v // H_qk
  # Build index maps from metadata
  idx_maps = _MetadataIndexMaps(meta)
  bounded_bt = pl.BoundedSlice(bt)

  qk_spec = pl.BlockSpec((bounded_bt, H_qk, K), idx_maps.token_map)
  g_spec = pl.BlockSpec((bounded_bt, H_v, K), idx_maps.token_map)
  v_spec = pl.BlockSpec((bounded_bt, H_v, V), idx_maps.token_map)
  o_spec = pl.BlockSpec((bounded_bt, H_v, V), idx_maps.token_map)
  if b_hbm is not None:
    b_last = b_hbm.shape[2]
    b_spec = pl.BlockSpec((bounded_bt, H_v, b_last), idx_maps.token_map)
  else:
    b_spec = None

  # ── Prologue: start h0 load for first sequence (don't wait) ──
  first_seq = meta.block_id_to_seq_idx[0]
  first_state_idx = state_indices_ref[first_seq]
  first_buf = first_seq % 2
  pltpu.make_async_copy(
      state_hbm.at[pl.ds(first_state_idx, 1), :, :, :],
      h_bufs.at[pl.ds(first_buf, 1), :, :, :],
      h_load_sems.at[first_buf],
  ).start()

  # ── Inner kernel ──
  def _inner_kernel_body(
      q_ref,  # [<=bt, H_qk, K]
      k_ref,  # [<=bt, H_qk, K]
      v_ref,  # [<=bt, H_v, V]
      g_ref,  # [<=bt, H_v, K]
      b_ref,  # [<=bt, H_v, num_lanes]
      a_log_ref,  # [H_v, num_lanes] or None
      dt_bias_ref,  # [H_v, num_lanes] or None
      o_ref,  # [<=bt, H_v, V]
      h_bufs_s,  # [2, H_v, K, V] VMEM scratch
      meta_s,  # GDNChunkIndices (SMEM)
      state_indices_s,  # [max_num_req] int32 (SMEM)
      has_initial_state_s,  # [max_num_req] int32 (SMEM)
      h_load_sems_s,  # [2] DMA semaphores
      h_store_sems_s,  # [2] DMA semaphores
  ):
    block_id = pl.program_id(0)
    seq_idx = meta_s.block_id_to_seq_idx[block_id]
    t_start = meta_s.block_id_to_t_offset[block_id]
    t_end = meta_s.block_id_to_t_offset[block_id + 1]
    block_len = t_end - t_start

    # Detect sequence start
    prev_seq_idx = meta_s.block_id_to_seq_idx[jnp.maximum(block_id - 1, 0)]
    is_new_seq = (block_id == 0) | (seq_idx != prev_seq_idx)

    # Look ahead: detect sequence end
    next_seq_idx = meta_s.block_id_to_seq_idx[block_id + 1]
    is_seq_end = seq_idx != next_seq_idx

    # Double-buffer index: alternate buffers per sequence
    buf_idx = seq_idx % 2
    safe_next_seq = jnp.maximum(next_seq_idx, 0)
    next_buf_idx = safe_next_seq % 2

    # Pool indices via state_indices
    state_idx = state_indices_s[seq_idx]
    safe_next_state_idx = state_indices_s[safe_next_seq]

    # ── Step 1: Prefetch next h0 & wait for current h0 load ──
    prefetch_cp = pltpu.make_async_copy(
        state_hbm.at[pl.ds(safe_next_state_idx, 1), :, :, :],
        h_bufs_s.at[pl.ds(next_buf_idx, 1), :, :, :],
        h_load_sems_s.at[next_buf_idx],
    )
    load_wait_cp = pltpu.make_async_copy(
        state_hbm.at[pl.ds(state_idx, 1), :, :, :],
        h_bufs_s.at[pl.ds(buf_idx, 1), :, :, :],
        h_load_sems_s.at[buf_idx],
    )

    @pl.when(is_seq_end & (next_seq_idx >= 0))
    def _prefetch():
      prefetch_cp.start()

    @pl.when(is_new_seq)
    def _wait_h0():
      load_wait_cp.wait()

    # If the request has no prior recurrent state (brand-new prefill
    # landing on a freshly-allocated mamba slot), the DMA-loaded h0
    # is stale data from a previous tenant. Overwrite with zeros so
    # the recurrent update starts from zero, mirroring the chunked
    # path's `init_states_for_seqs = jnp.where(has_initial_state,
    # ..., 0)` and GPU's `initial_state[~has_initial_state, ...] = 0`
    # in `gdn_linear_attn._forward_core`.
    has_init = has_initial_state_s[seq_idx]

    @pl.when(is_new_seq & (has_init == 0))
    def _zero_h0():
      h_bufs_s[buf_idx] = jnp.zeros((H_v, K, V), dtype=h_bufs_s.dtype)

    # ── Step 2: Compute ──
    h = h_bufs_s[buf_idx].astype(jnp.float32)

    if use_gate_in_kernel:
      a_val = jnp.exp(a_log_ref[:, 0].astype(jnp.float32))
      if dt_bias_ref is not None:
        dt_bias_tile = dt_bias_ref[...].astype(jnp.float32)  # [H_v, num_lanes]
        if K > dt_bias_tile.shape[-1]:
          dt_bias_val = jnp.concatenate(
              [dt_bias_tile] * (K // dt_bias_tile.shape[-1]), axis=-1
          )
        else:
          dt_bias_val = dt_bias_tile

    def step(local_t, h):
      q_t = q_ref[local_t].astype(jnp.float32)
      k_t = k_ref[local_t].astype(jnp.float32)
      v_t = v_ref[local_t].astype(jnp.float32)
      # Fused SiLU, matching the chunked path's `jax.nn.silu(qkv_chunk)`.
      if apply_silu:
        q_t = jax.nn.silu(q_t)
        k_t = jax.nn.silu(k_t)
        v_t = jax.nn.silu(v_t)
      g_t = g_ref[local_t].astype(jnp.float32)
      if b_ref is not None:
        b_tile = b_ref[local_t].astype(jnp.float32)  # [H_v, num_lanes]
        if V > b_tile.shape[-1]:
          beta_t = jax.nn.sigmoid(
              jnp.concatenate([b_tile] * (V // b_tile.shape[-1]), axis=-1)
          )  # [H_v, V]
        else:
          beta_t = jax.nn.sigmoid(b_tile)  # [H_v, num_lanes] (== [H_v, V])

      if use_qk_l2norm:
        q_t = q_t / jnp.sqrt(jnp.sum(q_t * q_t, axis=-1, keepdims=True) + 1e-6)
        k_t = k_t / jnp.sqrt(jnp.sum(k_t * k_t, axis=-1, keepdims=True) + 1e-6)
      q_t = q_t * scale

      # GQA: repeat q/k from H_qk to H_v heads
      if repeat_factor > 1:
        q_t = jnp.repeat(q_t, repeat_factor, axis=0)
        k_t = jnp.repeat(k_t, repeat_factor, axis=0)

      if use_gate_in_kernel:
        if dt_bias_ref is not None:
          g_t = g_t + dt_bias_val
        if lower_bound is not None:
          gk = lower_bound / (1.0 + jnp.exp(-(a_val[:, None] * g_t)))
        else:
          gk = -a_val[:, None] * jax.nn.softplus(g_t)
      else:
        gk = g_t

      # Same algebraic identity as fused_gdn_decode_kernel.py:
      # o = q @ h_pre + (q . k) * b_v
      # Lets MXU(o) and VPU(rank-1 update) run in parallel.
      h_pre = h * jnp.exp(gk[:, :, None])
      kh = jax.lax.dot_general(
          k_t.reshape(H_v, 1, K),
          h_pre,
          (((2,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      ).reshape(H_v, V)
      v_diff = v_t - kh
      b_v = beta_t * v_diff if b_ref is not None else v_diff

      o_step1 = jax.lax.dot_general(
          q_t.reshape(H_v, 1, K),
          h_pre,
          (((2,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      ).reshape(H_v, V)
      qk_dot = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
      o_t = o_step1 + qk_dot * b_v
      h = h_pre + k_t[:, :, None] * b_v[:, None, :]

      o_ref[local_t] = o_t.astype(o_ref.dtype)
      return h

    h = jax.lax.fori_loop(0, block_len, step, h, unroll=False)
    h_bufs_s[buf_idx] = h.astype(h_bufs_s.dtype)

    # ── Step 3: Wait prev store, start current store ──
    # Store updated state back at state_idx.
    store_cp = pltpu.make_async_copy(
        h_bufs_s.at[pl.ds(buf_idx, 1), :, :, :],
        state_hbm.at[pl.ds(state_idx, 1), :, :, :],
        h_store_sems_s.at[buf_idx],
    )
    # Wait for prev store from same buffer (S-2's store).
    # Skip for the first 2 sequences — no prior store on this sem.
    has_prev_same_buf = (seq_idx - first_seq) >= 2

    @pl.when(is_seq_end & has_prev_same_buf)
    def _wait_prev_store():
      store_cp.wait()

    # Start current store
    @pl.when(is_seq_end)
    def _start_store():
      store_cp.start()

  # Run pipeline — None specs/inputs are passed through as None refs
  if use_gate_in_kernel and a_log_hbm is not None:
    a_log_spec = pl.BlockSpec((H_v, a_log_hbm.shape[1]), lambda _: (0, 0))
  else:
    a_log_spec = None
  dt_bias_spec = (
      pl.BlockSpec((H_v, dt_bias_hbm.shape[1]), lambda _: (0, 0))
      if dt_bias_hbm is not None
      else None
  )

  pltpu.emit_pipeline(
      _inner_kernel_body,
      grid=(num_blocks,),
      in_specs=[
          qk_spec,
          qk_spec,
          v_spec,
          g_spec,
          b_spec,
          a_log_spec,
          dt_bias_spec,
      ],
      out_specs=o_spec,
  )(
      q_hbm,
      k_hbm,
      v_hbm,
      g_hbm,
      b_hbm,
      a_log_hbm,
      dt_bias_hbm,
      o_hbm,
      scratches=[
          h_bufs,
          meta,
          state_indices_ref,
          has_initial_state_ref,
          h_load_sems,
          h_store_sems,
      ],
  )

  # ── Epilogue: wait for outstanding stores ──
  last_seq = meta.block_id_to_seq_idx[jnp.maximum(num_blocks - 1, 0)]
  last_buf = last_seq % 2
  other_buf = 1 - last_buf
  # Always wait for last seq's store
  pltpu.make_async_copy(
      h_bufs.at[pl.ds(0, 1), :, :, :],
      state_hbm.at[pl.ds(0, 1), :, :, :],
      h_store_sems.at[last_buf],
  ).wait()
  # If >= 2 seqs, also wait for the other sem
  drain_other = pltpu.make_async_copy(
      h_bufs.at[pl.ds(0, 1), :, :, :],
      state_hbm.at[pl.ds(0, 1), :, :, :],
      h_store_sems.at[other_buf],
  )

  @pl.when(last_seq != first_seq)
  def _drain_other():
    drain_other.wait()


# ── Public API ──


def fused_recurrent_gdn(
    q,  # [T, H_qk, K]
    k,  # [T, H_qk, K]
    v,  # [T, H_v, V]
    cu_seqlens,  # [N+1] int32
    g,  # [T, H_v, K] float32
    initial_state,  # [num_states, H_v, K, V] float32
    state_indices,  # [max_num_req] int32
    b,  # [T, H_v, num_lanes] or None
    has_initial_state,  # [max_num_req] int32 (0/1)
    *,
    scale,  # float
    use_qk_l2norm,  # bool
    use_gate_in_kernel=False,  # bool
    apply_silu=False,  # bool
    A_log=None,  # [H_v, num_lanes] or None
    dt_bias=None,  # [H_v, num_lanes] or None
    lower_bound=None,  # float or None
    distribution,  # [2] int32
):
  """Run the pre-computed-metadata recurrent GDN pallas kernel.

  ``apply_silu`` folds the GDN activation into the kernel: q/k/v are
  read as raw (pre-activation) projections and SiLU is applied on load.
  Leave it ``False`` when the caller already applied the activation
  upstream (e.g. fused into a preceding causal conv1d).

  ``has_initial_state[i]`` indicates whether request ``i``'s recurrent
  slot already holds a valid prior state (continuation, prefix-cache
  hit) or whether the slot is freshly allocated and its contents must
  be treated as zeros for this call. Mirrors the chunked path's
  `init_states_for_seqs = jnp.where(has_initial_state, ..., 0)` and
  GPU's `initial_state[~has_initial_state, ...] = 0` in
  ``gdn_linear_attn._forward_core``. Pass an all-ones array if the
  caller wants the previous (no-op) behaviour.
  """
  T, H_qk, H_v, K, V, dtype, num_states, num_lanes, _ = validate_gdn_inputs(
      q,
      k,
      v,
      g,
      initial_state,
      state_indices,
      b=b,
      use_gate_in_kernel=use_gate_in_kernel,
      A_log=A_log,
      dt_bias=dt_bias,
  )
  max_num_req = cu_seqlens.shape[0] - 1

  vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
  bt = get_default_recurrent_block_sizes(
      H_qk,
      H_v,
      K,
      V,
      dtype,
      use_gate_in_kernel,
      dt_bias is not None,
      vmem_bytes_limit,
      state_dtype=initial_state.dtype,
  )

  # Worst case: cdiv(T, bt) base blocks + up to max_num_req-1 boundary splits
  max_num_blocks = (T + bt - 1) // bt + max_num_req - 1

  any_spec = pl.BlockSpec(memory_space=pl.ANY)
  smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

  o_shape = jax.ShapeDtypeStruct((T, H_v, V), dtype)
  state_shape = jax.ShapeDtypeStruct(
      (num_states, H_v, K, V), initial_state.dtype
  )

  meta = calculate_chunk_indices(cu_seqlens, distribution, max_num_blocks, bt)

  n_seqs = distribution[1] - distribution[0]
  grid_dim = jnp.where(n_seqs > 0, 1, 0)

  n_b = b is not None
  n_gate = (A_log is not None) + (dt_bias is not None)

  scope_name = f"recurrent_gdn-bt_{bt}"

  o, state = pl.pallas_call(
      functools.partial(
          _recurrent_gdn_main,
          H_qk=H_qk,
          H_v=H_v,
          K=K,
          V=V,
          scale=scale,
          use_qk_l2norm=use_qk_l2norm,
          use_gate_in_kernel=use_gate_in_kernel,
          apply_silu=apply_silu,
          lower_bound=lower_bound,
          bt=bt,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              GDNChunkIndices(
                  num_blocks=smem_spec,
                  block_id_to_seq_idx=smem_spec,
                  block_id_to_t_offset=smem_spec,
              ),
              *([any_spec] * 4),  # q, k, v, g
              any_spec if b is not None else None,  # b
              smem_spec,  # state_indices
              any_spec if A_log is not None else None,
              any_spec if dt_bias is not None else None,
              any_spec,  # state_init (= initial_state)
              smem_spec,  # has_initial_state
          ],
          out_specs=[any_spec, any_spec],
          grid=(grid_dim,),
          scratch_shapes=[
              # h_bufs match HBM dtype so the DMAs don't need conversion.
              # The compute path upcasts h_bufs to fp32 once per block
              # (h = h_bufs_s[buf_idx].astype(fp32) above) and carries it
              # as fp32 across the per-token fori_loop, so on-chip math is
              # fp32 regardless of HBM storage dtype.
              pltpu.VMEM(
                  (2, H_v, K, V), initial_state.dtype
              ),  # h_bufs (double buffer)
              pltpu.SemaphoreType.DMA((2,)),  # h_load_sems
              pltpu.SemaphoreType.DMA((2,)),  # h_store_sems
          ],
      ),
      # Aliases reference flat positional inputs. `meta` is a 3-leaf
      # pytree, so the absolute index of `v` is 5 (3 meta leaves + q, k)
      # and the absolute index of `initial_state` is 8 + n_b + n_gate.
      # `has_initial_state` is appended to the end and is not aliased.
      input_output_aliases={5: 0, 8 + n_b + n_gate: 1},
      out_shape=[o_shape, state_shape],
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
          vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
      ),
      name=scope_name,
  )(
      meta,
      q,
      k,
      v,
      g,
      b,
      state_indices,
      A_log,
      dt_bias,
      initial_state,
      has_initial_state,
  )

  return o, state


# ==============================================================================
# Inlined fused_gdn_decode_kernel.py
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
"""Fused recurrent GDN decoding kernel for TPU.

Processes ``bt`` decode tokens per pipeline step using ``emit_pipeline``
for q/k/v/g/b tiling, with bulk manual DMA for state load/store via
``state_indices``.
"""



import functools

import jax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# inlined validate_gdn_inputs


def get_default_decode_block_sizes(
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    dtype,
    use_gate_in_kernel: bool,
    has_dt_bias: bool,
    vmem_bytes_limit: int,
    state_dtype=jnp.float32,
) -> int:
  """Choose bt to maximize VMEM utilization within vmem_bytes_limit.

  Accounts for state scratch ``(bt, H_v, K, V)`` of ``state_dtype``, optional
  a_log / dt_bias, and bt-proportional tiles that ``emit_pipeline``
  double-buffers (q, k, v, g, b, o).
  """
  ibits = dtypes.itemsize_bits(dtype)
  sbits = dtypes.itemsize_bits(state_dtype)

  # Fixed (not bt-dependent), in bits
  num_lanes = pltpu.get_tpu_info().num_lanes
  fixed_bits = 0
  if use_gate_in_kernel:
    fixed_bits += 2 * H_v * num_lanes * 32  # a_log: (H_v, num_lanes) f32
  if has_dt_bias:
    fixed_bits += 2 * H_v * num_lanes * 32  # dt_bias: (H_v, num_lanes) f32

  # bt-proportional (in bits):
  #   state scratch: (2*bt, H_v, K, V) state_dtype (double buffer)
  #   pipeline tiles (×2 for emit_pipeline double buffering):
  #     q(bt,H_qk,K) + k(bt,H_qk,K)           -> 2·H_qk·K·ibits
  #     g(bt,H_v,K) float32                     -> H_v·K·32
  #     v(bt,H_v,V) + o(bt,H_v,V)              -> 2·H_v·V·ibits
  #     b(bt,H_v,num_lanes)                     -> H_v·num_lanes·ibits
  per_bt_bits = 2 * H_v * K * V * sbits + 2 * (
      2 * H_qk * K * ibits
      + H_v * K * 32
      + 2 * H_v * V * ibits
      + H_v * num_lanes * ibits
  )

  bt = max(1, (vmem_bytes_limit * 8 - fixed_bits) // per_bt_bits)
  # Round down to nearest power of 2
  return 1 << (bt.bit_length() - 1)


# ── Outer kernel ──────────────────────────────────────────────────────


def _decode_kernel_main(
    q_hbm,  # [T, H_qk, K]
    k_hbm,  # [T, H_qk, K]
    v_hbm,  # [T, H_v, V]
    g_hbm,  # [T, H_v, K] float32
    b_hbm,  # [T, H_v, num_lanes]
    state_indices_ref,  # [max_num_req] int32 (SMEM)
    a_log_hbm,  # [H_v, num_lanes] or None
    dt_bias_hbm,  # [H_v, num_lanes] or None
    distribution_ref,  # [2] int32 (SMEM)
    _state_init_ref,  # [num_states, H_v, K, V] aliased to state_hbm
    o_hbm,  # [T, H_v, V]
    state_hbm,  # [num_states, H_v, K, V]
    h_bufs,  # [2, bt, H_v, K, V] VMEM scratch
    h_load_sems,
    h_store_sems,
    *,
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    scale: float,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    apply_silu: bool,
    lower_bound: float | None,
    bt: int,
):
  decode_end = distribution_ref[0]
  nb_t = (decode_end + bt - 1) // bt
  repeat_factor = H_v // H_qk

  bounded_bt = pl.BoundedSlice(bt)

  def token_map(i):
    t_start = i * bt
    t_size = jnp.minimum(bt, decode_end - t_start)
    return (pl.ds(t_start, t_size), 0, 0)

  qk_spec = pl.BlockSpec((bounded_bt, H_qk, K), token_map)
  g_spec = pl.BlockSpec((bounded_bt, H_v, K), token_map)
  v_spec = pl.BlockSpec((bounded_bt, H_v, V), token_map)
  if b_hbm is not None:
    b_last = b_hbm.shape[2]
    b_spec = pl.BlockSpec((bounded_bt, H_v, b_last), token_map)
  else:
    b_spec = None

  if use_gate_in_kernel and a_log_hbm is not None:
    a_log_spec = pl.BlockSpec((H_v, a_log_hbm.shape[1]), lambda _: (0, 0))
  else:
    a_log_spec = None
  dt_bias_spec = (
      pl.BlockSpec((H_v, dt_bias_hbm.shape[1]), lambda _: (0, 0))
      if dt_bias_hbm is not None
      else None
  )

  # ── Prologue: start loading first bt-block's states ──
  for i_t in range(bt):

    @pl.when(i_t < decode_end)
    def _first_load():
      si = state_indices_ref[i_t]
      pltpu.make_async_copy(
          state_hbm.at[pl.ds(si, 1), :, :, :],
          h_bufs.at[0, pl.ds(i_t, 1), :, :, :],
          h_load_sems.at[0],
      ).start()

  # ── Inner kernel (runs per bt-block) ──
  def _inner_kernel(
      q_ref,  # [<=bt, H_qk, K]
      k_ref,  # [<=bt, H_qk, K]
      v_ref,  # [<=bt, H_v, V]
      g_ref,  # [<=bt, H_v, K]
      b_ref,  # [<=bt, H_v, num_lanes]
      a_log_ref,  # [H_v, num_lanes] or None
      dt_bias_ref,  # [H_v, num_lanes] or None
      o_ref,  # [<=bt, H_v, V]
      h_bufs_s,
      state_indices_s,  # [max_num_req] int32 (SMEM)
      h_load_sems_s,
      h_store_sems_s,
  ):
    block_id = pl.program_id(0)
    t_start = block_id * bt
    block_len = jnp.minimum(bt, decode_end - t_start)
    buf_idx = block_id % 2
    next_buf_idx = (block_id + 1) % 2

    if use_gate_in_kernel:
      a_val = jnp.exp(a_log_ref[:, 0].astype(jnp.float32))
      if dt_bias_ref is not None:
        dt_bias_tile = dt_bias_ref[...].astype(jnp.float32)  # [H_v, num_lanes]
        if K > dt_bias_tile.shape[-1]:
          dt_bias_val = jnp.concatenate(
              [dt_bias_tile] * (K // dt_bias_tile.shape[-1]), axis=-1
          )
        else:
          dt_bias_val = dt_bias_tile

    # ── Step 1: Prefetch next bt-block's states ──
    next_t_start = t_start + bt
    next_block_len = jnp.maximum(jnp.minimum(bt, decode_end - next_t_start), 0)
    for i_t in range(bt):

      @pl.when(i_t < next_block_len)
      def _prefetch():
        next_si = state_indices_s[next_t_start + i_t]
        pltpu.make_async_copy(
            state_hbm.at[pl.ds(next_si, 1), :, :, :],
            h_bufs_s.at[next_buf_idx, pl.ds(i_t, 1), :, :, :],
            h_load_sems_s.at[next_buf_idx],
        ).start()

    # ── Step 2: Wait for current bt-block's state loads ──
    pltpu.make_async_copy(
        state_hbm.at[pl.ds(0, block_len), :, :, :],
        h_bufs_s.at[buf_idx, pl.ds(0, block_len), :, :, :],
        h_load_sems_s.at[buf_idx],
    ).wait()

    # ── Step 3: Compute ──
    for i_t in range(bt):

      @pl.when(i_t < block_len)
      def _process_token():
        h0 = h_bufs_s[buf_idx, i_t].astype(jnp.float32)
        q_t = q_ref[i_t].astype(jnp.float32)
        k_t = k_ref[i_t].astype(jnp.float32)
        v_t = v_ref[i_t].astype(jnp.float32)
        # Fused SiLU, matching the chunked path's `jax.nn.silu(qkv_row)`.
        if apply_silu:
          q_t = jax.nn.silu(q_t)
          k_t = jax.nn.silu(k_t)
          v_t = jax.nn.silu(v_t)
        g_t = g_ref[i_t].astype(jnp.float32)
        if b_ref is not None:
          b_tile = b_ref[i_t].astype(jnp.float32)  # [H_v, num_lanes]
          if V > b_tile.shape[-1]:
            beta_t = jax.nn.sigmoid(
                jnp.concatenate([b_tile] * (V // b_tile.shape[-1]), axis=-1)
            )  # [H_v, V]
          else:
            beta_t = jax.nn.sigmoid(b_tile)  # [H_v, num_lanes] (== [H_v, V])

        if use_qk_l2norm:
          q_t = q_t / jnp.sqrt(
              jnp.sum(q_t * q_t, axis=-1, keepdims=True) + 1e-6
          )
          k_t = k_t / jnp.sqrt(
              jnp.sum(k_t * k_t, axis=-1, keepdims=True) + 1e-6
          )
        q_t = q_t * scale

        # GQA: repeat q/k from H_qk to H_v heads
        if repeat_factor > 1:
          q_t = jnp.repeat(q_t, repeat_factor, axis=0)
          k_t = jnp.repeat(k_t, repeat_factor, axis=0)

        if use_gate_in_kernel:
          g_val = g_t
          if dt_bias_ref is not None:
            g_val = g_val + dt_bias_val
          if lower_bound is not None:
            gk = lower_bound / (1.0 + jnp.exp(-(a_val[:, None] * g_val)))
          else:
            gk = -a_val[:, None] * jax.nn.softplus(g_val)
        else:
          gk = g_t

        h_pre = h0 * jnp.exp(gk[:, :, None])
        kh = jax.lax.dot_general(
            k_t.reshape(H_v, 1, K),
            h_pre,
            (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).reshape(H_v, V)
        v_diff = v_t - kh
        b_v = beta_t * v_diff if b_ref is not None else v_diff

        # Algebraic identity to skip the post-rank-1-update matmul:
        # o = q @ (h_pre + outer(k, b_v))
        #   = q @ h_pre + (q . k) * b_v
        # (q . k)[h] is a per-head scalar, so the second term is a
        # cheap HV*V scaled-add instead of a full HV*K*V matmul.
        # This lets MXU(o) and VPU(rank-1 update) run in parallel.
        o_step1 = jax.lax.dot_general(
            q_t.reshape(H_v, 1, K),
            h_pre,
            (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).reshape(H_v, V)
        qk_dot = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
        o_t = o_step1 + qk_dot * b_v
        h_new = h_pre + k_t[:, :, None] * b_v[:, None, :]

        o_ref[i_t] = o_t.astype(o_ref.dtype)
        h_bufs_s[buf_idx, i_t] = h_new.astype(h_bufs_s.dtype)

    # ── Step 4: Wait for stores from 2 blocks ago (same buffer set) ──
    prev_t_start = jnp.maximum((block_id - 2) * bt, 0)
    prev_block_len = jnp.where(
        block_id >= 2,
        jnp.minimum(bt, decode_end - prev_t_start),
        0,
    )

    @pl.when(prev_block_len > 0)
    def _wait_prev_store():
      pltpu.make_async_copy(
          h_bufs_s.at[buf_idx, pl.ds(0, prev_block_len), :, :, :],
          state_hbm.at[pl.ds(0, prev_block_len), :, :, :],
          h_store_sems_s.at[buf_idx],
      ).wait()

    # ── Step 5: Start storing current bt-block's states ──
    for i_t in range(bt):

      @pl.when(i_t < block_len)
      def _start_store():
        si = state_indices_s[t_start + i_t]
        pltpu.make_async_copy(
            h_bufs_s.at[buf_idx, pl.ds(i_t, 1), :, :, :],
            state_hbm.at[pl.ds(si, 1), :, :, :],
            h_store_sems_s.at[buf_idx],
        ).start()

  pltpu.emit_pipeline(
      _inner_kernel,
      grid=(nb_t,),
      in_specs=[
          qk_spec,
          qk_spec,
          v_spec,
          g_spec,
          b_spec,
          a_log_spec,
          dt_bias_spec,
      ],
      out_specs=v_spec,
  )(
      q_hbm,
      k_hbm,
      v_hbm,
      g_hbm,
      b_hbm,
      a_log_hbm,
      dt_bias_hbm,
      o_hbm,
      scratches=[h_bufs, state_indices_ref, h_load_sems, h_store_sems],
  )

  # ── Epilogue: drain outstanding stores ──
  last_buf_idx = (nb_t - 1) % 2
  other_buf_idx = nb_t % 2
  last_block_len = jnp.minimum(bt, decode_end - (nb_t - 1) * bt)
  pltpu.make_async_copy(
      h_bufs.at[last_buf_idx, pl.ds(0, last_block_len), :, :, :],
      state_hbm.at[pl.ds(0, last_block_len), :, :, :],
      h_store_sems.at[last_buf_idx],
  ).wait()

  other_block_len = jnp.where(
      nb_t >= 2,
      jnp.minimum(bt, decode_end - (nb_t - 2) * bt),
      0,
  )

  @pl.when(other_block_len > 0)
  def _drain_other():
    pltpu.make_async_copy(
        h_bufs.at[other_buf_idx, pl.ds(0, other_block_len), :, :, :],
        state_hbm.at[pl.ds(0, other_block_len), :, :, :],
        h_store_sems.at[other_buf_idx],
    ).wait()


# ── Public API ───────────────────────────────────────────────────────


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "apply_silu",
        "lower_bound",
    ],
)
def fused_decoding_gdn(
    q: jax.Array,  # [T, H_qk, K]
    k: jax.Array,  # [T, H_qk, K]
    v: jax.Array,  # [T, H_v, V]
    g: jax.Array,  # [T, H_v, K] float32
    initial_state: jax.Array,  # [num_states, H_v, K, V] float32
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [2] int32
    b: jax.Array | None,  # [T, H_v, num_lanes] or None
    *,
    scale: float,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    apply_silu: bool = False,
    A_log: jax.Array | None = None,  # [H_v, num_lanes] float32 or None
    dt_bias: jax.Array | None = None,  # [H_v, num_lanes] float32 or None
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array]:
  r"""Fused recurrent GDN single-step decode.

  ``apply_silu`` folds the GDN activation into the kernel: q/k/v are
  read as raw (pre-activation) projections and SiLU is applied on load.
  Leave it ``False`` when the caller already applied the activation
  upstream (e.g. fused into a preceding causal conv1d).

  Args:
      q: Queries ``[T, H_qk, K]``.
      k: Keys ``[T, H_qk, K]``.
      v: Values ``[T, H_v, V]``.
      g: Per-key gating ``[T, H_v, K]``, float32.
      initial_state: State cache ``[num_states, H_v, K, V]`` float32.
      state_indices: ``i32[max_num_req]`` — indices into the state cache.
      distribution: ``i32[2]`` — ``(decode_end, total)``.
      b: Raw betas ``[T, H_v, num_lanes]`` (sigmoid applied inside kernel).
      scale: Scale factor.
      use_qk_l2norm_in_kernel: L2-normalize q, k inside the kernel.
      use_gate_in_kernel: Apply gate transformation inside kernel.
      apply_silu: Apply the SiLU activation to q/k/v inside the kernel.
        Set this when q/k/v are raw projections; leave it ``False`` when
        the activation was already applied upstream.
      A_log: Per-head log gate ``[H_v, num_lanes]`` float32.
      dt_bias: Per-head bias ``[H_v, num_lanes]`` float32.
      lower_bound: If set, use sigmoid gate instead of softplus.

  Returns:
      ``(o, updated_state)`` — *o* is ``[T, H_v, V]``,
      *updated_state* is ``[num_states, H_v, K, V]``.
  """
  T, H_qk, H_v, K, V, dtype, num_states, num_lanes, _ = validate_gdn_inputs(
      q,
      k,
      v,
      g,
      initial_state,
      state_indices,
      b=b,
      use_gate_in_kernel=use_gate_in_kernel,
      A_log=A_log,
      dt_bias=dt_bias,
  )

  vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
  bt = get_default_decode_block_sizes(
      H_qk,
      H_v,
      K,
      V,
      dtype,
      use_gate_in_kernel,
      dt_bias is not None,
      vmem_bytes_limit,
      state_dtype=initial_state.dtype,
  )

  any_spec = pl.BlockSpec(memory_space=pl.ANY)
  smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

  decode_end = distribution[0]
  grid_dim = jnp.where(decode_end > 0, 1, 0)

  n_b = b is not None
  n_gate = (A_log is not None) + (dt_bias is not None)

  scope_name = f"decoding_gdn-bt_{bt}"

  o, state = pl.pallas_call(
      functools.partial(
          _decode_kernel_main,
          H_qk=H_qk,
          H_v=H_v,
          K=K,
          V=V,
          scale=scale,
          use_qk_l2norm=use_qk_l2norm_in_kernel,
          use_gate_in_kernel=use_gate_in_kernel,
          apply_silu=apply_silu,
          lower_bound=lower_bound,
          bt=bt,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              *([any_spec] * 4),  # q, k, v, g
              any_spec if b is not None else None,  # b
              smem_spec,  # state_indices
              any_spec if A_log is not None else None,
              any_spec if dt_bias is not None else None,
              smem_spec,  # distribution
              any_spec,  # state_init
          ],
          out_specs=[any_spec, any_spec],
          grid=(grid_dim,),
          scratch_shapes=[
              # h_bufs match HBM dtype so the DMAs don't need conversion.
              # The per-token compute path upcasts to fp32 on each load
              # (see h0 = h_bufs_s[..., i_t].astype(fp32) above), so on-chip
              # math is fp32 regardless of HBM storage dtype.
              pltpu.VMEM(
                  (2, bt, H_v, K, V), initial_state.dtype
              ),  # h_bufs (double buffer)
              pltpu.SemaphoreType.DMA((2,)),  # h_load_sems
              pltpu.SemaphoreType.DMA((2,)),  # h_store_sems
          ],
      ),
      input_output_aliases={2: 0, 6 + n_b + n_gate: 1},
      out_shape=[
          jax.ShapeDtypeStruct((T, H_v, V), dtype),
          jax.ShapeDtypeStruct((num_states, H_v, K, V), initial_state.dtype),
      ],
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
          vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
      ),
      name=scope_name,
  )(
      q,
      k,
      v,
      g,
      b,
      state_indices,
      A_log,
      dt_bias,
      distribution,
      initial_state,
  )

  return o, state


# ==============================================================================
# Inlined fused_gdn_kernel_wrapper.py
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
"""Fused GDN kernel wrapper — dispatch and public API."""



import functools

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# inlined
# inlined


def _dispatch_with_distribution(
    q,
    k,
    v,
    cu_seqlens,
    g,
    initial_state,
    state_indices,
    b,
    has_initial_state,
    *,
    scale,
    use_qk_l2norm,
    use_gate_in_kernel,
    apply_silu,
    A_log,
    dt_bias,
    lower_bound,
    distribution,
):
  """Dispatch to decode and recurrent kernels following the RPA pattern.

  Both kernels update the state cache in-place via ``input_output_aliases``.
  The decode kernel runs first, then its updated state and output are
  chained to the recurrent kernel.

  ``has_initial_state`` is consumed only by the recurrent kernel: decode
  tokens always have a valid prior state (a request must finish prefill
  before it can decode), so masking is unnecessary on the decode path.
  """
  # ── Decode kernel → updates state in-place ──
  # NOTE: pass `initial_state` through as-is. The kernels upcast to fp32
  # on VMEM load for compute precision; HBM storage stays at the array's
  # dtype. An `astype(jnp.float32)` here would materialize an fp32 copy
  # of a bf16 state and undo the storage win before the kernel runs.
  o_d, state_1 = fused_decoding_gdn(
      q,
      k,
      v,
      g.astype(jnp.float32),
      initial_state,
      state_indices,
      distribution,
      b,
      scale=scale,
      use_qk_l2norm_in_kernel=use_qk_l2norm,
      use_gate_in_kernel=use_gate_in_kernel,
      apply_silu=apply_silu,
      A_log=A_log,
      dt_bias=dt_bias,
      lower_bound=lower_bound,
  )

  # ── Recurrent kernel → updates state in-place ──
  # `o_d` arrives in the `v` slot: the decode kernel's output buffer
  # aliases `v`, so decode rows now hold decode outputs while
  # prefill/mixed rows still hold raw `v`. The recurrent grid starts at
  # `distribution[0]`, so it only ever reads the untouched rows — the
  # fused SiLU below is applied to real `v`, never to decode output.
  o_r, state_2 = fused_recurrent_gdn(
      q,
      k,
      o_d,
      cu_seqlens,
      g.astype(jnp.float32),
      state_1,
      state_indices,
      b,
      has_initial_state,
      scale=scale,
      use_qk_l2norm=use_qk_l2norm,
      use_gate_in_kernel=use_gate_in_kernel,
      apply_silu=apply_silu,
      A_log=A_log,
      dt_bias=dt_bias,
      lower_bound=lower_bound,
      distribution=distribution,
  )

  return o_r, state_2


# ── Public API ──


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "apply_silu",
        "lower_bound",
    ],
    donate_argnames=["v", "initial_state"],
)
def fused_gdn(
    q: jax.Array,  # [T, H_qk, K]
    k: jax.Array,  # [T, H_qk, K]
    v: jax.Array,  # [T, H_v, V]
    cu_seqlens: jax.Array,  # [max_num_req+1] int32
    g: jax.Array,  # [T, H_v, K] or [T, H_v]
    initial_state: jax.Array,  # [num_states, H_v, K, V]
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [2] int32
    b: jax.Array | None = None,  # [T, H_v] or None
    has_initial_state: jax.Array | None = None,  # [max_num_req] bool
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    apply_silu: bool = False,
    A_log: jax.Array | None = None,  # [H_v] float32 or None
    dt_bias: jax.Array | None = None,  # [H_v] float32 or None
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array]:
  r"""Fused recurrent GDN forward pass.

  Supports GQA: ``H_v`` (value heads from ``v``) can be a multiple of
  ``H_qk`` (query/key heads from ``q``/``k``).  The kernel repeats
  q/k internally.

  Args:
      q: Queries ``[T, H_qk, K]``.
      k: Keys ``[T, H_qk, K]``.
      v: Values ``[T, H_v, V]``.
      cu_seqlens: Cumulative sequence lengths ``[max_num_req+1]``.
      g: Gating ``[T, H_v, K]`` or ``[T, H_v]`` (broadcast to K).
      initial_state: State cache ``[num_states, H_v, K, V]``.
      state_indices: ``i32[max_num_req]`` — indices into the state cache.
      distribution: ``i32[2]`` — ``(decode_end, total)``.
      b: Raw betas ``[T, H_v]`` (sigmoid applied inside kernel). ``None`` means
        beta=1 (no beta gating).
      has_initial_state: Boolean tensor of shape ``[max_num_req]``. ``True``
        when the request's recurrent slot already holds a valid prior state
        (chunked-prefill continuation, prefix-cache hit, or running decode).
        ``False`` for brand-new prefills, whose slot is zeroed inside the
        recurrent kernel before the update so stale data from a previous tenant
        doesn't leak. ``None`` (default) is treated as all-True, preserving the
        pre-fix behaviour for callers that don't manage slot reuse.
      scale: Scale factor.  Default ``K ** -0.5``.
      use_qk_l2norm_in_kernel: L2-normalize q, k inside the kernel.
      use_gate_in_kernel: Apply gate transformation inside kernel.
      apply_silu: Apply the SiLU activation to q/k/v inside the kernel.
        Set this when q/k/v are raw projections; leave it ``False`` when
        the activation was already applied upstream.
      A_log: Per-head log gate ``[H_v]`` float32.
      dt_bias: Per-head bias ``[H_v]`` float32. Optional. Broadcast to ``[H_v,
        num_lanes]`` internally.
      lower_bound: If set, use sigmoid gate instead of softplus.

  Returns:
      ``(o, updated_state)`` — *o* is ``[T, H_v, V]``,
      *updated_state* is ``[num_states, H_v, K, V]`` with final states
      written back at the corresponding ``state_indices`` positions.
  """
  T, H_qk, K = q.shape
  H_v = v.shape[1]

  # Broadcast g from [T, H_v] to [T, H_v, K] if needed.
  if g.shape == (T, H_v):
    g = jnp.broadcast_to(g[..., None], (T, H_v, K))
  elif g.shape != (T, H_v, K):
    raise ValueError(
        f"g shape {g.shape} must be [{T}, {H_v}, {K}] or [{T}, {H_v}]"
    )

  # Validate pre-broadcast inputs.
  if b is not None and b.shape != (T, H_v):
    raise ValueError(f"b shape {b.shape} must be [{T}, {H_v}]")
  if A_log is not None and A_log.shape != (H_v,):
    raise ValueError(f"A_log shape {A_log.shape} must be [{H_v}]")
  if dt_bias is not None and dt_bias.shape != (H_v,):
    raise ValueError(f"dt_bias shape {dt_bias.shape} must be [{H_v}]")

  cu_seqlens = cu_seqlens.astype(jnp.int32)
  state_indices = state_indices.astype(jnp.int32)

  if scale is None:
    scale = K**-0.5
  num_lanes = pltpu.get_tpu_info().num_lanes
  if b is not None:
    b = jnp.broadcast_to(
        b[:, :, None], (T, H_v, num_lanes)
    )  # [T, H_v, num_lanes]
  if dt_bias is not None:
    dt_bias = jnp.broadcast_to(dt_bias[:, None], (H_v, num_lanes)).astype(
        jnp.float32
    )  # [H_v, num_lanes]
  distribution = distribution.astype(jnp.int32)

  if A_log is not None:
    A_log = jnp.broadcast_to(A_log[:, None], (H_v, num_lanes)).astype(
        jnp.float32
    )  # [H_v, num_lanes]

  # Public contract is Boolean (matching the chunked / ref impls);
  # cast to int32 here for SMEM compatibility — the recurrent kernel
  # checks `has_init == 0` to decide whether to zero h0. Default to
  # all-True (no masking), matching the pre-fix behaviour.
  max_num_req = state_indices.shape[0]
  if has_initial_state is None:
    has_initial_state = jnp.ones((max_num_req,), dtype=jnp.int32)
  else:
    has_initial_state = has_initial_state.astype(jnp.int32)

  o, state = _dispatch_with_distribution(
      q,
      k,
      v,
      cu_seqlens,
      g,
      initial_state,
      state_indices,
      b,
      has_initial_state,
      scale=scale,
      use_qk_l2norm=use_qk_l2norm_in_kernel,
      use_gate_in_kernel=use_gate_in_kernel,
      apply_silu=apply_silu,
      A_log=A_log,
      dt_bias=dt_bias,
      lower_bound=lower_bound,
      distribution=distribution,
  )

  return o, state


def ragged_gated_delta_rule(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state=None,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
):
  """Adapter matching the ragged_gated_delta_rule_{ref,chunked} interface.

  Internally reshapes inputs and delegates to :func:`fused_gdn`.

  Args:
      mixed_qkv: ``(num_tokens, 2*n_kq*d_k + n_v*d_v)`` post-conv, raw
        (pre-activation). SiLU is fused into the kernels via
        ``apply_silu=True``.
      b: ``(num_tokens, n_v)`` — raw beta (sigmoid applied in kernel).
      a: ``(num_tokens, n_v)`` — raw alpha (gate transform in kernel).
      recurrent_state: ``(num_states, n_v, d_k, d_v)``.
      A_log: ``(n_v,)`` float32.
      dt_bias: ``(n_v,)`` float32.
      query_start_loc: ``(num_seqs+1,)`` int32.
      state_indices: ``(num_seqs,)`` int32.
      distribution: ``(3,)`` int32 — ``(decode_end, prefill_end, mixed_end)``.
      has_initial_state: Optional Boolean tensor of shape ``(max_reqs,)``.
        ``True`` when the request's slot already holds a valid prior recurrent
        state; ``False`` for brand-new prefills (the recurrent kernel zeros h0
        for those slots so stale data from a reused mamba slot doesn't leak).
        ``None`` (default) is treated as all-True, preserving the pre-PR-#2408
        behaviour. Pass it when you want the same stale-slot guard the chunked
        and ref impls already enforce.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key dimension.
      d_v: Value dimension.

  Returns:
      ``(updated_recurrent_state, output)`` where
      *updated_recurrent_state* is ``(num_states, n_v, d_k, d_v)`` and
      *output* is ``(num_tokens, n_v*d_v)``.
  """
  num_tokens = mixed_qkv.shape[0]
  key_dim = n_kq * d_k

  q = mixed_qkv[..., :key_dim].reshape(num_tokens, n_kq, d_k)
  k = mixed_qkv[..., key_dim : key_dim * 2].reshape(num_tokens, n_kq, d_k)
  v = mixed_qkv[..., key_dim * 2 :].reshape(num_tokens, n_v, d_v)

  g = a

  # (decode_end, prefill_end, mixed_end) → (decode_end, total)
  fused_distribution = jnp.stack([distribution[0], distribution[2]])

  output, new_recurrent_state = fused_gdn(
      q,
      k,
      v,
      cu_seqlens=query_start_loc,
      g=g,
      initial_state=recurrent_state,
      state_indices=state_indices,
      distribution=fused_distribution,
      b=b,
      has_initial_state=has_initial_state,
      use_qk_l2norm_in_kernel=True,
      use_gate_in_kernel=True,
      apply_silu=True,
      A_log=A_log,
      dt_bias=dt_bias,
  )

  output = output.reshape(num_tokens, n_v * d_v)
  return new_recurrent_state, output



CONFIGS = {
    'gdn_small': {
        'name': 'gdn_small',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 256,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 2,
        'prefill_length': 128,
        'max_reqs': 32,
    },
    'gdn_prefill_medium': {
        'name': 'gdn_prefill_medium',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 2048,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 4,
        'prefill_length': 512,
        'max_reqs': 64,
    },
    'gdn_prefill_large': {
        'name': 'gdn_prefill_large',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 8192,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 8,
        'prefill_length': 1024,
        'max_reqs': 64,
    },
    'gdn_decode_b64': {
        'name': 'gdn_decode_b64',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 64,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 0,
        'prefill_length': 0,
        'num_decode_reqs': 64,
        'max_reqs': 64,
    },
    'gdn_mixed_decode_prefill': {
        'name': 'gdn_mixed_decode_prefill',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 1056,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 2,
        'prefill_length': 512,
        'num_decode_reqs': 32,
        'max_reqs': 64,
    },
}
CONFIG = CONFIGS['gdn_small']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (mixed_qkv, b, a, recurrent_state, A_log, dt_bias, query_start_loc, state_indices, distribution, has_initial_state)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

  num_tokens = cfg['num_tokens']
  n_kq = cfg['n_kq']
  n_v = cfg['n_v']
  d_k = cfg['d_k']
  d_v = cfg['d_v']
  num_prefill_reqs = cfg.get('num_prefill_reqs', 0)
  prefill_length = cfg.get('prefill_length', 0)
  num_decode_reqs = cfg.get('num_decode_reqs', 0)
  max_reqs = cfg['max_reqs']

  key_dim = n_kq * d_k
  val_dim = n_v * d_v
  mixed_dim = 2 * key_dim + val_dim

  mixed_qkv = jax.random.normal(k1, (num_tokens, mixed_dim), dtype=dtype)
  b = jax.random.normal(k2, (num_tokens, n_v), dtype=dtype)
  a = jax.random.normal(k3, (num_tokens, n_v), dtype=dtype)

  state_size = max_reqs + 1
  recurrent_state = jax.random.normal(
      k7, (state_size, n_v, d_k, d_v), dtype=jnp.float32
  )
  # Slot 0 is the null block that `state_indices` is zero-padded with
  # for the inactive request slots. Implementations differ on whether
  # they touch it: the ref zeroes every slot named in `state_indices`
  # whose `has_initial_state` is False (which includes the padding),
  # while the kernels skip inactive requests entirely. Seed it with
  # zeros so the null block's final contents are well defined either
  # way and don't show up as a spurious mismatch.
  recurrent_state = recurrent_state.at[0].set(0.0)

  A_log = jax.random.normal(k4, (n_v,), dtype=jnp.float32)
  dt_bias = jax.random.normal(k5, (n_v,), dtype=jnp.float32)

  lengths = [1] * num_decode_reqs + [prefill_length] * num_prefill_reqs
  num_seqs = len(lengths)
  query_start_loc = jnp.cumsum(jnp.array([0] + lengths, dtype=jnp.int32))
  last_val = query_start_loc[-1]
  query_start_loc = jnp.pad(
      query_start_loc,
      (0, max_reqs + 1 - len(query_start_loc)),
      constant_values=last_val,
  )

  available_indices = jax.random.permutation(
      k6, jnp.arange(1, max_reqs + 1, dtype=jnp.int32)
  )
  valid_state_indices = available_indices[:num_seqs]
  state_indices = jnp.pad(valid_state_indices, (0, max_reqs - num_seqs))

  distribution = jnp.array(
      [num_decode_reqs, num_seqs, num_seqs], dtype=jnp.int32
  )
  has_initial_state = jnp.pad(
      jnp.array(
          [True] * num_decode_reqs + [False] * num_prefill_reqs,
          dtype=jnp.bool_,
      ),
      (0, max_reqs - num_seqs),
  )

  return (
      mixed_qkv,
      b,
      a,
      recurrent_state,
      A_log,
      dt_bias,
      query_start_loc,
      state_indices,
      distribution,
      has_initial_state,
  )


def get_inputs(dtype=jnp.bfloat16):
  """Returns list of inputs across all defined configurations."""
  return [
      (list(create_inputs(dtype=dtype, config=cfg)), [])
      for cfg in CONFIGS.values()
  ]


def computation(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
):
  """Pallas fused GDN kernel."""
  n_v = b.shape[1]
  n_kq = 2
  d_k = 128
  d_v = 128

  return ragged_gated_delta_rule(
      mixed_qkv,
      b,
      a,
      recurrent_state,
      A_log,
      dt_bias,
      query_start_loc,
      state_indices,
      distribution,
      has_initial_state,
      n_kq=n_kq,
      n_v=n_v,
      d_k=d_k,
      d_v=d_v,
  )

