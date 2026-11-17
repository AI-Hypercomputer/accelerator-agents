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
"""Kimi Delta Attention (KDA) backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

`forward_pallas(q, k, v, g, beta, a_log, delta_time_bias)` is the Tokamax Pallas
TPU forward for KDA: a fused gate-activation and intra-chunk kernel, then a
fused state-propagation and output kernel, over chunks of 64 tokens. It returns
`(out, residuals)`, where `residuals` is a dict of the intermediates the
Tokamax forward keeps for a backward pass:
  q, k            `[H, B, T, K]`        L2-normalised queries and keys
  q_rstd, k_rstd  `[H, B, T]`           their reciprocal norms (float32)
  g_cumsum        `[H, B, T, K]`        chunk-local cumulative sum of the
                                        activated log-decay, in log2 units
                                        (float32)
  aqk             `[H, B, T, 64]`       per chunk, the causal, decayed and
                                        scaled query-key matrix
  akk             `[H, B, T, 64]`       per chunk, the inverse `(I + L)^-1` of
                                        the intra-chunk delta-rule system
  h               `[H, B, T/64, K, V]`  recurrent state entering each chunk
  v_new           `[H, B, T, V]`        per-token corrected values `u - w @ h`
Tokamax's residual record also holds the inputs `v`, `beta`, `g`, `a_log` and
`delta_time_bias` unchanged; they are not repeated in `residuals`.

`workload(q, k, v, g, beta, a_log, delta_time_bias, dout)` returns the
gradients `(dq, dk, dv, dg, dbeta, da_log, ddelta_time_bias)` for the output
cotangent `dout`. It is the correctness oracle: JAX autodiff (`jax.vjp`) of the
pure-JAX token-by-token recurrence `forward`. This file contains no backward
kernel and no hand-derived backward pass.

`forward` is the Tokamax KDA reference with its options fixed to the Kimi
Linear training configuration: query/key L2 normalisation inside the op, the
gate activated inside the op as `-5 * sigmoid(exp(a_log) * (g +
delta_time_bias))`, query scale `K ** -0.5`, fixed-length sequences, a zero
initial state, and no final state returned. Tensors are head-first,
`[H, B, T, D]`. `forward_pallas` runs the kernels in the same configuration.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` (it
may consume the saved residuals) so that your `workload` returns the same
gradients as this one.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/kda/pallas_mosaic_tpu_fwd_kernel.py  (forward kernels)
  tokamax/_src/ops/kda/common.py     (TPU limits, mini-batch heuristic)
  tokamax/_src/ops/kda/utils.py      (helpers)
  tokamax/_src/ops/kda/pallas_mosaic_tpu.py  (how the kernels are called)
  tokamax/_src/ops/kda/reference.py  (forward recurrence)
  tokamax/_src/ops/kda/base.py       (default query scale)
  tokamax/_src/ops/kda/arg_specs.py  (shapes and options)
  tokamax/_src/ops/kda/pallas_mosaic_tpu_kernel_test.py  (value ranges)
"""

# Dimensions.
#   H   number of attention heads
#   B   batch size
#   T   sequence length
#   K   key head dimension
#   V   value head dimension
#   BT  chunk size (64)
#   NT  number of chunks (T // BT)
#   MB  mini-batch of heads processed per grid point
#   GW  gate channel width (K here: one gate value per key channel)

import dataclasses
import functools
import json
import math
import os

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/kda/reference.py, Copyright 2026 Ant
# Group, Apache-2.0)
# ==============================================================================

# Lower bound of the sigmoid gate: the activated log-decay lies in (-5, 0).
_LOWER_BOUND = -5.0


def _accumulator_dtype(dtype):
  return jnp.promote_types(jnp.dtype(dtype), jnp.float32)


def _l2_normalize(x, acc_dtype):
  x_f = x.astype(acc_dtype)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + 1e-6)
  return x_f * rstd[..., None]


def _activate_gate(g, *, a_log, delta_time_bias, lower_bound):
  heads, _, _, key_dim = g.shape
  g_f = g.astype(jnp.float32)
  g_f = g_f + delta_time_bias.astype(jnp.float32).reshape(heads, 1, 1, key_dim)
  A = jnp.exp(a_log.astype(jnp.float32)).reshape(heads, 1, 1, 1)
  return lower_bound * jax.nn.sigmoid(A * g_f)


def forward(q, k, v, g, beta, a_log, delta_time_bias):
  """KDA forward pass: explicit token-by-token delta-rule recurrence.

  Args:
    q: Queries, `[H, B, T, K]`. L2-normalised over `K` inside the op.
    k: Keys, `[H, B, T, K]`. L2-normalised over `K` inside the op.
    v: Values, `[H, B, T, V]`.
    g: Raw per-channel gate input, `[H, B, T, K]`. The log-decay is
      `-5 * sigmoid(exp(a_log) * (g + delta_time_bias))`.
    beta: Per-token delta-rule learning rate, `[H, B, T]`, values in `[0, 1]`.
    a_log: Per-head log decay rate, `[H]`.
    delta_time_bias: Per-head, per-key-channel gate bias, `[H * K]`.

  Returns:
    The output, `[H, B, T, V]`, in the dtype of `q`. The state starts at zero
    and the final state is not returned. Queries are scaled by `K ** -0.5`.
  """
  heads, batch, seq_len, key_dim = q.shape
  value_dim = v.shape[-1]
  acc_dtype = _accumulator_dtype(q.dtype)
  output_dtype = q.dtype
  scale = key_dim**-0.5

  g = _activate_gate(
      g, a_log=a_log, delta_time_bias=delta_time_bias, lower_bound=_LOWER_BOUND
  )
  q_h = _l2_normalize(q, acc_dtype)
  k_h = _l2_normalize(k, acc_dtype)

  q_h = q_h * scale
  v_h = v.astype(acc_dtype)
  g_h = g.astype(acc_dtype)
  beta_h = beta.astype(acc_dtype)

  # One recurrent state per batch element and head.
  states = jnp.zeros((batch, 1, heads, key_dim, value_dim), dtype=acc_dtype)
  output_h = jnp.zeros((heads, batch, seq_len, value_dim), dtype=acc_dtype)

  def step_token(h, b, t, carry):
    states, output_h = carry
    previous_state = states[b, 0, h]
    state = previous_state * jnp.exp(g_h[h, b, t])[:, None]
    prediction = k_h[h, b, t] @ state
    residual = v_h[h, b, t] - prediction
    new_state = state + (
        beta_h[h, b, t] * k_h[h, b, t]
    )[:, None] * residual[None, :]
    out_t = q_h[h, b, t] @ new_state
    output_h = output_h.at[h, b, t].set(out_t)
    states = states.at[b, 0, h].set(new_state)
    return states, output_h

  def step_head(h, carry):
    states, output_h = carry

    def body_b(b, b_carry):
      def body_t(t, t_carry):
        return step_token(h, b, t, t_carry)

      return jax.lax.fori_loop(0, seq_len, body_t, b_carry)

    states, output_h = jax.lax.fori_loop(
        0, batch, body_b, (states, output_h)
    )
    return states, output_h

  states, output_h = jax.lax.fori_loop(
      0, heads, step_head, (states, output_h)
  )

  return output_h.astype(output_dtype)


# ==============================================================================
# Pallas TPU forward kernels (inlined from Tokamax; each source file is under
# its own banner below)
# ==============================================================================

# ------------------------------------------------------------------------------
# From tokamax/_src/ops/kda/utils.py
# Copyright 2026 Ant Group. All Rights Reserved. (Apache-2.0)
# Small numeric and indexing helpers.
# ------------------------------------------------------------------------------


RCP_LN2 = 1.0 / math.log(2)


def exp2(x):
  """Base-2 exponential, matching Triton's tl.exp2."""
  return jnp.exp2(x.astype(jnp.float32))


def get_interpret() -> bool:
  env = os.environ.get("PALLAS_INTERPRET", "")
  return env.strip().lower() in ("1", "true") or jax.default_backend() == "cpu"


def get_chain_precision() -> jax.lax.Precision:
  """Precision for the two matmuls on the forward's serial recurrent chain.

  The two matmuls are `w @ h` and `kg^T @ v_new` in the state-propagation
  kernel. Tokamax reads this from its `tokamax_kda_chain_precision` config
  option; `HIGHEST` is that option's default.
  """
  return jax.lax.Precision.HIGHEST


def cdiv(x, y: int):
  return (x + y - 1) // y


def l2norm_fwd(x: jax.Array, eps: float = 1e-6):
  x_f = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + eps)
  return (x_f * rstd[..., None]).astype(x.dtype), rstd.astype(jnp.float32)


def align_up(x, align: int):
  return cdiv(x, align) * align


def prepare_lens(cu_seqlens: jax.Array) -> jax.Array:
  return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
    max_T: int | jax.Array | None = None,
) -> jax.Array:
  """Compute per-chunk `(seq_id, block_id)` mapping from cu_seqlens."""
  if cu_seqlens.ndim == 2:
    rows = [
        prepare_chunk_indices(cu_seqlens[b], chunk_size, max_T=max_T)
        for b in range(cu_seqlens.shape[0])
    ]
    return jnp.stack(rows, axis=0)
  lens = prepare_lens(cu_seqlens)
  n_chunks = cdiv(lens, chunk_size)
  num_seqs = len(lens)
  target_max_T = cu_seqlens[-1] if max_T is None else max_T
  total_nt = int(target_max_T) // chunk_size
  seq_ids = jnp.repeat(
      jnp.arange(num_seqs, dtype=jnp.int32),
      n_chunks,
      total_repeat_length=total_nt,
  )
  prefix_chunks = jnp.concatenate([
      jnp.zeros(1, dtype=jnp.int32),
      jnp.cumsum(n_chunks),
  ])
  seq_offsets = jnp.repeat(
      prefix_chunks[:-1],
      n_chunks,
      total_repeat_length=total_nt,
  )
  block_ids = jnp.arange(total_nt, dtype=jnp.int32) - seq_offsets
  return jnp.stack([seq_ids, block_ids], axis=1)


# ------------------------------------------------------------------------------
# From tokamax/_src/ops/kda/common.py
# Copyright 2026 Ant Group. All Rights Reserved. (Apache-2.0)
# TPU limits and the mini-batch heuristic.
# ------------------------------------------------------------------------------


_TPU_VMEM_USAGE_FRACTION = 0.9
_HIGH_VMEM_THRESHOLD_BYTES = 64 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class _TpuLimits:
  """Subset of JAX TPU hardware information used by KDA kernels."""

  vmem_limit_bytes: int
  block_align_minor: int
  block_align_major: int


def get_tpu_limits() -> _TpuLimits:
  """Returns KDA limits for the TPU selected by Mosaic dispatch."""
  tpu_info = pltpu.get_tpu_info()
  return _TpuLimits(
      # Preserve headroom for compiler-managed VMEM allocations.
      vmem_limit_bytes=int(
          tpu_info.vmem_capacity_bytes * _TPU_VMEM_USAGE_FRACTION
      ),
      block_align_minor=int(tpu_info.num_sublanes),
      block_align_major=int(tpu_info.num_lanes),
  )


def get_vmem_limit_bytes() -> int:
  """Returns the usable VMEM budget in bytes (90% of hardware VMEM capacity)."""
  return get_tpu_limits().vmem_limit_bytes


def get_max_mini_batch(
    *,
    chunk_size: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> int:
  """Returns the upper bound on `mini_batch` for the hardware VMEM budget.

  When `chunk_size` (`BT`) is provided for kernels whose in-kernel `f32`
  `[MB, BT, BT]` intermediates grow quadratically with `BT`, pins `max_mb=1`
  for `BT > 64` on 64 MiB VMEM TPUs while allowing `max_mb=16` at `BT <= 128`
  on high-VMEM TPUs (`> 64 MiB` budget). When `chunk_size` is omitted, returns
  `32` on high-VMEM TPUs and `16` on 64 MiB VMEM TPUs.

  Args:
    chunk_size: Optional chunk size (`BT`) for kernels whose in-kernel
      intermediates scale quadratically with `BT`.
    vmem_limit_bytes: Optional VMEM budget in bytes; defaults to hardware VMEM.
  """
  if vmem_limit_bytes is None:
    vmem_limit_bytes = get_vmem_limit_bytes()
  high_vmem = vmem_limit_bytes > _HIGH_VMEM_THRESHOLD_BYTES
  if chunk_size is None:
    return 32 if high_vmem else 16
  if chunk_size <= 64:
    return 32
  if chunk_size <= 128 and high_vmem:
    return 16
  return 1


def estimate_mini_batch(
    per_tile_bytes: int,
    total: int,
    *,
    max_mb: int | None = None,
    chunk_size: int | None = None,
    vmem_budget: int | None = None,
    align_minor: int | None = None,
    multiple_of: int = 1,
) -> int:
  """Estimate the optimal mini-batch size to maximise VMEM utilisation.

  This mirrors the backward auto-tune pattern used throughout the KDA kernels:
  compute the largest MB that fits within the hardware VMEM budget, cap it,
  and then adjust downward so that ``total`` is evenly divisible by ``MB``.

  Args:
      per_tile_bytes: Estimated VMEM footprint (bytes) for **one** tile/head.
      total: Number of tiles (or heads) to partition.
      max_mb: Upper bound on MB. ``None`` (default) derives the cap from
          hardware VMEM capacity and ``chunk_size`` via ``get_max_mini_batch``.
      chunk_size: Optional chunk size (``BT``) for kernels whose in-kernel
          intermediates scale quadratically with ``BT``.
      vmem_budget: VMEM budget in bytes. ``None`` (default) queries
          ``get_tpu_limits().vmem_limit_bytes`` at call time.
      align_minor: TPU block_align_minor constraint. ``None`` (default)
          queries ``get_tpu_limits().block_align_minor``.  When provided,
          the function prefers MB values that satisfy this alignment.
      multiple_of: Minimum divisor constraint on ``MB`` (e.g. ``gqa_repeats``).

  Returns:
      Mini-batch size ``MB`` such that ``total % MB == 0`` (best-effort).
  """
  if vmem_budget is None or align_minor is None or max_mb is None:
    hw = get_tpu_limits()
    if vmem_budget is None:
      vmem_budget = hw.vmem_limit_bytes
    if align_minor is None:
      align_minor = hw.block_align_minor
    if max_mb is None:
      max_mb = get_max_mini_batch(
          chunk_size=chunk_size, vmem_limit_bytes=hw.vmem_limit_bytes
      )

  per_tile_bytes = max(1, per_tile_bytes)
  MB = max(1, vmem_budget // per_tile_bytes)
  MB = max(multiple_of, min(MB, total, max(max_mb, multiple_of)))

  # Find an MB that divides total evenly and is a multiple of multiple_of.
  while MB > multiple_of:
    if total % MB == 0 and MB % multiple_of == 0:
      break
    MB -= 1

  return MB


# ------------------------------------------------------------------------------
# From tokamax/_src/ops/kda/pallas_mosaic_tpu_fwd_kernel.py
# Copyright 2026 Ant Group. All Rights Reserved. (Apache-2.0)
# Forward kernels and their `pallas_call` wrappers.
# ------------------------------------------------------------------------------


# Rows of the previous chunk kept for the depthwise conv; one f32 sublane tile.
# The conv weights are left-padded to this many taps, so W <= _CONV_HALO.
_CONV_HALO = 8


def _solve_unit_lower_triangular_batched(A, b):
  """Solve (I + A) x = b exactly for batched inputs.

  Uses block forward substitution with block size 16 for TPU MXU
  utilization.

  Args:
      A: (MB, N, N) strictly lower triangular matrix (float32).
      b: (MB, N, D) right-hand side matrix (float32).

  Returns:
      x: (MB, N, D) exact solution matrix.
  """
  MB, N, D = b.shape
  BS = 16
  num_blocks = N // BS
  A = A.astype(jnp.float32)
  b = b.astype(jnp.float32)

  blocks = [b[:, i * BS : (i + 1) * BS, :] for i in range(num_blocks)]

  for i in range(num_blocks):
    start = i * BS
    end = (i + 1) * BS

    A_ii = A[:, start:end, start:end]  # [MB, BS, BS]
    x_block = blocks[i]  # [MB, BS, D]

    rows = [x_block[:, r, :] for r in range(BS)]  # list of [MB, D]
    for j in range(BS):
      if j > 0:
        vec = A_ii[:, j, :j]  # [MB, j]
        mat = jnp.stack(rows[:j], axis=1)  # [MB, j, D]
        # [MB, 1, j] @ [MB, j, D] → [MB, 1, D] → squeeze
        correction = jnp.matmul(vec[:, None, :], mat).squeeze(1)
        rows[j] = rows[j] - correction

    x_block = jnp.stack(rows, axis=1)  # [MB, BS, D]
    blocks[i] = x_block

    if i < num_blocks - 1:
      x_rest = jnp.concatenate(blocks[i + 1 :], axis=1)  # [MB, rest, D]
      A_rest = A[:, (i + 1) * BS :, start:end]  # [MB, rest, BS]

      update = jnp.matmul(
        A_rest,
        x_block,
        preferred_element_type=jnp.float32,
      )
      x_rest = x_rest - update

      remaining = num_blocks - 1 - i
      for idx in range(remaining):
        blocks[i + 1 + idx] = x_rest[:, idx * BS : (idx + 1) * BS, :]

  return jnp.concatenate(blocks, axis=1)


def _solve_unit_lower_triangular_neumann_batched(A, b):
  """Solve (I + A) x = b with a blocked finite Neumann series.

  The strictly lower triangular system is split into 8-token diagonal blocks.
  Each diagonal block is inverted with a finite Neumann series, then the
  block-off-diagonal system is inverted with Horner doubling.

  Args:
      A: (MB, N, N) strictly lower triangular matrix.
      b: (MB, N, D) right-hand side matrix.

  Returns:
      A pair containing the solution with shape (MB, N, D) and the inverse of
      (I + A) with shape (MB, N, N), both in float32.
  """
  N = A.shape[-1]
  block_size = 8
  num_blocks = N // block_size
  inv_dtype = jnp.float32
  identity = jnp.eye(N, dtype=inv_dtype)

  # Batched dot: [MB, M, K] @ [MB, K, N] -> [MB, M, N].
  def dot_batch(lhs, rhs):
    return jax.lax.dot_general(
        lhs,
        rhs,
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

  A = A.astype(inv_dtype)
  block_ids = jnp.arange(N, dtype=jnp.int32) // block_size
  same_block = (block_ids[:, None] == block_ids[None, :]).astype(inv_dtype)
  A_diag = A * same_block[None]
  A_off_diag = A - A_diag

  neg_A_diag = -A_diag
  diag_inverse = identity[None] + neg_A_diag
  power = neg_A_diag
  num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[block_size]
  for _ in range(num_diag_steps):
    power = dot_batch(power, power)
    diag_inverse = dot_batch(diag_inverse, identity[None] + power)

  if num_blocks == 1:
    return dot_batch(diag_inverse, b.astype(inv_dtype)), diag_inverse

  # Fuse P @ [F | b] to share a matmul, where P is the block-diagonal
  # inverse and F is the block-off-diagonal part of A.
  off_diag_and_rhs = jnp.concatenate(
      [A_off_diag, b.astype(inv_dtype)], axis=-1
  )
  merged = dot_batch(diag_inverse, off_diag_and_rhs)
  transformed_off_diag = merged[:, :, :N]
  transformed_rhs = merged[:, :, N:]

  # (I + G)^-1 = sum(k=0..num_blocks-1) (-G)^k. The block-strictly-lower
  # matrix G is nilpotent, so Horner doubling evaluates this finite series.
  neg_transformed_off_diag = -transformed_off_diag
  off_diag_inverse = identity[None] + neg_transformed_off_diag
  power = neg_transformed_off_diag
  log2_num_blocks = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[num_blocks]
  for _ in range(log2_num_blocks - 1):
    power = dot_batch(power, power)
    off_diag_inverse = off_diag_inverse + dot_batch(off_diag_inverse, power)

  result = dot_batch(off_diag_inverse, transformed_rhs)
  inverse = dot_batch(off_diag_inverse, diag_inverse)
  return result, inverse


def _scalar_gate_intra(
    q_f32, k_f32, g_cumsum, beta_f32, *, BT: int, scale: float
):
  """Computes intra-chunk Aqk and L for a scalar gate (Gated Delta Net)."""
  gs = g_cumsum[:, :, 0]  # [MB, BT]
  row = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
  col = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
  causal = row >= col
  strict = row > col

  # Mask pairwise differences before exp2 to prevent overflow on anti-causal entries.
  diff = gs[:, :, None] - gs[:, None, :]  # [MB, BT, BT]
  decay = jnp.exp2(jnp.where(causal[None], diff, jnp.float32(0.0)))

  qk = jax.lax.dot_general(
      q_f32,
      k_f32,
      (((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )  # [MB, BT, BT]
  kk = jax.lax.dot_general(
      k_f32,
      k_f32,
      (((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )  # [MB, BT, BT]

  zero = jnp.float32(0.0)
  Aqk = jnp.where(causal[None], qk * decay * scale, zero)
  L = jnp.where(strict[None], kk * decay * beta_f32, zero)
  return Aqk, L


def _make_chunk_halo(
    x_r: jax.Array,
    aligned_segment_ids: jax.Array | None = None,
    halo: int = _CONV_HALO,
) -> jax.Array:
  """Extracts the last `halo` tokens of chunk l-1 for each chunk l."""
  prev = x_r[:, :, :-1, -halo:, :]
  if aligned_segment_ids is not None:
    H, B, NC, BT, _ = x_r.shape
    seg_r = aligned_segment_ids.reshape(B, NC, BT)
    same_seg = (seg_r[:, 1:, 0] == seg_r[:, :-1, -1]) & (seg_r[:, 1:, 0] >= 0)
    prev = jnp.where(same_seg[None, :, :, None, None], prev, 0)
  return jnp.pad(prev, ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))


def _apply_causal_conv1d_silu(
    x_curr: jax.Array,
    x_prev: jax.Array,
    w_pad: jax.Array,
    bias: jax.Array,
    W: int,
) -> jax.Array:
  """Computes depthwise causal conv1d (W <= _CONV_HALO) + SiLU in fp32."""
  x_curr_f32 = x_curr.astype(jnp.float32)
  x_prev_f32 = x_prev.astype(jnp.float32)
  w_f32 = w_pad.astype(jnp.float32)
  acc = x_curr_f32 * w_f32[:, _CONV_HALO - 1 : _CONV_HALO, :]
  for lag in range(1, W):
    x_lag = jnp.concatenate(
        [x_prev_f32[:, _CONV_HALO - lag :, :], x_curr_f32[:, :-lag, :]],
        axis=1,
    )
    acc = acc + x_lag * w_f32[:, _CONV_HALO - 1 - lag : _CONV_HALO - lag, :]
  acc = acc + bias.astype(jnp.float32)[:, None, :]
  return jax.nn.silu(acc)


def _fused_gate_intra_kernel(
    # Inputs. All refs are single VMEM blocks.
    q_ref,  # [MB, 1, 1, BT, K]
    k_ref,  # [MB, 1, 1, BT, K]
    g_ref,  # [MB, 1, 1, 1, BT]  scalar gate, width-1 axis kept
    # off lanes; else [MB, 1, 1, BT, K]
    beta_ref,  # [MB, 1, 1, 1, BT]  width-1 axis kept off lanes
    v_ref,  # [MB, 1, 1, BT, V]
    a_log_ref,  # [MB, 1, 1, 1, 1] f32  per head
    delta_time_bias_ref,  # [MB, 1, 1, 1, K] f32  per head
    # With use_conv1d_in_kernel, nine conv inputs come first, all f32 except
    # the halos (input dtype); `HALO` is `_CONV_HALO`:
    #   q_prev_ref, k_prev_ref  [MB, 1, 1, HALO, K]  last HALO rows of the
    #   v_prev_ref              [MB, 1, 1, HALO, V]  previous chunk, zeroed
    #                                                across segment starts
    #   w_q_ref, w_k_ref        [MB, 1, 1, HALO, K]  W taps left-padded to
    #   w_v_ref                 [MB, 1, 1, HALO, V]  HALO; same block for
    #                                                every chunk
    #   b_q_ref, b_k_ref        [MB, 1, 1, 1, K]
    #   b_v_ref                 [MB, 1, 1, 1, V]
    # Outputs (k dtype unless noted):
    #   u_out_ref               [MB, 1, 1, BT, V]
    #   w_out_ref, kg_out_ref   [MB, 1, 1, BT, K]
    #   Aqk_out_ref             [MB, 1, 1, BT, BT]
    #   Akk_inv_out_ref         [MB, 1, 1, BT, BT]
    #   g_cumsum_out_ref        [MB, 1, 1, BT, GW] f32
    # and, with use_conv1d_in_kernel, q_norm_ref [MB, 1, 1, BT, K]; with
    # return_conv_intermediates also k_norm_ref [MB, 1, 1, BT, K] and
    # v_silu_ref [MB, 1, 1, BT, V] (v dtype); with use_qk_l2norm also
    # q_rstd_ref, k_rstd_ref [MB, 1, 1, 1, BT] f32.
    *conv_in_and_out_refs,
    chunk_size: int,
    head_dim: int,
    value_dim: int,
    scale: float,
    cumsum_scale: float,
    fuse_cumsum: bool,
    disable_recompute: bool,
    safe_gate: bool,
    per_channel_gate: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    mini_batch: int = 1,
    use_conv1d_in_kernel: bool = False,
    use_qk_l2norm: bool = False,
    return_conv_intermediates: bool = False,
    conv_kernel_size: int = 4,
):
  """Fused Pallas kernel: gate activation + cumsum + BC=16 Aqk/L + Neumann inversion.

  When fuse_cumsum=True, g_ref contains raw gate values (not cumsum'd).
  The kernel applies gate activation (if use_gate_in_kernel) and chunk-local
  prefix sum via tril matmul before proceeding to the Aqk/L construction.

  For bfloat16 inputs, uses Neumann series inversion.
  For float32 inputs, falls back to exact forward substitution.

  Ref shapes are listed next to the parameters.

  MB heads are processed simultaneously via batch-vectorized ops.

  Args:
      mini_batch: int — number of heads processed per grid point (MB).
  """
  dtype = q_ref.dtype
  BT = chunk_size
  BC = 16
  NC = BT // BC
  K = head_dim
  V = value_dim
  MB = mini_batch

  q_prev_ref = k_prev_ref = v_prev_ref = None
  w_q_ref = w_k_ref = w_v_ref = None
  b_q_ref = b_k_ref = b_v_ref = None
  if use_conv1d_in_kernel:
    (
        q_prev_ref,
        k_prev_ref,
        v_prev_ref,
        w_q_ref,
        w_k_ref,
        w_v_ref,
        b_q_ref,
        b_k_ref,
        b_v_ref,
        *out_refs,
    ) = conv_in_and_out_refs
  else:
    out_refs = conv_in_and_out_refs

  # `qg` is not an output: it is not part of `KdaResiduals` and the backward
  # recomputes it.
  (
      u_out_ref,
      w_out_ref,
      kg_out_ref,
      Aqk_out_ref,
      Akk_inv_out_ref,
      g_cumsum_out_ref,
  ) = out_refs[:6]
  extra_out_refs = out_refs[6:]

  # Load all MB heads at once
  q_in = q_ref[:, 0, 0]  # [MB, BT, K]
  k_in = k_ref[:, 0, 0]  # [MB, BT, K]
  # Transpose width-1 inputs stored as [.., 1, BT] back to [MB, BT, 1].
  if g_ref.shape[-2] == 1:
    g = g_ref[:, 0, 0, 0][..., None]  # [MB, BT] -> [MB, BT, 1]
  else:
    g = g_ref[:, 0, 0]  # [MB, BT, K]
  beta = beta_ref[:, 0, 0, 0][..., None]  # [MB, BT] -> [MB, BT, 1]
  v_in = v_ref[:, 0, 0]  # [MB, BT, V]

  if use_conv1d_in_kernel:
    assert (
        q_prev_ref is not None
        and k_prev_ref is not None
        and v_prev_ref is not None
        and w_q_ref is not None
        and w_k_ref is not None
        and w_v_ref is not None
        and b_q_ref is not None
        and b_k_ref is not None
        and b_v_ref is not None
    )
    q_prev = q_prev_ref[:, 0, 0]  # [MB, _CONV_HALO, K]
    k_prev = k_prev_ref[:, 0, 0]  # [MB, _CONV_HALO, K]
    v_prev = v_prev_ref[:, 0, 0]  # [MB, _CONV_HALO, V]
    w_q = w_q_ref[:, 0, 0]  # [MB, _CONV_HALO, K]
    w_k = w_k_ref[:, 0, 0]  # [MB, _CONV_HALO, K]
    w_v = w_v_ref[:, 0, 0]  # [MB, _CONV_HALO, V]
    b_q = b_q_ref[:, 0, 0, 0]  # [MB, K]
    b_k = b_k_ref[:, 0, 0, 0]  # [MB, K]
    b_v = b_v_ref[:, 0, 0, 0]  # [MB, V]

    q_silu = _apply_causal_conv1d_silu(
        q_in, q_prev, w_q, b_q, conv_kernel_size
    ).astype(dtype)
    k_silu = _apply_causal_conv1d_silu(
        k_in, k_prev, w_k, b_k, conv_kernel_size
    ).astype(dtype)
    v_silu = _apply_causal_conv1d_silu(
        v_in, v_prev, w_v, b_v, conv_kernel_size
    ).astype(dtype)

    q_rstd_val = k_rstd_val = None
    if use_qk_l2norm:
      q_f32_s = q_silu.astype(jnp.float32)
      k_f32_s = k_silu.astype(jnp.float32)
      q_rstd_val = jax.lax.rsqrt(
          jnp.sum(q_f32_s * q_f32_s, axis=-1, keepdims=True) + 1e-6
      )
      k_rstd_val = jax.lax.rsqrt(
          jnp.sum(k_f32_s * k_f32_s, axis=-1, keepdims=True) + 1e-6
      )
      q = (q_f32_s * q_rstd_val).astype(dtype)
      k = (k_f32_s * k_rstd_val).astype(dtype)
    else:
      q = q_silu
      k = k_silu
    v = v_silu

    q_norm_ref = extra_out_refs[0]
    q_norm_ref[:, 0, 0] = q
    if return_conv_intermediates:
      k_norm_ref = extra_out_refs[1]
      v_silu_ref = extra_out_refs[2]
      k_norm_ref[:, 0, 0] = k
      v_silu_ref[:, 0, 0] = v
      if use_qk_l2norm:
        assert q_rstd_val is not None and k_rstd_val is not None
        q_rstd_ref = extra_out_refs[3]
        k_rstd_ref = extra_out_refs[4]
        q_rstd_ref[:, 0, 0, 0] = q_rstd_val[..., 0]
        k_rstd_ref[:, 0, 0, 0] = k_rstd_val[..., 0]
  else:
    q = q_in
    k = k_in
    v = v_in

  # --- Gate activation + cumsum ---
  g_f32 = g.astype(jnp.float32)

  if use_gate_in_kernel:
    assert per_channel_gate, "use_gate_in_kernel requires per_channel_gate=True"
    dt_b = delta_time_bias_ref[:, 0, 0, 0]        # [MB, K]
    g_f32 = g_f32 + dt_b[:, None, :]       # [MB, BT, K]
    A_val = a_log_ref[:, 0, 0, 0, 0]       # [MB]
    if lower_bound is None:
      g_f32 = -jnp.exp(A_val)[:, None, None] * jax.nn.softplus(g_f32)
    else:
      g_f32 = lower_bound * jax.nn.sigmoid(
        jnp.exp(A_val)[:, None, None] * g_f32
      )

  if fuse_cumsum:
    tril = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))
    # tril[BT,BT] @ g_f32[MB,BT,K] → [MB,BT,K]: contract on BT, batch on MB
    g_cumsum = jax.lax.dot_general(
      tril, g_f32, (((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
    ).transpose(1, 0, 2) * cumsum_scale  # [BT,MB,K] → [MB,BT,K]
  else:
    g_cumsum = g_f32

  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  beta_f32 = beta.astype(jnp.float32)

  # --- BC=16 sub-block factored Aqk/L (fla safe-gate style, j-batched) ---
  # Per i_sc, batch the j_sc loop into a single (MB, 2*BC, K) x (MB, BT, K) matmul:
  #   q_eg[m,r,k]   = q_i[m,r,k] * exp2(g_i[m,r,k] - gn_ref[m,k])
  #   k_eg[m,r,k]   = k_i[m,r,k] * exp2(g_i[m,r,k] - gn_ref[m,k])    (Akk left)
  #   k_eng[m,t,k]  = k[m,t,k]   * exp2(gn_ref[m,k] - g_cumsum[m,t,k]) (full BT)
  # so Aqk[m,r,t]  = sum_k q_i[m,r,k] * k[m,t,k] * exp2(g_i[m,r,k] - g_cumsum[m,t,k]).
  # Stack q_eg and k_eg along row dim and do one matmul -> split Aqk/Akk.
  #
  # Anti-causal rows (t >= (i_sc+1)*BC) have positive `gn_ref - g_cumsum[t]`
  # for negative-gate cumsum, which can overflow exp2. Mask those diffs to 0
  # BEFORE exp2, then zero the resulting rows so they contribute 0 to matmul.
  #
  # Use broadcasted_iota for 2D indices to avoid Mosaic-unsupported i1
  # shape casts (e.g. (BT,) -> (BT, 1) on bool tensors).
  ref_idx = BC // 2 if safe_gate else 0
  row_iota_bt_k = jax.lax.broadcasted_iota(jnp.int32, (BT, K), dimension=0)
  row_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), dimension=0)
  col_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), dimension=1)

  Aqk_rows = []
  L_rows = []
  for i_sc in range(NC) if per_channel_gate else ():
    i_s = i_sc * BC
    q_i = q_f32[:, i_s : i_s + BC]       # [MB, BC, K]
    k_i = k_f32[:, i_s : i_s + BC]       # [MB, BC, K]
    g_i = g_cumsum[:, i_s : i_s + BC]    # [MB, BC, K]
    beta_i = beta_f32[:, i_s : i_s + BC]  # [MB, BC, 1]

    gn_ref = g_i[:, ref_idx : ref_idx + 1, :]   # [MB, 1, K] — query sub-block ref
    diff_i = g_i - gn_ref                        # [MB, BC, K], <= 0
    exp_diff_i = jnp.exp2(diff_i)
    q_eg = q_i * exp_diff_i     # [MB, BC, K]
    k_eg = k_i * exp_diff_i     # [MB, BC, K]

    # j-side: full BT rows. Mask anti-causal rows BEFORE exp2 (avoids
    # overflow when gn_ref - g_cumsum[t] is large positive). The 2D mask
    # is built from broadcasted_iota to bypass the i1 reshape limitation.
    valid_j = (row_iota_bt_k < (i_s + BC)).astype(jnp.float32)   # [BT, K]
    diff_j_safe = (gn_ref - g_cumsum) * valid_j[None]              # [MB, BT, K]
    k_eng_full = k_f32 * jnp.exp2(diff_j_safe) * valid_j[None]    # [MB, BT, K]

    # Stack q_eg / k_eg and do one matmul: [MB, 2*BC, K] x [MB, K, BT] -> [MB, 2*BC, BT]
    qk_eg = jnp.concatenate([q_eg, k_eg], axis=1)  # [MB, 2*BC, K]
    qk_dot = jax.lax.dot_general(
      qk_eg, k_eng_full, (((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
    )  # [MB, 2*BC, BT]
    Aqk_row = qk_dot[:, :BC] * scale      # [MB, BC, BT]
    Akk_row = qk_dot[:, BC:] * beta_i     # [MB, BC, BT]

    # Apply diagonal-block masks (causal for Aqk, strict-lower for Akk).
    # All masks built from 2D iota; no i1 shape casts needed.
    in_diag = (col_iota_bc_bt >= i_s) & (col_iota_bc_bt < i_s + BC)  # [BC, BT]
    col_local = col_iota_bc_bt - i_s                                   # [BC, BT]
    causal_diag = row_iota_bc_bt >= col_local                          # [BC, BT]
    strict_diag = row_iota_bc_bt > col_local                           # [BC, BT]
    aqk_keep = (~in_diag) | causal_diag
    akk_keep = (~in_diag) | strict_diag
    Aqk_row = jnp.where(aqk_keep[None], Aqk_row, jnp.float32(0.0))
    Akk_row = jnp.where(akk_keep[None], Akk_row, jnp.float32(0.0))

    Aqk_rows.append(Aqk_row)
    L_rows.append(Akk_row)

  if per_channel_gate:
    Aqk = jnp.concatenate(Aqk_rows, axis=1).astype(dtype)  # [MB, BT, BT]
    L = jnp.concatenate(L_rows, axis=1)  # [MB, BT, BT]
  else:
    Aqk, L = _scalar_gate_intra(
        q_f32, k_f32, g_cumsum, beta_f32, BT=BT, scale=scale
    )
    Aqk = Aqk.astype(dtype)

  # --- Solve (I + L) x = rhs ---
  v_beta = v.astype(jnp.float32) * beta_f32          # [MB, BT, V]
  k_eg_beta = k_f32 * jnp.exp2(g_cumsum) * beta_f32  # [MB, BT, K]
  rhs = jnp.concatenate([v_beta, k_eg_beta], axis=-1)  # [MB, BT, V+K]

  use_neumann = dtype != jnp.float32

  if use_neumann:
    result, A_inv = _solve_unit_lower_triangular_neumann_batched(L, rhs)
  else:
    # --- Exact forward substitution (fp32) ---
    I_bt = jnp.eye(BT, dtype=jnp.float32)
    I_bt_batch = jnp.broadcast_to(I_bt, (MB, BT, BT))
    combined_b = jnp.concatenate(
      [rhs, I_bt_batch], axis=-1
    )  # [MB, BT, V+K+BT]
    full_result = _solve_unit_lower_triangular_batched(L, combined_b)
    result = full_result[:, :, : V + K]
    A_inv = full_result[:, :, V + K :]

  u = result[:, :, :V]         # [MB, BT, V]
  w = result[:, :, V : V + K]  # [MB, BT, K]

  # --- kg ---
  g_last = g_cumsum[:, BT - 1 : BT, :]  # [MB, 1, K]
  kg = k_f32 * exp2(g_last - g_cumsum)

  # --- Store all MB heads ---
  u_out_ref[:, 0, 0] = u.astype(u_out_ref.dtype)
  w_out_ref[:, 0, 0] = w.astype(w_out_ref.dtype)
  kg_out_ref[:, 0, 0] = kg.astype(kg_out_ref.dtype)
  Aqk_out_ref[:, 0, 0] = Aqk.astype(Aqk_out_ref.dtype)
  Akk_inv_out_ref[:, 0, 0] = A_inv.astype(Akk_inv_out_ref.dtype)
  g_cumsum_out_ref[:, 0, 0] = g_cumsum


def _compute_intra_fused_mini_batch(H, BT, K, V, dtype=None):
  """Auto-compute mini-batch size for intra fused kernel.

  Each head needs VMEM for intermediates: BT*K + BT*V + 2*BT*BT (Aqk, L).
  Uses hardware VMEM budget via ``estimate_mini_batch``.
  """
  elem_size = dtype.itemsize if dtype is not None else 4
  per_head = (BT * K + BT * V + 2 * BT * BT) * elem_size
  return estimate_mini_batch(per_head, H)


@functools.partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "scale",
        "safe_gate",
        "per_channel_gate",
        "disable_recompute",
        "cumsum_scale",
        "use_gate_in_kernel",
        "lower_bound",
        "mini_batch",
        "use_conv1d_in_kernel",
        "use_qk_l2norm",
        "return_conv_intermediates",
    ],
)
def pallas_kda_fwd_intra_fused(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float,
    chunk_size: int = 64,
    safe_gate: bool = True,
    per_channel_gate: bool = True,
    disable_recompute: bool = False,
    cumsum_scale: float = RCP_LN2,
    a_log: jax.Array | None = None,
    delta_time_bias: jax.Array | None = None,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    mini_batch: int | None = None,
    conv_weight_q: jax.Array | None = None,
    conv_weight_k: jax.Array | None = None,
    conv_weight_v: jax.Array | None = None,
    conv_bias_q: jax.Array | None = None,
    conv_bias_k: jax.Array | None = None,
    conv_bias_v: jax.Array | None = None,
    use_conv1d_in_kernel: bool = False,
    use_qk_l2norm: bool = False,
    return_conv_intermediates: bool = False,
    aligned_segment_ids: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
  """Fuses gate cumsum with the fixed-length intra-chunk solve.

  Heads are mini-batched to amortize DMA. When `use_conv1d_in_kernel` is
  true, causal conv1d + SiLU (+ optional L2 norm) are fused into the kernel.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  NC = T // BT

  if use_gate_in_kernel:
    assert a_log is not None, "a_log required when use_gate_in_kernel=True"

  if mini_batch is None:
    MB = _compute_intra_fused_mini_batch(H, BT, K, V, dtype=q.dtype)
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  # [H, B, T, K] -> [H, B, NC, BT, K]
  q_r = q.reshape(H, B, NC, BT, K)
  k_r = k.reshape(H, B, NC, BT, K)
  GW = g.shape[-1]
  scalar_gate = GW == 1
  # Store width-1 axes as [.., 1, BT] to avoid 128-lane minor-axis padding in HBM.
  g_r = (
      g.reshape(H, B, NC, 1, BT) if scalar_gate else g.reshape(H, B, NC, BT, GW)
  )
  beta_r = beta.reshape(H, B, NC, 1, BT)
  v_r = v.reshape(H, B, NC, BT, V)

  if use_gate_in_kernel:
    assert a_log is not None
    a_log_r = a_log.astype(jnp.float32).reshape(H, 1, 1, 1, 1)
    if delta_time_bias is not None:
      delta_time_bias_r = delta_time_bias.astype(jnp.float32).reshape(H, 1, 1, 1, K)
    else:
      delta_time_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)
  else:
    a_log_r = jnp.zeros((H, 1, 1, 1, 1), dtype=jnp.float32)
    delta_time_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)

  grid = (H // MB, B, NC)

  def _make_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, l: (i, j, l, 0, 0),
      block_shape=(MB, 1, 1, BT, last_dim),
    )

  def _make_halo_spec(last_dim):
    return pl.BlockSpec(
        index_map=lambda i, j, l: (i, j, l, 0, 0),
        block_shape=(MB, 1, 1, _CONV_HALO, last_dim),
    )

  def _make_weight_spec(last_dim):
    return pl.BlockSpec(
        index_map=lambda i, j, l: (i, 0, 0, 0, 0),
        block_shape=(MB, 1, 1, _CONV_HALO, last_dim),
    )

  def _make_per_head_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, l: (i, 0, 0, 0, 0),
      block_shape=(MB, 1, 1, 1, last_dim),
    )

  def _make_narrow_spec():
    """For an operand stored `[.., 1, BT]` to keep its size-1 axis off lanes."""
    return pl.BlockSpec(
        index_map=lambda i, j, l: (i, j, l, 0, 0),
        block_shape=(MB, 1, 1, 1, BT),
    )

  in_args = [q_r, k_r, g_r, beta_r, v_r, a_log_r, delta_time_bias_r]
  in_specs = [
      _make_spec(K),
      _make_spec(K),
      _make_narrow_spec() if scalar_gate else _make_spec(GW),
      _make_narrow_spec(),
      _make_spec(V),
      _make_per_head_spec(1),
      _make_per_head_spec(K),
  ]
  out_shape = [
      jax.ShapeDtypeStruct((H, B, NC, BT, V), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, GW), jnp.float32),
  ]
  out_specs = [
      _make_spec(V),
      _make_spec(K),
      _make_spec(K),
      _make_spec(BT),
      _make_spec(BT),
      _make_spec(GW),
  ]

  conv_kernel_size = 4
  if use_conv1d_in_kernel:
    assert (
        conv_weight_q is not None
        and conv_weight_k is not None
        and conv_weight_v is not None
    )
    H_Q, W, _ = conv_weight_q.shape
    H_K, _, _ = conv_weight_k.shape
    # Internal backstop; the Mosaic backend rejects other widths up front.
    assert (
        1 <= W <= _CONV_HALO
    ), f"conv_weight kernel size W={W} must be in [1, {_CONV_HALO}]"
    conv_kernel_size = W
    q_prev_r = _make_chunk_halo(q_r, aligned_segment_ids, halo=_CONV_HALO)
    k_prev_r = _make_chunk_halo(k_r, aligned_segment_ids, halo=_CONV_HALO)
    v_prev_r = _make_chunk_halo(v_r, aligned_segment_ids, halo=_CONV_HALO)
    w_q_exp = (
        jnp.repeat(conv_weight_q, H // H_Q, axis=0)
        if H_Q < H
        else conv_weight_q
    )
    w_k_exp = (
        jnp.repeat(conv_weight_k, H // H_K, axis=0)
        if H_K < H
        else conv_weight_k
    )
    b_q_exp = (
        (
            jnp.repeat(conv_bias_q, H // conv_bias_q.shape[0], axis=0)
            if conv_bias_q.shape[0] < H
            else conv_bias_q
        )
        if conv_bias_q is not None
        else None
    )
    b_k_exp = (
        (
            jnp.repeat(conv_bias_k, H // conv_bias_k.shape[0], axis=0)
            if conv_bias_k.shape[0] < H
            else conv_bias_k
        )
        if conv_bias_k is not None
        else None
    )
    w_q_pad = jnp.pad(
        w_q_exp.astype(jnp.float32), ((0, 0), (_CONV_HALO - W, 0), (0, 0))
    ).reshape(H, 1, 1, _CONV_HALO, K)
    w_k_pad = jnp.pad(
        w_k_exp.astype(jnp.float32), ((0, 0), (_CONV_HALO - W, 0), (0, 0))
    ).reshape(H, 1, 1, _CONV_HALO, K)
    w_v_pad = jnp.pad(
        conv_weight_v.astype(jnp.float32),
        ((0, 0), (_CONV_HALO - W, 0), (0, 0)),
    ).reshape(H, 1, 1, _CONV_HALO, V)
    b_q_r = (
        b_q_exp.astype(jnp.float32).reshape(H, 1, 1, 1, K)
        if b_q_exp is not None
        else jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)
    )
    b_k_r = (
        b_k_exp.astype(jnp.float32).reshape(H, 1, 1, 1, K)
        if b_k_exp is not None
        else jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)
    )
    b_v_r = (
        conv_bias_v.astype(jnp.float32).reshape(H, 1, 1, 1, V)
        if conv_bias_v is not None
        else jnp.zeros((H, 1, 1, 1, V), dtype=jnp.float32)
    )

    in_args.extend([
        q_prev_r,
        k_prev_r,
        v_prev_r,
        w_q_pad,
        w_k_pad,
        w_v_pad,
        b_q_r,
        b_k_r,
        b_v_r,
    ])
    in_specs.extend([
        _make_halo_spec(K),
        _make_halo_spec(K),
        _make_halo_spec(V),
        _make_weight_spec(K),
        _make_weight_spec(K),
        _make_weight_spec(V),
        _make_per_head_spec(K),
        _make_per_head_spec(K),
        _make_per_head_spec(V),
    ])

    out_shape.append(jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype))
    out_specs.append(_make_spec(K))
    if return_conv_intermediates:
      out_shape.extend([
          jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
          jax.ShapeDtypeStruct((H, B, NC, BT, V), v.dtype),
      ])
      out_specs.extend([
          _make_spec(K),
          _make_spec(V),
      ])
      if use_qk_l2norm:
        out_shape.extend([
            jax.ShapeDtypeStruct((H, B, NC, 1, BT), jnp.float32),
            jax.ShapeDtypeStruct((H, B, NC, 1, BT), jnp.float32),
        ])
        out_specs.extend([
            _make_narrow_spec(),
            _make_narrow_spec(),
        ])

  pallas_outs = pl.pallas_call(
      functools.partial(
          _fused_gate_intra_kernel,
          chunk_size=BT,
          head_dim=K,
          value_dim=V,
          scale=scale,
          cumsum_scale=cumsum_scale,
          fuse_cumsum=True,
          disable_recompute=disable_recompute,
          safe_gate=safe_gate,
          per_channel_gate=per_channel_gate,
          use_gate_in_kernel=use_gate_in_kernel,
          lower_bound=lower_bound,
          mini_batch=MB,
          use_conv1d_in_kernel=use_conv1d_in_kernel,
          use_qk_l2norm=use_qk_l2norm,
          return_conv_intermediates=return_conv_intermediates,
          conv_kernel_size=conv_kernel_size,
      ),
      interpret=get_interpret(),
      out_shape=out_shape,
      in_specs=in_specs,
      out_specs=out_specs,
      grid=grid,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
      ),
  )(*in_args)

  u_r, w_r, kg_r, Aqk_r, Akk_inv_r, g_cumsum_r = pallas_outs[:6]
  extra_outs = pallas_outs[6:]

  # --- Reshape back to [H, B, T, D] (head-first) ---
  w_out = w_r.reshape(H, B, T, K)
  u_out = u_r.reshape(H, B, T, V)
  kg_out = kg_r.reshape(H, B, T, K)
  Aqk_flat = Aqk_r.reshape(H, B, NC * BT, BT)
  Akk_flat = Akk_inv_r.reshape(H, B, NC * BT, BT)
  g_cumsum_out = g_cumsum_r.reshape(H, B, T, GW)

  q_norm_out = k_norm_out = v_silu_out = q_rstd_out = k_rstd_out = None
  if use_conv1d_in_kernel:
    q_norm_out = extra_outs[0].reshape(H, B, T, K)
    if return_conv_intermediates:
      k_norm_out = extra_outs[1].reshape(H, B, T, K)
      v_silu_out = extra_outs[2].reshape(H, B, T, V)
      if use_qk_l2norm:
        q_rstd_out = extra_outs[3].reshape(H, B, T)
        k_rstd_out = extra_outs[4].reshape(H, B, T)

  return (
      w_out,
      u_out,
      kg_out,
      Aqk_flat,
      Akk_flat,
      g_cumsum_out,
      q_norm_out,
      k_norm_out,
      v_silu_out,
      q_rstd_out,
      k_rstd_out,
  )


def kda_fwd_intra_fused(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
    chunk_indices: jax.Array | None = None,
    safe_gate: bool = True,
    per_channel_gate: bool = True,
    disable_recompute: bool = False,
    cumsum_scale: float = RCP_LN2,
    a_log: jax.Array | None = None,
    delta_time_bias: jax.Array | None = None,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    conv_weight_q: jax.Array | None = None,
    conv_weight_k: jax.Array | None = None,
    conv_weight_v: jax.Array | None = None,
    conv_bias_q: jax.Array | None = None,
    conv_bias_k: jax.Array | None = None,
    conv_bias_v: jax.Array | None = None,
    use_conv1d_in_kernel: bool = False,
    use_qk_l2norm: bool = False,
    return_conv_intermediates: bool = False,
    aligned_segment_ids: jax.Array | None = None,
    mini_batch: int | None = None,
):
  """Fused gate cumsum + intra-chunk solve entry.

  Args:
      g: [H, B, T, K] for a per-channel gate or [H, B, T, 1] for a scalar gate
        -- raw gate input (NOT cumsum'd). Other args configure the intra-chunk
        solve and gate activation.

  Returns:
      11-tuple: (w, u, kg, Aqk, Akk, g_cumsum, q_norm, k_norm, v_silu, q_rstd,
      k_rstd).
  """
  assert (
      per_channel_gate is False or chunk_size == 64
  ), f"per-channel gate is validated at chunk_size=64 only; got {chunk_size}"
  assert chunk_size in (
      16,
      32,
      64,
      128,
      256,
      512,
  ), f"chunk_size must be a power of two in [16, 512]; got {chunk_size}"

  return pallas_kda_fwd_intra_fused(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      scale=scale,
      chunk_size=chunk_size,
      safe_gate=safe_gate,
      per_channel_gate=per_channel_gate,
      disable_recompute=disable_recompute,
      cumsum_scale=cumsum_scale,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
      mini_batch=mini_batch,
      conv_weight_q=conv_weight_q,
      conv_weight_k=conv_weight_k,
      conv_weight_v=conv_weight_v,
      conv_bias_q=conv_bias_q,
      conv_bias_k=conv_bias_k,
      conv_bias_v=conv_bias_v,
      use_conv1d_in_kernel=use_conv1d_in_kernel,
      use_qk_l2norm=use_qk_l2norm,
      return_conv_intermediates=return_conv_intermediates,
      aligned_segment_ids=aligned_segment_ids,
  )


def _chunk_kda_fwd_h_o_varlen_kernel(
    seqlens_ref,  # scalar prefetch: cu_seqlens [N+1]
    chunk_to_seq_ref,  # scalar prefetch: chunk -> seq mapping [NT]
    # Stage 3 inputs
    w_ref,  # [MB, 1, BT, K_PADSIZE]
    u_ref,  # [MB, 1, BT, V_ALIGNED]
    kg_ref,  # [MB, 1, BT, K_PADSIZE]
    gk_ref,  # [MB, 1, BT, K_PADSIZE]  -- g_cumsum; [MB, 1, BT, 1] scalar gate
    # Stage 4 inputs
    q_ref,  # [MB, 1, BT, K_PADSIZE]
    A_ref,  # [MB, 1, BT, BT]
    # Optional initial state
    h0_ref,  # [1, MB, K_PADSIZE, V_ALIGNED] or None
    # Outputs
    o_ref,  # [MB, 1, BT, V_ALIGNED]
    ht_ref,  # [1, MB, K_PADSIZE, V_ALIGNED] or None
    h_out_ref,  # [MB, 1, 1, K_PADSIZE, V_ALIGNED] or None  -- per-chunk pre-update h
    v_new_out_ref,  # [MB, 1, BT, V_ALIGNED] or None         -- per-token v_new
    # Scratch
    scratch_ref,  # [MB, K_PADSIZE, V_ALIGNED]
    *,
    BT,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    STORE_H,
    STORE_V_NEW,
    MB,
    OUTPUT_PRECISION,
):
  """Fused Stage 3+4 Pallas kernel body for varlen with H-dim mini-batch.

  Grid is (H // MB, B, NT). For each program point (h_group, i_b, i_c):
    - MB heads [h_group*MB .. h_group*MB+MB) are processed per grid point
      via batched matmuls over the MB dimension.
    - i_b is the batch index, i_c is the chunk index within that batch.
    - seq_idx = chunk_to_seq[i_b, i_c] identifies which sequence this chunk
      belongs to within batch i_b.
    - At t0 == bos: init scratch (h0 or zeros) for this sequence
    - At t0 + BT >= eos: store final state for this sequence
  """
  i_b = pl.program_id(1)
  i_c = pl.program_id(2)
  seq_idx = chunk_to_seq_ref[i_b, i_c]

  bos = seqlens_ref[i_b, seq_idx]
  eos = seqlens_ref[i_b, seq_idx + 1]
  t0 = i_c * BT

  K = w_ref.shape[-1]
  V = u_ref.shape[-1]

  # === Init state (first chunk of THIS sequence) — uniform across all MB heads ===
  @pl.when(t0 == bos)
  def _():
    scratch_ref[:] = jnp.zeros([MB, K, V], dtype=jnp.float32)
    if USE_INITIAL_STATE:
      scratch_ref[:] = h0_ref[0, 0].astype(jnp.float32)  # [MB, K, V]

  # === Stage 3+4 work — batched over MB heads ===
  # h: pre-update state for all MB heads in this tile.
  b_h = scratch_ref[:]   # [MB, K, V]

  # Spill pre-update h (saved residual for bwd; tagged with
  # checkpoint_name("kda_residuals") at the custom_vjp boundary).
  if STORE_H:
    h_out_ref[:, 0, 0] = b_h.astype(h_out_ref.dtype)  # [MB, K, V]

  b_w = w_ref[:,0,:]   # [MB, BT, K]
  b_u = u_ref[:,0,:]   # [MB, BT, V]

  # Stage 3 delta correction: v_new = u - w @ h
  # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
  # v_new feeds directly into the recursive state update, so this matmul and
  # the state update below are the two on the serial chain. Their precision
  # is switchable; see `get_chain_precision`.
  b_v_new = b_u.astype(jnp.float32) - jnp.matmul(
      b_w.astype(jnp.float32),
      b_h,
      precision=get_chain_precision(),
      preferred_element_type=jnp.float32,
  )  # [MB, BT, V]

  # Spill v_new (used by bwd Stage 0 when disable_recompute=True).
  if STORE_V_NEW:
    v_new_out_ref[:,0,:] = b_v_new.astype(v_new_out_ref.dtype)

  # Stage 4 inter-chunk output: scale * (q * exp2(g - g_ref)) @ (h * exp2(g_ref))
  # Reference-point stabilization mirrors the production varlen kernel's
  # g_ref = g[0] choice for bit-identical numerics.
  b_q = q_ref[:,0,:]                              # [MB, BT, K]
  b_g = gk_ref[:,0,:].astype(jnp.float32)         # [MB, BT, K]
  b_A = A_ref[:,0,:]                               # [MB, BT, BT]

  b_g_ref_row = b_g[:, 0:1, :]                # [MB, 1, K]
  b_qg = b_q.astype(jnp.float32) * jnp.exp2(
    jnp.maximum(b_g - b_g_ref_row, -126.0)
  )                                            # [MB, BT, K]
  b_h_scaled = b_h * jnp.exp2(
    jnp.maximum(b_g_ref_row[:, 0, :], -126.0)
  )[:, :, None]                                # [MB, K, V]

  # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
  # Output-only GEMM: bf16 inputs use DEFAULT (no extra precision to preserve),
  # fp32 inputs use HIGHEST to maintain full mantissa fidelity.
  b_o = jnp.matmul(
    b_qg, b_h_scaled,
    precision=OUTPUT_PRECISION,
    preferred_element_type=jnp.float32,
  ) * scale                                    # [MB, BT, V]

  # Stage 4 intra-chunk: A @ v_new
  # Apply lower-triangular mask: Aqk is causal by construction, but masking
  # here matches the original per-head loop behaviour and guards against
  # any tiny upper-triangle fp noise from the Neumann intra-chunk solve.
  m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]  # [BT, BT]
  b_A_f32 = jnp.where(m_s[None, :, :], b_A.astype(jnp.float32), 0.0)
  # [MB, BT, BT] @ [MB, BT, V] -> [MB, BT, V]
  b_o = b_o + jnp.matmul(
    b_A_f32, b_v_new,
    precision=OUTPUT_PRECISION,
    preferred_element_type=jnp.float32,
  )                                            # [MB, BT, V]

  o_ref[:,0,:] = b_o.astype(o_ref.dtype)

  # Stage 3 state update: h = decay(h) + kg^T @ v_new
  # On the serial chain: accumulates directly into the recursive hidden state,
  # so its precision comes from `get_chain_precision` (HIGHEST by default).
  b_gk_last = gk_ref[:,0,:][:, BT - 1, :].astype(jnp.float32)  # [MB, K]
  b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None]          # [MB, K, V] decay

  b_kg = kg_ref[:,0,:]   # [MB, BT, K]
  # [MB, K, BT] @ [MB, BT, V] -> [MB, K, V]
  b_h_new = b_h_new + jnp.matmul(
      b_kg.astype(jnp.float32).transpose(0, 2, 1),
      b_v_new,
      precision=get_chain_precision(),
      preferred_element_type=jnp.float32,
  )
  scratch_ref[:] = b_h_new

  # === Final state (last chunk of THIS sequence) ===
  @pl.when(t0 + BT >= eos)
  def _():
    if STORE_FINAL_STATE:
      ht_ref[0, 0] = scratch_ref[:].astype(ht_ref.dtype)  # [MB, K, V]


@functools.partial(
  jax.jit,
  static_argnames=[
    "output_final_state",
    "scale",
    "chunk_size",
    "store_h",
    "store_v_new",
    "store_intermediates",
    "mini_batch",
  ],
)
def chunk_kda_fwd_h_o_varlen(
    w: jax.Array,
    u: jax.Array,
    kg: jax.Array,
    gk: jax.Array,
    q: jax.Array,
    A: jax.Array,
    cu_seqlens: jax.Array,
    chunk_indices: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    scale: float = 1.0,
    chunk_size: int = 64,
    store_h: bool = False,
    store_v_new: bool = False,
    store_intermediates: bool | None = None,
    mini_batch: int | None = None,
) -> tuple[
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
  """Fuses varlen state propagation with output projection.

  The recurrent state remains in VMEM. `store_h` and `store_v_new` spill
  only the intermediates needed by backward; `store_intermediates` is the
  legacy alias that enables both.
  """
  # Back-compat shim: old single-flag callers map to both stores.
  if store_intermediates is not None:
    store_h = store_h or store_intermediates
    store_v_new = store_v_new or store_intermediates
  H, B, T, K = q.shape
  V = u.shape[-1]
  BT = chunk_size

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  # Ensure cu_seqlens is 2D [B, N+1] for kernel block specs
  if cu_seqlens.ndim == 1:
    cu_seqlens = jnp.broadcast_to(cu_seqlens[None, :], (B, cu_seqlens.shape[0]))
  assert A.shape[-1] == BT, (
      f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}"
  )

  N = cu_seqlens.shape[-1] - 1
  if initial_state is not None:
    assert initial_state.shape[1] == N, (
        f"initial_state has N={initial_state.shape[1]}, expected {N}"
    )
  assert K <= 256, "current kernel does not support K > 256."

  hw = get_tpu_limits()
  K_PADSIZE = int(align_up(K, hw.block_align_major))
  V_ALIGNED = int(align_up(V, hw.block_align_major))

  # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
  if mini_batch is None:
    elem_size = q.dtype.itemsize
    # scratch per head: h_state[K_PADSIZE*V_ALIGNED] in f32 + i/o buffers
    in_bytes = (BT * K_PADSIZE + BT * V_ALIGNED + BT) * elem_size
    out_bytes = BT * V_ALIGNED * elem_size
    scratch_bytes = K_PADSIZE * V_ALIGNED * 4  # float32 accumulator
    per_head = in_bytes + out_bytes + scratch_bytes
    MB = estimate_mini_batch(per_head, H)  # pylint: disable=invalid-name
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  # Generate chunk_to_seq mapping from chunk_indices
  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
  NT = chunk_indices.shape[-2]
  chunk_to_seq = chunk_indices[..., 0].astype(jnp.int32)  # [NT] or [B, NT]
  # Kernel block specs index with [b, c], so ensure 2D [B, NT].
  if chunk_to_seq.ndim == 1:
    chunk_to_seq = jnp.broadcast_to(chunk_to_seq[None, :], (B, NT))

  # grid = (H // MB, NT) with NT = T // BT, so chunk index c ranges
  # 0..NT-1 and the maximum T offset accessed is (NT-1)*BT+BT = T.
  # No extra trailing chunk is ever touched, so T_alloc == T suffices.
  T_alloc = T

  # Pad K (last dim) to K_PADSIZE if needed, then transpose to
  # [B=1, H, T, K_PADSIZE]. No T-dim padding required (see T_alloc above).
  # Inputs stay in their original dtype (typically bf16); the kernel casts
  # tile-level slices to f32 on the fly in VMEM, avoiding a full-tensor
  # bf16->f32 cast in HBM that doubles memory and DMA bandwidth.
  def _pad_kdim_then_t(x, dim_pad):
    if dim_pad > 0:
      x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, dim_pad)))
    return x

  GW = gk.shape[-1]  # K for a per-channel gate, 1 for a scalar one
  w_t  = _pad_kdim_then_t(w,  K_PADSIZE - K)
  kg_t = _pad_kdim_then_t(kg, K_PADSIZE - K)
  gk_t = _pad_kdim_then_t(gk, (K_PADSIZE - K) if GW == K else 0)
  q_t  = _pad_kdim_then_t(q,  K_PADSIZE - K)
  u_t  = _pad_kdim_then_t(u,  V_ALIGNED - V)

  # A is [B, T, H, BT]; transpose to [B=1, H, T, BT]. No T padding needed.
  A_t = A  # [H, B, T, BT]

  # h0: [B, N, H, K, V] -> pad K and V -> [B, N, H, K_PADSIZE, V_ALIGNED]
  if initial_state is not None:
    h0 = initial_state
    if V_ALIGNED > V:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    if K_PADSIZE > K:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
  else:
    h0 = None

  # Index maps with MB heads per grid point. Scalar prefetch order:
  # (seqlens_ref, chunk_to_seq_ref).
  # Inputs are [H, B, T_alloc, X]; BlockSpec slices MB heads at h*MB.
  def _t_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0)

  def _A_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0)

  bspec_k = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
  # Narrow when the gate is scalar; identical to bspec_k otherwise.
  bspec_gk = pl.BlockSpec(
      [MB, 1, BT, K_PADSIZE if GW == K else GW], index_map=_t_index_map
  )
  bspec_v = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
  bspec_a = pl.BlockSpec([MB, 1, BT, BT],        index_map=_A_index_map)
  bspec_h0 = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (
        b, chunk_to_seq_ref[b, c], h, 0, 0
      ),
    )
    if h0 is not None else None
  )

  # Output specs.
  o_spec = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
  ht_spec = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (
        b, chunk_to_seq_ref[b, c], h, 0, 0
      ),
    )
    if output_final_state else None
  )
  # Per-chunk h spill (pre-update, used by bwd save-h fast path). Layout
  # [H, B, NT, K_PADSIZE, V_ALIGNED]
  h_out_spec = (
    pl.BlockSpec(
      [MB, 1, 1, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (
        h, b, c, 0, 0
      ),
    )
    if store_h else None
  )
  # Per-token v_new spill, [H, B, T_alloc, V_ALIGNED].
  v_new_out_spec = (
    pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
    if store_v_new else None
  )

  o_shape = jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], jnp.float32)
  ht_shape = (
    jax.ShapeDtypeStruct([B, N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
    if output_final_state else None
  )
  # h_per_chunk dtype mirrors u (matches unfused chunk_gated_delta_rule_fwd_h
  # which produces h in u.dtype, typically bf16/fp32).
  h_out_shape = (
    jax.ShapeDtypeStruct([H, B, NT, K_PADSIZE, V_ALIGNED], u.dtype)
    if store_h else None
  )
  v_new_out_shape = (
    jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], u.dtype)
    if store_v_new else None
  )

  scratch = pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32)
  grid = (H // MB, B, NT)
  interpret = get_interpret()

  # bf16 inputs: DEFAULT precision is lossless (operands already have ~7-bit
  # mantissa); fp32 inputs: HIGHEST preserves full 23-bit mantissa fidelity.
  # This applies to the output matmuls only. The two serial-chain matmuls
  # (`w @ h` and the state update) use `get_chain_precision` instead.
  _output_prec = (
    jax.lax.Precision.DEFAULT
    if q.dtype == jnp.bfloat16
    else jax.lax.Precision.HIGHEST
  )

  o_out, ht_out, h_out, v_new_out = pl.pallas_call(
      functools.partial(
          _chunk_kda_fwd_h_o_varlen_kernel,
          BT=BT,
          scale=scale,
          USE_INITIAL_STATE=(h0 is not None),
          STORE_FINAL_STATE=output_final_state,
          STORE_H=store_h,
          STORE_V_NEW=store_v_new,
          MB=MB,
          OUTPUT_PRECISION=_output_prec,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          grid=grid,
          in_specs=[
              bspec_k,  # w
              bspec_v,  # u
              bspec_k,  # kg
              bspec_gk,  # gk
              bspec_k,  # q
              bspec_a,  # A
              bspec_h0,  # h0
          ],
          out_specs=[o_spec, ht_spec, h_out_spec, v_new_out_spec],
          scratch_shapes=[scratch],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[o_shape, ht_shape, h_out_shape, v_new_out_shape],
      interpret=interpret,
  )(
      cu_seqlens.astype(jnp.int32),
      chunk_to_seq,
      w_t,
      u_t,
      kg_t,
      gk_t,
      q_t,
      A_t,
      h0,
  )

  # Post-process: o is [H, 1, T, V_ALIGNED]
  if V_ALIGNED > V:
    o_out = o_out[..., :V]
  o_out = o_out.astype(u.dtype)

  if output_final_state and ht_out is not None:
    if V_ALIGNED > V:
      ht_out = ht_out[..., :V]
    if K_PADSIZE > K:
      ht_out = ht_out[..., :K, :]

    # Handle empty sequences: sequences with no chunks never execute kernel code,
    # so their final_state is uninitialized. Fill them with initial_state or zeros.
    seq_lens = jnp.diff(cu_seqlens, axis=-1)
    empty_mask = (seq_lens == 0)  # [B, N]
    if initial_state is not None:
      # For empty sequences, final_state should equal initial_state
      fill_value = initial_state[:, :, :, :K, :V]
    else:
      # For empty sequences without initial_state, final_state should be zeros
      fill_value = jnp.zeros((B, N, H, K, V), dtype=ht_out.dtype)
    # Use where to selectively replace empty sequence states
    ht_out = jnp.where(empty_mask[:, :, None, None, None], fill_value, ht_out)
  else:
    ht_out = None

  # Post-process spilled intermediates (independently controlled).
  if store_h:
    # h_out: [1, NT, H, K_PADSIZE, V_ALIGNED] -> trim K/V padding back.
    if V_ALIGNED > V:
      h_out = h_out[..., :V]
    if K_PADSIZE > K:
      h_out = h_out[..., :K, :]

  else:
    h_out = None
  if store_v_new:
    if V_ALIGNED > V:
      v_new_out = v_new_out[..., :V]
  else:
    v_new_out = None

  return o_out, ht_out, h_out, v_new_out


# ==============================================================================
# Forward on the Pallas kernels
# (call sequence of `PallasMosaicTpuKimiDeltaAttention._fwd` in
#  tokamax/_src/ops/kda/pallas_mosaic_tpu.py and `chunk_kda_fwd_custom` in
#  tokamax/_src/ops/kda/pallas_mosaic_tpu_fwd_kernel.py, with the options fixed
#  as in `forward`)
# ==============================================================================

# Chunk length. The Tokamax TPU kernels require 64 for a per-channel gate.
_CHUNK_SIZE = 64

# Off TPU the kernels run in Pallas interpret mode, but their block sizing still
# reads `pltpu.get_tpu_info()`, which needs a target chip. Tokamax supports
# these kernels on TPU generation 6 and newer.
_INTERPRET_TARGET_TPU = 'TPU v6e'


def _forward_pallas(q, k, v, g, beta, a_log, delta_time_bias):
  """Runs the two forward kernels and collects the saved intermediates."""
  _, B, T, K = q.shape
  BT = _CHUNK_SIZE
  scale = K**-0.5

  # Queries and keys are L2-normalised before the kernels.
  q_norm, q_rstd = l2norm_fwd(q)
  k_norm, k_rstd = l2norm_fwd(k)

  # Stage 1 + 2: gate activation, chunk-local gate cumsum and intra-chunk
  # solve. The mini-batch of heads is left to the kernels' VMEM heuristic.
  w, u, kg, Aqk, Akk, g_cumsum, _, _, _, _, _ = kda_fwd_intra_fused(
      q=q_norm,
      k=k_norm,
      v=v,
      g=g,
      beta=beta,
      scale=scale,
      cu_seqlens=None,
      chunk_size=BT,
      chunk_indices=None,
      safe_gate=True,
      per_channel_gate=True,
      disable_recompute=True,
      cumsum_scale=RCP_LN2,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      use_gate_in_kernel=True,
      lower_bound=_LOWER_BOUND,
      use_conv1d_in_kernel=False,
      use_qk_l2norm=True,
      return_conv_intermediates=False,
      aligned_segment_ids=None,
      mini_batch=None,
  )

  # Stage 3 + 4: inter-chunk state propagation and output. A fixed-length
  # batch is represented as one sequence `[0, T]` per batch element.
  stage34_cu_seqlens = jnp.broadcast_to(
      jnp.asarray([0, T], dtype=jnp.int32), (B, 2)
  )
  block_ids = jnp.arange(T // BT, dtype=jnp.int32)
  fixed_chunk_indices = jnp.stack(
      [jnp.zeros_like(block_ids), block_ids], axis=-1
  )
  stage34_chunk_indices = jnp.broadcast_to(
      fixed_chunk_indices[None, :, :], (B, T // BT, 2)
  )
  o, _, h, v_new = chunk_kda_fwd_h_o_varlen(
      w=w,
      u=u,
      kg=kg,
      gk=g_cumsum,
      q=q_norm,
      A=Aqk,
      cu_seqlens=stage34_cu_seqlens,
      chunk_indices=stage34_chunk_indices,
      initial_state=None,
      output_final_state=False,
      scale=scale,
      chunk_size=BT,
      store_h=True,
      store_v_new=True,
      mini_batch=None,
  )

  out = o.astype(q.dtype)
  residuals = {
      'q': q_norm,
      'k': k_norm,
      'q_rstd': q_rstd,
      'k_rstd': k_rstd,
      'g_cumsum': g_cumsum,
      'aqk': Aqk,
      'akk': Akk,
      'h': h,
      'v_new': v_new,
  }
  return out, residuals


def forward_pallas(q, k, v, g, beta, a_log, delta_time_bias):
  """KDA forward pass on the Pallas TPU kernels.

  Takes the same arguments as `forward`; `T` must be a multiple of 64 and `K`
  at most 256. Returns `(out, residuals)`; see the module docstring for the
  entries of `residuals`. Off TPU the kernels run in Pallas interpret mode
  with block sizes chosen for `_INTERPRET_TARGET_TPU`.
  """
  if jax.default_backend() == 'tpu':
    return _forward_pallas(q, k, v, g, beta, a_log, delta_time_bias)
  target = jax.sharding.AbstractDevice(
      device_kind=_INTERPRET_TARGET_TPU, num_cores=1, platform='tpu'
  )
  mesh = jax.sharding.AbstractMesh((), (), abstract_device=target)
  with jax.sharding.use_abstract_mesh(mesh):
    return _forward_pallas(q, k, v, g, beta, a_log, delta_time_bias)


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # The Kimi Linear 48B-A3B training shape of the tokamax arg_specs
    # (fixed_training_b1_t8192_bf16) with the number of heads and the sequence
    # length reduced, so that the states saved by the autodiff oracle fit in
    # TPU memory: `workload` keeps close to four float32 `K x V` states per
    # token and head, about 4 GB here. The figure is an estimate from XLA
    # memory analysis on CPU, not a TPU measurement.
    'kda_b1_t4096_h4': {
        'name': 'kda_b1_t4096_h4_bwd_pallas_fwd',
        'model': 'Kimi-Linear-48B-A3B',
        'operator': 'kda_backward',
        'num_heads': 4,
        'batch': 1,
        'seq_len': 4096,
        'key_dim': 128,
        'val_dim': 128,
    },
    # tokamax arg_specs: fixed_training_b1_t8192_bf16 (Kimi Linear 48B-A3B) at
    # full size. By the same estimate the autodiff oracle needs about 56 GB for
    # this shape.
    'kda_b1_t8192': {
        'name': 'kda_b1_t8192_bwd_pallas_fwd',
        'model': 'Kimi-Linear-48B-A3B',
        'operator': 'kda_backward',
        'num_heads': 32,
        'batch': 1,
        'seq_len': 8192,
        'key_dim': 128,
        'val_dim': 128,
    },
    # Tiny shapes for CPU smoke tests. `seq_len` stays a multiple of 64, the
    # chunk size of the Tokamax Pallas kernels for a per-channel gate.
    'smoke': {
        'name': 'kda_smoke_bwd_pallas_fwd',
        'model': 'Kimi-Linear-48B-A3B',
        'operator': 'kda_backward',
        'num_heads': 2,
        'batch': 2,
        'seq_len': 128,
        'key_dim': 16,
        'val_dim': 8,
    },
}
CONFIG = CONFIGS['kda_b1_t4096_h4']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(q, k, v, g, beta, a_log, delta_time_bias, dout)`.

  `dout` is the output cotangent. `a_log` and `delta_time_bias` are model
  parameters and stay float32 for every `dtype`, as in the Tokamax arg specs.
  """
  cfg = _resolve_config(config)
  H, B, T = cfg['num_heads'], cfg['batch'], cfg['seq_len']
  K, V = cfg['key_dim'], cfg['val_dim']
  keys = jax.random.split(jax.random.key(42), 8)
  qk_shape = (H, B, T, K)
  v_shape = (H, B, T, V)
  # Value ranges follow the production-range ("prod") input profile of the
  # Tokamax KDA kernel tests, which keeps the recurrence well-conditioned:
  # SiLU-activated queries/keys, learning rates in (0.05, 0.95), and gate
  # parameters for which most channels decay slowly.
  q = jax.nn.silu(jax.random.normal(keys[0], qk_shape))
  k = jax.nn.silu(jax.random.normal(keys[1], qk_shape))
  v = 1.5 * jax.random.normal(keys[2], v_shape)
  g = jax.random.uniform(keys[3], qk_shape, minval=-4.5, maxval=4.5)
  beta = jax.random.uniform(keys[4], (H, B, T), minval=0.05, maxval=0.95)
  a_log = jax.random.uniform(keys[5], (H,), minval=0.2, maxval=3.0)
  delta_time_bias = jax.random.uniform(
      keys[6], (H * K,), minval=-8.0, maxval=-1.5
  )
  dout = jax.random.normal(keys[7], v_shape)
  # Sample in float32 and cast, so every dtype sees the same values.
  q, k, v, g, beta, dout = (x.astype(dtype) for x in (q, k, v, g, beta, dout))
  return q, k, v, g, beta, a_log, delta_time_bias, dout


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 7, 'diff_argnums': (0, 1, 2, 3, 4, 5, 6)}


def workload(q, k, v, g, beta, a_log, delta_time_bias, dout):
  """Returns the VJP of `forward` with `dout`.

  The result is `(dq, dk, dv, dg, dbeta, da_log, ddelta_time_bias)`, each with
  the shape and dtype of its primal.
  """
  _, vjp_fn = jax.vjp(forward, q, k, v, g, beta, a_log, delta_time_bias)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  Per token and head the forward recurrence does a `K x V` state decay (KV),
  a `K`-vector by `K x V` matmul (2KV), a rank-1 state update (KV) and a
  second vector-matrix product (2KV): about 6KV. The VJP runs the forward once
  and a backward that costs about twice the forward, so 3x in total.
  """
  cfg = _resolve_config(config)
  H, B, T = cfg['num_heads'], cfg['batch'], cfg['seq_len']
  K, V = cfg['key_dim'], cfg['val_dim']
  return int(3 * 6 * B * T * H * K * V)


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
