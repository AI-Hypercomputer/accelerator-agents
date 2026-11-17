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
"""RWKV-7 (DPLR delta rule) backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

`forward_pallas(q, k, v, alpha, beta, gk)` is the Tokamax Pallas TPU forward
kernel. It returns `(out, residuals)`, where `residuals = (a_ab_inv_c,
merged_out_c, s_in)` are the per-chunk intermediates the kernel saves for a
backward pass:
  a_ab_inv_c    `[B, H, T/64, 64, 64]`     `(I - A_ab)^-1` of the intra-chunk
                                           system, in the dtype of `q`
  merged_out_c  `[B, H, T/64, 64, K + V]`  the solved factors `w` (first `K`
                                           columns) and `u` (last `V`), float32
  s_in          `[B, H, T/64, K, V]`       recurrent state entering each chunk,
                                           float32

`workload(q, k, v, alpha, beta, gk, dout)` returns the gradients `(dq, dk, dv,
dalpha, dbeta, dgk)` for the output cotangent `dout`. It is the correctness
oracle: JAX autodiff (`jax.vjp`) of the pure-JAX chunked recurrence `forward`.
This file contains no backward kernel and no hand-derived backward pass.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` (it
may consume the saved residuals) so that your `workload` returns the same
gradients as this one.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/experimental/rwkv7/pallas_mosaic_tpu_kernel.py  (forward kernel)
  tokamax/_src/ops/experimental/rwkv7/pallas_mosaic_tpu.py  (residuals kept)
  tokamax/_src/ops/experimental/rwkv7/base.py       (forward recurrence)
  tokamax/_src/ops/experimental/rwkv7/arg_specs.py  (shapes, options)
  tokamax/_src/ops/experimental/rwkv7/pallas_mosaic_tpu_test.py  (value ranges)
The only change to the reference maths: `forward` sums the log-decay `gk` over
each chunk in float32 instead of the input dtype.
"""

# Dimensions.
#   B   batch size
#   T   sequence length (nt * tt)
#   H   number of attention heads
#   K   key head dimension
#   V   value head dimension
#   BH  flattened batch-head count (B * H)
#   tb  stream micro-batch tile extent dividing BH
#   tt  sequence chunk extent (chunk size)
#   nb  grid extent over BH: nb = BH // tb
#   nt  grid extent over T (number of chunks): nt = T // tt

import functools
import json
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/experimental/rwkv7/base.py)
# ==============================================================================

_CHUNK_SIZE = 64


def _chunkify(x: jax.Array, B: int, nt: int, tt: int) -> jax.Array:
  # [B, T, H, ...] -> [B, H, nt, tt, ...]
  _, _, H, *rest = x.shape
  x = x.reshape((B, nt, tt, H, *rest))
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _move_nt_front(x: jax.Array) -> jax.Array:
  return jnp.moveaxis(x, 2, 0)


def _mm(x: jax.Array, y: jax.Array) -> jax.Array:
  # Pin HIGHEST precision on f32 contractions so TPU MXU does not downcast
  # reference matmuls to bf16.
  return jnp.matmul(
      x,
      y,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def _scan_body(
    s: jax.Array, chunk_inputs: tuple[Any, ...]
) -> tuple[jax.Array, jax.Array]:
  q_i, k_i, v_i, beta_i, gk_cs_i, aqk_i, aqb_i, u_i, w_i = chunk_inputs
  v2_i = u_i + _mm(w_i, s)
  o_1 = _mm(aqk_i, v_i)
  o_2 = _mm(aqb_i, v2_i)
  o_3 = _mm(q_i * jnp.exp(jnp.minimum(gk_cs_i, 0.0)), s)
  o_i = o_1 + o_2 + o_3
  g_last = gk_cs_i[-1]
  decay_last = jnp.exp(jnp.minimum(g_last[None, :] - gk_cs_i, 0.0))
  s_new = (
      s * jnp.exp(jnp.minimum(g_last, 0.0))[:, None]
      + _mm((k_i * decay_last).T, v_i)
      + _mm((beta_i * decay_last).T, v2_i)
  )
  return s_new.astype(s.dtype), o_i.astype(u_i.dtype)


def _intra_chunk_one_plain(
    q_i: jax.Array,
    k_i: jax.Array,
    v_i: jax.Array,
    alpha_i: jax.Array,
    beta_i: jax.Array,
    gk_i: jax.Array,
    gk_cs_i: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  tt, _ = q_i.shape
  _, V = v_i.shape
  row_idx = jnp.arange(tt)[:, None]
  col_idx = jnp.arange(tt)[None, :]
  incl_causal = (col_idx <= row_idx).astype(jnp.float32)
  strict_causal = (col_idx < row_idx).astype(jnp.float32)

  decay_incl = jnp.exp(
      jnp.minimum(gk_cs_i[:, None, :] - gk_cs_i[None, :, :], 0.0)
  )
  aqk = (
      jnp.sum(q_i[:, None, :] * k_i[None, :, :] * decay_incl, axis=-1)
      * incl_causal
  )
  aqb = (
      jnp.sum(q_i[:, None, :] * beta_i[None, :, :] * decay_incl, axis=-1)
      * incl_causal
  )

  gk_excl = gk_cs_i - gk_i
  decay_excl = jnp.exp(
      jnp.minimum(gk_excl[:, None, :] - gk_cs_i[None, :, :], 0.0)
  )
  a_ab = (
      jnp.sum(alpha_i[:, None, :] * beta_i[None, :, :] * decay_excl, axis=-1)
      * strict_causal
  )
  a_ak = (
      jnp.sum(alpha_i[:, None, :] * k_i[None, :, :] * decay_excl, axis=-1)
      * strict_causal
  )

  # Invert (I - A_ab) via Neumann series doubling since A_ab is strictly lower
  # triangular (nilpotent with A_ab^tt = 0).
  eye = jnp.eye(tt, dtype=jnp.float32)
  a_ab_f32 = a_ab.astype(jnp.float32)
  p = eye + a_ab_f32
  curr = a_ab_f32
  num_doubling_steps = max(0, math.ceil(math.log2(tt)) - 1)
  for _ in range(num_doubling_steps):
    curr = _mm(curr, curr)
    p = _mm(p, eye + curr)
  a_ab_inv = p

  ak_v = _mm(a_ak, v_i)
  decayed_alpha = jnp.exp(jnp.minimum(gk_excl, 0.0)) * alpha_i
  merged_rhs = jnp.concat([ak_v, decayed_alpha], axis=-1)
  merged_uw = _mm(a_ab_inv, merged_rhs)
  u_i = merged_uw[:, :V]
  w_i = merged_uw[:, V:]

  return (
      aqk.astype(q_i.dtype),
      aqb.astype(q_i.dtype),
      u_i.astype(q_i.dtype),
      w_i.astype(q_i.dtype),
  )


def forward(q, k, v, alpha, beta, gk):
  """RWKV-7 forward pass: chunked diagonal-plus-low-rank delta-rule recurrence.

  Per head, with a `[K, V]` state `S` that starts at zero, token `t` computes

    S_t = (S_{t-1} * exp(gk_t)[:, None]
           + beta_t[:, None] * (alpha_t @ S_{t-1})[None, :]
           + k_t[:, None] * v_t[None, :])
    o_t = (q_t * K ** -0.5) @ S_t

  evaluated `tt = 64` tokens at a time: the intra-chunk interactions are solved
  in closed form and the state is carried from chunk to chunk by a scan. The
  two forms agree for `gk <= 0`, where the `jnp.minimum(..., 0.0)` clamps on
  the log-decay differences are inactive.

  Args:
    q: Queries, `[B, T, H, K]`.
    k: Keys, `[B, T, H, K]`.
    v: Values, `[B, T, H, V]`.
    alpha: Low-rank removal vectors, `[B, T, H, K]`.
    beta: Low-rank replacement vectors, `[B, T, H, K]`.
    gk: Log-space diagonal decay, `[B, T, H, K]`, values `<= 0`.

  Returns:
    The output, `[B, T, H, V]`, in the dtype of `q`. `T` must be a multiple of
    64. The state starts at zero and the final state is not returned. Queries
    are scaled by `K ** -0.5`.
  """
  B, T, H, K = q.shape
  _, _, _, V = v.shape
  tt = _CHUNK_SIZE
  nt = T // tt
  scale = float(K**-0.5)

  q_c = _chunkify(q, B, nt, tt) * scale
  k_c = _chunkify(k, B, nt, tt)
  v_c = _chunkify(v, B, nt, tt)
  alpha_c = _chunkify(alpha, B, nt, tt)
  beta_c = _chunkify(beta, B, nt, tt)
  gk_c = _chunkify(gk, B, nt, tt)
  # Unlike the Tokamax reference, which sums `gk` in the input dtype, the
  # cumulative log-decay is summed in float32, as the Tokamax Pallas kernel
  # does: a bfloat16 running sum over a 64-token chunk loses the decay.
  gk_cs_c = jnp.cumsum(gk_c.astype(jnp.float32), axis=-2)

  vmap_intra = jax.vmap(jax.vmap(jax.vmap(_intra_chunk_one_plain)))
  aqk, aqb, u, w = vmap_intra(q_c, k_c, v_c, alpha_c, beta_c, gk_c, gk_cs_c)

  q_s, k_s, v_s, beta_s, gk_cs_s, aqk_s, aqb_s, u_s, w_s = map(
      _move_nt_front, (q_c, k_c, v_c, beta_c, gk_cs_c, aqk, aqb, u, w)
  )

  scan_body_batched = jax.vmap(jax.vmap(_scan_body))

  s0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)

  _, o_stacked = jax.lax.scan(
      scan_body_batched,
      s0,
      (q_s, k_s, v_s, beta_s, gk_cs_s, aqk_s, aqb_s, u_s, w_s),
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(B, T, H, V)
  return o


# ==============================================================================
# Pallas TPU forward kernel
# (inlined from tokamax/_src/ops/experimental/rwkv7/pallas_mosaic_tpu_kernel.py,
#  Copyright 2026 Google LLC, Apache-2.0)
# ==============================================================================

# `_CHUNK_SIZE`, `_chunkify` and `_mm` in the kernel file are the same as the
# definitions from base.py above and are not repeated here.


def _pick_tb(BH: int) -> int:
  for candidate in (4, 2, 1):
    if BH % candidate == 0:
      return candidate
  return 1


def _rwkv7_fwd_kernel(
    # Inputs.
    q_ref,  # [1, 1, tb, tt, K] dtype
    k_ref,  # [1, 1, tb, tt, K] dtype
    v_ref,  # [1, 1, tb, tt, V] dtype
    alpha_ref,  # [1, 1, tb, tt, K] dtype
    beta_ref,  # [1, 1, tb, tt, K] dtype
    gk_ref,  # [1, 1, tb, tt, K] dtype
    gk_cs_ref,  # [1, 1, tb, tt, K] f32
    s0_ref,  # [1, tb, K, V] f32
    # Outputs.
    o_ref,  # [1, 1, tb, tt, V] dtype
    a_ab_inv_ref,  # [1, 1, tb, tt, tt] dtype
    merged_out_ref,  # [1, 1, tb, tt, K + V] f32
    s_in_ref,  # [1, 1, tb, K, V] f32
    s_final_ref,  # [1, tb, K, V] f32
    # Scratch.
    s_scratch_ref,  # [tb, K, V] f32  persists across the nt grid axis
    *,
    nt: int,
):
  """Single-pass fused RWKV-7 forward Pallas kernel."""
  t_i = pl.program_id(1)
  _, _, _, tt, K = q_ref.shape

  @pl.when(t_i == 0)
  def _():
    s_scratch_ref[...] = s0_ref[0].astype(jnp.float32)

  q_i = q_ref[0, 0].astype(jnp.float32)
  k_i = k_ref[0, 0].astype(jnp.float32)
  v_i = v_ref[0, 0].astype(jnp.float32)
  alpha_i = alpha_ref[0, 0].astype(jnp.float32)
  beta_i = beta_ref[0, 0].astype(jnp.float32)
  gk_i = gk_ref[0, 0].astype(jnp.float32)
  gk_cs = gk_cs_ref[0, 0].astype(jnp.float32)

  # Broadcasted iota avoids host-to-device constant materialization for masks.
  row_idx = jax.lax.broadcasted_iota(jnp.int32, (1, tt, tt), 1)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (1, tt, tt), 2)
  incl_causal = (col_idx <= row_idx).astype(jnp.float32)
  strict_causal = (col_idx < row_idx).astype(jnp.float32)
  eye = (row_idx == col_idx).astype(jnp.float32)

  # Center cumulative log-decay around the chunk midpoint g0 so both positive
  # and negative exponentials stay within f32 dynamic range before MXU matmul.
  gk_excl = gk_cs - gk_i
  g0 = gk_cs[:, tt // 2 : tt // 2 + 1, :]
  exp_pos_incl = jnp.exp(jnp.clip(gk_cs - g0, -80.0, 80.0))
  exp_pos_excl = jnp.exp(jnp.clip(gk_excl - g0, -80.0, 80.0))
  exp_neg = jnp.exp(jnp.clip(g0 - gk_cs, -80.0, 80.0))

  q_hat = q_i * exp_pos_incl
  alpha_hat = alpha_i * exp_pos_excl
  k_hat = k_i * exp_neg
  beta_hat = beta_i * exp_neg

  a_qk = _mm(q_hat, k_hat.swapaxes(-1, -2)) * incl_causal
  a_qb = _mm(q_hat, beta_hat.swapaxes(-1, -2)) * incl_causal
  a_ab = _mm(alpha_hat, beta_hat.swapaxes(-1, -2)) * strict_causal
  a_ak = _mm(alpha_hat, k_hat.swapaxes(-1, -2)) * strict_causal

  # Invert (I - A_ab) via Neumann series repeated squaring on the MXU in
  # log2(tt) steps rather than sequential triangular substitution on the VPU.
  p_mat = eye + a_ab
  curr = a_ab
  num_doubling_steps = max(0, math.ceil(math.log2(tt)) - 1)
  for _ in range(num_doubling_steps):
    curr = _mm(curr, curr)
    p_mat = _mm(p_mat, eye + curr)
  a_ab_inv = p_mat
  a_ab_inv_ref[0, 0] = a_ab_inv.astype(a_ab_inv_ref.dtype)

  # Fuse w_i and u_i solves into a single [tt, tt] @ [tt, K + V] matmul to
  # maximize MXU utilization.
  ak_v = _mm(a_ak, v_i)
  decayed_alpha = jnp.exp(jnp.minimum(gk_excl, 0.0)) * alpha_i
  merged_rhs = jnp.concatenate([decayed_alpha, ak_v], axis=-1)
  merged_uw = _mm(a_ab_inv, merged_rhs)
  merged_out_ref[0, 0] = merged_uw.astype(merged_out_ref.dtype)
  w_i = merged_uw[:, :, :K]
  u_i = merged_uw[:, :, K:]

  s_prev = s_scratch_ref[...]
  s_in_ref[0, 0] = s_prev.astype(s_in_ref.dtype)

  v2_i = u_i + _mm(w_i, s_prev)
  q_exp = q_i * jnp.exp(jnp.minimum(gk_cs, 0.0))
  o_i = _mm(q_exp, s_prev) + _mm(a_qk, v_i) + _mm(a_qb, v2_i)
  o_ref[0, 0] = o_i.astype(o_ref.dtype)

  g_last = gk_cs[:, -1:, :]
  exp_g_last = jnp.exp(jnp.minimum(g_last, 0.0)).swapaxes(-1, -2)
  decay_last = jnp.exp(jnp.minimum(g_last - gk_cs, 0.0))
  dk_i = k_i * decay_last
  db_i = beta_i * decay_last
  s_new = (
      s_prev * exp_g_last
      + _mm(dk_i.swapaxes(-1, -2), v_i)
      + _mm(db_i.swapaxes(-1, -2), v2_i)
  )
  s_scratch_ref[...] = s_new

  @pl.when(t_i == nt - 1)
  def _():
    s_final_ref[0] = s_new.astype(s_final_ref.dtype)


def _rwkv7_fwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    alpha_c: jax.Array,
    beta_c: jax.Array,
    gk_c: jax.Array,
    gk_cs_c: jax.Array,
    s0: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Executes single-pass fused RWKV-7 forward kernel."""
  B, H, nt, tt, K = q_c.shape
  _, _, _, _, V = v_c.shape
  BH = B * H
  tb = _pick_tb(BH)
  nb = BH // tb

  def _prep_in(x: jax.Array, D: int) -> jax.Array:
    return jnp.swapaxes(x.reshape(nb, tb, nt, tt, D), 1, 2)

  q_in = _prep_in(q_c, K)
  k_in = _prep_in(k_c, K)
  v_in = _prep_in(v_c, V)
  alpha_in = _prep_in(alpha_c, K)
  beta_in = _prep_in(beta_c, K)
  gk_in = _prep_in(gk_c, K)
  gk_cs_in = _prep_in(gk_cs_c, K)
  s0_in = s0.reshape(nb, tb, K, V).astype(jnp.float32)

  grid = (nb, nt)
  bspec_k = pl.BlockSpec(
      (1, 1, tb, tt, K), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_v = pl.BlockSpec(
      (1, 1, tb, tt, V), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_tt = pl.BlockSpec(
      (1, 1, tb, tt, tt), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_kv = pl.BlockSpec(
      (1, 1, tb, tt, K + V), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_s_in = pl.BlockSpec(
      (1, 1, tb, K, V), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_s = pl.BlockSpec((1, tb, K, V), lambda b_i, t_i: (b_i, 0, 0, 0))

  interpret = jax.default_backend() != "tpu"
  o_out, a_ab_inv_out, merged_out, s_in_out, s_final_out = pl.pallas_call(
      functools.partial(_rwkv7_fwd_kernel, nt=nt),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              bspec_k,
              bspec_k,
              bspec_v,
              bspec_k,
              bspec_k,
              bspec_k,
              bspec_k,
              bspec_s,
          ],
          out_specs=[bspec_v, bspec_tt, bspec_kv, bspec_s_in, bspec_s],
          scratch_shapes=[pltpu.VMEM((tb, K, V), jnp.float32)],
      ),
      out_shape=[
          jax.ShapeDtypeStruct((nb, nt, tb, tt, V), v_c.dtype),
          jax.ShapeDtypeStruct((nb, nt, tb, tt, tt), q_c.dtype),
          jax.ShapeDtypeStruct((nb, nt, tb, tt, K + V), jnp.float32),
          jax.ShapeDtypeStruct((nb, nt, tb, K, V), jnp.float32),
          jax.ShapeDtypeStruct((nb, tb, K, V), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary")
      ),
      interpret=interpret,
  )(q_in, k_in, v_in, alpha_in, beta_in, gk_in, gk_cs_in, s0_in)

  o_c = jnp.swapaxes(o_out, 1, 2).reshape(B, H, nt, tt, V)
  a_ab_inv_c = jnp.swapaxes(a_ab_inv_out, 1, 2).reshape(B, H, nt, tt, tt)
  merged_out_c = jnp.swapaxes(merged_out, 1, 2).reshape(B, H, nt, tt, K + V)
  s_in = jnp.swapaxes(s_in_out, 1, 2).reshape(B, H, nt, K, V)
  s_final = s_final_out.reshape(B, H, K, V)
  return o_c, a_ab_inv_c, merged_out_c, s_in, s_final


def rwkv7_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    gk: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> tuple[Any, ...]:
  """Computes RWKV-7 forward pass using single-pass fused Pallas TPU kernel."""
  B, T, H, K = q.shape
  _, _, _, V = v.shape
  tt = chunk_size
  if tt > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {tt}."
    )
  nt = T // tt
  if scale is None:
    scale = K**-0.5

  q_c = _chunkify(q, B, nt, tt) * scale
  k_c = _chunkify(k, B, nt, tt)
  v_c = _chunkify(v, B, nt, tt)
  alpha_c = _chunkify(alpha, B, nt, tt)
  beta_c = _chunkify(beta, B, nt, tt)
  gk_c = _chunkify(gk, B, nt, tt)
  gk_cs_c = jnp.cumsum(gk_c.astype(jnp.float32), axis=-2)

  if initial_state is None:
    s0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32)

  o_c, a_ab_inv_c, merged_out_c, s_in, s_final = _rwkv7_fwd_pallas(
      q_c, k_c, v_c, alpha_c, beta_c, gk_c, gk_cs_c, s0
  )

  o = jnp.transpose(o_c, (0, 2, 3, 1, 4)).reshape(B, T, H, V).astype(q.dtype)
  final_state = s_final.astype(q.dtype) if output_final_state else None
  if return_residuals:
    return o, final_state, a_ab_inv_c, merged_out_c, s_in
  return o, final_state


def forward_pallas(q, k, v, alpha, beta, gk):
  """RWKV-7 forward pass on the Pallas TPU kernel.

  Takes the same arguments as `forward`. Returns `(out, residuals)` with
  `residuals = (a_ab_inv_c, merged_out_c, s_in)`; see the module docstring.
  Off TPU the kernel runs in Pallas interpret mode.
  """
  out, _, a_ab_inv_c, merged_out_c, s_in = rwkv7_pallas_fwd(
      q, k, v, alpha, beta, gk, chunk_size=_CHUNK_SIZE, return_residuals=True
  )
  return out, (a_ab_inv_c, merged_out_c, s_in)


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # Tokamax arg spec training_b1_t2048_h32_k128_v128_bf16 with 8 heads
    # instead of 32, so that the autodiff oracle fits in TPU memory. `forward`
    # builds `[tt, tt, K]` float32 decay tensors per chunk and head, and XLA
    # plans about 5.9 GiB of temporaries for `workload` here (memory analysis
    # on CPU; not measured on a TPU).
    'rwkv7_b1_t2048_h8': {
        'name': 'rwkv7_b1_t2048_h8_bwd_pallas_fwd',
        'model': 'RWKV-7',
        'operator': 'rwkv7_backward',
        'batch': 1,
        'seq_len': 2048,
        'num_heads': 8,
        'key_dim': 128,
        'val_dim': 128,
    },
    # The full arg spec. By the same estimate `workload` needs about 23.5 GiB
    # of temporaries.
    'rwkv7_b1_t2048_h32': {
        'name': 'rwkv7_b1_t2048_h32_bwd_pallas_fwd',
        'model': 'RWKV-7',
        'operator': 'rwkv7_backward',
        'batch': 1,
        'seq_len': 2048,
        'num_heads': 32,
        'key_dim': 128,
        'val_dim': 128,
    },
    # Tiny shapes for CPU smoke tests.
    'smoke': {
        'name': 'rwkv7_smoke_bwd_pallas_fwd',
        'model': 'RWKV-7',
        'operator': 'rwkv7_backward',
        'batch': 2,
        'seq_len': 128,
        'num_heads': 4,
        'key_dim': 16,
        'val_dim': 8,
    },
}
CONFIG = CONFIGS['rwkv7_b1_t2048_h8']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(q, k, v, alpha, beta, gk, dout)`; `dout` is the output cotangent."""
  cfg = _resolve_config(config)
  B, T, H = cfg['batch'], cfg['seq_len'], cfg['num_heads']
  K, V = cfg['key_dim'], cfg['val_dim']
  keys = jax.random.split(jax.random.key(42), 7)
  qk_shape = (B, T, H, K)
  v_shape = (B, T, H, V)
  # Value ranges follow the Tokamax RWKV-7 tests, which keep the recurrence
  # stable: unit-norm keys, low-rank vectors of norm about one, log-decay in
  # (-1, 0). Queries are standard normal; `forward` applies the `K ** -0.5`
  # scale.
  q = jax.random.normal(keys[0], qk_shape)
  k = jax.random.normal(keys[1], qk_shape)
  k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
  v = jax.random.normal(keys[2], v_shape)
  alpha = jax.random.normal(keys[3], qk_shape) * K**-0.5
  beta = jax.random.normal(keys[4], qk_shape) * K**-0.5
  gk = -jax.nn.sigmoid(jax.random.normal(keys[5], qk_shape))
  # The cotangent is scaled by 0.25 to keep dk, which is about ten times
  # larger than the other gradients, at an rms of about 0.3; the absolute
  # bfloat16 rounding error of the gradients grows with their scale.
  dout = 0.25 * jax.random.normal(keys[6], v_shape)
  # Sample in float32 and cast, so every dtype sees the same values.
  return tuple(x.astype(dtype) for x in (q, k, v, alpha, beta, gk, dout))


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 6, 'diff_argnums': (0, 1, 2, 3, 4, 5)}


def workload(q, k, v, alpha, beta, gk, dout):
  """Returns `(dq, dk, dv, dalpha, dbeta, dgk)`, the VJP of `forward` with `dout`."""
  _, vjp_fn = jax.vjp(forward, q, k, v, alpha, beta, gk)
  return vjp_fn(dout)


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
