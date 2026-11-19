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
"""Raven Gated Slot Attention (GSA) backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

Raven GSA is a two-stage linear-attention recurrence. Stage 1 writes keys into
`M` memory slots and reads slot logits with the queries; the logits go through
a softmax over the slots; stage 2 writes values into the same slots and reads
them with the softmax weights. Both stages share the slot write weights `s` and
the per-slot log-retention gates `g`.

`forward_pallas(q, k, s, g, v)` is the Tokamax Pallas TPU forward: one kernel
per stage, with the slot softmax between them in plain JAX. It returns
`(out, residuals)`, where `residuals = (s1_in_c, qv, s2_in_c)` are the
intermediates the Tokamax forward saves for a backward pass, next to the inputs
themselves:
  s1_in_c  `[B, H, T/64, K, M]`  float32  stage-1 state entering each chunk
  qv       `[B, T, H, M]`        dtype of `v`  softmax of the stage-1 slot logits
  s2_in_c  `[B, H, T/64, M, V]`  float32  stage-2 state entering each chunk

`workload(q, k, s, g, v, dout)` returns the gradients `(dq, dk, ds, dg, dv)`
for the output cotangent `dout`. It is the correctness oracle: JAX autodiff
(`jax.vjp`) of the pure-JAX chunked forward reference `forward`. This file
contains no backward kernel and no hand-derived backward pass.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` (it
may consume the saved residuals) so that your `workload` returns the same
gradients as this one.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/experimental/raven/pallas_mosaic_tpu_kernel.py  (forward kernels)
  tokamax/_src/ops/experimental/raven/pallas_mosaic_tpu.py  (saved residuals)
  tokamax/_src/ops/experimental/raven/base.py       (forward reference)
  tokamax/_src/ops/experimental/raven/arg_specs.py  (shapes and dtypes)
  tokamax/_src/ops/experimental/raven/pallas_mosaic_tpu_test.py  (value ranges)
"""

# Dimensions.
#   B      batch size
#   T      sequence length
#   H      number of attention heads
#   K      key/query head dimension
#   M      number of memory slots
#   V      value head dimension
#   BH     flattened batch * heads (B * H)
#   tb     tile extent of BH (stream-batching factor per Pallas program)
#   tt     tile extent of T (chunk size)
#   nb     grid extent along BH: nb = BH // tb
#   nt     grid extent along T: nt = T // tt

import functools
import json
import os
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/experimental/raven/base.py)
#
# The Tokamax options are fixed to chunk_size=64, the default query scale
# `K ** -0.5`, zero initial states and no returned final state.
# ==============================================================================

_CHUNK_SIZE = 64


def _chunkify(x, B, nt, tt):
  # [B, T, H, ...] -> [B, H, nt, tt, ...]
  _, _, H, *rest = x.shape
  x = x.reshape((B, nt, tt, H, *rest))
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _move_nt_front(x):
  return jnp.moveaxis(x, 2, 0)


def _stage1_scan_step(s_state, inputs):
  q_i, g_i, y_diag_i, state_contrib_i, g_last_i = inputs
  y_off = jnp.matmul(
      q_i,
      s_state,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  ) * jnp.exp(g_i)
  y = y_diag_i + y_off
  s_new = s_state * jnp.exp(g_last_i)[None, :] + state_contrib_i
  return s_new.astype(s_state.dtype), y.astype(y_diag_i.dtype)


def _stage1_intra_one_plain(q_i, k_i, s_i, g_i):
  tt, _ = q_i.shape
  mask_upper = jnp.triu(jnp.ones((tt, tt), dtype=bool), k=1)
  qk = jnp.where(
      mask_upper,
      0.0,
      jnp.matmul(
          q_i,
          k_i.T,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      ),
  )
  diff = jnp.where(
      mask_upper[:, :, None], 0.0, g_i[:, None, :] - g_i[None, :, :]
  )
  decay = jnp.exp(diff)
  y_diag = jnp.sum(qk[:, :, None] * decay * s_i[None, :, :], axis=1)
  g_last = g_i[-1]
  mask_last = (jnp.arange(tt) == (tt - 1))[:, None]
  decay_tail = jnp.exp(jnp.where(mask_last, 0.0, g_last[None, :] - g_i))
  state_contrib = jnp.matmul(
      k_i.T,
      s_i * decay_tail,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return y_diag, state_contrib, g_last[None, :]


def raven_stage1_reference(q, k, s, g):
  """Stage 1 reference: decay on slot output (M) axis.

  Returns the slot logits, `[B, T, H, M]`, in the dtype of `s`.
  """
  B, T, H, K = q.shape
  _, _, _, M = s.shape
  tt = _CHUNK_SIZE
  nt = T // tt

  scale = float(K**-0.5)
  q_scaled = (q.astype(jnp.float32) * scale).astype(q.dtype)

  q_c = _chunkify(q_scaled.astype(jnp.float32), B, nt, tt)
  k_c = _chunkify(k.astype(jnp.float32), B, nt, tt)
  s_c = _chunkify(s.astype(jnp.float32), B, nt, tt)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  )

  vmap_intra = jax.vmap(jax.vmap(jax.vmap(_stage1_intra_one_plain)))
  y_diag, state_contrib, g_last_block = vmap_intra(q_c, k_c, s_c, g_c)
  g_last = g_last_block[..., 0, :]

  q_s, g_s, y_diag_s, state_contrib_s, g_last_s = map(
      _move_nt_front, (q_c, g_c, y_diag, state_contrib, g_last)
  )

  scan_body_batched = jax.vmap(jax.vmap(_stage1_scan_step))

  initial_recurrent_state = jnp.zeros((B, H, K, M), dtype=jnp.float32)

  _, o_stacked = jax.lax.scan(
      scan_body_batched,
      initial_recurrent_state,
      (q_s, g_s, y_diag_s, state_contrib_s, g_last_s),
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(B, T, H, M).astype(s.dtype)
  return o


def _stage2_scan_step(s_state, inputs):
  qv_i, g_i, y_diag_i, state_contrib_i, g_last_i = inputs
  y_off = jnp.matmul(
      qv_i * jnp.exp(g_i),
      s_state,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  y = y_diag_i + y_off
  s_new = s_state * jnp.exp(g_last_i)[:, None] + state_contrib_i
  return s_new.astype(s_state.dtype), y.astype(y_diag_i.dtype)


def _stage2_intra_one_plain(qv_i, s_i, v_i, g_i):
  tt, _ = qv_i.shape
  mask_upper = jnp.triu(jnp.ones((tt, tt), dtype=bool), k=1)
  diff = jnp.where(
      mask_upper[:, :, None], 0.0, g_i[:, None, :] - g_i[None, :, :]
  )
  decay = jnp.exp(diff)
  aqk_raw = jnp.einsum(
      "tm,nm,tnm->tn",
      qv_i,
      s_i,
      decay,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  aqk = jnp.where(mask_upper, 0.0, aqk_raw)
  y_diag = jnp.matmul(
      aqk,
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  g_last = g_i[-1]
  mask_last = (jnp.arange(tt) == (tt - 1))[:, None]
  decay_tail = jnp.exp(jnp.where(mask_last, 0.0, g_last[None, :] - g_i))
  state_contrib = jnp.matmul(
      (s_i * decay_tail).T,
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return y_diag, state_contrib, g_last[None, :]


def raven_stage2_reference(qv, s, v, g):
  """Stage 2 reference: decay on slot input (M) axis, output on value (V) axis.

  Returns the output, `[B, T, H, V]`, in the dtype of `v`.
  """
  B, T, H, M = qv.shape
  _, _, _, V = v.shape
  tt = _CHUNK_SIZE
  nt = T // tt

  qv_c = _chunkify(qv.astype(jnp.float32), B, nt, tt)
  s_c = _chunkify(s.astype(jnp.float32), B, nt, tt)
  v_c = _chunkify(v.astype(jnp.float32), B, nt, tt)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  )

  vmap_intra = jax.vmap(jax.vmap(jax.vmap(_stage2_intra_one_plain)))
  y_diag, state_contrib, g_last_block = vmap_intra(qv_c, s_c, v_c, g_c)
  g_last = g_last_block[..., 0, :]

  qv_s, g_s, y_diag_s, state_contrib_s, g_last_s = map(
      _move_nt_front, (qv_c, g_c, y_diag, state_contrib, g_last)
  )

  scan_body_batched = jax.vmap(jax.vmap(_stage2_scan_step))

  initial_recurrent_state = jnp.zeros((B, H, M, V), dtype=jnp.float32)

  _, o_stacked = jax.lax.scan(
      scan_body_batched,
      initial_recurrent_state,
      (qv_s, g_s, y_diag_s, state_contrib_s, g_last_s),
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(B, T, H, V).astype(v.dtype)
  return o


def forward(q, k, s, g, v):
  """Raven GSA forward pass: stage 1 -> slot softmax -> stage 2.

  Per token `t`, with both states starting at zero and `q` scaled by
  `K ** -0.5`, the chunked computation below is equivalent to

    S1_t = S1_{t-1} * exp(g_t)[None, :] + k_t[:, None] * s_t[None, :]  # [K, M]
    qv_t = softmax(q_t @ S1_t)                                         # [M]
    S2_t = S2_{t-1} * exp(g_t)[:, None] + s_t[:, None] * v_t[None, :]  # [M, V]
    out_t = qv_t @ S2_t                                                # [V]

  Args:
    q: Queries, `[B, T, H, K]`.
    k: Keys, `[B, T, H, K]`.
    s: Slot write weights, `[B, T, H, M]`.
    g: Per-slot log-retention gates, `[B, T, H, M]`, values `<= 0`.
    v: Values, `[B, T, H, V]`.

  Returns:
    The output, `[B, T, H, V]`, in the dtype of `v`. `T` must be a multiple of
    the chunk size (64). The final states are not returned.
  """
  _, T, _, _ = q.shape
  if T % _CHUNK_SIZE != 0:
    raise ValueError(
        f"seq_len ({T}) must be divisible by chunk_size ({_CHUNK_SIZE})."
    )
  ok = raven_stage1_reference(q, k, s, g)
  qv = jax.nn.softmax(ok.astype(jnp.float32), axis=-1).astype(v.dtype)
  return raven_stage2_reference(qv, s, v, g)


# ==============================================================================
# Pallas TPU forward kernel
# (inlined from tokamax/_src/ops/experimental/raven/pallas_mosaic_tpu_kernel.py,
#  Copyright 2026 Google LLC, Apache-2.0)
#
# `_CHUNK_SIZE` and `_chunkify` are identical to the definitions above and are
# not repeated here.
# ==============================================================================


def _resolve_interpret(interpret: bool | None = None) -> bool:
  """Resolves whether ``pallas_call`` should run in CPU interpret mode."""
  if interpret is not None:
    return interpret
  env = os.environ.get("PALLAS_INTERPRET", "").strip().lower()
  if env in ("0", "false"):
    return False
  if env in ("1", "true"):
    return True
  return jax.default_backend() != "tpu"


def _validate_chunk_size(T: int, tt: int) -> None:
  """Validates that ``chunk_size`` fits TPU VMEM layout."""
  del T
  if tt > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {tt}."
    )


def _select_tb(BH: int) -> int:
  """Selects the largest stream-batching factor ``tb <= 16`` dividing ``BH``."""
  for d in (16, 8, 4, 2):
    if BH % d == 0:
      return d
  return 1


def _unchunk(
    x_c: jax.Array,
    rest_dim: int,
    B: int,
    nt: int,
    tt: int,
    H: int,
) -> jax.Array:
  x_ = jnp.transpose(x_c, (0, 2, 3, 1, 4))
  return x_.reshape(B, nt * tt, H, rest_dim)


def _decayed_gram(
    a: jax.Array, b: jax.Array, g_a: jax.Array, g_b: jax.Array
) -> jax.Array:
  """Computes decayed Gram matrix ``(a @ b^T) * exp(g_a[:, None] - g_b[None, :])``."""
  # Factor exp(g_a - g_b) through the chunk-initial baseline g0 so that both
  # factors can be absorbed into the operands before a single MXU matmul, while
  # clamping exponents to avoid f32 overflow.
  g0 = g_b[..., :1, :]
  lhs = (a * jnp.exp(jnp.minimum(g_a - g0, 80.0))).astype(jnp.float32)
  rhs = (b * jnp.exp(jnp.minimum(g0 - g_b, 80.0))).astype(jnp.float32)
  out = jnp.matmul(
      lhs,
      jnp.swapaxes(rhs, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return out.astype(a.dtype)


def _decayed_apply(qk: jax.Array, s: jax.Array, g: jax.Array) -> jax.Array:
  """Computes ``out[r, m] = sum_c qk[r, c] * exp(g[r, m] - g[c, m]) * s[c, m]``."""
  # Factor the slot-dependent decay exp(g[r, m] - g[c, m]) around g0 = g[0, m]
  # so the c-dependent term scales s before the MXU contraction and the
  # r-dependent term scales the output afterward.
  g0 = g[..., :1, :]
  rhs = (s * jnp.exp(jnp.minimum(g0 - g, 80.0))).astype(jnp.float32)
  out = jnp.matmul(
      qk.astype(jnp.float32),
      rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return (out * jnp.exp(jnp.minimum(g - g0, 80.0))).astype(s.dtype)


def _stage1_fwd_kernel(
    # Inputs.
    q_ref,  # [tb, 1, tt, K] bf16/f32
    k_ref,  # [tb, 1, tt, K] bf16/f32
    s_ref,  # [tb, 1, tt, M] bf16/f32
    g_ref,  # [tb, 1, tt, M] f32
    s0_ref,  # [tb, K, M] f32
    # Outputs.
    ok_ref,  # [tb, 1, tt, M] bf16/f32
    s1_final_ref,  # [tb, K, M] f32
    s1_in_ref,  # [tb, 1, K, M] f32
    # Scratch.
    s1_scratch_ref,  # [tb, K, M] f32   persists across the nt grid axis
    *,
    nt: int,
) -> None:
  """Fused single-pass tb-batched compute kernel for Raven Stage 1."""
  t_i = pl.program_id(1)
  _, _, tt, _ = q_ref.shape

  @pl.when(t_i == 0)
  def _():
    s1_scratch_ref[...] = s0_ref[...].astype(jnp.float32)

  # Keep the recurrent state in f32 in VMEM across all nt chunks to prevent
  # precision drift over long sequences.
  s1 = s1_scratch_ref[...]  # [tb, K, M]
  s1_in_ref[:, 0] = s1.astype(s1_in_ref.dtype)

  q_i = q_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  k_i = k_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  s_i = s_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  g_i = g_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)

  qk = (
      jnp.matmul(
          q_i,
          jnp.swapaxes(k_i, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * causal
  )
  y_diag = _decayed_apply(qk, s_i, g_i)
  y_off = jnp.matmul(
      q_i,
      s1,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  ) * jnp.exp(g_i)
  ok_i = y_diag + y_off

  g_last = g_i[:, -1, :]  # [tb, M]
  decay_tail = jnp.exp(jnp.minimum(g_last[:, None, :] - g_i, 0.0))
  state_contrib = jnp.matmul(
      jnp.swapaxes(k_i, -1, -2),
      s_i * decay_tail,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s1_new = s1 * jnp.exp(g_last)[:, None, :] + state_contrib

  s1_scratch_ref[...] = s1_new
  ok_ref[:, 0] = ok_i.astype(ok_ref.dtype)

  @pl.when(t_i == nt - 1)
  def _():
    s1_final_ref[...] = s1_new.astype(s1_final_ref.dtype)


def raven_pallas_stage1_fwd(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> (
    tuple[jax.Array, jax.Array | None]
    | tuple[tuple[jax.Array, jax.Array | None], jax.Array]
):
  """Computes Raven Stage 1 forward pass using a single-pass tb-batched Pallas kernel."""
  B, T, H, K = q.shape
  _, _, _, M = s.shape
  tt = chunk_size
  _validate_chunk_size(T, tt)
  nt = T // tt
  BH = B * H
  tb = _select_tb(BH)
  nb = BH // tb

  if scale is None:
    scale = float(K**-0.5)
  q_scaled = (q.astype(jnp.float32) * scale).astype(q.dtype)

  q_c = _chunkify(q_scaled, B, nt, tt).reshape(BH, nt, tt, K)
  k_c = _chunkify(k, B, nt, tt).reshape(BH, nt, tt, K)
  s_c = _chunkify(s, B, nt, tt).reshape(BH, nt, tt, M)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  ).reshape(BH, nt, tt, M)

  if initial_state is None:
    s0 = jnp.zeros((BH, K, M), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32).reshape(BH, K, M)

  grid = (nb, nt)
  bspec_k = pl.BlockSpec((tb, 1, tt, K), lambda b_i, t_i: (b_i, t_i, 0, 0))
  bspec_m = pl.BlockSpec((tb, 1, tt, M), lambda b_i, t_i: (b_i, t_i, 0, 0))
  bspec_s = pl.BlockSpec((tb, K, M), lambda b_i, t_i: (b_i, 0, 0))
  bspec_s_in = pl.BlockSpec((tb, 1, K, M), lambda b_i, t_i: (b_i, t_i, 0, 0))

  interpret = _resolve_interpret()
  ok_c, s1_final, s1_in_c = pl.pallas_call(
      functools.partial(_stage1_fwd_kernel, nt=nt),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_m, bspec_m, bspec_s],
      out_specs=[bspec_m, bspec_s, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((tb, K, M), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((BH, nt, tt, M), s.dtype),
          jax.ShapeDtypeStruct((BH, K, M), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, K, M), jnp.float32),
      ],
      interpret=interpret,
  )(q_c, k_c, s_c, g_c, s0)

  ok = _unchunk(ok_c.reshape(B, H, nt, tt, M), M, B, nt, tt, H)
  s1_final_4d = s1_final.reshape(B, H, K, M)
  s1_in_5d = s1_in_c.reshape(B, H, nt, K, M)
  final_state = s1_final_4d.astype(q.dtype) if output_final_state else None
  if return_residuals:
    return (ok, final_state), s1_in_5d
  return ok, final_state


def _stage2_fwd_kernel(
    # Inputs.
    qv_ref,  # [tb, 1, tt, M] bf16/f32
    s_ref,  # [tb, 1, tt, M] bf16/f32
    v_ref,  # [tb, 1, tt, V] bf16/f32
    g_ref,  # [tb, 1, tt, M] f32
    s0_ref,  # [tb, M, V] f32
    # Outputs.
    out_ref,  # [tb, 1, tt, V] bf16/f32
    s2_final_ref,  # [tb, M, V] f32
    s2_in_ref,  # [tb, 1, M, V] f32
    # Scratch.
    s2_scratch_ref,  # [tb, M, V] f32   persists across the nt grid axis
    *,
    nt: int,
) -> None:
  """Fused single-pass tb-batched compute kernel for Raven Stage 2."""
  t_i = pl.program_id(1)
  _, _, tt, _ = qv_ref.shape

  @pl.when(t_i == 0)
  def _():
    s2_scratch_ref[...] = s0_ref[...].astype(jnp.float32)

  s2 = s2_scratch_ref[...]  # [tb, M, V]
  s2_in_ref[:, 0] = s2.astype(s2_in_ref.dtype)

  qv_i = qv_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  s_i = s_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  v_i = v_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]
  g_i = g_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)

  aqk = _decayed_gram(qv_i, s_i, g_i, g_i) * causal
  y_diag = jnp.matmul(
      aqk,
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  y_off = jnp.matmul(
      qv_i * jnp.exp(g_i),
      s2,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  out_i = y_diag + y_off

  g_last = g_i[:, -1, :]  # [tb, M]
  decay_tail = jnp.exp(jnp.minimum(g_last[:, None, :] - g_i, 0.0))
  state_contrib = jnp.matmul(
      jnp.swapaxes(s_i * decay_tail, -1, -2),
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s2_new = s2 * jnp.exp(g_last)[:, :, None] + state_contrib

  s2_scratch_ref[...] = s2_new
  out_ref[:, 0] = out_i.astype(out_ref.dtype)

  @pl.when(t_i == nt - 1)
  def _():
    s2_final_ref[...] = s2_new.astype(s2_final_ref.dtype)


def raven_pallas_stage2_fwd(
    qv: jax.Array,
    s: jax.Array,
    v: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> (
    tuple[jax.Array, jax.Array | None]
    | tuple[tuple[jax.Array, jax.Array | None], jax.Array]
):
  """Computes Raven Stage 2 forward pass using a single-pass tb-batched Pallas kernel."""
  B, T, H, M = qv.shape
  _, _, _, V = v.shape
  tt = chunk_size
  _validate_chunk_size(T, tt)
  nt = T // tt
  BH = B * H
  tb = _select_tb(BH)
  nb = BH // tb

  qv_c = _chunkify(qv, B, nt, tt).reshape(BH, nt, tt, M)
  s_c = _chunkify(s, B, nt, tt).reshape(BH, nt, tt, M)
  v_c = _chunkify(v, B, nt, tt).reshape(BH, nt, tt, V)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  ).reshape(BH, nt, tt, M)

  if initial_state is None:
    s0 = jnp.zeros((BH, M, V), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32).reshape(BH, M, V)

  grid = (nb, nt)
  bspec_m = pl.BlockSpec((tb, 1, tt, M), lambda b_i, t_i: (b_i, t_i, 0, 0))
  bspec_v = pl.BlockSpec((tb, 1, tt, V), lambda b_i, t_i: (b_i, t_i, 0, 0))
  bspec_s = pl.BlockSpec((tb, M, V), lambda b_i, t_i: (b_i, 0, 0))
  bspec_s_in = pl.BlockSpec((tb, 1, M, V), lambda b_i, t_i: (b_i, t_i, 0, 0))

  interpret = _resolve_interpret()
  out_c, s2_final, s2_in_c = pl.pallas_call(
      functools.partial(_stage2_fwd_kernel, nt=nt),
      grid=grid,
      in_specs=[bspec_m, bspec_m, bspec_v, bspec_m, bspec_s],
      out_specs=[bspec_v, bspec_s, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((tb, M, V), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((BH, nt, tt, V), v.dtype),
          jax.ShapeDtypeStruct((BH, M, V), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, M, V), jnp.float32),
      ],
      interpret=interpret,
  )(qv_c, s_c, v_c, g_c, s0)

  out = _unchunk(out_c.reshape(B, H, nt, tt, V), V, B, nt, tt, H)
  s2_final_4d = s2_final.reshape(B, H, M, V)
  s2_in_5d = s2_in_c.reshape(B, H, nt, M, V)
  final_state = s2_final_4d.astype(v.dtype) if output_final_state else None
  if return_residuals:
    return (out, final_state), s2_in_5d
  return out, final_state


def raven_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    v: jax.Array | None = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: (
        jax.Array | tuple[jax.Array | None, jax.Array | None] | None
    ) = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> Any:
  """Computes Raven GSA forward pass (Stage 1 when ``v is None``, or full 2-stage)."""
  if v is None:
    if isinstance(initial_state, tuple):
      s1_init = initial_state[0]
    else:
      s1_init = initial_state
    if return_residuals:
      (ok, s1_final), s1_in_c = raven_pallas_stage1_fwd(
          q,
          k,
          s,
          g,
          chunk_size=chunk_size,
          scale=scale,
          initial_state=s1_init,
          output_final_state=output_final_state,
          return_residuals=True,
      )
      return (ok, s1_final), (s1_in_c, None, None)
    return raven_pallas_stage1_fwd(
        q,
        k,
        s,
        g,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=s1_init,
        output_final_state=output_final_state,
        return_residuals=False,
    )

  if isinstance(initial_state, tuple):
    s1_init, s2_init = initial_state
  else:
    s1_init, s2_init = initial_state, None

  (ok, s1_final), s1_in_c = raven_pallas_stage1_fwd(
      q,
      k,
      s,
      g,
      chunk_size=chunk_size,
      scale=scale,
      initial_state=s1_init,
      output_final_state=output_final_state,
      return_residuals=True,
  )
  qv = jax.nn.softmax(ok.astype(jnp.float32), axis=-1).astype(v.dtype)
  (out, s2_final), s2_in_c = raven_pallas_stage2_fwd(
      qv,
      s,
      v,
      g,
      chunk_size=chunk_size,
      initial_state=s2_init,
      output_final_state=output_final_state,
      return_residuals=True,
  )
  final_state = (s1_final, s2_final) if output_final_state else None
  if return_residuals:
    return (out, final_state), (s1_in_c, qv, s2_in_c)
  return out, final_state


def forward_pallas(q, k, s, g, v):
  """Raven GSA forward pass on the Pallas TPU kernels.

  Takes the same arguments as `forward`. Returns `(out, residuals)` with
  `residuals = (s1_in_c, qv, s2_in_c)`; see the module docstring. Off TPU the
  kernels run in Pallas interpret mode.
  """
  (out, _), residuals = raven_pallas_fwd(
      q, k, s, g, v, chunk_size=_CHUNK_SIZE, return_residuals=True
  )
  return out, residuals


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: raven_gsa_b2_t1024_h8_k128_m64_v128_bf16.
    'raven_b2_t1024': {
        'name': 'raven_b2_t1024_bwd_pallas_fwd',
        'model': 'Raven-GSA',
        'operator': 'raven_gsa_backward',
        'batch': 2,
        'seq_len': 1024,
        'num_heads': 8,
        'key_dim': 128,
        'num_slots': 64,
        'val_dim': 128,
    },
    # Tiny shapes for CPU smoke tests (two chunks of 64 tokens).
    'smoke': {
        'name': 'raven_smoke_bwd_pallas_fwd',
        'model': 'Raven-GSA',
        'operator': 'raven_gsa_backward',
        'batch': 2,
        'seq_len': 128,
        'num_heads': 4,
        'key_dim': 16,
        'num_slots': 8,
        'val_dim': 32,
    },
}
CONFIG = CONFIGS['raven_b2_t1024']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(q, k, s, g, v, dout)`; `dout` is the output cotangent.

  `g` is always float32, as in the Tokamax arg spec; the other arrays are cast
  to `dtype`.
  """
  cfg = _resolve_config(config)
  B, T, H = cfg['batch'], cfg['seq_len'], cfg['num_heads']
  K, M, V = cfg['key_dim'], cfg['num_slots'], cfg['val_dim']
  keys = jax.random.split(jax.random.key(42), 6)
  qk_shape = (B, T, H, K)
  m_shape = (B, T, H, M)
  v_shape = (B, T, H, V)
  # Value ranges follow the Tokamax two-stage Raven test: standard-normal
  # queries, keys and values, slot weights in (0, 1) and small negative
  # log-retention gates.
  q = jax.random.normal(keys[0], qk_shape)
  k = jax.random.normal(keys[1], qk_shape)
  s = jax.nn.sigmoid(jax.random.normal(keys[2], m_shape))
  g = -jnp.abs(jax.random.normal(keys[3], m_shape)) * 0.1
  v = jax.random.normal(keys[4], v_shape)
  # The VJP is linear in `dout`. With a standard deviation of 0.5 the gradients
  # at the default dimensions have rms from about 0.05 (dq, dk) to 0.7 (dv),
  # which keeps the bfloat16 rounding error of the largest ones (ds, dg) small
  # in absolute terms.
  dout = 0.5 * jax.random.normal(keys[5], v_shape)
  # Sample in float32 and cast, so every dtype sees the same values.
  q, k, s, v, dout = (x.astype(dtype) for x in (q, k, s, v, dout))
  return q, k, s, g, v, dout


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 5, 'diff_argnums': (0, 1, 2, 3, 4)}


def workload(q, k, s, g, v, dout):
  """Returns `(dq, dk, ds, dg, dv)`, the VJP of `forward` with `dout`."""
  _, vjp_fn = jax.vjp(forward, q, k, s, g, v)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  Counts the contractions in `forward` per head and per chunk of `tt` tokens,
  at two FLOPs per multiply-add, ignoring elementwise work:
    stage 1: `q @ k^T` (2 tt^2 K), the decayed apply of that to `s`
      (2 tt^2 M), `q @ S1` (2 tt K M) and the state update `k^T @ s`
      (2 tt K M);
    stage 2: the decayed Gram of `qv` and `s` (2 tt^2 M), its product with
      `v` (2 tt^2 V), `qv @ S2` (2 tt M V) and the state update `s^T @ v`
      (2 tt M V).
  The VJP runs the forward once and a backward that costs about twice the
  forward, so 3x in total.
  """
  cfg = _resolve_config(config)
  B, T, H = cfg['batch'], cfg['seq_len'], cfg['num_heads']
  K, M, V = cfg['key_dim'], cfg['num_slots'], cfg['val_dim']
  tt = _CHUNK_SIZE
  per_chunk = 2 * tt * tt * (K + 2 * M + V) + 4 * tt * M * (K + V)
  return int(3 * B * H * (T // tt) * per_chunk)


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
