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
"""Gated Delta Net 2 (GDN-2) backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

`forward_pallas(q, k, v, g, b, w)` is the Tokamax Pallas TPU forward kernel. It
returns `(out, residuals)`, where `residuals = (a_mat_c, merged_out_c, s_in_c)`
are the per-chunk intermediates the kernel saves for a backward pass:
  a_mat_c       `[B*H, T/64, 64, 64]`     inverse of the intra-chunk WY system
  merged_out_c  `[B*H, T/64, 64, K + V]`  the two WY factors, concatenated
  s_in_c        `[B*H, T/64, K, V]`       recurrent state entering each chunk

`workload(q, k, v, g, b, w, dout)` returns the gradients `(dq, dk, dv, dg, db,
dw)` for the output cotangent `dout`. It is the correctness oracle: JAX autodiff
(`jax.vjp`) of the pure-JAX token-by-token recurrence `forward`. This file
contains no backward kernel and no hand-derived backward pass.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` (it
may consume the saved residuals) so that your `workload` returns the same
gradients as this one.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/experimental/gdn2/pallas_mosaic_tpu_kernel.py  (forward kernel)
  tokamax/_src/ops/experimental/gdn2/base.py       (forward recurrence)
  tokamax/_src/ops/experimental/gdn2/arg_specs.py  (shapes)
  tokamax/_src/ops/experimental/gdn2/pallas_mosaic_tpu_test.py  (value ranges)
"""

# Dimensions.
#   B   batch size
#   T   sequence length
#   H   number of attention heads
#   K   key head dimension
#   V   value head dimension
#   BH  flattened batch-head count (B * H)
#   tb  tile extent of BH (micro-batch size dividing BH)
#   tt  tile extent of T (chunk size)
#   nb  grid extent along BH (BH // tb)
#   nt  grid extent along T (T // tt, number of chunks)

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
# Forward reference (from tokamax/_src/ops/experimental/gdn2/base.py)
# ==============================================================================


def _step_token(state, inputs):
  """Single token recurrence step for GDN-2."""
  q_t, k_t, v_t, g_t, b_t, w_t = inputs
  state_decay = state * jnp.exp(g_t)[:, None]
  beta = b_t * k_t
  # Pin HIGHEST precision so TPU does not lower f32 reference matmuls to bf16.
  v_pred = jnp.matmul(
      beta,
      state_decay,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  v_new = (w_t * v_t) - v_pred
  state_new = state_decay + k_t[:, None] * v_new[None, :]
  out_t = jnp.matmul(
      q_t,
      state_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return state_new, out_t


def _scan_head(q_h, k_h, v_h, g_h, b_h, w_h, s0):
  """Scans GDN-2 token-by-token recurrence across one head."""
  s_final, out_seq = jax.lax.scan(
      _step_token, s0, (q_h, k_h, v_h, g_h, b_h, w_h)
  )
  return out_seq, s_final


def forward(q, k, v, g, b, w):
  """GDN-2 forward pass: token-by-token delta-rule recurrence.

  Args:
    q: Queries, `[B, T, H, K]`.
    k: Keys, `[B, T, H, K]`.
    v: Values, `[B, T, H, V]`.
    g: Per-channel log-decay (erase gate), `[B, T, H, K]`, values `<= 0`.
    b: Per-channel write strength, `[B, T, H, K]`, values in `[0, 1]`.
    w: Per-channel value gate, `[B, T, H, V]`, values in `[0, 1]`.

  Returns:
    The output, `[B, T, H, V]`, in the dtype of `q`. The state starts at zero
    and the final state is not returned. Queries are scaled by `K ** -0.5`.
  """
  B, _, H, K = q.shape
  _, _, _, V = v.shape
  scale = float(K**-0.5)

  q_bh = jnp.transpose(q, (0, 2, 1, 3)) * scale
  k_bh = jnp.transpose(k, (0, 2, 1, 3))
  v_bh = jnp.transpose(v, (0, 2, 1, 3))
  g_bh = jnp.transpose(g, (0, 2, 1, 3))
  b_bh = jnp.transpose(b, (0, 2, 1, 3))
  w_bh = jnp.transpose(w, (0, 2, 1, 3))

  s0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)

  vmap_scan = jax.vmap(jax.vmap(_scan_head))
  out_bh, _ = vmap_scan(q_bh, k_bh, v_bh, g_bh, b_bh, w_bh, s0)
  return jnp.transpose(out_bh, (0, 2, 1, 3)).astype(q.dtype)


# ==============================================================================
# Pallas TPU forward kernel
# (inlined from tokamax/_src/ops/experimental/gdn2/pallas_mosaic_tpu_kernel.py,
#  Copyright 2026 Google LLC, Apache-2.0)
# ==============================================================================

_CHUNK_SIZE = 64
_MAX_TB = 16


def _select_tb(BH: int, max_tb: int = _MAX_TB) -> int:
  """Selects largest divisor of `BH` up to `max_tb` for stream batching."""
  for tb in range(min(BH, max_tb), 0, -1):
    if BH % tb == 0:
      return tb
  return 1


def _chunkify(x: jax.Array, B: int, nt: int, tt: int) -> jax.Array:
  # [B, T, H, ...] -> [BH, nt, tt, ...]
  _, _, H, *rest = x.shape
  rest_shape = tuple(rest)
  x = x.reshape((B, nt, tt, H) + rest_shape)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  x_t = jnp.transpose(x, perm)
  return x_t.reshape((B * H, nt, tt) + rest_shape)


def _unchunk(
    x_c: jax.Array,
    D: int,
    B: int,
    nt: int,
    tt: int,
    H: int,
) -> jax.Array:
  x_ = x_c.reshape((B, H, nt, tt, D))
  x_ = jnp.transpose(x_, (0, 2, 3, 1, 4))
  return x_.reshape(B, nt * tt, H, D)


def _decayed_gram(
    lhs_in: jax.Array,
    rhs_in: jax.Array,
    g_lhs: jax.Array,
    g_rhs: jax.Array,
) -> jax.Array:
  """Computes decayed Gram matrix `(lhs_in @ rhs_in^T) * exp(g_lhs - g_rhs)`."""
  # Shift log-decays by the chunk's first token `g0` and clamp at 80.0 so the
  # factored `exp(g_lhs - g0)` and `exp(g0 - g_rhs)` terms never overflow f32
  # before contracting on the MXU.
  g0 = g_rhs[..., :1, :]
  lhs = (lhs_in * jnp.exp(jnp.minimum(g_lhs - g0, 80.0))).astype(jnp.float32)
  rhs = (rhs_in * jnp.exp(jnp.minimum(g0 - g_rhs, 80.0))).astype(jnp.float32)
  out = jnp.matmul(
      lhs,
      jnp.swapaxes(rhs, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return out.astype(lhs_in.dtype)


def _unit_lower_inverse(a0: jax.Array) -> jax.Array:
  """Computes `(I - a0)^{-1}` for strictly lower-triangular `a0` via Schulz doubling."""
  *_, tt, _ = a0.shape
  power = a0.astype(jnp.float32)
  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  eye = (row_idx == col_idx).astype(jnp.float32)
  inv = eye + power
  # Because `a0` is strictly lower-triangular of size `tt`, `a0^tt == 0`, so
  # `ceil(log2(tt)) - 1` squaring steps compute the exact inverse on the MXU.
  num_doubling_steps = max(0, math.ceil(math.log2(tt)) - 1)
  for _ in range(num_doubling_steps):
    power = jnp.matmul(
        power,
        power,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    inv = inv + jnp.matmul(
        inv,
        power,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
  return inv.astype(a0.dtype)


def _gdn2_fwd_kernel(
    # Inputs.
    q_ref,  # [tb, 1, tt, K]
    k_ref,  # [tb, 1, tt, K]
    v_ref,  # [tb, 1, tt, V]
    g_ref,  # [tb, 1, tt, K] f32
    b_ref,  # [tb, 1, tt, K]
    w_ref,  # [tb, 1, tt, V]
    s0_ref,  # [tb, K, V] f32
    # Outputs.
    out_ref,  # [tb, 1, tt, V]
    s_final_ref,  # [tb, K, V] f32
    a_mat_ref,  # [tb, 1, tt, tt]
    merged_out_ref,  # [tb, 1, tt, K + V] f32
    s_in_ref,  # [tb, 1, K, V] f32
    # Scratch.
    s_scratch_ref,  # [tb, K, V] f32   persists across the nt chunk grid axis
    *,
    nt: int,
) -> None:
  """Single-pass `tb`-batched intra-chunk WY + in-VMEM state recurrence kernel."""
  t_i = pl.program_id(1)
  _, _, tt, K = q_ref.shape

  # Seed the persistent VMEM state buffer from `s0_ref` on the first chunk;
  # subsequent chunks along axis 1 reuse `s_scratch_ref` in place without HBM round-trips.
  @pl.when(t_i == 0)
  def _():
    s_scratch_ref[...] = s0_ref[...].astype(jnp.float32)

  s_prev = s_scratch_ref[...]
  s_in_ref[:, 0] = s_prev.astype(s_in_ref.dtype)

  q_tile = q_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  k_tile = k_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  v_tile = v_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]
  g_tile = g_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  b_tile = b_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  w_tile = w_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  strictly_lower = (col_idx < row_idx).astype(jnp.float32)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)

  bk = b_tile * k_tile
  t_lower = -_decayed_gram(bk, k_tile, g_tile, g_tile) * strictly_lower
  a_mat = _unit_lower_inverse(t_lower)

  exp_g = jnp.exp(g_tile)
  k_g_b = exp_g * bk
  wv = w_tile * v_tile
  # Concatenate `k_g_b` and `wv` along the feature axis so `(I + T)^{-1}` is
  # applied to both WY factors in a single `[tb, tt, tt] x [tb, tt, K + V]` MXU matmul.
  merged_rhs = jnp.concatenate([k_g_b, wv], axis=-1)
  merged_out = jnp.matmul(
      a_mat,
      merged_rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  aqk_tile = _decayed_gram(q_tile, k_tile, g_tile, g_tile) * causal_keep

  a_mat_ref[:, 0] = a_mat.astype(a_mat_ref.dtype)
  merged_out_ref[:, 0] = merged_out.astype(merged_out_ref.dtype)

  wwy_tile = merged_out[:, :, :K]
  u_tile = merged_out[:, :, K:]

  v_new = u_tile - jnp.matmul(
      wwy_tile,
      s_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  out_tile = jnp.matmul(
      q_tile * exp_g,
      s_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  ) + jnp.matmul(
      aqk_tile,
      v_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_tile[:, -1, :]  # [tb, K]
  decayed_k = jnp.exp(g_last[:, None, :] - g_tile) * k_tile
  s_update = jnp.matmul(
      jnp.swapaxes(decayed_k, -1, -2),
      v_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s_new = s_prev * jnp.exp(g_last)[:, :, None] + s_update

  s_scratch_ref[...] = s_new
  out_ref[:, 0] = out_tile.astype(out_ref.dtype)

  @pl.when(t_i == nt - 1)
  def _():
    s_final_ref[...] = s_new.astype(s_final_ref.dtype)


def _gdn2_fwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    g_c: jax.Array,
    b_c: jax.Array,
    w_c: jax.Array,
    s0: jax.Array,
    out_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs fused single-pass `tb`-batched intra-chunk WY + state recurrence."""
  BH, nt, tt, K = q_c.shape
  _, _, _, V = v_c.shape
  tb = _select_tb(BH)
  nb = BH // tb
  grid = (nb, nt)
  idx = lambda b_i, t_i: (b_i, t_i, 0, 0)  # noqa: E731
  bspec_k = pl.BlockSpec((tb, 1, tt, K), idx)
  bspec_v = pl.BlockSpec((tb, 1, tt, V), idx)
  bspec_tt = pl.BlockSpec((tb, 1, tt, tt), idx)
  bspec_merged = pl.BlockSpec((tb, 1, tt, K + V), idx)
  bspec_s = pl.BlockSpec((tb, K, V), lambda b_i, t_i: (b_i, 0, 0))
  bspec_s_in = pl.BlockSpec((tb, 1, K, V), idx)

  interpret = jax.default_backend() != "tpu"
  return pl.pallas_call(
      functools.partial(_gdn2_fwd_kernel, nt=nt),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_v, bspec_k, bspec_k, bspec_v, bspec_s],
      out_specs=[bspec_v, bspec_s, bspec_tt, bspec_merged, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((tb, K, V), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((BH, nt, tt, V), out_dtype),
          jax.ShapeDtypeStruct((BH, K, V), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, tt, tt), q_c.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, K + V), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, K, V), jnp.float32),
      ],
      interpret=interpret,
  )(q_c, k_c, v_c, g_c, b_c, w_c, s0)


def gdn2_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> tuple[Any, ...]:
  """Computes GDN-2 forward pass using fused `tb`-batched Pallas kernel."""
  B, T, H, K = q.shape
  _, _, _, V = v.shape
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )

  BH = B * H
  tt = chunk_size
  nt = T // tt
  if scale is None:
    scale = float(K**-0.5)

  q_c = _chunkify(q.astype(jnp.float32) * scale, B, nt, tt).astype(q.dtype)
  k_c = _chunkify(k, B, nt, tt)
  v_c = _chunkify(v, B, nt, tt)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  )
  b_c = _chunkify(b, B, nt, tt)
  w_c = _chunkify(w, B, nt, tt)

  if initial_state is None:
    s0 = jnp.zeros((BH, K, V), dtype=jnp.float32)
  else:
    s0 = initial_state.reshape((BH, K, V)).astype(jnp.float32)

  out_c, s_final, a_mat_c, merged_out_c, s_in_c = _gdn2_fwd_pallas(
      q_c, k_c, v_c, g_c, b_c, w_c, s0, v_c.dtype
  )

  out = _unchunk(out_c, V, B, nt, tt, H)
  final_state = (
      s_final.reshape((B, H, K, V)).astype(q.dtype)
      if output_final_state
      else None
  )
  if return_residuals:
    return (out, final_state), (a_mat_c, merged_out_c, s_in_c)
  return out, final_state


def forward_pallas(q, k, v, g, b, w):
  """GDN-2 forward pass on the Pallas TPU kernel.

  Takes the same arguments as `forward`. Returns `(out, residuals)` with
  `residuals = (a_mat_c, merged_out_c, s_in_c)`; see the module docstring.
  Off TPU the kernel runs in Pallas interpret mode.
  """
  (out, _), residuals = gdn2_pallas_fwd(
      q, k, v, g, b, w, chunk_size=_CHUNK_SIZE, return_residuals=True
  )
  return out, residuals


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # Tokamax arg spec training_b1_t2048_h32_k128_v128_bf16 with 8 heads
    # instead of 32. Autodiff of the token-by-token recurrence saves about
    # three float32 `K x V` states per head and token (XLA memory analysis on
    # CPU), which is about 3 GiB here; not measured on a TPU.
    'gdn2_b1_t2048_h8': {
        'name': 'gdn2_b1_t2048_h8_bwd_pallas_fwd',
        'model': 'Gated-DeltaNet-2',
        'operator': 'gdn2_backward',
        'batch': 1,
        'seq_len': 2048,
        'num_heads': 8,
        'key_dim': 128,
        'val_dim': 128,
    },
    # The full arg spec. By the same estimate `workload` needs about 12 GiB
    # of temporaries.
    'gdn2_b1_t2048_h32': {
        'name': 'gdn2_b1_t2048_h32_bwd_pallas_fwd',
        'model': 'Gated-DeltaNet-2',
        'operator': 'gdn2_backward',
        'batch': 1,
        'seq_len': 2048,
        'num_heads': 32,
        'key_dim': 128,
        'val_dim': 128,
    },
    # Tiny shapes for CPU smoke tests.
    'smoke': {
        'name': 'gdn2_smoke_bwd_pallas_fwd',
        'model': 'Gated-DeltaNet-2',
        'operator': 'gdn2_backward',
        'batch': 2,
        'seq_len': 128,
        'num_heads': 2,
        'key_dim': 16,
        'val_dim': 8,
    },
}
CONFIG = CONFIGS['gdn2_b1_t2048_h8']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(q, k, v, g, b, w, dout)`; `dout` is the output cotangent."""
  cfg = _resolve_config(config)
  B, T, H = cfg['batch'], cfg['seq_len'], cfg['num_heads']
  K, V = cfg['key_dim'], cfg['val_dim']
  keys = jax.random.split(jax.random.key(42), 7)
  qk_shape = (B, T, H, K)
  v_shape = (B, T, H, V)
  # Value ranges follow the Tokamax GDN-2 tests, which keep the delta-rule
  # recurrence stable: unit-norm keys, log-decay in (-1, 0), gates in (0, 1).
  # Queries are standard normal; `forward` applies the `K ** -0.5` scale.
  q = jax.random.normal(keys[0], qk_shape)
  k = jax.random.normal(keys[1], qk_shape)
  k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
  v = jax.random.normal(keys[2], v_shape)
  g = -jax.nn.sigmoid(jax.random.normal(keys[3], qk_shape))
  b = jax.nn.sigmoid(jax.random.normal(keys[4], qk_shape))
  w = jax.nn.sigmoid(jax.random.normal(keys[5], v_shape))
  dout = jax.random.normal(keys[6], v_shape)
  # Sample in float32 and cast, so every dtype sees the same values.
  return tuple(x.astype(dtype) for x in (q, k, v, g, b, w, dout))


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 6, 'diff_argnums': (0, 1, 2, 3, 4, 5)}


def workload(q, k, v, g, b, w, dout):
  """Returns `(dq, dk, dv, dg, db, dw)`, the VJP of `forward` with `dout`."""
  _, vjp_fn = jax.vjp(forward, q, k, v, g, b, w)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  Per token and head the forward recurrence does a `K x V` state decay (KV),
  a `K`-vector by `K x V` matmul (2KV), a rank-1 state update (KV) and a
  second vector-matrix product (2KV): about 6KV. The VJP runs the forward once
  and a backward that costs about twice the forward, so 3x in total.
  """
  cfg = _resolve_config(config)
  B, T, H = cfg['batch'], cfg['seq_len'], cfg['num_heads']
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
