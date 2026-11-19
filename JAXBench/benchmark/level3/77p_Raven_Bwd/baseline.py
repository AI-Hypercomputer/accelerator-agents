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
"""Raven Gated Slot Attention (GSA) backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

Raven GSA is a two-stage linear-attention recurrence. Stage 1 writes keys into
`M` memory slots and reads slot logits with the queries; the logits go through
a softmax over the slots; stage 2 writes values into the same slots and reads
them with the softmax weights. Both stages share the slot write weights `s` and
the per-slot log-retention gates `g`.

`workload(q, k, s, g, v, dout)` returns the gradients `(dq, dk, ds, dg, dv)` of
the two-stage forward pass for the output cotangent `dout`. The gradients are
computed by JAX autodiff (`jax.vjp`) of the chunked forward reference
`forward`; this file contains no hand-derived backward pass.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
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
#   tt     tile extent of T (chunk size)
#   nt     grid extent along T: nt = T // tt

import json

import jax
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
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: raven_gsa_b2_t1024_h8_k128_m64_v128_bf16.
    'raven_b2_t1024': {
        'name': 'raven_b2_t1024_bwd',
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
        'name': 'raven_smoke_bwd',
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
