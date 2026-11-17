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
"""Gated Delta Net 2 (GDN-2) backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

`workload(q, k, v, g, b, w, dout)` returns the gradients `(dq, dk, dv, dg, db,
dw)` of the GDN-2 forward pass for the output cotangent `dout`. The gradients
are computed by JAX autodiff (`jax.vjp`) of the token-by-token forward
recurrence `forward`; this file contains no hand-derived backward pass.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
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

import json

import jax
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
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # Tokamax arg spec training_b1_t2048_h32_k128_v128_bf16 with 8 heads
    # instead of 32. Autodiff of the token-by-token recurrence saves about
    # three float32 `K x V` states per head and token (XLA memory analysis on
    # CPU), which is about 3 GiB here; not measured on a TPU.
    'gdn2_b1_t2048_h8': {
        'name': 'gdn2_b1_t2048_h8_bwd',
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
        'name': 'gdn2_b1_t2048_h32_bwd',
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
        'name': 'gdn2_smoke_bwd',
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
