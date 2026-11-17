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
"""Causal Splash attention backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

`workload(q, k, v, dout)` returns the gradients `(dq, dk, dv)` of causal,
grouped-query dot-product attention for the output cotangent `dout`. The
gradients are computed by JAX autodiff (`jax.vjp`) of the forward reference
`forward`; this file contains no hand-derived backward pass.

`forward` is Tokamax's Splash attention reference with its options fixed:
causal mask, unscaled logits, and no segment ids, attention sinks, logit soft
cap or dropout.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/splash_attention/reference.py  (forward reference)
  tokamax/_src/ops/splash_attention/base.py       (causal mask, output dtype)
  tokamax/_src/ops/splash_attention/arg_specs.py  (shapes)
  tokamax/_src/ops/splash_attention/splash_attention_kernel_test.py
      (value ranges)
  The only change to the reference maths: key/value heads are repeated in
  float32 (`_repeat_kv_heads`), which changes how bfloat16 dk and dv round.
"""

# Dimensions.
#   T   query sequence length
#   t   key/value sequence length (equal to T here)
#   H   number of query heads
#   h   number of key/value heads (divides H)
#   D   query/key head dimension
#   d   value head dimension

import functools
import json

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference
# (from tokamax/_src/ops/splash_attention/reference.py and
#  tokamax/_src/ops/splash_attention/base.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype('float32')).max)


def _attention_reference_impl(q, k, v, mask, mask_value):
  """Single-head attention: `q` `[T, D]`, `k` `[t, D]`, `v` `[t, d]`."""
  logits = jnp.einsum('sd,td->st', q.astype(jnp.float32), k.astype(jnp.float32))

  logits = jnp.where(mask, logits, mask_value)
  m = logits.max(axis=-1)
  s = jnp.exp(logits - m[..., None])
  l = s.sum(axis=-1)
  p = s / l[..., None]

  o = jnp.einsum('st,td->sd', p, v.astype(jnp.float32))
  return o


def _repeat_kv_heads(x, repeats):
  """Repeats each key/value head `repeats` times along the head axis."""
  # This step differs from the Tokamax Splash reference, which repeats `x` in
  # its own dtype, so that autodiff rounds the gradient of every query head to
  # that dtype and then sums over the head group. The repeat is done in float32
  # here, as in the Tokamax dot-product attention reference: the forward values
  # are unchanged, and for bfloat16 inputs dk and dv are summed over the group
  # in float32 and rounded once, in line with Tokamax's own Splash VJP.
  dtype = x.dtype
  x = x.astype(jnp.promote_types(dtype, jnp.float32))
  return jnp.repeat(x, repeats, axis=0).astype(dtype)


def forward(q, k, v):
  """Causal dot-product attention with grouped key/value heads.

  Computes `softmax(q @ k.T) @ v` per head, where query `i` attends to keys
  `j <= i`. The logits are not scaled. Query heads
  `[g * H // h, (g + 1) * H // h)` share key/value head `g`.

  Args:
    q: Queries, `[H, T, D]`.
    k: Keys, `[h, t, D]`; `h` divides `H`.
    v: Values, `[h, t, d]`.

  Returns:
    The output, `[H, T, d]`, in the dtype of `q`.
  """
  q_seq_len = q.shape[-2]
  kv_seq_len = k.shape[-2]
  q_idx = jnp.arange(q_seq_len)[:, None]
  kv_idx = jnp.arange(kv_seq_len)[None, :]
  mask = q_idx >= kv_idx

  # In grouped attention (1 < num_kv_heads && num_kv_heads < num_q_heads).
  # We interleave the KV heads across the Q heads.
  # For example: for 8 Q heads and 4 KV heads:
  # Q head [0, 1] see KV head 0
  # Q head [2, 3] see KV head 1
  # Q head [4, 5] see KV head 2
  # Q head [6, 7] see KV head 3

  kv_heads, q_heads = k.shape[0], q.shape[0]
  assert q_heads % kv_heads == 0

  if kv_heads < q_heads:
    # Repeat K and V heads to match the number of Q heads.
    q_heads_per_kv = q_heads // kv_heads
    k = _repeat_kv_heads(k, q_heads_per_kv)
    v = _repeat_kv_heads(v, q_heads_per_kv)

  attn_impl = functools.partial(
      _attention_reference_impl, mask_value=DEFAULT_MASK_VALUE
  )
  out = jax.vmap(attn_impl, in_axes=(0, 0, 0, None))(q, k, v, mask)
  return out.astype(q.dtype)


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # Default: the primary shape below with a quarter of the heads and half the
    # sequence length; the head dimensions and the number of query heads per
    # key/value head are unchanged. It is reduced so that the pure-JAX
    # `workload`, which holds `[H, T, t]` float32 logits, fits in memory: XLA's
    # CPU memory plan for `jit(workload)` is 5.1 GiB of temporaries here and
    # 80.8 GiB for the full shape. Both figures are XLA estimates on CPU, not
    # TPU measurements.
    'q16_kv2_s4096': {
        'name': 'splash_q16_kv2_s4096_bwd',
        'model': 'Splash-Attention',
        'operator': 'splash_attention_backward',
        'num_q_heads': 16,
        'num_kv_heads': 2,
        'seq_len': 4096,
        'head_dim_qk': 128,
        'head_dim_v': 128,
    },
    # tokamax arg_specs: q64_kv8_s8192_dqk128_dv128 (tagged primary).
    'q64_kv8_s8192': {
        'name': 'splash_q64_kv8_s8192_bwd',
        'model': 'Splash-Attention',
        'operator': 'splash_attention_backward',
        'num_q_heads': 64,
        'num_kv_heads': 8,
        'seq_len': 8192,
        'head_dim_qk': 128,
        'head_dim_v': 128,
    },
    # Small shapes for CPU smoke tests. The sequence spans two 1024-token
    # blocks of the Tokamax TPU kernel, and the number of query heads per
    # key/value head is that of the default config.
    'smoke': {
        'name': 'splash_smoke_bwd',
        'model': 'Splash-Attention',
        'operator': 'splash_attention_backward',
        'num_q_heads': 16,
        'num_kv_heads': 2,
        'seq_len': 2048,
        'head_dim_qk': 128,
        'head_dim_v': 128,
    },
}
CONFIG = CONFIGS['q16_kv2_s4096']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(q, k, v, dout)`; `dout` is the output cotangent."""
  cfg = _resolve_config(config)
  H, h, T = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['seq_len']
  D, d = cfg['head_dim_qk'], cfg['head_dim_v']
  keys = jax.random.split(jax.random.key(42), 4)
  # Normal values with standard deviation 0.5, as in the Tokamax Splash kernel
  # tests; `forward` does not scale the logits. Sample in float32 and cast, so
  # every dtype sees the same values.
  q = (jax.random.normal(keys[0], (H, T, D)) * 0.5).astype(dtype)
  k = (jax.random.normal(keys[1], (h, T, D)) * 0.5).astype(dtype)
  v = (jax.random.normal(keys[2], (h, T, d)) * 0.5).astype(dtype)
  dout = (jax.random.normal(keys[3], (H, T, d)) * 0.5).astype(dtype)
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

  Per query head, `forward` computes the `T x t` logits with a `T x D` by
  `D x t` matmul (2*T*t*D) and the output with a `T x t` by `t x d` matmul
  (2*T*t*d). Both are dense: the causal mask is applied to the logits
  afterwards. The VJP runs the forward once and a backward that costs about
  twice the forward, so 3x in total.
  """
  cfg = _resolve_config(config)
  H, T = cfg['num_q_heads'], cfg['seq_len']
  D, d = cfg['head_dim_qk'], cfg['head_dim_v']
  return int(3 * H * (2 * T * T * D + 2 * T * T * d))


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
