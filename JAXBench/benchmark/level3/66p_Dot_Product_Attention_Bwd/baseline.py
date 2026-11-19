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
"""Causal dot-product attention backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

`workload(q, k, v, dout)` returns the gradients `(dq, dk, dv)` of causal,
grouped-query scaled dot-product attention for the output cotangent `dout`.
The gradients are computed by JAX autodiff (`jax.vjp`) of the forward reference
`forward`; this file contains no hand-derived backward pass.

`forward` is Tokamax's XLA attention with its options fixed: causal mask,
logits scaled by `D ** -0.5`, default dot precision, and no bias, logit soft
cap, dropout or paging. In Tokamax the softmax carries a custom JVP rule; here
it is the plain function, differentiated by JAX.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
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

import json
import math

import jax
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
        'name': 'mixtral_8x7b_b1_h16_attention_bwd',
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
        'name': 'mixtral_8x7b_attention_bwd',
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
        'name': 'attention_smoke_bwd',
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
