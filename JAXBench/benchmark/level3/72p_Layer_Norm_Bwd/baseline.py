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
"""Layer normalization backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

`workload(x, scale, offset, dout)` returns the gradients `(dx, dscale,
doffset)` of layer normalization over the last axis, with a learned per-channel
scale and offset, for the output cotangent `dout`. The gradients are computed
by JAX autodiff (`jax.vjp`) of the forward pass `forward`; this file contains
no hand-derived backward pass.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Tokamax ships no Pallas TPU kernel for this op (its only kernel is a Triton
GPU kernel), so there is no hand-written TPU baseline to compare speed
against; the reference here is the XLA implementation.

The op's options are fixed to standard layer norm: `axis=-1`,
`subtract_mean=True`, `epsilon=1e-6`, `scale_offset=0.0`, with both `scale`
and `offset` present. As in Tokamax, the computation runs in at least float32
and the result is cast back to the dtype of `x`.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/normalization/base.py       (forward, `Normalization._fwd`)
  tokamax/_src/ops/normalization/arg_specs.py  (shapes and dtypes)
  tokamax/_src/ops/normalization/test_base.py  (value ranges)
"""

# Dimensions.
#   R   rows
#   C   columns
#   D   channels (the normalized axis)

import json

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/normalization/base.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================

# Added to the variance before the reciprocal square root.
EPSILON = 1e-6


def forward(x, scale, offset):
  """Layer normalization over the last axis with learned scale and offset.

  Args:
    x: The input, `[..., D]`.
    scale: Per-channel scale factors, `[D]`.
    offset: Per-channel offsets, `[D]`.

  Returns:
    The normalized array, with the shape and dtype of `x`.
  """
  axis = x.ndim - 1
  # Always calculate in at least `float32` precision.
  y = x.astype(jnp.promote_types(x.dtype, jnp.float32))
  mean = jnp.mean(y, axis, keepdims=True)
  y -= mean

  var = jnp.mean(jnp.square(y), axis, keepdims=True)
  rstddev = jax.lax.rsqrt(var + EPSILON)
  y *= rstddev

  param_bcast_axes = [a for a in range(x.ndim) if a != axis]
  scale = scale.astype(y.dtype)
  y *= jnp.expand_dims(scale, param_bcast_axes)
  offset = offset.astype(y.dtype)
  y += jnp.expand_dims(offset, param_bcast_axes)

  return y.astype(x.dtype)


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: alphafold_768res_128chan.
    'alphafold_768res_128chan': {
        'name': 'layer_norm_alphafold_768res_128chan_bwd',
        'model': 'AlphaFold',
        'operator': 'layer_norm_backward',
        'rows': 768,
        'cols': 768,
        'channels': 128,
    },
    # tokamax arg_specs: alphafold_384res_128chan.
    'alphafold_384res_128chan': {
        'name': 'layer_norm_alphafold_384res_128chan_bwd',
        'model': 'AlphaFold',
        'operator': 'layer_norm_backward',
        'rows': 384,
        'cols': 384,
        'channels': 128,
    },
    # tokamax arg_specs: alphafold_384res_64chan.
    'alphafold_384res_64chan': {
        'name': 'layer_norm_alphafold_384res_64chan_bwd',
        'model': 'AlphaFold',
        'operator': 'layer_norm_backward',
        'rows': 384,
        'cols': 384,
        'channels': 64,
    },
    # Tiny shapes for CPU smoke tests.
    'smoke': {
        'name': 'layer_norm_smoke_bwd',
        'model': 'AlphaFold',
        'operator': 'layer_norm_backward',
        'rows': 16,
        'cols': 24,
        'channels': 32,
    },
}
CONFIG = CONFIGS['alphafold_768res_128chan']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(x, scale, offset, dout)`; `dout` is the output cotangent.

  `x` and `dout` have dtype `dtype`. `scale` and `offset` are always float32,
  as in the Tokamax arg specs (`param_dtype='float32'`).
  """
  cfg = _resolve_config(config)
  R, C, D = cfg['rows'], cfg['cols'], cfg['channels']
  keys = jax.random.split(jax.random.key(42), 4)
  # Value ranges follow the Tokamax normalization tests: standard normal
  # activations and cotangent, parameters uniform in [0, 1).
  x = jax.random.normal(keys[0], (R, C, D))
  scale = jax.random.uniform(keys[1], (D,))
  offset = jax.random.uniform(keys[2], (D,))
  dout = jax.random.normal(keys[3], (R, C, D))
  # Sample in float32 and cast, so every dtype sees the same values.
  return x.astype(dtype), scale, offset, dout.astype(dtype)


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 3, 'diff_argnums': (0, 1, 2)}


def workload(x, scale, offset, dout):
  """Returns `(dx, dscale, doffset)`, the VJP of `forward` with `dout`."""
  _, vjp_fn = jax.vjp(forward, x, scale, offset)
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
