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
"""Triangle multiplication backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

The op is the "outgoing edges" triangle multiplicative update of AlphaFold
(Jumper et al. 2021, Supplementary Algorithm 11) on a pair representation `x`
of shape `[N, N, C]`: layer norm, a gated projection to two `[H, N, N]` planes,
the triangle contraction `sum_k left[c, i, k] * right[c, j, k]`, a second layer
norm, an output projection and an output gate.

`workload(x, mask, projection_in_weights, gate_in_weights,
projection_out_weights, gate_out_weights, layernorm_in_scale,
layernorm_in_offset, layernorm_out_scale, layernorm_out_offset, dout)` returns
the gradients of the forward pass with respect to `x` and the eight weight
arrays (nine arrays, in argument order) for the output cotangent `dout`. The
boolean `mask` is not differentiated. The gradients are computed by JAX
autodiff (`jax.vjp`) of the forward reference `forward`; this file contains no
hand-derived backward pass.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/triangle_multiplication/base.py       (forward)
  tokamax/_src/ops/normalization/base.py                 (layer norm)
  tokamax/_src/ops/gated_linear_unit/base.py             (gated linear unit)
  tokamax/_src/ops/triangle_multiplication/arg_specs.py  (shapes)
  tokamax/_src/ops/triangle_multiplication/pallas_mosaic_tpu_test.py
                                                         (value ranges)
"""

# Dimensions.
#   N   number of residues (the pair representation is N x N)
#   C   input channels
#   H   hidden channels of the triangle contraction
#   D   output channels

import json

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference
# (from tokamax/_src/ops/triangle_multiplication/base.py,
#  tokamax/_src/ops/normalization/base.py and
#  tokamax/_src/ops/gated_linear_unit/base.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================

# Options of the Tokamax op, fixed to the configuration benchmarked here.
_TRIANGLE_TYPE = 'outgoing'  # Algorithm 11; 'incoming' is Algorithm 12.
_EPSILON = 1e-6  # Tokamax default.
_PRECISION = None  # Tokamax default: the backend's default matmul precision.


def _layer_norm(x, scale, offset):
  """Layer norm over the last axis, computed in at least float32.

  From `Normalization._fwd` in tokamax/_src/ops/normalization/base.py with
  `axis=-1`, `subtract_mean=True` and `scale_offset=0.0`.
  """
  axis = x.ndim - 1
  # Always calculate in at least `float32` precision.
  y = x.astype(jnp.promote_types(x.dtype, jnp.float32))
  mean = jnp.mean(y, axis, keepdims=True)
  y -= mean

  var = jnp.mean(jnp.square(y), axis, keepdims=True)
  rstddev = jax.lax.rsqrt(var + _EPSILON)
  y *= rstddev

  param_bcast_axes = [a for a in range(x.ndim) if a != axis]
  scale = scale.astype(y.dtype)
  y *= jnp.expand_dims(scale, param_bcast_axes)
  offset = offset.astype(y.dtype)
  y += jnp.expand_dims(offset, param_bcast_axes)

  return y.astype(x.dtype)


def _gated_linear_unit(x, gate_weights, projection_weights):
  """Computes `sigmoid(x @ gate_weights) * (x @ projection_weights)`.

  From `GatedLinearUnit._fwd` in tokamax/_src/ops/gated_linear_unit/base.py
  with unfused weights and `activation=jax.nn.sigmoid`.
  """
  projection = jnp.matmul(x, projection_weights, precision=_PRECISION)
  gate = jnp.matmul(x, gate_weights, precision=_PRECISION)
  # Apply activation and compute product of FP8/FP16/BF16 in FP32.
  dtype = jnp.promote_types(x.dtype, jnp.float32)
  gate = gate.astype(dtype)
  projection = projection.astype(dtype)
  gate = jax.nn.sigmoid(gate)
  return (gate * projection).astype(x.dtype)


def forward(
    x,
    mask,
    projection_in_weights,
    gate_in_weights,
    projection_out_weights,
    gate_out_weights,
    layernorm_in_scale,
    layernorm_in_offset,
    layernorm_out_scale,
    layernorm_out_offset,
):
  """Triangle multiplicative update ("outgoing edges").

  Implements Supplementary Algorithm 11 of 'Highly accurate protein structure
  prediction with AlphaFold', Jumper et. al. 2021.

  Args:
    x: The input array of shape `[N, N, C]`.
    mask: A boolean mask of shape `[N, N]`.
    projection_in_weights: Weights for projection layer `[C, 2, H]`.
    gate_in_weights: Weights for gate layer `[C, 2, H]`.
    projection_out_weights: Weights for the output projection layer `[H, D]`.
    gate_out_weights: Weights for the output gate layer `[C, D]`.
    layernorm_in_scale: Scale for the input layer normalization `[C]`.
    layernorm_in_offset: Offset for the input layer normalization `[C]`.
    layernorm_out_scale: Scale for the output layer normalization `[H]`.
    layernorm_out_offset: Offset for the output layer normalization `[H]`.

  Returns:
    The result of triangle multiplication of shape `[N, N, D]`, in the dtype of
    `x`.
  """
  h_dim = layernorm_out_scale.shape[0]

  left_act = _layer_norm(x, layernorm_in_scale, layernorm_in_offset)

  channel_dim = len(gate_in_weights)
  gate_weights = jnp.reshape(gate_in_weights, (channel_dim, -1))  # (C, 2 * H)
  projection_weights = jnp.reshape(projection_in_weights, (channel_dim, -1))
  proj_act = _gated_linear_unit(left_act, gate_weights, projection_weights)
  proj_act = proj_act.reshape(proj_act.shape[0], proj_act.shape[1], 2, h_dim)
  proj_act = jnp.transpose(proj_act, (2, 3, 0, 1))  # 2 C N N
  proj_act = mask * proj_act
  left_proj_act, right_proj_act = jnp.split(proj_act, 2, axis=0)
  left_proj_act = jnp.squeeze(left_proj_act, axis=0)
  right_proj_act = jnp.squeeze(right_proj_act, axis=0)

  equation = 'cik,cjk->ijc' if _TRIANGLE_TYPE == 'outgoing' else 'ckj,cki->ijc'
  act = jnp.einsum(
      equation, left_proj_act, right_proj_act, precision=_PRECISION
  )

  act = _layer_norm(act, layernorm_out_scale, layernorm_out_offset)

  act = jnp.dot(act, projection_out_weights, precision=_PRECISION)

  gate_values = jnp.dot(left_act, gate_out_weights, precision=_PRECISION)
  act *= jax.nn.sigmoid(gate_values)

  return act


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: seq_len=384, channels=64, triangle_type='outgoing'
    # (project 'alphafold'); the smallest of the 384/640/1024/2048 series.
    'trimul_n384_c64': {
        'name': 'trimul_outgoing_n384_c64_bwd',
        'model': 'AlphaFold',
        'operator': 'triangle_multiplication_backward',
        'seq_len': 384,
        'channels': 64,
        'hidden_channels': 64,
        'out_channels': 64,
    },
    # Small shapes for CPU smoke tests. N is a multiple of 128 and C, H, D
    # are in {16, 32, 64, 128}, as in the default config; the three channel
    # counts differ so that a swapped axis shows up as a shape error.
    'smoke': {
        'name': 'trimul_outgoing_smoke_bwd',
        'model': 'AlphaFold',
        'operator': 'triangle_multiplication_backward',
        'seq_len': 128,
        'channels': 32,
        'hidden_channels': 16,
        'out_channels': 64,
    },
}
CONFIG = CONFIGS['trimul_n384_c64']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns the ten arguments of `forward` followed by the cotangent `dout`.

  The cotangent is scaled by `N ** -0.5` so that the gradient with respect to
  `x` is above the absolute tolerance (1e-2) of the benchmark harness.
  """
  cfg = _resolve_config(config)
  N, C = cfg['seq_len'], cfg['channels']
  H, D = cfg['hidden_channels'], cfg['out_channels']
  keys = jax.random.split(jax.random.key(42), 11)
  # Value ranges follow the fan-in-scaled parameters of the Tokamax triangle
  # multiplication tests: standard normal activations, weights drawn with
  # standard deviation `fan_in ** -0.5`, layer norm scales near one and offsets
  # near zero, and a mask that keeps each pair with probability 0.5. The
  # cotangent is standard normal times `N ** -0.5`: the weight gradients are
  # sums over the N^2 pairs and about N times larger than the gradient with
  # respect to `x`, and this scale places the two on either side of one.
  x = jax.random.normal(keys[0], (N, N, C))
  mask = jax.random.bernoulli(keys[1], 0.5, (N, N))
  projection_in_weights = jax.random.normal(keys[2], (C, 2, H)) * C**-0.5
  gate_in_weights = jax.random.normal(keys[3], (C, 2, H)) * C**-0.5
  projection_out_weights = jax.random.normal(keys[4], (H, D)) * H**-0.5
  gate_out_weights = jax.random.normal(keys[5], (C, D)) * C**-0.5
  layernorm_in_scale = 1.0 + 0.1 * jax.random.normal(keys[6], (C,))
  layernorm_in_offset = 0.1 * jax.random.normal(keys[7], (C,))
  layernorm_out_scale = 1.0 + 0.1 * jax.random.normal(keys[8], (H,))
  layernorm_out_offset = 0.1 * jax.random.normal(keys[9], (H,))
  dout = jax.random.normal(keys[10], (N, N, D)) * N**-0.5
  # Sample in float32 and cast, so every dtype sees the same values. The mask
  # stays boolean.
  cast = lambda a: a.astype(dtype)
  return (
      cast(x),
      mask,
      cast(projection_in_weights),
      cast(gate_in_weights),
      cast(projection_out_weights),
      cast(gate_out_weights),
      cast(layernorm_in_scale),
      cast(layernorm_in_offset),
      cast(layernorm_out_scale),
      cast(layernorm_out_offset),
      cast(dout),
  )


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`
# (every argument except the boolean `mask`).
VJP_SPEC = {'num_primals': 10, 'diff_argnums': (0, 2, 3, 4, 5, 6, 7, 8, 9)}


def workload(
    x,
    mask,
    projection_in_weights,
    gate_in_weights,
    projection_out_weights,
    gate_out_weights,
    layernorm_in_scale,
    layernorm_in_offset,
    layernorm_out_scale,
    layernorm_out_offset,
    dout,
):
  """Returns the VJP of `forward` with `dout` for `x` and the eight weights.

  The result is the tuple `(dx, dprojection_in_weights, dgate_in_weights,
  dprojection_out_weights, dgate_out_weights, dlayernorm_in_scale,
  dlayernorm_in_offset, dlayernorm_out_scale, dlayernorm_out_offset)`.
  """
  fwd = lambda x, *weights: forward(x, mask, *weights)
  _, vjp_fn = jax.vjp(
      fwd,
      x,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  The forward pass has four matmuls, counted at two FLOPs per multiply-add:
  the two input projections `[N, N, C] @ [C, 2H]` (8 N^2 C H together), the
  triangle contraction of two `[H, N, N]` planes (2 H N^3), the output
  projection `[N, N, H] @ [H, D]` (2 N^2 H D) and the output gate
  `[N, N, C] @ [C, D]` (2 N^2 C D). The VJP runs the forward once and a
  backward that costs about twice the forward, so 3x in total.
  """
  cfg = _resolve_config(config)
  N, C = cfg['seq_len'], cfg['channels']
  H, D = cfg['hidden_channels'], cfg['out_channels']
  forward_flops = 2 * H * N**3 + N**2 * (8 * C * H + 2 * H * D + 2 * C * D)
  return int(3 * forward_flops)


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
