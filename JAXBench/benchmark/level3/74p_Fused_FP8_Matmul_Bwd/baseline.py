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
"""Fused FP8 matmul backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

The op computes `lhs @ rhs` with both operands quantised to `float8_e4m3fn`
inside the op: `lhs` (activations, `[M, K]`) with one absmax scale per row,
`rhs` (weights, `[K, N]`) with one absmax scale per output column. `forward`
is the pure-JAX forward reference.

`workload(lhs, rhs, dout)` returns the gradients `(dlhs, drhs)` for the output
cotangent `dout`.

The gradient of this op is a definition, not a derivative. The forward rounds
to FP8, so differentiating it is not meaningful: `jax.vjp(forward)` sends the
cotangent back through the FP8 casts unscaled, is not linear in `dout`
(small cotangents underflow FP8, large ones overflow to NaN) and differs from
the op's backward by roughly 20% in relative L2 norm at unit-scale `dout`. The
op instead defines its backward as the matmul gradient of the quantised
operands, treating the rounding and the absmax scales as constants:
  dlhs = quantize_rows(dout * rhs_scale) @ rhs_q^T   (an FP8 x FP8 matmul)
  drhs = (lhs_q * lhs_scale)^T @ dout                (accumulated in float32)
`workload` therefore implements that definition directly (the functions under
"Reference backward", copied from Tokamax) instead of calling `jax.vjp`.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.
Elements that sit on an FP8 rounding boundary can round either way depending
on how the scaling multiply is compiled, so implementations of the same
quantisation agree in relative L2 norm rather than bit for bit; the Tokamax
tests accept 2e-2.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/experimental/fused_fp8_matmul/base.py       (forward and
      backward references)
  tokamax/_src/ops/experimental/fused_fp8_matmul/test_base.py  (value ranges)
  tokamax/_src/ops/experimental/fused_fp8_matmul/pallas_mosaic_tpu_test.py
      (production shape)
"""

# Dimensions.
#   M   number of rows of `lhs` (batch * sequence)
#   K   contracting dimension
#   N   number of output columns

import json

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference
# (from tokamax/_src/ops/experimental/fused_fp8_matmul/base.py)
# ==============================================================================

FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8_DTYPE).max)  # 448.0


def quantize_rows_fp8(x, eps=1e-7):
  """Quantizes `x` to FP8 with one absmax scale per row (last axis).

  The scaling is done in `x`'s dtype, the scale is float32.

  Args:
    x: The array to quantize.
    eps: Lower bound on the per-row absmax, so an all-zero row gets a finite
      scale.

  Returns:
    `(xq, scale)` with `scale` of shape `x.shape[:-1] + (1,)` in float32 such
    that `x ~= xq * scale`.
  """
  max_abs = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True), eps)
  scale = (max_abs / FP8_MAX).astype(jnp.float32)
  inv = (FP8_MAX / max_abs).astype(x.dtype)
  return (x * inv).astype(FP8_DTYPE), scale


def quantize_cols_fp8(w, eps=1e-7):
  """Quantizes `w` to FP8 with one absmax scale per column (axis 0 reduced).

  Args:
    w: The array to quantize.
    eps: Lower bound on the per-column absmax, so an all-zero column gets a
      finite scale.

  Returns:
    `(wq, scale)` with `scale` of shape `w.shape[1:]` in float32 such that
    `w ~= wq * scale`.
  """
  max_abs = jnp.maximum(jnp.max(jnp.abs(w), axis=0), eps)
  scale = (max_abs / FP8_MAX).astype(jnp.float32)
  inv = (FP8_MAX / max_abs).astype(w.dtype)
  return (w * inv).astype(FP8_DTYPE), scale


def fused_fp8_matmul_reference(lhs, rhs):
  """XLA reference. Returns `(out, residuals)`.

  `residuals = (lhs_q, lhs_scale, rhs_q, rhs_scale)` are the FP8 operands and
  their scales:
    lhs_q      `[M, K]`  float8_e4m3fn
    lhs_scale  `[M, 1]`  float32
    rhs_q      `[K, N]`  float8_e4m3fn
    rhs_scale  `[N]`     float32
  """
  lhs_q, lhs_scale = quantize_rows_fp8(lhs)
  rhs_q, rhs_scale = quantize_cols_fp8(rhs)
  acc = jnp.matmul(lhs_q, rhs_q, preferred_element_type=jnp.float32)
  out = (acc * (lhs_scale * rhs_scale)).astype(lhs.dtype)
  return out, (lhs_q, lhs_scale, rhs_q, rhs_scale)


def forward(lhs, rhs):
  """Fused FP8 matmul forward pass.

  Args:
    lhs: Activations, `[M, K]`, `bfloat16` or `float32`. Quantized to
      `float8_e4m3fn` with one scale per row.
    rhs: Weights, `[K, N]`, unquantized. Quantized to `float8_e4m3fn` with one
      scale per output column.

  Returns:
    `(lhs_q @ rhs_q) * (lhs_scale * rhs_scale)`, `[M, N]`, in the dtype of
    `lhs`.
  """
  out, _ = fused_fp8_matmul_reference(lhs, rhs)
  return out


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # The op has no arg_specs.py. (32768, 3072, 4096) is the shape the Tokamax
    # kernel and its tests name as the production shape of the training
    # workload the op was built for.
    'fp8_matmul_m32768_k3072_n4096': {
        'name': 'fused_fp8_matmul_m32768_k3072_n4096_bwd',
        'model': 'FP8-Linear-Projection',
        'operator': 'fused_fp8_matmul_backward',
        'm': 32768,
        'k': 3072,
        'n': 4096,
    },
    # Tiny shapes for CPU smoke tests.
    'smoke': {
        'name': 'fused_fp8_matmul_smoke_bwd',
        'model': 'FP8-Linear-Projection',
        'operator': 'fused_fp8_matmul_backward',
        'm': 768,
        'k': 256,
        'n': 384,
    },
}
CONFIG = CONFIGS['fp8_matmul_m32768_k3072_n4096']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(lhs, rhs, dout)`; `dout` is the output cotangent."""
  cfg = _resolve_config(config)
  M, K, N = cfg['m'], cfg['k'], cfg['n']
  keys = jax.random.split(jax.random.key(42), 3)
  # Value ranges follow the Tokamax fused FP8 matmul tests: unit-normal
  # activations and cotangent, weights with standard deviation 0.02.
  lhs = jax.random.normal(keys[0], (M, K))
  rhs = jax.random.normal(keys[1], (K, N)) * 0.02
  dout = jax.random.normal(keys[2], (M, N))
  # Sample in float32 and cast, so every dtype sees the same values.
  return tuple(x.astype(dtype) for x in (lhs, rhs, dout))


# ==============================================================================
# Reference backward
# (from tokamax/_src/ops/experimental/fused_fp8_matmul/base.py)
#
# This is the definition of the op's gradient, not the derivative of `forward`.
# ==============================================================================


def fused_fp8_dlhs_reference(dout, rhs_q, rhs_scale):
  """XLA reference for the activation gradient `dlhs = dout @ rhs^T` in FP8.

  The contraction runs over `n`, so the weight's per-column scale cannot be
  factored out of it: it is folded into the cotangent first. The scaled
  cotangent is then quantized with one scale per row, exactly like the
  activation in the forward pass, and that row scale is applied to the FP8
  product.

  Args:
    dout: Cotangent of the output, shape `(m, n)`, `bfloat16` or `float32`.
    rhs_q: The `(k, n)` FP8 weight.
    rhs_scale: The `(n,)` float32 per-column weight scales.

  Returns:
    `dlhs` of shape `(m, k)` in `dout.dtype`.
  """
  dout_q, dout_scale = quantize_rows_fp8(dout * rhs_scale.astype(dout.dtype))
  # Contract the last dims directly rather than spelling it `rhs_q.T`: on TPUs
  # without an FP8 MXU (v6e), XLA's handling of a transposed FP8 operand made
  # the jitted VJP take minutes to compile.
  acc = jax.lax.dot_general(
      dout_q,
      rhs_q,
      dimension_numbers=(((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  return (acc * dout_scale).astype(dout.dtype)


def fused_fp8_drhs_reference(lhs_q, lhs_scale, dout):
  """XLA reference for the weight gradient `drhs = lhs^T @ dout`.

  The activation is reconstructed from its FP8 residual and the product is
  accumulated in float32. The cotangent is not quantized.

  Args:
    lhs_q: The `(m, k)` FP8 activation.
    lhs_scale: The `(m, 1)` float32 per-row activation scales.
    dout: Cotangent of the output, shape `(m, n)`.

  Returns:
    `drhs` of shape `(k, n)` in `dout.dtype`.
  """
  lhs = lhs_q.astype(jnp.float32) * lhs_scale
  return jnp.matmul(lhs.T, dout.astype(jnp.float32)).astype(dout.dtype)


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {
    'num_primals': 2,
    'diff_argnums': (0, 1),
    'autodiff': False,
    'why_not_autodiff': (
        'The op defines its backward as FP8-quantised matmuls of the saved'
        ' operands with the rounding and absmax scales held constant;'
        ' jax.vjp of the forward pushes the cotangent through the FP8 casts'
        ' and gives a different, cotangent-scale-dependent result.'
    ),
}


def workload(lhs, rhs, dout):
  """Returns `(dlhs, drhs)`, the op's backward for the output cotangent `dout`.

  `dlhs` has the shape and dtype of `lhs`, `drhs` those of `rhs`.
  """
  _, (lhs_q, lhs_scale, rhs_q, rhs_scale) = fused_fp8_matmul_reference(lhs, rhs)
  dlhs = fused_fp8_dlhs_reference(dout, rhs_q, rhs_scale)
  drhs = fused_fp8_drhs_reference(lhs_q, lhs_scale, dout)
  return dlhs.astype(lhs.dtype), drhs.astype(rhs.dtype)


def get_flops(config=None):
  """Estimated FLOPs of the forward plus the backward pass.

  The forward is one `[M, K] x [K, N]` matmul: 2MKN. The backward is two
  matmuls of the same size, `[M, N] x [N, K]` for `dlhs` and `[K, M] x [M, N]`
  for `drhs`: 4MKN, twice the forward. Quantisation and scaling are
  elementwise and not counted.
  """
  cfg = _resolve_config(config)
  M, K, N = cfg['m'], cfg['k'], cfg['n']
  return int(3 * 2 * M * K * N)


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
