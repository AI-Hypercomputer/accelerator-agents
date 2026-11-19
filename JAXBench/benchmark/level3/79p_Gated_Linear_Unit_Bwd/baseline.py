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
"""Gated linear unit (SwiGLU) backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

The op computes `silu(x @ weights[:, 0]) * (x @ weights[:, 1])` with the gate
and projection weights fused into one `[K, 2, N]` array.

`workload(x, weights, dout)` returns the gradients `(dx, dweights)` of the
forward pass for the output cotangent `dout`. The gradients are computed by JAX
autodiff (`jax.vjp`) of the forward reference `forward`; this file contains no
hand-derived backward pass.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Tokamax ships no Pallas TPU kernel for this op (its kernels are for GPU only),
so there is no hand-written TPU baseline to compare speed against.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/gated_linear_unit/base.py       (forward reference)
  tokamax/_src/ops/gated_linear_unit/arg_specs.py  (shapes, activation,
                                                    precision)
  tokamax/_src/ops/gated_linear_unit/test_base.py  (input distributions; the
                                                    weights are additionally
                                                    scaled to fan-in here)
"""

# Dimensions.
#   *B  leading batch dimensions (two in the configs below)
#   M   rows (last batch dimension)
#   K   input features
#   N   output features

import json

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/gated_linear_unit/base.py)
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
# ==============================================================================

# Options of the Tokamax op, fixed to the values of the arg spec named at
# `CONFIGS` below.
_ACTIVATION = jax.nn.silu
_PRECISION = jax.lax.DotAlgorithmPreset.BF16_BF16_F32


def forward(x, weights):
  """Gated linear unit forward pass (https://arxiv.org/abs/1612.08083).

  Computes `silu(x @ weights[:, 0]) * (x @ weights[:, 1])`, i.e. SwiGLU
  (https://arxiv.org/abs/2002.05202).

  Args:
    x: The input array, `[*B, M, K]`.
    weights: The fused weights of the two linear transformations, `[K, 2, N]`.
      `weights[:, 0]` is the gate and `weights[:, 1]` the projection.

  Returns:
    The output, `[*B, M, N]`, in the dtype of `x`.
  """
  y = jnp.einsum('...k,kjn->...jn', x, weights, precision=_PRECISION)
  # Apply activation and compute product of FP8/FP16/BF16 in FP32.
  dtype = jnp.promote_types(x.dtype, jnp.float32)
  gate, projection = jnp.split(
      jax.lax.collapse(y, -2).astype(dtype),
      2,
      axis=-1,
  )
  gate = _ACTIVATION(gate)
  return (gate * projection).astype(x.dtype)


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs (project alphafold_3):
    # batch_1=512_batch_2=512_n_in=128_n_out=512_vmap_size=None_activation=silu_dtype=bfloat16.
    # No gated linear unit arg spec is tagged 'primary'; this is the smallest
    # batch of the sweep (512..5120) for the 128 -> 512 bfloat16 SwiGLU shape.
    'glu_512x512_k128_n512': {
        'name': 'glu_512x512_k128_n512_bwd',
        'model': 'AlphaFold-3',
        'operator': 'gated_linear_unit_backward',
        'batch_1': 512,
        'batch_2': 512,
        'n_in': 128,
        'n_out': 512,
    },
    # Tiny shapes for CPU smoke tests.
    'smoke': {
        'name': 'glu_smoke_bwd',
        'model': 'AlphaFold-3',
        'operator': 'gated_linear_unit_backward',
        'batch_1': 16,
        'batch_2': 8,
        'n_in': 32,
        'n_out': 64,
    },
}
CONFIG = CONFIGS['glu_512x512_k128_n512']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(x, weights, dout)`; `dout` is the output cotangent."""
  cfg = _resolve_config(config)
  B1, B2 = cfg['batch_1'], cfg['batch_2']
  K, N = cfg['n_in'], cfg['n_out']
  keys = jax.random.split(jax.random.key(42), 3)
  # Standard-normal `x` and `dout`, as in the Tokamax gated linear unit tests.
  # The tests also draw standard-normal weights; here they are scaled by
  # `K ** -0.5` (fan-in), so the gate pre-activations have unit variance and
  # the activation is evaluated in its curved region instead of saturating.
  x = jax.random.normal(keys[0], (B1, B2, K))
  weights = jax.random.normal(keys[1], (K, 2, N)) * K**-0.5
  dout = jax.random.normal(keys[2], (B1, B2, N))
  # Sample in float32 and cast, so every dtype sees the same values.
  return tuple(a.astype(dtype) for a in (x, weights, dout))


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 2, 'diff_argnums': (0, 1)}


def workload(x, weights, dout):
  """Returns `(dx, dweights)`, the VJP of `forward` with `dout`."""
  _, vjp_fn = jax.vjp(forward, x, weights)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  With `R = B1 * B2` rows, the forward is one `[R, K] x [K, 2N]` matmul:
  `2 * R * K * 2N` FLOPs. The backward is two matmuls of the same size,
  `dx = dy @ weights^T` (`[R, 2N] x [2N, K]`) and `dweights = x^T @ dy`
  (`[K, R] x [R, 2N]`), i.e. twice the forward, so 3x in total. The
  elementwise activation and product are not counted.
  """
  cfg = _resolve_config(config)
  R = cfg['batch_1'] * cfg['batch_2']
  K, N = cfg['n_in'], cfg['n_out']
  return int(3 * 2 * R * K * 2 * N)


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
