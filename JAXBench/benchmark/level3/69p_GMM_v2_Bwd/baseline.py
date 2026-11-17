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
"""GMM v2 backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

GMM v2 is the newer Tokamax Pallas TPU kernel for ragged dot, the grouped
matmul (GMM) of Mixture-of-Experts layers. The rows of `lhs` are sorted by
group (expert) and `group_sizes[g]` consecutive rows belong to group `g`. Each
row is multiplied by the weight matrix of its group:
`out[m] = lhs[m] @ rhs[g(m)]`.

`workload(lhs, rhs, group_sizes, dout)` returns the gradients `(dlhs, drhs)` of
the GMM forward pass for the output cotangent `dout`. The gradients are
computed by JAX autodiff (`jax.vjp`) of `forward`; this file contains no
hand-derived backward pass. `group_sizes` is an integer input and has no
gradient.

Tokamax has no pure-JAX implementation of `gmm_v2` itself. `forward` is the
pure-JAX reference of the ragged dot op that `gmm_v2` implements
(`jax.lax.ragged_dot_general`), with the argument order of `gmm_v2`.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernels are both yours to design) that returns the same gradients.

The configuration is the plain one: no fused activation, no quantisation, no
weight scales or bias, no partial sum, no group offset.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/ragged_dot/base.py       (forward reference `RaggedDot._fwd`)
  tokamax/_src/ops/ragged_dot/arg_specs.py  (shapes, group-size distribution)
  tokamax/_src/ops/ragged_dot/test_base.py  (value ranges)
"""

# Dimensions.
#   M   number of lhs rows (tokens routed to experts), sorted by group
#   K   contracting dimension (input features)
#   N   output features
#   G   number of groups (experts)

import json

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/ragged_dot/base.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================

DEFAULT_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [1]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)


def forward(lhs, rhs, group_sizes):
  """GMM forward pass: `out[m] = lhs[m] @ rhs[group of row m]`.

  Args:
    lhs: Activations, `[M, K]`, rows sorted by group.
    rhs: Per-group weights, `[G, K, N]`.
    group_sizes: Number of rows in each group, int32 `[G]`, summing to `M`.

  Returns:
    The output, `[M, N]`, in the result dtype of `lhs` and `rhs`. Products are
    accumulated in float32 before the cast.
  """
  out_dtype = jnp.result_type(lhs, rhs)
  # Tokamax's canonical form of the default matmul precision.
  precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)

  # NOTE: `preferred_element_type` changes the accumulation type when using
  # `jax.lax.Precision`.
  is_integer = jnp.issubdtype(out_dtype, jnp.integer)
  acc_dtype = jnp.int32 if is_integer else jnp.float32
  preferred_element_type = jnp.promote_types(out_dtype, acc_dtype)
  dot_out = jax.lax.ragged_dot_general(
      lhs,
      rhs,
      group_sizes=group_sizes,
      ragged_dot_dimension_numbers=DEFAULT_RAGGED_DOT_DIM_NUMS,
      precision=precision,
      preferred_element_type=preferred_element_type,
  ).astype(out_dtype)
  return dot_out


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: gmm_v2_maxtext_8192x7168_64x7168x1024, with plain
    # (unquantised) bf16 operands.
    'maxtext_8192x7168_64x7168x1024': {
        'name': 'gmm_v2_maxtext_8192x7168_64x7168x1024_bwd',
        'model': 'MaxText-MoE',
        'operator': 'gmm_v2_backward',
        'num_rows': 8192,
        'in_features': 7168,
        'out_features': 1024,
        'num_groups': 64,
    },
    # Tiny shapes for CPU smoke tests.
    'smoke': {
        'name': 'gmm_v2_smoke_bwd',
        'model': 'MaxText-MoE',
        'operator': 'gmm_v2_backward',
        'num_rows': 2048,
        'in_features': 256,
        'out_features': 128,
        'num_groups': 6,
    },
}
CONFIG = CONFIGS['maxtext_8192x7168_64x7168x1024']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(lhs, rhs, group_sizes, dout)`; `dout` is the output cotangent."""
  cfg = _resolve_config(config)
  M, K = cfg['num_rows'], cfg['in_features']
  N, G = cfg['out_features'], cfg['num_groups']
  keys = jax.random.split(jax.random.key(42), 4)
  # Value ranges follow the Tokamax ragged dot tests: normal(0, 0.1) operands.
  lhs = jax.random.normal(keys[0], (M, K)) * 0.1
  rhs = jax.random.normal(keys[1], (G, K, N)) * 0.1
  # Each row is assigned to a uniformly random group, as Tokamax does for the
  # group sizes of its benchmark specs (`generate_group_sizes`). The sizes are
  # uneven and sum to M.
  group_ids = jax.random.randint(keys[2], (M,), 0, G)
  group_sizes = jnp.bincount(group_ids, length=G).astype(jnp.int32)
  # The VJP is linear in the cotangent; normal(0, 0.1) keeps the gradients
  # between about 0.1 and 1 in rms for the configs above.
  dout = jax.random.normal(keys[3], (M, N)) * 0.1
  # Sample in float32 and cast, so every dtype sees the same values.
  return lhs.astype(dtype), rhs.astype(dtype), group_sizes, dout.astype(dtype)


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 3, 'diff_argnums': (0, 1)}


def workload(lhs, rhs, group_sizes, dout):
  """Returns `(dlhs, drhs)`, the VJP of `forward` with `dout`."""
  _, vjp_fn = jax.vjp(lambda l, r: forward(l, r, group_sizes), lhs, rhs)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  The forward multiplies every row of `lhs` (`K` values) by one `K x N` weight
  matrix: `2 * M * K * N`. The backward is two products of the same size,
  `dlhs[m] = dout[m] @ rhs[g].T` and `drhs[g] = lhs[rows of g].T @ dout[rows of
  g]`, so it costs twice the forward and the total is 3x.
  """
  cfg = _resolve_config(config)
  M, K, N = cfg['num_rows'], cfg['in_features'], cfg['out_features']
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
