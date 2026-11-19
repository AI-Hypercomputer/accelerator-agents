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
"""Kimi Delta Attention (KDA) backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

`workload(q, k, v, g, beta, a_log, delta_time_bias, dout)` returns the
gradients `(dq, dk, dv, dg, dbeta, da_log, ddelta_time_bias)` of the KDA forward
pass for the output cotangent `dout`. The gradients are computed by JAX autodiff
(`jax.vjp`) of the token-by-token forward recurrence `forward`; this file
contains no hand-derived backward pass.

`forward` is the Tokamax KDA reference with its options fixed to the Kimi
Linear training configuration: query/key L2 normalisation inside the op, the
gate activated inside the op as `-5 * sigmoid(exp(a_log) * (g +
delta_time_bias))`, query scale `K ** -0.5`, fixed-length sequences, a zero
initial state, and no final state returned. Tensors are head-first,
`[H, B, T, D]`.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/kda/reference.py  (forward recurrence)
  tokamax/_src/ops/kda/base.py       (default query scale)
  tokamax/_src/ops/kda/arg_specs.py  (shapes and options)
  tokamax/_src/ops/kda/pallas_mosaic_tpu_kernel_test.py  (value ranges)
"""

# Dimensions.
#   H   number of attention heads
#   B   batch size
#   T   sequence length
#   K   key head dimension
#   V   value head dimension

import json

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/kda/reference.py, Copyright 2026 Ant
# Group, Apache-2.0)
# ==============================================================================

# Lower bound of the sigmoid gate: the activated log-decay lies in (-5, 0).
_LOWER_BOUND = -5.0


def _accumulator_dtype(dtype):
  return jnp.promote_types(jnp.dtype(dtype), jnp.float32)


def _l2_normalize(x, acc_dtype):
  x_f = x.astype(acc_dtype)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + 1e-6)
  return x_f * rstd[..., None]


def _activate_gate(g, *, a_log, delta_time_bias, lower_bound):
  heads, _, _, key_dim = g.shape
  g_f = g.astype(jnp.float32)
  g_f = g_f + delta_time_bias.astype(jnp.float32).reshape(heads, 1, 1, key_dim)
  A = jnp.exp(a_log.astype(jnp.float32)).reshape(heads, 1, 1, 1)
  return lower_bound * jax.nn.sigmoid(A * g_f)


def forward(q, k, v, g, beta, a_log, delta_time_bias):
  """KDA forward pass: explicit token-by-token delta-rule recurrence.

  Args:
    q: Queries, `[H, B, T, K]`. L2-normalised over `K` inside the op.
    k: Keys, `[H, B, T, K]`. L2-normalised over `K` inside the op.
    v: Values, `[H, B, T, V]`.
    g: Raw per-channel gate input, `[H, B, T, K]`. The log-decay is
      `-5 * sigmoid(exp(a_log) * (g + delta_time_bias))`.
    beta: Per-token delta-rule learning rate, `[H, B, T]`, values in `[0, 1]`.
    a_log: Per-head log decay rate, `[H]`.
    delta_time_bias: Per-head, per-key-channel gate bias, `[H * K]`.

  Returns:
    The output, `[H, B, T, V]`, in the dtype of `q`. The state starts at zero
    and the final state is not returned. Queries are scaled by `K ** -0.5`.
  """
  heads, batch, seq_len, key_dim = q.shape
  value_dim = v.shape[-1]
  acc_dtype = _accumulator_dtype(q.dtype)
  output_dtype = q.dtype
  scale = key_dim**-0.5

  g = _activate_gate(
      g, a_log=a_log, delta_time_bias=delta_time_bias, lower_bound=_LOWER_BOUND
  )
  q_h = _l2_normalize(q, acc_dtype)
  k_h = _l2_normalize(k, acc_dtype)

  q_h = q_h * scale
  v_h = v.astype(acc_dtype)
  g_h = g.astype(acc_dtype)
  beta_h = beta.astype(acc_dtype)

  # One recurrent state per batch element and head.
  states = jnp.zeros((batch, 1, heads, key_dim, value_dim), dtype=acc_dtype)
  output_h = jnp.zeros((heads, batch, seq_len, value_dim), dtype=acc_dtype)

  def step_token(h, b, t, carry):
    states, output_h = carry
    previous_state = states[b, 0, h]
    state = previous_state * jnp.exp(g_h[h, b, t])[:, None]
    prediction = k_h[h, b, t] @ state
    residual = v_h[h, b, t] - prediction
    new_state = state + (
        beta_h[h, b, t] * k_h[h, b, t]
    )[:, None] * residual[None, :]
    out_t = q_h[h, b, t] @ new_state
    output_h = output_h.at[h, b, t].set(out_t)
    states = states.at[b, 0, h].set(new_state)
    return states, output_h

  def step_head(h, carry):
    states, output_h = carry

    def body_b(b, b_carry):
      def body_t(t, t_carry):
        return step_token(h, b, t, t_carry)

      return jax.lax.fori_loop(0, seq_len, body_t, b_carry)

    states, output_h = jax.lax.fori_loop(
        0, batch, body_b, (states, output_h)
    )
    return states, output_h

  states, output_h = jax.lax.fori_loop(
      0, heads, step_head, (states, output_h)
  )

  return output_h.astype(output_dtype)


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # The Kimi Linear 48B-A3B training shape of the tokamax arg_specs
    # (fixed_training_b1_t8192_bf16) with the number of heads and the sequence
    # length reduced, so that the states saved by the autodiff oracle fit in
    # TPU memory: `workload` keeps close to four float32 `K x V` states per
    # token and head, about 4 GB here. The figure is an estimate from XLA
    # memory analysis on CPU, not a TPU measurement.
    'kda_b1_t4096_h4': {
        'name': 'kda_b1_t4096_h4_bwd',
        'model': 'Kimi-Linear-48B-A3B',
        'operator': 'kda_backward',
        'num_heads': 4,
        'batch': 1,
        'seq_len': 4096,
        'key_dim': 128,
        'val_dim': 128,
    },
    # tokamax arg_specs: fixed_training_b1_t8192_bf16 (Kimi Linear 48B-A3B) at
    # full size. By the same estimate the autodiff oracle needs about 56 GB for
    # this shape.
    'kda_b1_t8192': {
        'name': 'kda_b1_t8192_bwd',
        'model': 'Kimi-Linear-48B-A3B',
        'operator': 'kda_backward',
        'num_heads': 32,
        'batch': 1,
        'seq_len': 8192,
        'key_dim': 128,
        'val_dim': 128,
    },
    # Tiny shapes for CPU smoke tests. `seq_len` stays a multiple of 64, the
    # chunk size of the Tokamax Pallas kernels for a per-channel gate.
    'smoke': {
        'name': 'kda_smoke_bwd',
        'model': 'Kimi-Linear-48B-A3B',
        'operator': 'kda_backward',
        'num_heads': 2,
        'batch': 2,
        'seq_len': 128,
        'key_dim': 16,
        'val_dim': 8,
    },
}
CONFIG = CONFIGS['kda_b1_t4096_h4']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(q, k, v, g, beta, a_log, delta_time_bias, dout)`.

  `dout` is the output cotangent. `a_log` and `delta_time_bias` are model
  parameters and stay float32 for every `dtype`, as in the Tokamax arg specs.
  """
  cfg = _resolve_config(config)
  H, B, T = cfg['num_heads'], cfg['batch'], cfg['seq_len']
  K, V = cfg['key_dim'], cfg['val_dim']
  keys = jax.random.split(jax.random.key(42), 8)
  qk_shape = (H, B, T, K)
  v_shape = (H, B, T, V)
  # Value ranges follow the production-range ("prod") input profile of the
  # Tokamax KDA kernel tests, which keeps the recurrence well-conditioned:
  # SiLU-activated queries/keys, learning rates in (0.05, 0.95), and gate
  # parameters for which most channels decay slowly.
  q = jax.nn.silu(jax.random.normal(keys[0], qk_shape))
  k = jax.nn.silu(jax.random.normal(keys[1], qk_shape))
  v = 1.5 * jax.random.normal(keys[2], v_shape)
  g = jax.random.uniform(keys[3], qk_shape, minval=-4.5, maxval=4.5)
  beta = jax.random.uniform(keys[4], (H, B, T), minval=0.05, maxval=0.95)
  a_log = jax.random.uniform(keys[5], (H,), minval=0.2, maxval=3.0)
  delta_time_bias = jax.random.uniform(
      keys[6], (H * K,), minval=-8.0, maxval=-1.5
  )
  dout = jax.random.normal(keys[7], v_shape)
  # Sample in float32 and cast, so every dtype sees the same values.
  q, k, v, g, beta, dout = (x.astype(dtype) for x in (q, k, v, g, beta, dout))
  return q, k, v, g, beta, a_log, delta_time_bias, dout


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 7, 'diff_argnums': (0, 1, 2, 3, 4, 5, 6)}


def workload(q, k, v, g, beta, a_log, delta_time_bias, dout):
  """Returns the VJP of `forward` with `dout`.

  The result is `(dq, dk, dv, dg, dbeta, da_log, ddelta_time_bias)`, each with
  the shape and dtype of its primal.
  """
  _, vjp_fn = jax.vjp(forward, q, k, v, g, beta, a_log, delta_time_bias)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  Per token and head the forward recurrence does a `K x V` state decay (KV),
  a `K`-vector by `K x V` matmul (2KV), a rank-1 state update (KV) and a
  second vector-matrix product (2KV): about 6KV. The VJP runs the forward once
  and a backward that costs about twice the forward, so 3x in total.
  """
  cfg = _resolve_config(config)
  H, B, T = cfg['num_heads'], cfg['batch'], cfg['seq_len']
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
