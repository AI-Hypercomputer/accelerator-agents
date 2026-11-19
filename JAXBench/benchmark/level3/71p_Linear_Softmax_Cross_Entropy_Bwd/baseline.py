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
"""Linear softmax cross-entropy loss backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

The op fuses the output projection `x @ w` of a language model with the softmax
cross-entropy loss against integer labels, and returns the loss averaged over
tokens as a float32 scalar.

`workload(x, labels, w, dloss)` returns `(dx, dw)`, the gradients of that mean
loss with respect to `x` and `w` multiplied by the scalar cotangent `dloss`.
`labels` is an integer input and has no gradient. The gradients are computed by
JAX autodiff (`jax.vjp`) of the forward reference `forward`; this file contains
no hand-derived backward pass.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py   (forward reference)
  tokamax/_src/ops/linear_softmax_cross_entropy_loss/arg_specs.py   (shapes, reduction)
  tokamax/_src/ops/linear_softmax_cross_entropy_loss/test_utils.py  (input distributions)
"""

# Dimensions.
#   B   number of tokens (batch and sequence flattened)
#   H   hidden dimension
#   V   vocabulary size

import json

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference
# (from tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================


def forward(x, labels, w):
  """Mean softmax cross-entropy loss of the logits `x @ w`.

  Args:
    x: Activations of the last layer, `[B, H]`.
    labels: Target class indices, `[B]`, integers in `[0, V)`.
    w: Output projection weights, `[H, V]`.

  Returns:
    The loss averaged over the `B` tokens, a float32 scalar. The logits and the
    log-softmax are computed in the dtype of the inputs; the sums over the
    vocabulary and over tokens accumulate in float32.
  """
  logits = x @ w
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  labels_one_hot = jax.nn.one_hot(labels, num_classes=w.shape[1], dtype=x.dtype)
  loss = -labels_one_hot * log_probs
  loss_per_sample = jnp.sum(loss, axis=-1, dtype=jnp.float32)
  return jnp.mean(loss_per_sample, dtype=jnp.float32)


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: the llama3.1-8b entry tagged 'primary' (batch_dim=1024,
    # hidden_dim=4096, output_dim=128256, reduction='mean', bfloat16).
    'llama3_8b_b1024': {
        'name': 'lsce_llama3_8b_b1024_bwd',
        'model': 'Llama-3.1-8B',
        'operator': 'linear_softmax_cross_entropy_loss_backward',
        'batch_tokens': 1024,
        'hidden_dim': 4096,
        'vocab_size': 128256,
    },
    # Small shapes for CPU smoke tests. Like the default, every dimension is a
    # whole number of the blocks the Tokamax TPU kernel tiles it into (1024
    # tokens, 512 hidden units, vocabulary blocks that are multiples of 128).
    'smoke': {
        'name': 'lsce_smoke_bwd',
        'model': 'Llama-3.1-8B',
        'operator': 'linear_softmax_cross_entropy_loss_backward',
        'batch_tokens': 1024,
        'hidden_dim': 1024,
        'vocab_size': 8192,
    },
}
CONFIG = CONFIGS['llama3_8b_b1024']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(x, labels, w, dloss)`; `dloss` is the float32 scalar cotangent.

  The gradient of a mean loss is O(1/B), far below the absolute tolerance of
  1e-2 that the benchmark harness compares with. The cotangent is therefore
  `B * sqrt(H) / 4` instead of one, which scales the gradient entries to order
  one: large enough that zeros fail that tolerance and small enough that
  bfloat16 rounding in the reference does not exceed it. The VJP is linear in
  the cotangent.
  """
  cfg = _resolve_config(config)
  B, H, V = cfg['batch_tokens'], cfg['hidden_dim'], cfg['vocab_size']
  k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
  # Standard-normal activations and uniformly drawn labels, as in the Tokamax
  # tests of this op (test_utils.generate_random_data). The weights are scaled
  # to fan-in, `H ** -0.5`, so the logits are O(1) and the softmax is not
  # saturated.
  x = jax.random.normal(k1, (B, H))
  labels = jax.random.randint(k2, (B,), 0, V)
  w = jax.random.normal(k3, (H, V)) * H**-0.5
  dloss = jnp.asarray(B * H**0.5 / 4, dtype=jnp.float32)
  # Sample in float32 and cast, so every dtype sees the same values. The
  # cotangent keeps the float32 dtype of the loss.
  return x.astype(dtype), labels, w.astype(dtype), dloss


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 3, 'diff_argnums': (0, 2)}


def workload(x, labels, w, dloss):
  """Returns `(dx, dw)`, the VJP of `forward` in `x` and `w` with `dloss`."""
  _, vjp_fn = jax.vjp(lambda x_, w_: forward(x_, labels, w_), x, w)
  return vjp_fn(dloss)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  The forward is dominated by the `[B, H] x [H, V]` projection, `2 * B * H * V`
  FLOPs. The backward does two matmuls of the same size, one per gradient
  (`[B, V] x [V, H]` for `dx` and `[H, B] x [B, V]` for `dw`), so it costs
  twice the forward: 3x in total. The elementwise softmax terms are `O(B * V)`
  and are not counted.
  """
  cfg = _resolve_config(config)
  B, H, V = cfg['batch_tokens'], cfg['hidden_dim'], cfg['vocab_size']
  return int(3 * 2 * B * H * V)


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
