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
"""mHC-lite fused post-and-pre backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

mHC-lite (Manifold-Constrained Hyper-Connections) carries `S` residual streams
per token. Between two wrapped layers the fused `post_and_pre` op does:
  post  `x_next[j] = sum_k residual[j, k] * x[k] + h_post[j] * layer_output`
  pre   RMS-normalises `x_next`, projects it with the learned weights of the
        next layer into three gates, and returns
          layer_input    `sum_k h_pre[k] * x_next[k]`, `h_pre = sigmoid + eps`
          h_post_next    `2 * sigmoid`
          residual_next  a softmax-weighted mix of the `S!` permutation matrices
`forward` returns `(layer_input, x_next, h_post_next, residual_next)`.

`workload(layer_output, x, h_post, residual, norm_scale, w_pre, b_pre,
alpha_pre, w_post, b_post, alpha_post, w_res, b_res, alpha_res, permutations,
d_layer_input, d_x_next, d_h_post_next, d_residual_next)` returns the 14
gradients `(d_layer_output, d_x, d_h_post, d_residual, d_norm_scale, d_w_pre,
d_b_pre, d_alpha_pre, d_w_post, d_b_post, d_alpha_post, d_w_res, d_b_res,
d_alpha_res)` for the four output cotangents. The gradients are computed by JAX
autodiff (`jax.vjp`) of `forward`; this file contains no hand-derived backward
pass. `permutations` is a constant and has no gradient.

`h_post`, `h_post_next` and their cotangent and gradient are float32 in every
precision, and `permutations` is float32, as in the Tokamax op. All other
arrays take the benchmark dtype.

In bfloat16 the gradients are those of the op whose pre stage consumes its own
bfloat16 `x_next` output. A kernel that feeds an unrounded float32 `x_next`
into the pre stage differs from `workload` by more than 1e-2 on a few percent
of the elements of the pre-gate weight gradient `d_w_pre`.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/experimental/mhc_lite/mhc_xla.py    (forward reference)
  tokamax/_src/ops/experimental/mhc_lite/base.py       (op outputs, epsilons)
  tokamax/_src/ops/experimental/mhc_lite/common.py     (FLOP model)
  tokamax/_src/ops/experimental/mhc_lite/arg_specs.py  (shapes, dtypes)
  tokamax/_src/ops/experimental/mhc_lite/mhc_test.py   (value ranges)
"""

# Dimensions.
#   T   number of tokens (batch * sequence)
#   S   number of residual streams (expansion rate)
#   E   embedding dimension
#   P   number of permutation matrices (S!)

import itertools
import json
import math

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/experimental/mhc_lite/mhc_xla.py and
# base.py)
# ==============================================================================

# Defaults of the Tokamax op.
_RMS_EPSILON = 1e-5
_PRE_MAPPING_EPSILON = 1e-6


def _gate_logits(normalized, w, alpha, b):
  """Computes float32 gate logits `alpha * (normalized @ w) + b`."""
  with jax.named_scope("gate_logits"):
    return alpha * jnp.dot(normalized, w) + b


def _pre_xla_raw(
    x,
    norm_scale,
    w_pre,
    b_pre,
    alpha_pre,
    w_post,
    b_post,
    alpha_post,
    w_res,
    b_res,
    alpha_res,
    permutations,
    *,
    rms_epsilon=1e-5,
    pre_mapping_epsilon=1e-6,
):
  """Computes `(layer_input, x_2d, h_post, residual_f32, h_pre)` in JAX/XLA."""
  if x.ndim == 2:
    tokens, flattened_size = x.shape
    streams = permutations.shape[1]
    embedding = flattened_size // streams
    layer_input_shape = (tokens, embedding)
  elif x.ndim == 4:
    batch, sequence, streams, embedding = x.shape
    tokens = batch * sequence
    flattened_size = streams * embedding
    layer_input_shape = (batch, sequence, embedding)
  else:
    raise ValueError(
        "Expected x to have shape (batch, sequence, streams, embedding) or"
        f" (tokens, streams * embedding); got {x.shape}."
    )
  dtype = jnp.promote_types(x.dtype, jnp.float32)
  with jax.named_scope("rms_norm"):
    x_f32 = x.reshape(tokens, streams, embedding).astype(dtype)
    flat_f32 = x_f32.reshape(tokens, flattened_size)
    inv_rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(flat_f32), axis=-1, keepdims=True) + rms_epsilon
    )
    normalized = flat_f32 * inv_rms * norm_scale

  with jax.named_scope("h_pre_gate"):
    h_pre = (
        jax.nn.sigmoid(_gate_logits(normalized, w_pre, alpha_pre, b_pre))
        + pre_mapping_epsilon
    )
  with jax.named_scope("stream_collapse"):
    layer_input = (
        jnp.sum(h_pre[:, :, None] * x_f32, axis=1)
        .astype(x.dtype)
        .reshape(layer_input_shape)
    )

  with jax.named_scope("h_post_gate"):
    h_post = 2.0 * jax.nn.sigmoid(
        _gate_logits(normalized, w_post, alpha_post, b_post)
    )
  with jax.named_scope("residual_bvn_mix"):
    weights_res = jax.nn.softmax(
        _gate_logits(normalized, w_res, alpha_res, b_res),
        axis=-1,
    )
    residual = jnp.einsum("tp,pkj->tjk", weights_res, permutations)
  return layer_input, x.reshape(tokens, flattened_size), h_post, residual, h_pre


def _post_xla(layer_output, x, h_post, residual):
  """Reference JAX-only implementation of the mHC-lite post-branch stage."""
  with jax.named_scope("mhc_post_xla"):
    embedding = layer_output.shape[-1]
    streams = h_post.shape[-1]
    tokens = (
        layer_output.shape[0]
        if layer_output.ndim == 2
        else layer_output.shape[0] * layer_output.shape[1]
    )
    out_shape = (
        (tokens, streams * embedding)
        if layer_output.ndim == 2
        else (*layer_output.shape[:2], streams, embedding)
    )
    dtype = jnp.promote_types(x.dtype, jnp.float32)
    x_f32 = x.reshape(tokens, streams, embedding).astype(dtype)
    layer_output_f32 = layer_output.reshape(tokens, embedding).astype(dtype)
    residual = residual.reshape(tokens, streams, streams)
    with jax.named_scope("residual_stream_mix"):
      residual_mix = jnp.einsum("tjk,tkd->tjd", residual, x_f32)
    with jax.named_scope("post_gate_scale"):
      post_mix = h_post[:, :, None] * layer_output_f32[:, None, :]
    with jax.named_scope("combine_post_output"):
      return (residual_mix + post_mix).astype(x.dtype).reshape(out_shape)


def forward(
    layer_output,
    x,
    h_post,
    residual,
    norm_scale,
    w_pre,
    b_pre,
    alpha_pre,
    w_post,
    b_post,
    alpha_post,
    w_res,
    b_res,
    alpha_res,
    permutations,
):
  """mHC-lite fused post-branch mixing and next-layer pre-branch gating.

  Args:
    layer_output: Output of the wrapped layer, `[T, E]`.
    x: Residual streams entering the wrapped layer, `[T, S * E]` (stream-major:
      stream `k` is `x[:, k * E:(k + 1) * E]`).
    h_post: Post gate of the wrapped layer, `[T, S]`, float32, values in
      `(0, 2)`.
    residual: Stream mixing matrices of the wrapped layer, `[T, S * S]`
      (row-major `[S, S]`, doubly stochastic).
    norm_scale: RMSNorm channel scale of the next layer, `[S * E]`.
    w_pre: Pre-gate projection, `[S * E, S]`.
    b_pre: Pre-gate bias, `[S]`.
    alpha_pre: Pre-gate logit scale, `[1]`.
    w_post: Post-gate projection, `[S * E, S]`.
    b_post: Post-gate bias, `[S]`.
    alpha_post: Post-gate logit scale, `[1]`.
    w_res: Residual-mix projection, `[S * E, P]`.
    b_res: Residual-mix bias, `[P]`.
    alpha_res: Residual-mix logit scale, `[1]`.
    permutations: All `S x S` permutation matrices, `[P, S, S]`, float32.

  Returns:
    `(layer_input, x_next, h_post_next, residual_next)`:
      layer_input    `[T, E]`      input of the next wrapped layer, dtype of `x`
      x_next         `[T, S * E]`  mixed residual streams, dtype of `x`
      h_post_next    `[T, S]`      post gate of the next layer, float32
      residual_next  `[T, S * S]`  mixing matrices of the next layer, dtype of
                                   `x`
  """
  with jax.named_scope("mhc_post_and_pre_xla"):
    x_next = _post_xla(layer_output, x, h_post, residual)
    layer_input, x_2d, h_post_next, residual_f32, _ = _pre_xla_raw(
        x_next,
        norm_scale,
        w_pre,
        b_pre,
        alpha_pre,
        w_post,
        b_post,
        alpha_post,
        w_res,
        b_res,
        alpha_res,
        permutations,
        rms_epsilon=_RMS_EPSILON,
        pre_mapping_epsilon=_PRE_MAPPING_EPSILON,
    )
    tokens = x_2d.shape[0]
    residual_next = residual_f32.astype(x_next.dtype).reshape(tokens, -1)
    return layer_input, x_2d, h_post_next, residual_next


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs (primary): t8192_s4_e512. The default: it is the
    # largest primary spec for which the memory plan of the pure-JAX `workload`
    # stays small. XLA plans 0.75 GiB of temporaries and 0.08 GiB of arguments
    # for it. The memory figures in this table are XLA estimates from compiling
    # `jax.jit(workload)` on CPU in bfloat16, not TPU measurements.
    'mhc_lite_t8192_s4_e512': {
        'name': 'mhc_lite_t8192_s4_e512_bwd',
        'model': 'mHC-lite',
        'operator': 'mhc_lite_post_and_pre_backward',
        'tokens': 8192,
        'streams': 4,
        'embedding': 512,
    },
    # tokamax arg_specs (primary): t196608_s4_e512, i.e. batch 128 x sequence
    # 1536, the shape of the Tokamax mHC-lite benchmark. Not the default
    # because of its memory plan: 18.0 GiB of temporaries and 1.9 GiB of
    # arguments (the plan grows linearly with the token count).
    'mhc_lite_t196608_s4_e512': {
        'name': 'mhc_lite_t196608_s4_e512_bwd',
        'model': 'mHC-lite',
        'operator': 'mhc_lite_post_and_pre_backward',
        'tokens': 196608,
        'streams': 4,
        'embedding': 512,
    },
    # tokamax arg_specs (primary, ci_tests): t512_s4_e256. 0.02 GiB of
    # temporaries.
    'mhc_lite_t512_s4_e256': {
        'name': 'mhc_lite_t512_s4_e256_bwd',
        'model': 'mHC-lite',
        'operator': 'mhc_lite_post_and_pre_backward',
        'tokens': 512,
        'streams': 4,
        'embedding': 256,
    },
    # Tiny shapes for CPU smoke tests.
    'smoke': {
        'name': 'mhc_lite_smoke_bwd',
        'model': 'mHC-lite',
        'operator': 'mhc_lite_post_and_pre_backward',
        'tokens': 256,
        'streams': 4,
        'embedding': 512,
    },
}
CONFIG = CONFIGS['mhc_lite_t8192_s4_e512']


def _select_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns the 15 arguments of `forward` followed by the 4 output cotangents.

  `(layer_output, x, h_post, residual, norm_scale, w_pre, b_pre, alpha_pre,
  w_post, b_post, alpha_post, w_res, b_res, alpha_res, permutations,
  d_layer_input, d_x_next, d_h_post_next, d_residual_next)`.
  """
  cfg = _select_config(config)
  T, S, E = cfg['tokens'], cfg['streams'], cfg['embedding']
  P = math.factorial(S)
  keys = jax.random.split(jax.random.key(42), 16)

  # Sample in float32 and cast, so every dtype sees the same values. Each
  # array is cast as soon as it is drawn to bound peak memory.
  def normal(key, shape, scale=1.0, dtype=dtype):
    return (scale * jax.random.normal(key, shape)).astype(dtype)

  permutations = jnp.eye(S, dtype=jnp.float32)[
      jnp.array(list(itertools.permutations(range(S))))
  ]

  # Activations follow the Tokamax mHC-lite tests: unit-normal streams and
  # layer output. `h_post` and `residual` are what a preceding pre stage
  # emits: a `2 * sigmoid` gate and a convex mix of permutation matrices.
  layer_output = normal(keys[0], (T, E))
  x = normal(keys[1], (T, S * E))
  h_post = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (T, S)))
  mix = jax.nn.softmax(jax.random.normal(keys[3], (T, P)), axis=-1)
  residual = (
      jnp.einsum('tp,pkj->tjk', mix, permutations)
      .reshape(T, S * S)
      .astype(dtype)
  )

  # Weights follow the Tokamax mHC-lite tests: unit-normal norm scale,
  # projections and biases scaled by 0.1, logit scales of one.
  norm_scale = normal(keys[4], (S * E,))
  w_pre = normal(keys[5], (S * E, S), 0.1)
  b_pre = normal(keys[6], (S,), 0.1)
  alpha_pre = jnp.ones((1,), dtype)
  w_post = normal(keys[7], (S * E, S), 0.1)
  b_post = normal(keys[8], (S,), 0.1)
  alpha_post = jnp.ones((1,), dtype)
  w_res = normal(keys[9], (S * E, P), 0.1)
  b_res = normal(keys[10], (P,), 0.1)
  alpha_res = jnp.ones((1,), dtype)

  # Output cotangents, scaled by 0.01 as in the Tokamax tests.
  d_layer_input = normal(keys[11], (T, E), 0.01)
  d_x_next = normal(keys[12], (T, S * E), 0.01)
  d_h_post_next = normal(keys[13], (T, S), 0.01, jnp.float32)
  d_residual_next = normal(keys[14], (T, S * S), 0.01)

  return (
      layer_output,
      x,
      h_post,
      residual,
      norm_scale,
      w_pre,
      b_pre,
      alpha_pre,
      w_post,
      b_post,
      alpha_post,
      w_res,
      b_res,
      alpha_res,
      permutations,
      d_layer_input,
      d_x_next,
      d_h_post_next,
      d_residual_next,
  )


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangents, and returns one gradient per entry of `diff_argnums`.
# `permutations` (argument 14) is a constant and is not differentiated.
VJP_SPEC = {
    'num_primals': 15,
    'diff_argnums': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
}


def workload(
    layer_output,
    x,
    h_post,
    residual,
    norm_scale,
    w_pre,
    b_pre,
    alpha_pre,
    w_post,
    b_post,
    alpha_post,
    w_res,
    b_res,
    alpha_res,
    permutations,
    d_layer_input,
    d_x_next,
    d_h_post_next,
    d_residual_next,
):
  """Returns the VJP of `forward` for every argument except `permutations`.

  The result is `(d_layer_output, d_x, d_h_post, d_residual, d_norm_scale,
  d_w_pre, d_b_pre, d_alpha_pre, d_w_post, d_b_post, d_alpha_post, d_w_res,
  d_b_res, d_alpha_res)`, each with the shape and dtype of its primal.
  """
  primals = (
      layer_output,
      x,
      h_post,
      residual,
      norm_scale,
      w_pre,
      b_pre,
      alpha_pre,
      w_post,
      b_post,
      alpha_post,
      w_res,
      b_res,
      alpha_res,
  )
  _, vjp_fn = jax.vjp(lambda *args: forward(*args, permutations), *primals)
  return vjp_fn((d_layer_input, d_x_next, d_h_post_next, d_residual_next))


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  Per token the forward does: the residual stream mix, an `S x S` by `S x E`
  product (2*S*S*E); the post-gate scale and add (2*S*E); the three gate
  projections of the `S*E` normalised vector onto `2*S + P` columns
  (2*S*E*(2*S + P)); the stream collapse (2*S*E); and the `P`-term mix of
  `S x S` permutation matrices (2*P*S*S). This is the forward FLOP model of
  Tokamax's `post_and_pre_cost`. The VJP runs the forward once and a backward
  that costs about twice the forward, so 3x in total.
  """
  cfg = _select_config(config)
  T, S, E = cfg['tokens'], cfg['streams'], cfg['embedding']
  P = math.factorial(S)
  post = 2 * S * (S + 1) * E
  pre = 2 * (S * E * (2 * S + P + 1) + P * S * S)
  return int(3 * T * (post + pre))


def benchmark(num_warmup=5, num_iters=100, config=None):
  """Benchmark and return results dict."""
  import time

  cfg = _select_config(config)
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
