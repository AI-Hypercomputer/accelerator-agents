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
"""Native Sparse Attention (NSA) backward pass -- pure-JAX reference.

JAXBench level 3, track 1 (pure-JAX reference -> Pallas TPU kernel).

NSA sums three causal attention branches over the same `q`, `k`, `v`, each
weighted by its own gate: a compressed branch (attention over block-mean keys
and values), a selected branch (attention over the query's own block plus the
`TOPK` earlier blocks with the highest routing score) and a sliding-window
branch (attention over the last `WINDOW` tokens).

`workload(q, k, v, g_cmp, g_slc, g_swa, dout)` returns the gradients `(dq, dk,
dv, dg_cmp, dg_slc, dg_swa)` of the NSA forward pass for the output cotangent
`dout`. The gradients are computed by JAX autodiff (`jax.vjp`) of the forward
reference `forward`; this file contains no hand-derived backward pass.

The top-k block selection of the selected branch is not differentiable. The
selected block indices are treated as constants (`stop_gradient`), so no
gradient flows through the routing scores. Tokamax's own VJP does the same.

Tokamax ships no Pallas backward kernel for this op: its VJP is `jax.vjp` of
the same pure-JAX reference. There is therefore no hand-written backward to
compare speed against.

Task: write a Pallas TPU implementation of `workload` (the forward and the
backward kernel are both yours to design) that returns the same gradients.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/experimental/nsa/base.py       (forward reference)
  tokamax/_src/ops/experimental/nsa/arg_specs.py  (shapes and options)
  tokamax/_src/ops/experimental/nsa/base_test.py  (value ranges)
"""

# Dimensions.
#   B   batch size
#   H   number of attention heads
#   T   sequence length
#   D   head dimension
#   W   local causal sliding window size
#   tt  query/KV block tile extent along T (chunk_size)
#   nt  number of query/KV blocks along T (nt = T // tt)
#   tc  compression block tile extent along T (cmp_block_size)
#   nc  number of compressed blocks along T (nc = T // tc)
#   tk  number of selected historical blocks (min(topk, nt))
#   nk  selection block extent per query block (nk = tk + 1)

import json

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/experimental/nsa/base.py)
# ==============================================================================

# Operator options, fixed to the values used by every Tokamax NSA arg spec.
# `T` must be a multiple of both `CHUNK_SIZE` and `CMP_BLOCK_SIZE`.
CHUNK_SIZE = 256  # tt: block size of the selected branch.
TOPK = 4  # Number of earlier blocks each query block attends to.
WINDOW = 128  # W: a query at `t` attends to keys `t - W .. t`.
CMP_BLOCK_SIZE = 64  # tc: block size of the compressed branch.


def _compute_cmp_reference(q, k, v, scale, cmp_block_size=64):
  """Coarse block-compression attention branch."""
  B, H, T, D = q.shape
  tc = cmp_block_size
  nc = T // tc

  k_cmp = jnp.mean(k.reshape(B, H, nc, tc, D), axis=3)
  v_cmp = jnp.mean(v.reshape(B, H, nc, tc, D), axis=3)

  logits = (
      jnp.einsum(
          "bhtd,bhcd->bhtc",
          q,
          k_cmp,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )

  q_pos = jnp.arange(T)[:, None]
  c_pos = (jnp.arange(nc) * tc)[None, :]
  causal_mask = (c_pos + tc - 1) <= q_pos

  logits = jnp.where(causal_mask[None, None], logits, -1e9)
  weights = jax.nn.softmax(logits, axis=-1)
  weights = jnp.where((q_pos >= tc - 1)[None, None], weights, 0.0)
  return jnp.einsum(
      "bhtc,bhcd->bhtd",
      weights,
      v_cmp,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def _compute_slc_reference(q, k, v, scale, topk=4, chunk_size=256):
  """Fine-grained top-k block selection attention branch."""
  B, H, T, D = q.shape
  tt = chunk_size
  nt = T // tt
  tk = min(topk, nt)
  nk = tk + 1

  q_blocks = q.reshape(B, H, nt, tt, D)
  k_blocks = k.reshape(B, H, nt, tt, D)
  v_blocks = v.reshape(B, H, nt, tt, D)

  q_mean = jnp.mean(q_blocks, axis=3)
  k_mean = jnp.mean(k_blocks, axis=3)

  routing_scores = (
      jnp.einsum(
          "bhqd,bhnd->bhqn",
          q_mean,
          k_mean,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )
  q_idx = jnp.arange(nt)[:, None]
  k_idx = jnp.arange(nt)[None, :]
  historical_mask = k_idx < q_idx
  masked_routing = jnp.where(
      historical_mask[None, None], routing_scores, -jnp.inf
  )
  _, topk_idx = jax.lax.top_k(masked_routing, tk)
  topk_idx = jax.lax.stop_gradient(topk_idx)

  def gather_head(x_h, idx_h):
    diag = x_h[:, None, :, :]
    hist = x_h[idx_h]
    return jnp.concatenate([diag, hist], axis=1).reshape(nt, nk * tt, D)

  gather_fn = jax.vmap(jax.vmap(gather_head))
  k_sub = gather_fn(k_blocks, topk_idx)
  v_sub = gather_fn(v_blocks, topk_idx)

  causal_diag = jnp.tril(jnp.ones((tt, tt), dtype=bool))
  ranks = jnp.arange(tk)

  def make_mask_i(q_i):
    valid_hist = jnp.broadcast_to(
        jnp.repeat(ranks < q_i, tt)[None, :],
        (tt, tk * tt),
    )
    return jnp.concatenate([causal_diag, valid_hist], axis=1)

  mask_sub = jax.vmap(make_mask_i)(jnp.arange(nt))

  logits = (
      jnp.einsum(
          "bhqtd,bhqkd->bhqtk",
          q_blocks,
          k_sub,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )
  logits = jnp.where(mask_sub[None, None], logits, -1e9)
  weights = jax.nn.softmax(logits, axis=-1)
  out_blocks = jnp.einsum(
      "bhqtk,bhqkd->bhqtd",
      weights,
      v_sub,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return out_blocks.reshape(B, H, T, D)


def _compute_swa_reference(q, k, v, scale, window=128):
  """Local sliding-window attention branch."""
  _, _, T, _ = q.shape
  logits = (
      jnp.einsum(
          "bhtd,bhsd->bhts",
          q,
          k,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )

  q_pos = jnp.arange(T)[:, None]
  k_pos = jnp.arange(T)[None, :]
  win_mask = (k_pos <= q_pos) & (k_pos >= q_pos - window)

  logits = jnp.where(win_mask[None, None], logits, -1e9)
  weights = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum(
      "bhts,bhsd->bhtd",
      weights,
      v,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def forward(q, k, v, g_cmp, g_slc, g_swa):
  """NSA forward pass: gated sum of three causal attention branches.

  Args:
    q: Queries, `[B, H, T, D]`.
    k: Keys, `[B, H, T, D]`.
    v: Values, `[B, H, T, D]`.
    g_cmp: Gate of the compressed branch, `[B, H, T, D]`.
    g_slc: Gate of the selected branch, `[B, H, T, D]`.
    g_swa: Gate of the sliding-window branch, `[B, H, T, D]`.

  Returns:
    The output, `[B, H, T, D]`, in the dtype of `q`. All branches are computed
    in float32 with a softmax scale of `D ** -0.5`.
  """
  _, _, _, D = q.shape
  orig_dtype = q.dtype
  scale = D**-0.5

  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  v_f32 = v.astype(jnp.float32)

  o_cmp = _compute_cmp_reference(
      q_f32, k_f32, v_f32, scale, cmp_block_size=CMP_BLOCK_SIZE
  )
  o_slc = _compute_slc_reference(
      q_f32, k_f32, v_f32, scale, topk=TOPK, chunk_size=CHUNK_SIZE
  )
  o_swa = _compute_swa_reference(q_f32, k_f32, v_f32, scale, window=WINDOW)

  g_cmp = g_cmp.astype(jnp.float32)
  g_slc = g_slc.astype(jnp.float32)
  g_swa = g_swa.astype(jnp.float32)

  out = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa
  return out.astype(orig_dtype)


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: training_b2_t4096_h8_d64_bf16. Of the Tokamax training
    # arg specs this is the default because the pure-JAX reference builds dense
    # `T x T` tensors per head: XLA plans about 3.3 GiB of temporaries for
    # `workload` at this shape. The memory figures here are XLA estimates from
    # compiling for CPU, not TPU measurements.
    'nsa_b2_t4096': {
        'name': 'nsa_b2_t4096_bwd',
        'model': 'Native-Sparse-Attention',
        'operator': 'nsa_backward',
        'batch': 2,
        'num_heads': 8,
        'seq_len': 4096,
        'head_dim': 64,
    },
    # tokamax arg_specs: training_b1_t8192_h16_d128_bf16. XLA plans about
    # 13.9 GiB of temporaries for `workload` at this shape (CPU estimate).
    'nsa_b1_t8192': {
        'name': 'nsa_b1_t8192_bwd',
        'model': 'Native-Sparse-Attention',
        'operator': 'nsa_backward',
        'batch': 1,
        'num_heads': 16,
        'seq_len': 8192,
        'head_dim': 128,
    },
    # Small shapes for CPU smoke tests. Eight blocks of `CHUNK_SIZE`, so the
    # later query blocks have more than `TOPK` earlier blocks to choose from.
    'smoke': {
        'name': 'nsa_smoke_bwd',
        'model': 'Native-Sparse-Attention',
        'operator': 'nsa_backward',
        'batch': 2,
        'num_heads': 2,
        'seq_len': 2048,
        'head_dim': 16,
    },
}
CONFIG = CONFIGS['nsa_b2_t4096']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(q, k, v, g_cmp, g_slc, g_swa, dout)`; `dout` is the cotangent.

  `q` and `k` carry block structure so that the top-k block selection of the
  selected branch does not depend on rounding. Within a head, every key block
  has a one-hot code along its own dimension as its mean. The mean of a query
  block is the sum of the codes of `TOPK` earlier blocks, chosen at random,
  minus the codes of all other earlier blocks. The routing score of a query
  block is therefore `+D ** -0.5` for the blocks it selects and `-D ** -0.5`
  for every other earlier block. The rest of `q` and `k` is standard-normal
  noise with zero mean within each block.

  With unstructured random `q` and `k` the routing scores are noise, and the
  smallest gap between a selected and a rejected block is smaller than the
  difference between float32 and bfloat16 arithmetic, so two correct
  implementations could select different blocks.

  Requires `T // CHUNK_SIZE <= D`.
  """
  cfg = _resolve_config(config)
  B, H = cfg['batch'], cfg['num_heads']
  T, D = cfg['seq_len'], cfg['head_dim']
  nt = T // CHUNK_SIZE
  if nt > D:
    raise ValueError(f'Need T // CHUNK_SIZE <= D, got {nt} blocks and D={D}.')
  keys = jax.random.split(jax.random.key(42), 9)
  shape = (B, H, T, D)

  def block_noise(key):
    # Standard-normal noise with zero mean within each block of `CHUNK_SIZE`.
    x = jax.random.normal(key, (B, H, nt, CHUNK_SIZE, D))
    return x - jnp.mean(x, axis=3, keepdims=True)

  # One-hot code of each key block, along a random dimension that no other
  # block of the same head uses: [B, H, nt, D].
  dims = jnp.argsort(jax.random.uniform(keys[7], (B, H, D)), axis=-1)[..., :nt]
  codes = jax.nn.one_hot(dims, D)
  # Each query block draws `TOPK` of its earlier blocks (all of them if there
  # are fewer): +1 for the drawn blocks, -1 for the other earlier blocks, 0 for
  # the block itself and later blocks.
  earlier = jnp.arange(nt)[None, :] < jnp.arange(nt)[:, None]
  draw = jax.random.uniform(keys[8], (B, H, nt, nt))
  draw = jnp.where(earlier, draw, -1.0)
  rank = jnp.argsort(jnp.argsort(-draw, axis=-1), axis=-1)
  sign = jnp.where(earlier, jnp.where(rank < TOPK, 1.0, -1.0), 0.0)
  q_mean = jnp.einsum('bhqn,bhnd->bhqd', sign, codes)

  q = (block_noise(keys[0]) + q_mean[:, :, :, None, :]).reshape(shape)
  k = (block_noise(keys[1]) + codes[:, :, :, None, :]).reshape(shape)
  # v and the gates follow the Tokamax NSA tests: standard-normal values and
  # gates in (0, 1).
  v = jax.random.normal(keys[2], shape)
  g_cmp = jax.nn.sigmoid(jax.random.normal(keys[3], shape))
  g_slc = jax.nn.sigmoid(jax.random.normal(keys[4], shape))
  g_swa = jax.nn.sigmoid(jax.random.normal(keys[5], shape))
  # The cotangent is scaled by 2 so that the rms of every gradient is well above
  # the benchmark harness's absolute tolerance of 1e-2. The VJP is linear in the
  # cotangent, so this only rescales the gradients.
  dout = 2.0 * jax.random.normal(keys[6], shape)
  # Sample in float32 and cast, so every dtype sees the same values.
  return tuple(x.astype(dtype) for x in (q, k, v, g_cmp, g_slc, g_swa, dout))


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 6, 'diff_argnums': (0, 1, 2, 3, 4, 5)}


def workload(q, k, v, g_cmp, g_slc, g_swa, dout):
  """Returns `(dq, dk, dv, dg_cmp, dg_slc, dg_swa)`, the VJP of `forward`."""
  _, vjp_fn = jax.vjp(forward, q, k, v, g_cmp, g_slc, g_swa)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward), matmuls only.

  Counts the einsums of `forward` as written, at 2 FLOPs per multiply-add. Each
  branch has a logits einsum and an output einsum of the same size:
    compressed      `T x nc` logits per head:       2 * 2 * T * nc * D
    selected        `T x (nk * tt)` logits per head: 2 * 2 * T * nk * tt * D
    sliding window  `T x T` logits per head:         2 * 2 * T * T * D
  plus the `nt x nt` routing scores of the selected branch (2 * nt * nt * D).
  The reference evaluates the sliding-window branch densely over all `T x T`
  pairs and masks afterwards, so that term dominates; a kernel that only visits
  the window does less work than this count. The VJP runs the forward once and
  a backward that costs about twice the forward, so 3x in total.
  """
  cfg = _resolve_config(config)
  B, H = cfg['batch'], cfg['num_heads']
  T, D = cfg['seq_len'], cfg['head_dim']
  nc = T // CMP_BLOCK_SIZE
  nt = T // CHUNK_SIZE
  nk = min(TOPK, nt) + 1
  fwd = B * H * D * (4 * T * (nc + nk * CHUNK_SIZE + T) + 2 * nt * nt)
  return int(3 * fwd)


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
